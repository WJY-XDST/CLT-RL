"""Drive both legs beyond the geometric stroke with and without stop contacts."""
import argparse
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher
ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--contacts', action='store_true')
parser.add_argument('--protected-tree', action='store_true')
parser.add_argument('--collision-groups', action='store_true')
parser.add_argument('--usd', type=Path)
parser.add_argument('--seconds', type=float, default=12.)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main():
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from omni.physx import get_physx_simulation_interface
    from pxr import PhysicsSchemaTools, UsdPhysics
    from isaaclab.utils.math import quat_apply, quat_mul, quat_conjugate
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.mine_env_cfg import MineWheelLeggedVMCFlatEnvCfg, MODEL
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv

    class BenchEnv(WheelLeggedVMCFlatEnv):
        def _get_dones(self):
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    cfg = MineWheelLeggedVMCFlatEnvCfg()
    cfg.scene.num_envs = 4
    cfg.seed = 43
    cfg.sim.device = args.device or 'cuda:0'
    cfg.leg_control_mode = 'implicit_joint_reference'
    cfg.leg_joint_stiffness, cfg.leg_joint_damping = 300., 3.
    cfg.height_reference_blend = cfg.height_feedback_gain = 0.
    asset = 'mine_closed_chain_training_protected_contacts.usd' if args.protected_tree else 'mine_closed_chain_base_box_stops.usd'
    if args.collision_groups:
        asset = 'mine_closed_chain_training_protected_contacts_groups.usd'
    cfg.robot.spawn.usd_path = str(ROOT / 'wheel_legged_isaaclab/assets/robots/mine/export_20261003_181515' / asset)
    if args.usd:
        cfg.robot.spawn.usd_path = str(args.usd.resolve())
    if args.protected_tree:
        cfg.robot.actuators['passive'].joint_names_expr = ['[LR]L[123]_joint', '[LR]L3_[LR]S_closure']
    cfg.robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
    cfg.robot.spawn.articulation_props.fix_root_link = True
    cfg.robot.spawn.articulation_props.enabled_self_collisions = args.contacts
    cfg.robot.init_state.pos = (0., 0., .8)
    cfg.feedforward_force = 0.
    cfg.reset_velocity_initial = cfg.reset_velocity_final = 0.
    cfg.commands.heading_command = False
    cfg.commands.ranges_lin_vel_x = (0., 0.)
    cfg.commands.ranges_height = (.3, .3)
    cfg.l0_ref_min, cfg.l0_ref_max = .12, .42
    cfg.action_scale_l0, cfg.l0_offset = .15, .27
    print('STOP_BENCH creating environment', flush=True)
    env = BenchEnv(cfg)
    print('STOP_BENCH environment ready', flush=True)
    env.reset()
    robot = env._robot
    contact_rows = []
    timestep = [0.]
    allowed = {frozenset(('LB_link', 'LS_link')), frozenset(('RB_link', 'RS_link'))}
    def on_contact(headers, data):
        for header in headers:
            a = str(PhysicsSchemaTools.intToSdfPath(header.actor0))
            b = str(PhysicsSchemaTools.intToSdfPath(header.actor1))
            pair = frozenset((a.rsplit('/', 1)[-1], b.rsplit('/', 1)[-1]))
            for index in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data):
                contact = data[index]
                contact_rows.append([timestep[0], a, b, pair in allowed, float(contact.separation),
                                     float(np.linalg.norm(contact.impulse)),
                                     str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                                     str(PhysicsSchemaTools.intToSdfPath(header.collider1))])
    subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    closures = []
    for prim in env.sim.stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
            eid = int(str(prim.GetPath()).split('/env_')[1].split('/')[0])
            frames = []
            joint = UsdPhysics.Joint(prim)
            for side in (0, 1):
                target = getattr(joint, f'GetBody{side}Rel')().GetTargets()[0]
                bid = robot.body_names.index(target.name)
                pos = torch.tensor(list(getattr(joint, f'GetLocalPos{side}Attr')().Get()), device=env.device, dtype=torch.float64)
                frames.append((bid, pos))
            closures.append((eid, frames))
    goals = torch.tensor([.12, .18, .36, .42], device=env.device)
    rows = []
    protected_body_names = ['LB_link', 'LS_link', 'RB_link', 'RS_link']
    protected_body_ids = [robot.body_names.index(name) for name in protected_body_names]
    protected_poses = []
    maximum_gap = 0.
    knee_frames = []
    for name in ('LS_joint', 'RS_joint'):
        joint = UsdPhysics.Joint(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/joints/' + name))
        frames = []
        for side in (0, 1):
            target = getattr(joint, f'GetBody{side}Rel')().GetTargets()[0]
            rot = getattr(joint, f'GetLocalRot{side}Attr')().Get()
            frames.append((robot.body_names.index(target.name), torch.tensor([rot.GetReal(), *rot.GetImaginary()], device=env.device, dtype=torch.float64)))
        knee_frames.append(frames)
    try:
        for step in range(round(args.seconds / env.step_dt)):
            t = step * env.step_dt
            if step % 100 == 0:
                print(f'STOP_BENCH step={step} time={t:.2f}s contacts={len(contact_rows)}', flush=True)
            timestep[0] = t
            ramp = min(max((t - 2.) / 5., 0.), 1.)
            target = .27 + ramp * (goals - .27)
            action = torch.zeros((4, 6), device=env.device)
            action[:, (0, 3)] = .052219456 / cfg.action_scale_theta
            action[:, (1, 4)] = ((target - cfg.l0_offset) / cfg.action_scale_l0)[:, None]
            env.step(action)
            env._update_forward_kinematics()
            # The all-body sensor supports net forces, not many-to-many
            # filtered contacts. Contact callbacks remain available on CPU;
            # absence of callback points on GPU is recorded as unknown depth.
            if not torch.isfinite(robot.data.joint_pos).all() or not env._five_bar_valid.all():
                raise ValueError('Invalid mechanism state during stop sweep')
            poses = robot.root_physx_view.get_link_transforms().double()
            protected_poses.append(poses[:, protected_body_ids].cpu().numpy())
            quat = poses[..., (6, 3, 4, 5)]
            quat = quat / torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
            knee_angles = []
            for (bid0, rot0), (bid1, rot1) in knee_frames:
                q0 = quat_mul(quat[:, bid0], rot0.expand(4, -1))
                q1 = quat_mul(quat[:, bid1], rot1.expand(4, -1))
                relative = quat_mul(quat_conjugate(q0), q1)
                knee_angles.append(2 * torch.atan2(relative[:, 3], relative[:, 0]))
            gaps = []
            for eid, frames in closures:
                endpoints = [poses[eid, bid, :3] + quat_apply(quat[eid, bid], pos) for bid, pos in frames]
                gaps.append(float(torch.linalg.vector_norm(endpoints[0] - endpoints[1])))
            maximum_gap = max(maximum_gap, max(gaps))
            forces = env._contact_sensor.data.net_forces_w
            bids = [robot.body_names.index(n) for n in ('LB_link', 'LS_link', 'RB_link', 'RS_link')]
            peak_force = torch.linalg.vector_norm(forces[:, bids], dim=-1).max(dim=-1).values
            for eid in range(4):
                rows.append([t, eid, float(goals[eid]), float(target[eid]),
                             *env._L0[eid].tolist(), *[float(q[eid]) for q in knee_angles],
                             float(peak_force[eid]), float(robot.data.applied_torque[eid].abs().max()), max(gaps)])
        args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output/'protected_cad_poses.npz', poses=np.stack(protected_poses),
                            body_names=protected_body_names, policy_dt=env.step_dt)
        with (args.output / 'trace.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s', 'env', 'goal_m', 'reference_m', 'left_length_m', 'right_length_m',
                             'LS_angle_rad', 'RS_angle_rad', 'peak_contact_force_N', 'peak_effort_Nm', 'closure_gap_m'])
            writer.writerows(rows)
        with (args.output / 'contacts.csv').open('w') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s', 'actor0', 'actor1', 'allowed_pair', 'separation_m', 'impulse_Ns', 'collider0', 'collider1'])
            writer.writerows(contact_rows)
        trials = []
        for eid in range(4):
            tail = np.array([r for r in rows if r[1] == eid and r[0] >= args.seconds - 2.])
            trials.append({'goal_m': float(goals[eid]), 'left_steady_m': float(tail[:, 4].mean()),
                           'right_steady_m': float(tail[:, 5].mean()),
                           'steady_peak_contact_force_N': float(tail[:, 8].max()),
                           'steady_LS_rad': float(tail[:, 6].mean()), 'steady_RS_rad': float(tail[:, 7].mean())})
        separation = [r[4] for r in contact_rows if r[3]]
        report = {'contacts_enabled': args.contacts, 'trials': trials, 'contact_points_reported': len(contact_rows),
                  'asset': cfg.robot.spawn.usd_path, 'protected_tree': args.protected_tree,
                  'maximum_contact_penetration_mm': max(0., -min(separation)) * 1000 if separation else None,
                  'unwanted_contact_points': sum(not r[3] for r in contact_rows), 'maximum_closure_gap_mm': maximum_gap * 1000,
                  'scope': 'Fixed base, zero gravity; deliberate overstroke with 300/3 force Drive and 10 Nm limit. Not a policy replay.'}
        (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print('STOP_REPORT ' + json.dumps(report), flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
        subscription = None
        env.close()


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        import traceback
        import faulthandler
        traceback.print_exc()
        faulthandler.cancel_dump_traceback_later()
        app.close(skip_cleanup=True)
        raise
    finally:
        app.close()
