"""Standing-policy ablations with physical FK and pre-reset contact telemetry.

30 Nm cases are diagnostic only: they never change the model's production limits.
"""
import argparse
import csv
import json
from pathlib import Path
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--seconds', type=float, default=5.)
p.add_argument('--force-probe', action='store_true', help='Fixed elevated base, no gravity, independent effort impulses')
p.add_argument('--gain-sweep', action='store_true')
p.add_argument('--stiffness-sweep', action='store_true')
p.add_argument('--coupled-gain-sweep', action='store_true')
p.add_argument('--balance-feedback', action='store_true', help='Diagnostic wheel pitch feedback, not a learned policy')
p.add_argument('--wheel-frame-compensation', action='store_true')
p.add_argument('--wheel-drive-test', action='store_true')
p.add_argument('--physics-dt', type=float, default=.005)
p.add_argument('--upright-reset', action='store_true', help='Rotate both coaxial motors equally so the virtual leg starts vertical')
p.add_argument('--vertical-guide', action='store_true', help='Diagnostic rail fixes chassis attitude and horizontal position while allowing vertical support motion')
p.add_argument('--ground-force-probe', action='store_true', help='Fix chassis at normal ground height without gravity and measure transmitted wheel support force')
p.add_argument('--force-leg-drive', action='store_true', help='Diagnostic zero-gain force Drive on active leg motors')
p.add_argument('--solver-sweep', action='store_true', help='Compare external-loop constraint convergence with zero actions and production torque limits')
p.add_argument('--contact-last', action='store_true')
p.add_argument('--finite-difference-vmc', action='store_true')
p.add_argument('--support-force-sweep', action='store_true', help='Diagnostic constant radial force, 30 Nm capacity, with angle gains 0/10/50')
p.add_argument('--free-wheels', action='store_true', help='Remove wheel motor torque to isolate support from velocity control')
p.add_argument('--joint-hold', action='store_true', help='Diagnostic implicit position hold at CAD q=0 with 10/30/100 Nm limits')
p.add_argument('--explicit-joint-hold', action='store_true', help='Same joint hold as explicit torques, isolating drive integration')
p.add_argument('--implicit-vmc-sweep', action='store_true', help='Diagnostic inverse virtual-leg references with implicit joint drives')
p.add_argument('--implicit-balance', action='store_true', help='Diagnostic wheel pitch feedback alongside implicit inverse leg references')
p.add_argument('--balance-integral', type=float, default=0., help='Diagnostic pitch integral gain in wheel velocity reference')
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app

def main():
    import torch
    import yaml
    from rsl_rl.modules import ActorCritic
    from isaaclab.utils.math import quat_apply_inverse
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.mine_env_cfg import MineWheelLeggedVMCFlatEnvCfg, MODEL
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv

    names = ['policy_10Nm', 'zero_10Nm', 'zero_30Nm', 'policy_30Nm', 'policy_angle_zero_10Nm']
    if args.force_probe:
        names = ['B_plus_1Nm', 'L_plus_1Nm', 'radial_plus_47N', 'angular_plus_5Nm', 'zero_effort']
    gains = [(kt,kl) for kt in (.1,.3,1.,3.) for kl in (10.,30.,90.)]
    if args.gain_sweep: names = [f'kd_angle_{kt}_kd_length_{kl}' for kt,kl in gains]
    stiffness = [(kp,kd) for kp in (0.,5.,10.,20.,50.) for kd in (.1,.5,1.,3.)]
    if args.stiffness_sweep: names = [f'kp_angle_{kp}_kd_angle_{kd}' for kp,kd in stiffness]
    coupled = [(kp,kd) for kp in (5.,10.,20.,50.) for kd in (0.,5.,10.,30.,90.,180.,300.)]
    if args.coupled_gain_sweep: names = [f'kp_angle_{kp}_kd_length_{kd}' for kp,kd in coupled]
    balance_gains = [(kp,kd) for kp in (2.,4.,8.) for kd in (.3,.6,1.)]
    if args.balance_feedback: names = [f'wheel_pitch_kp_{kp}_kd_{kd}' for kp,kd in balance_gains]
    if args.wheel_drive_test: names = ['wheel_negative', 'wheel_zero', 'wheel_positive', 'wheel_fast', 'wheel_policy']
    solver_counts = [(64,8),(64,64),(64,128),(8,64),(8,128),(4,64),(4,128),(128,64),(128,128)]
    if args.solver_sweep: names = [f'position_{pos}_velocity_{vel}' for pos,vel in solver_counts]
    support_forces = [(f,kp) for f in (0.,47.,100.,200.) for kp in (0.,10.,50.)]
    if args.support_force_sweep: names = [f'force_{f}_kp_angle_{kp}' for f,kp in support_forces]
    if args.joint_hold or args.explicit_joint_hold: names = ['joint_hold_10Nm','joint_hold_30Nm','joint_hold_100Nm','joint_hold_10Nm_lowgain','joint_hold_30Nm_lowgain']
    implicit_gains=[(100.,1.),(100.,3.),(100.,10.),(300.,1.),(300.,3.),(300.,10.),(1000.,3.),(1000.,10.)]
    if args.implicit_vmc_sweep: names=[f'implicit_kp_{kp}_kd_{kd}' for kp,kd in implicit_gains]
    cfg = MineWheelLeggedVMCFlatEnvCfg()
    cfg.scene.num_envs = len(names)
    if args.vertical_guide or args.solver_sweep or args.implicit_vmc_sweep: cfg.scene.replicate_physics = False
    if args.solver_sweep: cfg.kp_theta, cfg.kd_theta = 10., .5
    cfg.sim.physx.solve_articulation_contact_last = args.contact_last
    cfg.sim.dt = args.physics_dt
    cfg.decimation = round(.02/args.physics_dt)
    cfg.seed = 43
    if args.free_wheels:
        cfg.wheel_damping = 0.
        cfg.robot.actuators['wheels'].damping = 0.
        cfg.robot.actuators['wheels'].effort_limit_sim = 0.
    cfg.reset_velocity_initial = cfg.reset_velocity_final = 0.
    cfg.commands.ranges_lin_vel_x = (0., 0.)
    cfg.commands.ranges_height = (.30, .30)
    cfg.commands.ranges_ang_vel_yaw = (0., 0.)
    cfg.commands.heading_command = False
    cfg.commands.grouped_training = False
    cfg.episode_length_s = args.seconds + 2
    if args.upright_reset:
        angle = .05221904
        cfg.robot.init_state.joint_pos = {'LB_joint':-angle, 'LL_joint':-angle,
                                         'RB_joint':angle, 'RL_joint':angle}
        cfg.robot.init_state.pos = (0.,0.,MODEL['nominal_leg_length']-MODEL['hip_z']+MODEL['wheel_radius'])
    # A common actuator capacity; per-env control clamps enforce the ablation.
    capacity_ablation = not (args.gain_sweep or args.stiffness_sweep or args.coupled_gain_sweep or args.balance_feedback or args.force_probe or args.wheel_drive_test or args.solver_sweep or args.implicit_vmc_sweep)
    cfg.robot.actuators['legs'].effort_limit_sim = 30. if capacity_ablation else 10.
    if args.joint_hold or args.implicit_vmc_sweep:
        cfg.robot.actuators['legs'].stiffness=1000.
        cfg.robot.actuators['legs'].damping=10.
        cfg.robot.actuators['legs'].effort_limit_sim=100. if args.joint_hold else 10.
    if args.force_probe:
        import isaaclab.sim as sim_utils
        cfg.robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
        cfg.robot.spawn.articulation_props.fix_root_link = True
        cfg.robot.init_state.pos = (0., 0., .65)
    if args.ground_force_probe:
        import isaaclab.sim as sim_utils
        cfg.robot.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
        cfg.robot.spawn.articulation_props.fix_root_link = True

    class Env(WheelLeggedVMCFlatEnv):
        def _update_forward_kinematics(self, env_ids=None):
            super()._update_forward_kinematics(env_ids)
            if args.finite_difference_vmc:
                counter=self._sim_step_counter
                if env_ids is not None or not hasattr(self,'fd_counter'):
                    self.fd_counter=counter
                    self.fd_length=self._L0.clone(); self.fd_angle=self._theta0.clone()
                    self.fd_ldot=torch.zeros_like(self._L0); self.fd_tdot=self.fd_ldot.clone()
                elif counter != self.fd_counter:
                    dt=(counter-self.fd_counter)*self.physics_dt
                    self.fd_ldot=(self._L0-self.fd_length)/dt
                    delta=self._theta0-self.fd_angle
                    self.fd_tdot=torch.atan2(delta.sin(),delta.cos())/dt
                    self.fd_counter=counter
                    self.fd_length=self._L0.clone(); self.fd_angle=self._theta0.clone()
                self._L0_dot[:]=self.fd_ldot;self._theta0_dot[:]=self.fd_tdot

        def _setup_scene(self):
            super()._setup_scene()
            if args.solver_sweep:
                from pxr import Usd, PhysxSchema
                for i,(pos,vel) in enumerate(solver_counts):
                    root=self.sim.stage.GetPrimAtPath(f'/World/envs/env_{i}/Robot')
                    found=False
                    for prim in Usd.PrimRange(root):
                        if prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
                            api=PhysxSchema.PhysxArticulationAPI(prim)
                            api.CreateSolverPositionIterationCountAttr(pos)
                            api.CreateSolverVelocityIterationCountAttr(vel)
                            found=True
                    if not found: raise RuntimeError(f'No articulation solver API in {root.GetPath()}')
            if args.force_leg_drive or args.joint_hold or args.implicit_vmc_sweep:
                from pxr import UsdPhysics
                for i in range(self.num_envs):
                    for name in cfg.leg_joint_names:
                        prim=self.sim.stage.GetPrimAtPath(f'/World/envs/env_{i}/Robot/joints/{name}')
                        drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
                        drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(cfg.robot.actuators['legs'].stiffness)
                        drive.CreateDampingAttr(cfg.robot.actuators['legs'].damping);drive.CreateMaxForceAttr(cfg.robot.actuators['legs'].effort_limit_sim)
            if args.vertical_guide:
                from pxr import UsdPhysics, Gf
                for i,origin in enumerate(self._terrain.env_origins.cpu().tolist()):
                    path=f'/World/envs/env_{i}/Robot'
                    joint=UsdPhysics.PrismaticJoint.Define(self.sim.stage,path+'/diagnostic_vertical_guide')
                    joint.CreateBody1Rel().SetTargets([path+'/base_link'])
                    joint.CreateAxisAttr('Z')
                    joint.CreateLocalPos0Attr(Gf.Vec3f(*[origin[j]+cfg.robot.init_state.pos[j] for j in range(3)]))
                    joint.CreateLocalPos1Attr(Gf.Vec3f(0.,0.,0.))
                    joint.CreateExcludeFromArticulationAttr(True)
                    joint.CreateCollisionEnabledAttr(False)

        def _apply_action(self):
            super()._apply_action()
            if args.implicit_vmc_sweep:
                from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_inverse
                theta_ref=self._actions[:,(0,3)]*cfg.action_scale_theta
                b,l,valid=five_bar_inverse(self._l0_ref_applied,theta_ref,**cfg.five_bar_geometry)
                if not valid.all(): raise RuntimeError('Invalid inverse reference')
                target=torch.stack((b[:,0],l[:,0],b[:,1],l[:,1]),-1)
                self._robot.set_joint_effort_target(torch.zeros_like(self._torque_limits),joint_ids=self._torque_joint_ids)
                self._robot.set_joint_position_target(target,joint_ids=self._leg_joint_ids)
            if args.joint_hold:
                self._robot.set_joint_effort_target(torch.zeros_like(self._torque_limits),joint_ids=self._torque_joint_ids)
                self._robot.set_joint_position_target(torch.zeros(self.num_envs,4,device=self.device),joint_ids=self._leg_joint_ids)
            if args.explicit_joint_hold:
                q=self._robot.data.joint_pos[:,self._leg_joint_ids]
                dq=self._robot.data.joint_vel[:,self._leg_joint_ids]
                kp=q.new_tensor([1000.,1000.,1000.,100.,100.])[:,None]
                kd=q.new_tensor([10.,10.,10.,1.,1.])[:,None]
                limit=q.new_tensor([10.,30.,100.,10.,30.])[:,None]
                tau=(-kp*q-kd*dq).clamp(-limit,limit)
                efforts=torch.stack((tau[:,0],tau[:,1],torch.zeros_like(tau[:,0]),tau[:,2],tau[:,3],torch.zeros_like(tau[:,0])),-1)
                self._robot.set_joint_effort_target(efforts,joint_ids=self._torque_joint_ids)
            if args.support_force_sweep:
                from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_torques
                force=self._L0.new_tensor([f for f,kp in support_forces])[:,None].expand(-1,2)
                kp=self._L0.new_tensor([kp for f,kp in support_forces])[:,None]
                angular=-kp*self._theta0-.5*self._theta0_dot
                b,l=five_bar_torques(self._five_bar_jacobian,force,angular)
                efforts=torch.stack((b[:,0],l[:,0],torch.zeros_like(b[:,0]),b[:,1],l[:,1],torch.zeros_like(b[:,1])),-1).clamp(-30,30)
                self._robot.set_joint_effort_target(efforts,joint_ids=self._torque_joint_ids)
            if args.force_probe:
                efforts = torch.zeros(self.num_envs, 6, device=self.device)
                efforts[0,0] = 1.; efforts[1,1] = 1.
                from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_torques
                f = torch.zeros(self.num_envs, 2, device=self.device); t = f.clone()
                f[2,0] = 47.; t[3,0] = 5.
                b,l = five_bar_torques(self._five_bar_jacobian,f,t)
                efforts[2:4,0] = b[2:4,0]; efforts[2:4,1] = l[2:4,0]
                self._robot.set_joint_effort_target(efforts, joint_ids=self._torque_joint_ids)

        def _get_dones(self):
            died, timeout = super()._get_dones()
            self.raw_died = died.clone()
            # Keep measuring after the first unsafe event without reset artifacts.
            return self._termination_reasons['numerical'], timeout

    env = Env(cfg)
    if args.implicit_vmc_sweep:
        property_ids=env._robot._ALL_INDICES.cpu()
        stiffness=env._robot.root_physx_view.get_dof_stiffnesses().clone()
        damping=env._robot.root_physx_view.get_dof_dampings().clone()
        stiffness[:,env._leg_joint_ids]=stiffness.new_tensor([g[0] for g in implicit_gains])[:,None]
        damping[:,env._leg_joint_ids]=damping.new_tensor([g[1] for g in implicit_gains])[:,None]
        env._robot.root_physx_view.set_dof_stiffnesses(stiffness,property_ids)
        env._robot.root_physx_view.set_dof_dampings(damping,property_ids)
    if args.explicit_joint_hold:
        limits=env._robot.root_physx_view.get_dof_max_forces().clone()
        limits[:,env._leg_joint_ids]=limits.new_tensor([10.,30.,100.,10.,30.])[:,None]
        env._robot.root_physx_view.set_dof_max_forces(limits,env._robot._ALL_INDICES.cpu())
    if args.joint_hold:
        property_ids=env._robot._ALL_INDICES.cpu()
        limits=env._robot.root_physx_view.get_dof_max_forces().clone()
        limits[:,env._leg_joint_ids]=limits.new_tensor([10.,30.,100.,10.,30.])[:,None]
        env._robot.root_physx_view.set_dof_max_forces(limits,property_ids)
        stiffness=env._robot.root_physx_view.get_dof_stiffnesses().clone()
        damping=env._robot.root_physx_view.get_dof_dampings().clone()
        stiffness[3:,env._leg_joint_ids]=100.
        damping[3:,env._leg_joint_ids]=1.
        env._robot.root_physx_view.set_dof_stiffnesses(stiffness,property_ids)
        env._robot.root_physx_view.set_dof_dampings(damping,property_ids)
    if args.support_force_sweep: env._torque_limits[:,(0,1,3,4)] = 30.
    if capacity_ablation:
        env._torque_limits[2:4, (0, 1, 3, 4)] = 30.
    obs, _ = env.reset()
    if args.gain_sweep:
        cfg.kd_theta = torch.tensor([g[0] for g in gains],device=env.device)[:,None]
        cfg.kd_l0 = torch.tensor([g[1] for g in gains],device=env.device)[:,None]
    if args.stiffness_sweep:
        cfg.kp_theta = torch.tensor([g[0] for g in stiffness],device=env.device)[:,None]
        cfg.kd_theta = torch.tensor([g[1] for g in stiffness],device=env.device)[:,None]
    if args.coupled_gain_sweep:
        cfg.kp_theta = torch.tensor([g[0] for g in coupled],device=env.device)[:,None]
        cfg.kd_theta = .5
        cfg.kd_l0 = torch.tensor([g[1] for g in coupled],device=env.device)[:,None]
    agent = yaml.safe_load((args.checkpoint.parent / 'params/agent.yaml').read_text())
    policy_cfg = dict(agent['policy']); policy_cfg.pop('class_name')
    actor = ActorCritic(obs, agent['obs_groups'], 6, **policy_cfg).to(env.device)
    actor.load_state_dict(torch.load(args.checkpoint, map_location=env.device, weights_only=True)['model_state_dict'], strict=True)
    actor.eval()
    wheel_ids = [env._robot.body_names.index(n) for n in ('LW_link', 'RW_link')]
    wheel_contact_ids, _ = env._contact_sensor.find_bodies(['LW_link','RW_link'], preserve_order=True)
    rs_ids = [env._robot.body_names.index(n) for n in ('LS_link','RS_link')]
    fields = ['time_s', 'case', 'height_m', 'pitch_rad', 'roll_rad', 'speed_m_s',
              'left_length_m', 'right_length_m', 'left_angle_rad', 'right_angle_rad',
              'left_ref_m', 'right_ref_m', 'left_angle_ref_rad', 'right_angle_ref_rad',
              'physical_left_angle_rad', 'physical_right_angle_rad', 'fk_error_mm',
              'base_contact_N', 'max_leg_contact_N', 'raw_failure']
    fields += ['q_' + n for n in env._robot.joint_names]
    fields += ['tau_' + env._robot.joint_names[i] for i in env._torque_joint_ids]
    fields += ['raw_action_' + str(i) for i in range(6)]
    fields += ['reward_' + k for k in env._episode_sums]
    fields += ['left_wheel_contact_N','right_wheel_contact_N']
    fields += ['dq_' + n for n in env._robot.joint_names]
    fields += ['left_wheel_contact_x_N','right_wheel_contact_x_N','left_wheel_world_rate','right_wheel_world_rate','left_carrier_world_rate','right_carrier_world_rate']
    fields += ['measured_tau_'+env._robot.joint_names[i] for i in env._torque_joint_ids]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'manifest.json').write_text(json.dumps({'checkpoint': str(args.checkpoint), 'cases': names, 'reset_velocity': 0., 'diagnostic_only_30Nm': True, 'cfg': cfg.to_dict(), 'contact_body_names':env._contact_sensor.body_names,'wheel_contact_ids':wheel_contact_ids, 'physx_max_forces':env._robot.root_physx_view.get_dof_max_forces().tolist(), 'physx_damping':env._robot.root_physx_view.get_dof_dampings().tolist()}, indent=2, default=str))
    reports = {name: {'first_failure_s': None, 'min_height_m': 1., 'max_fk_error_mm': 0.} for name in names}
    pitch_integral=torch.zeros(env.num_envs,device=env.device)
    with (args.output / 'trace.csv').open('w', newline='') as stream, torch.inference_mode():
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for step in range(round(args.seconds / env.step_dt)):
            actions = actor.act_inference(obs).clone()
            actions[1:3] = 0.
            actions[4, (0, 3)] = 0.
            if args.force_probe or args.gain_sweep or args.stiffness_sweep or args.coupled_gain_sweep or args.balance_feedback or args.solver_sweep or args.support_force_sweep or args.implicit_vmc_sweep: actions[:] = 0.
            if args.implicit_balance:
                pitch=torch.asin(env._robot.data.projected_gravity_b[:,0].clamp(-1,1))
                pitch_integral=(pitch_integral+pitch*env.step_dt).clamp(-.1,.1)
                velocity=4.*pitch+.6*env._robot.data.root_ang_vel_b[:,1]+args.balance_integral*pitch_integral
                actions[:,(2,5)]=(velocity/cfg.wheel_radius/cfg.action_scale_vel)[:,None]
            if args.wheel_drive_test:
                actions[:4] = 0.
                actions[:4,(2,5)] = actions.new_tensor([-.5,0.,.5,1.])[:,None]
            if args.balance_feedback:
                pitch = torch.asin(env._robot.data.projected_gravity_b[:,0].clamp(-1,1))
                kp = pitch.new_tensor([g[0] for g in balance_gains])
                kd = pitch.new_tensor([g[1] for g in balance_gains])
                velocity = kp*pitch+kd*env._robot.data.root_ang_vel_b[:,1]
                actions[:,(2,5)] = (velocity/cfg.wheel_radius/cfg.action_scale_vel)[:,None]
                if args.wheel_frame_compensation:
                    from isaaclab.utils.math import quat_apply
                    axis = torch.zeros(len(names),3,device=env.device); axis[:,1] = 1.
                    axis = quat_apply(env._robot.data.root_quat_w,axis)
                    carrier_rate = (env._robot.data.body_ang_vel_w[:,rs_ids]*axis[:,None]).sum(-1)
                    actions[:,(2,5)] -= carrier_rate/cfg.action_scale_vel
            obs, *_ = env.step(actions)
            env._update_forward_kinematics()
            d = env._robot.data
            poses = env._robot.root_physx_view.get_link_transforms()
            rel = poses[:, wheel_ids, :3] - d.root_pos_w[:, None, :]
            rel = quat_apply_inverse(d.root_quat_w[:, None, :].expand(-1, 2, -1), rel)
            actual = rel[..., (0, 2)].clone(); actual[..., 1] -= MODEL['hip_z']
            predicted = torch.stack((-env._L0 * env._theta0.sin(), -env._L0 * env._theta0.cos()), -1)
            fk = torch.linalg.vector_norm(actual - predicted, dim=-1).amax(-1) * 1000
            angle = torch.atan2(-actual[..., 0], -actual[..., 1])
            forces = env._contact_sensor.data.net_forces_w
            base = torch.linalg.vector_norm(forces[:, env._base_id], dim=-1).amax(-1)
            legs = torch.linalg.vector_norm(forces[:, env._leg_contact_ids], dim=-1).amax(-1)
            for i, name in enumerate(names):
                row = dict(zip(fields[:20], [(step + 1)*env.step_dt, name, d.root_pos_w[i, 2].item(),
                    torch.asin(d.projected_gravity_b[i, 0].clamp(-1,1)).item(),
                    torch.atan2(-d.projected_gravity_b[i,1],-d.projected_gravity_b[i,2]).item(),
                    env._get_heading_frame_horizontal_velocity()[i,0].item(),
                    *env._L0[i].tolist(), *env._theta0[i].tolist(), *env._l0_ref_applied[i].tolist(),
                    *(env._actions[i,(0,3)]*cfg.action_scale_theta).tolist(), *angle[i].tolist(),
                    fk[i].item(), base[i].item(), legs[i].item(), int(env.raw_died[i].item())]))
                row.update({'q_'+n: d.joint_pos[i,j].item() for j,n in enumerate(env._robot.joint_names)})
                row.update({'tau_'+env._robot.joint_names[j]: d.applied_torque[i,j].item() for j in env._torque_joint_ids})
                measured=env._robot.root_physx_view.get_dof_projected_joint_forces()
                row.update({'measured_tau_'+env._robot.joint_names[j]:measured[i,j].item() for j in env._torque_joint_ids})
                row.update({'raw_action_'+str(j): actions[i,j].item() for j in range(6)})
                row.update({'reward_'+k: v[i].item() for k,v in env._last_reward_terms.items()})
                wheel_forces = torch.linalg.vector_norm(forces[i,wheel_contact_ids],dim=-1)
                row.update(dict(zip(('left_wheel_contact_N','right_wheel_contact_N'),wheel_forces.tolist())))
                row.update({'dq_'+n:d.joint_vel[i,j].item() for j,n in enumerate(env._robot.joint_names)})
                from isaaclab.utils.math import quat_apply
                axis = torch.tensor([0.,1.,0.],device=env.device)
                axis = quat_apply(d.root_quat_w[i],axis)
                row.update(dict(zip(('left_wheel_contact_x_N','right_wheel_contact_x_N'),forces[i,wheel_contact_ids,0].tolist())))
                row.update(dict(zip(('left_wheel_world_rate','right_wheel_world_rate'),(d.body_ang_vel_w[i,wheel_ids]*axis).sum(-1).tolist())))
                row.update(dict(zip(('left_carrier_world_rate','right_carrier_world_rate'),(d.body_ang_vel_w[i,rs_ids]*axis).sum(-1).tolist())))
                writer.writerow(row)
                report=reports[name]
                report['min_height_m']=min(report['min_height_m'],row['height_m'])
                report['max_fk_error_mm']=max(report['max_fk_error_mm'],row['fk_error_mm'])
                if row['raw_failure'] and report['first_failure_s'] is None: report['first_failure_s']=row['time_s']
            if step % 50 == 0:
                stream.flush(); print('DIAG_PROGRESS',step,flush=True)
    (args.output / 'report.json').write_text(json.dumps(reports, indent=2))
    print('BALANCE_DIAGNOSIS',json.dumps(reports),flush=True)
    env.close()

try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
