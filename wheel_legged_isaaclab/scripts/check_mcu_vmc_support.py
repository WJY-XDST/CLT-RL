"""Read-only simulator diagnostic: physical wheel anchors versus five-bar FK.

Only temporary simulations and requested result files are created. Training weights,
project configuration, permanent USD assets and hardware are not modified.
"""
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--overrides', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--mode', choices=['free', 'vertical_guide', 'slider_guide', 'hanging', 'pulse', 'fixed_step'], default='free')
parser.add_argument('--seconds', type=float, default=1.0)
parser.add_argument('--physics-hz', type=int, default=200)
parser.add_argument('--control-hz', type=int, default=None,
                    help='Hold VMC torques between updates; defaults to physics-hz for historical diagnostics')
parser.add_argument('--position-iterations', type=int, default=None)
parser.add_argument('--length-reference', type=float, default=None,
                    help='Fixed virtual length in m, bypassing height blending/feedback for a controlled experiment')
parser.add_argument('--length-step', type=float, default=0., help='Small fixed-reference step in m')
parser.add_argument('--angle-step', type=float, default=0., help='Small fixed-reference step in rad')
parser.add_argument('--step-time', type=float, default=2., help='Reference step time on the simulation clock')
parser.add_argument('--freefall-probe', action='store_true',
                    help='Separate collision-free 1 kg body; checks gravity without altering the robot')
parser.add_argument('--zero-gravity', action='store_true',
                    help='Isolate closed-chain actuator dynamics without weight or gravity feedforward')
parser.add_argument('--stop-on-failure', action='store_true', help='Do not continue collecting reset episodes')
parser.add_argument('--gain-set', choices=['original', 'joint_pd_linearized', 'firmware_length'], default='original')
parser.add_argument('--execution', choices=['vmc'], default='vmc',
                    help='VMC support diagnosis; direct motor-target tests use check_direct_joint_pd.py')
parser.add_argument('--hardware-case', choices=['legacy', 'limits', 'gravity', 'combined'], default='combined',
                    help='Single-variable ablation of MCU caps / gravity; never sends hardware commands')
parser.add_argument('--case-index', type=int, default=0,
                    help='Run one case only, avoiding world-joint clone ambiguity')
parser.add_argument('--wheel-mode', choices=['configured', 'disabled', 'external'], default='configured',
                    help='Diagnostic wheel-drive ablation; does not change the training configuration')
parser.add_argument('--virtual-force', type=float, default=None,
                    help='Optional constant outward force in N per leg, bypassing virtual length PD')
parser.add_argument('--virtual-torque', type=float, default=None,
                    help='Optional constant virtual swing torque in Nm per leg')
parser.add_argument('--angle-reference', type=float, default=0.,
                    help='Fixed virtual swing-angle reference in rad for this diagnostic only')
parser.add_argument('--solver', choices=['configured', 'pgs', 'tgs'], default='configured')
parser.add_argument('--velocity-iterations', type=int, default=None)
parser.add_argument('--external-forces-every-iteration', action=argparse.BooleanOptionalAction, default=None)
parser.add_argument('--joint-speed-limit', type=float, default=None,
                    help='Diagnostic speed cap for all non-wheel DOFs, in rad/s; not a hardware calibration')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not math.isfinite(args.seconds) or args.seconds <= 0:
    parser.error('seconds must be positive and finite')
if not math.isfinite(args.angle_reference) or abs(args.angle_reference) > .20:
    parser.error('angle-reference must be finite and inside [-0.20, 0.20] rad')
if any(value is not None and not math.isfinite(value)
       for value in (args.virtual_force, args.virtual_torque)):
    parser.error('virtual force/torque must be finite')
if args.velocity_iterations is not None and not 0 <= args.velocity_iterations <= 32:
    parser.error('velocity-iterations must be inside [0, 32]')
if args.position_iterations is not None and not 1 <= args.position_iterations <= 255:
    parser.error('position-iterations must be inside [1, 255]')
from wheel_legged_gym_isaaclab.diagnostic_timing import control_stride, is_control_update
try:
    args.control_hz = args.physics_hz if args.control_hz is None else args.control_hz
    vmc_stride = control_stride(args.physics_hz, args.control_hz)
except ValueError as error:
    parser.error(str(error))
if any(not math.isfinite(x) for x in (args.length_step, args.angle_step, args.step_time)) or args.step_time < 0:
    parser.error('Step parameters must be finite and step-time non-negative')
if abs(args.angle_reference + args.angle_step) > .20:
    parser.error('Stepped angle reference must remain inside [-0.20, 0.20] rad')
if args.length_reference is not None and (not math.isfinite(args.length_reference) or args.length_reference <= 0):
    parser.error('length-reference must be positive and finite')
if args.length_step and args.length_reference is None:
    parser.error('length-step requires an explicit length-reference')
if args.joint_speed_limit is not None and (
        not math.isfinite(args.joint_speed_limit) or args.joint_speed_limit <= 0):
    parser.error('joint-speed-limit must be positive and finite')
app = AppLauncher(args).app


def main():
    import torch
    from pxr import Gf, Usd, UsdPhysics, PhysxSchema
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.utils.math import quat_apply, quat_apply_inverse
    from wheel_legged_gym_isaaclab.config_overrides import apply_config_overrides
    from wheel_legged_gym_isaaclab.robot_model import MODEL
    from wheel_legged_gym_isaaclab.mcu_hardware import (
        HARDWARE_PROFILE, LEG_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT,
        GRAVITY_FORCE, GRAVITY_ANGLE_BIAS, GRAVITY_FORCE_LIMIT,
    )
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env_cfg import WheelLeggedVMCFlatEnvCfg
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv

    gains = [(50., 3., 900., 90.), (50., 3., 900., 15.),
             (50., 3., 900., 2.), (10., 1., 900., 15.),
             (0., 0., 900., 90.), (0., 0., 900., 15.),
             (0., 0., 900., 2.), (5., .2, 900., 15.)]
    if args.gain_set == 'joint_pd_linearized':
        # CPU linearization of joint Kp=300, Kd=3 at CAD zero.
        # Diagnostic comparison only; this is not a recommended calibration.
        gains = [(50., 3., 900., 90.), (600., 6., 6906.48677, 69.064868),
                 (600., 6., 900., 90.), (300., 6., 900., 90.),
                 (100., 6., 900., 90.), (50., 3., 6906.48677, 69.064868),
                 (100., 3., 1800., 30.), (50., 3., 900., 15.)]
    if args.gain_set == 'firmware_length':
        # Isolate the MCU fixed-reference P/D coefficients; its integral term
        # and LQR outer angular controller are NOT represented by this test.
        gains = [(50., 3., 8000., 44.)]
    pulse_controls = [(0., 0.), (10., 0.), (-10., 0.), (20., 0.),
                      (0., 2.), (0., -2.), (20., 2.), (-20., -2.)]
    if args.case_index is not None:
        if not 0 <= args.case_index < len(gains):
            raise ValueError('case-index must select an existing diagnostic case')
        gains = [gains[args.case_index]]
        pulse_controls = [pulse_controls[args.case_index]]
    cfg = WheelLeggedVMCFlatEnvCfg()
    apply_config_overrides(cfg, json.loads(args.overrides.read_text()))
    if args.external_forces_every_iteration is not None:
        cfg.enable_external_forces_every_iteration = args.external_forces_every_iteration
    if args.solver != 'configured':
        cfg.sim.physx.solver_type = 0 if args.solver == 'pgs' else 1
    if args.velocity_iterations is not None:
        cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = args.velocity_iterations
        cfg.sim.physx.min_velocity_iteration_count = args.velocity_iterations
        cfg.sim.physx.max_velocity_iteration_count = args.velocity_iterations
    if args.position_iterations is not None:
        cfg.robot.spawn.articulation_props.solver_position_iteration_count = args.position_iterations
        cfg.sim.physx.min_position_iteration_count = args.position_iterations
        cfg.sim.physx.max_position_iteration_count = args.position_iterations
    if args.length_reference is not None:
        if not all(cfg.l0_ref_min <= x <= cfg.l0_ref_max for x in
                   (args.length_reference, args.length_reference + args.length_step)):
            raise ValueError('Fixed length references must stay inside the existing VMC range')
    if args.wheel_mode == 'disabled':
        cfg.wheel_control_mode = 'implicit_velocity'
        cfg.wheel_damping = 0.
        cfg.robot.actuators['wheels'].damping = 0.
    elif args.wheel_mode == 'external':
        cfg.wheel_control_mode = 'explicit_velocity'
        cfg.robot.actuators['wheels'].stiffness = 0.
        cfg.robot.actuators['wheels'].damping = 0.
    # Override both caps and feedforward explicitly: old saved run parameters
    # must not acquire new defaults implicitly during the baseline test.
    new_limits = args.hardware_case in ('limits', 'combined')
    new_gravity = args.hardware_case in ('gravity', 'combined')
    cfg.hardware_profile = HARDWARE_PROFILE if new_limits else None
    cfg.leg_effort_limit = LEG_EFFORT_LIMIT if new_limits else MODEL['leg_effort_limit']
    cfg.wheel_effort_limit = WHEEL_EFFORT_LIMIT if new_limits else MODEL['wheel_effort_limit']
    cfg.feedforward_force = GRAVITY_FORCE if new_gravity else MODEL['supported_mass_kg'] * 9.81 / 2
    cfg.feedforward_mode = 'mcu_pitch_compensated' if new_gravity else 'constant'
    cfg.feedforward_angle_bias = GRAVITY_ANGLE_BIAS if new_gravity else 0.
    cfg.feedforward_force_limit = GRAVITY_FORCE_LIMIT
    cfg.scene.num_envs = len(gains)
    cfg.scene.clone_in_fabric = False
    cfg.sim.device = args.device or 'cuda:0'
    if args.physics_hz < 50 or args.physics_hz % 50:
        raise ValueError('Diagnostic physics-hz must be a positive multiple of 50')
    cfg.sim.dt = 1. / args.physics_hz
    cfg.decimation = args.physics_hz // 50
    cfg.sim.render_interval = cfg.decimation
    cfg.contact_sensor.update_period = cfg.sim.dt
    cfg.seed = 43
    cfg.reset_velocity_initial = cfg.reset_velocity_final = 0.
    cfg.commands.ranges_lin_vel_x = (0., 0.)
    cfg.commands.ranges_ang_vel_yaw = (0., 0.)
    cfg.commands.ranges_height = (.30, .30)
    cfg.commands.grouped_training = False
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1000.
    cfg.episode_length_s = args.seconds + 2.
    if args.mode in ('hanging', 'pulse', 'fixed_step'):
        cfg.robot.spawn.articulation_props.fix_root_link = True
        cfg.robot.init_state.pos = (0., 0., .8)
        cfg.height_feedback_gain = 0.
    if args.mode in ('pulse', 'fixed_step'):
        cfg.sim.gravity = (0., 0., 0.)
    if args.zero_gravity:
        cfg.sim.gravity = (0., 0., 0.)
        cfg.feedforward_force = 0.
    if args.mode == 'fixed_step':
        cfg.feedforward_force = 0.
    cfg.leg_control_mode = 'explicit_vmc'
    gain_tensor = torch.tensor(gains, device=cfg.sim.device)

    class DiagnosticEnv(WheelLeggedVMCFlatEnv):
        recording = False
        pre_control = None
        physics_index = 0
        physics_records = []
        diagnostic_substep = 0
        controller_updates = 0
        hinge_spec = None
        freefall_probe = None

        def _setup_scene(self):
            super()._setup_scene()
            if args.mode == 'slider_guide':
                for i in range(self.scene.cfg.num_envs):
                    root = f'/World/envs/env_{i}/Robot'
                    guide = UsdPhysics.PrismaticJoint.Define(self.sim.stage, root + '/joints/diagnostic_world_slider')
                    guide.CreateBody1Rel().SetTargets([root + '/base_link'])
                    guide.CreateAxisAttr('Z')
                    origin = self._terrain.env_origins[i].cpu().tolist()
                    origin = [origin[k] + self.cfg.robot.init_state.pos[k] for k in range(3)]
                    guide.CreateLocalPos0Attr(Gf.Vec3f(*origin))
                    guide.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
                    guide.CreateLocalRot0Attr(Gf.Quatf(1., Gf.Vec3f(0.)))
                    guide.CreateLocalRot1Attr(Gf.Quatf(1., Gf.Vec3f(0.)))
                    guide.CreateExcludeFromArticulationAttr(True)
            if args.freefall_probe:
                self.freefall_probe = RigidObject(RigidObjectCfg(
                    prim_path='/World/DiagnosticFreefall',
                    spawn=sim_utils.SphereCfg(
                        radius=.05,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            disable_gravity=False, linear_damping=0., angular_damping=0.,
                            solver_position_iteration_count=cfg.robot.spawn.articulation_props.solver_position_iteration_count,
                            solver_velocity_iteration_count=cfg.robot.spawn.articulation_props.solver_velocity_iteration_count,
                            sleep_threshold=0., stabilization_threshold=0.,
                        ),
                        mass_props=sim_utils.MassPropertiesCfg(mass=1.),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(20., 20., 20.)),
                ))
                self.scene.rigid_objects['diagnostic_freefall'] = self.freefall_probe

        def _modify_leg_references(self, length, angle):
            if args.mode == 'fixed_step':
                return torch.full_like(length, .2691946125), torch.full_like(angle, .0522195)
            stepped = self.diagnostic_substep * self.cfg.sim.dt + 1e-12 >= args.step_time
            if args.length_reference is not None:
                length = torch.full_like(length, args.length_reference + (args.length_step if stepped else 0.))
            self.diagnostic_angle_ref = args.angle_reference + (args.angle_step if stepped else 0.)
            return length, torch.full_like(angle, self.diagnostic_angle_ref)

        def hinge_errors(self):
            d = self._robot.data
            ids0, ids1, p0, p1, a0, a1 = self.hinge_spec
            q0, q1 = d.body_link_quat_w[:, ids0], d.body_link_quat_w[:, ids1]
            w0 = d.body_link_pos_w[:, ids0] + quat_apply(q0, p0[None].expand(self.num_envs, -1, -1))
            w1 = d.body_link_pos_w[:, ids1] + quat_apply(q1, p1[None].expand(self.num_envs, -1, -1))
            aw0 = quat_apply(q0, a0[None].expand(self.num_envs, -1, -1))
            aw1 = quat_apply(q1, a1[None].expand(self.num_envs, -1, -1))
            gap = torch.linalg.vector_norm(w0 - w1, dim=-1)
            # Parallel and antiparallel directions define the same hinge axis.
            axis_error = torch.acos(torch.sum(aw0 * aw1, dim=-1).abs().clamp(0., 1.))
            return gap, axis_error

        def _apply_action(self):
            if args.mode == 'vertical_guide':
                # Ideal diagnostic guide: retain vertical pose and velocity;
                # hold only horizontal translation and body attitude. This is
                # NOT a valid balance controller or an acceptance condition.
                pose = self._robot.data.root_link_pose_w.clone()
                pose[:, :2] = self._terrain.env_origins[:, :2]
                pose[:, 3:] = self._robot.data.default_root_state[:, 3:7]
                velocity = self._robot.data.root_link_vel_w.clone()
                velocity[:, :2] = 0.
                velocity[:, 3:] = 0.
                self._robot.write_root_pose_to_sim(pose)
                self._robot.write_root_velocity_to_sim(velocity)
            update_control = not self.recording or is_control_update(self.diagnostic_substep, vmc_stride)
            if update_control:
                super()._apply_action()
                if self.recording:
                    self.controller_updates += 1
            else:
                # DirectRLEnv still calls scene.write_data_to_sim() every physics
                # step, re-sending the held effort / wheel velocity target. Only
                # state recording is refreshed here, not the controller.
                self._update_forward_kinematics()
            if self.recording:
                self.physics_index += 1
                actual = self._robot.data.joint_effort_target[:, self._leg_joint_ids]
                equivalent = torch.linalg.solve(self._five_bar_jacobian.transpose(-1,-2),
                                                actual.reshape(self.num_envs,2,2,1)).squeeze(-1)
                self.physics_records.append({
                    'time_s': self.physics_index * self.cfg.sim.dt,
                    'state_time_s': self.diagnostic_substep * self.cfg.sim.dt,
                    'control_update': update_control,
                    'joint_pos_rad': self._robot.data.joint_pos[:, self._leg_joint_ids].cpu().tolist(),
                    'joint_vel_rad_s': self._robot.data.joint_vel[:, self._leg_joint_ids].cpu().tolist(),
                    'all_joint_pos_rad': self._robot.data.joint_pos.cpu().tolist(),
                    'all_joint_vel_rad_s': self._robot.data.joint_vel.cpu().tolist(),
                    'height_m': self._robot.data.root_pos_w[:, 2].cpu().tolist(),
                    'root_velocity_world_m_s': self._robot.data.root_lin_vel_w.cpu().tolist(),
                    'length_m': self._L0.cpu().tolist(),
                    'angle_rad': self._theta0.cpu().tolist(),
                    'length_rate_m_s': self._L0_dot.cpu().tolist(),
                    'angle_rate_rad_s': self._theta0_dot.cpu().tolist(),
                    'motor_target_nm': actual.cpu().tolist(),
                    'equivalent_F_T': equivalent.cpu().tolist(),
                    'pre_control': self.pre_control,
                    'ref_length_m': self._l0_ref_applied.cpu().tolist(),
                    'ref_angle_rad': getattr(self, 'diagnostic_angle_ref', args.angle_reference),
                    'hinge_gap_m': self.hinge_errors()[0].cpu().tolist(),
                    'hinge_axis_error_rad': self.hinge_errors()[1].cpu().tolist(),
                    'freefall_position_m': self.freefall_probe.data.root_pos_w.cpu().tolist() if self.freefall_probe else None,
                    'freefall_velocity_m_s': self.freefall_probe.data.root_lin_vel_w.cpu().tolist() if self.freefall_probe else None,
                })
                self.diagnostic_substep += 1

        def _vmc(self, unused_force, unused_torque):
            kt, dt, kl, dl = (gain_tensor[:, i:i+1] for i in range(4))
            angle_ref = torch.full_like(self._theta0, getattr(self, 'diagnostic_angle_ref', args.angle_reference))
            if args.mode == 'fixed_step':
                angle_ref = torch.full_like(angle_ref, .0522195)
            force = kl * (self._l0_ref_applied-self._L0) - dl*self._L0_dot + self._get_gravity_feedforward()
            torque = kt * (angle_ref-self._theta0) - dt*self._theta0_dot
            if args.mode == 'pulse':
                control = torch.tensor(pulse_controls, device=self.device)
                force = control[:, 0:1].expand(-1, 2)
                torque = control[:, 1:2].expand(-1, 2)
            if args.virtual_force is not None:
                force = torch.full_like(force, args.virtual_force)
            if args.virtual_torque is not None:
                torque = torch.full_like(torque, args.virtual_torque)
            self.pre_control = {
                'force_n': force.detach().cpu().tolist(),
                'gravity_feedforward_n': self._get_gravity_feedforward().detach().cpu().tolist(),
                'swing_torque_nm': torque.detach().cpu().tolist(),
                'length_rate_m_s': self._L0_dot.detach().cpu().tolist(),
                'angle_rate_rad_s': self._theta0_dot.detach().cpu().tolist(),
            }
            result = super()._vmc(force, torque)
            self.pre_control['unclipped_motor_nm'] = torch.stack(
                (result[0][:, 0], result[1][:, 0], result[0][:, 1], result[1][:, 1]), -1
            ).detach().cpu().tolist()
            return result

        def physical_leg(self):
            data = self._robot.data
            quats = data.body_link_quat_w[:, self.wheel_body_ids]
            local = self.wheel_local[None].expand(self.num_envs, -1, -1)
            wheel_world = data.body_link_pos_w[:, self.wheel_body_ids] + quat_apply(quats, local)
            root_quat = data.root_link_quat_w[:, None].expand(-1, 2, -1)
            base = quat_apply_inverse(root_quat, wheel_world-data.root_link_pos_w[:, None])
            relative = base-self.hip_local[None]
            physical_length = torch.linalg.vector_norm(relative[:, :, (0, 2)], dim=-1)
            physical_angle = torch.atan2(-relative[:, :, 0], -relative[:, :, 2])
            return physical_length, physical_angle, wheel_world, relative

        def _get_dones(self):
            self._update_forward_kinematics()
            died, timeout = super()._get_dones()
            if self.recording:
                d = self._robot.data
                physical_length, physical_angle, wheel_world, relative = self.physical_leg()
                g = d.projected_gravity_b
                self.snapshot = dict(
                    time_s=float(self.episode_length_buf[0].item()*self.step_dt),
                    height_m=d.root_pos_w[:, 2].cpu().tolist(),
                    vertical_velocity_m_s=d.root_lin_vel_w[:, 2].cpu().tolist(),
                    pitch_rad=torch.asin(g[:, 0].clamp(-1, 1)).cpu().tolist(),
                    roll_rad=torch.atan2(-g[:, 1], -g[:, 2]).cpu().tolist(),
                    fk_length_m=self._L0.cpu().tolist(),
                    physical_length_m=physical_length.cpu().tolist(),
                    fk_angle_rad=self._theta0.cpu().tolist(),
                    physical_angle_rad=physical_angle.cpu().tolist(),
                    wheel_anchor_world_m=wheel_world.cpu().tolist(),
                    wheel_relative_hip_m=relative.cpu().tolist(),
                    joint_pos_rad=d.joint_pos[:, self._leg_joint_ids].cpu().tolist(),
                    joint_vel_rad_s=d.joint_vel[:, self._leg_joint_ids].cpu().tolist(),
                    motor_target_nm=d.joint_effort_target[:, self._leg_joint_ids].cpu().tolist(),
                    motor_sim_buffer_nm=self._robot._joint_effort_target_sim[:, self._leg_joint_ids].cpu().tolist(),
                    motor_estimate_nm=d.applied_torque[:, self._leg_joint_ids].cpu().tolist(),
                    # Independent simulator input readback, unlike the
                    # software actuator estimate above. Constraint reactions
                    # and contact loads must be inspected separately.
                    physx_actuation_nm=self._robot.root_physx_view.get_dof_actuation_forces()[:, self._leg_joint_ids].cpu().tolist(),
                    physx_projected_joint_nm=self._robot.root_physx_view.get_dof_projected_joint_forces()[:, self._leg_joint_ids].cpu().tolist(),
                    contact_body_names=self._contact_sensor.body_names,
                    contact_forces_world_n=self._contact_sensor.data.net_forces_w.cpu().tolist(),
                    raw_wheel_velocity_rad_s=d.joint_vel[:, self._wheel_joint_ids].cpu().tolist(),
                    ref_length_m=self._l0_ref_applied.cpu().tolist(),
                    died=died.cpu().tolist(),
                    reasons={k: v.cpu().tolist() for k, v in self._termination_reasons.items()},
                    pre_control=self.pre_control,
                )
            return died, timeout

    env = DiagnosticEnv(cfg)
    if cfg.sim.gravity == (0., 0., 0.):
        # With zero simulation gravity Isaac Lab's normalized gravity vector
        # is zero, so atan2(-0,-0) falsely marks a fixed upright base as fallen.
        # A unit down vector is an ORIENTATION reference only; no force is
        # applied and the native scene gravity remains exactly zero.
        env._robot.data.GRAVITY_VEC_W[:] = torch.tensor((0., 0., -1.), device=env.device)
    if args.joint_speed_limit is not None:
        speed_limits = env._robot.root_physx_view.get_dof_max_velocities().clone()
        ids = [i for i in range(env._robot.num_joints) if i not in env._wheel_joint_ids]
        speed_limits[:, ids] = args.joint_speed_limit
        env._robot.root_physx_view.set_dof_max_velocities(
            speed_limits, torch.arange(env.num_envs, dtype=torch.int32, device='cpu')
        )
    env.wheel_body_ids, names = env._robot.find_bodies(['LW_link', 'RW_link'], preserve_order=True)
    hip, wheel = [], []
    for prefix in ['L', 'R']:
        joint = UsdPhysics.RevoluteJoint(env.sim.stage.GetPrimAtPath(f'/World/envs/env_0/Robot/joints/{prefix}W_joint'))
        wheel.append(list(joint.GetLocalPos1Attr().Get()))
        joint = UsdPhysics.RevoluteJoint(env.sim.stage.GetPrimAtPath(f'/World/envs/env_0/Robot/joints/{prefix}L_joint'))
        hip.append(list(joint.GetLocalPos0Attr().Get()))
    env.hip_local = torch.tensor(hip, device=env.device)
    env.wheel_local = torch.tensor(wheel, device=env.device)
    # Inspect actual tree membership, not the joint-name suffix: the protected
    # asset intentionally exchanges a knee hinge and a named closure in its tree.
    hinges = []
    body_indices = {name: index for index, name in enumerate(env._robot.body_names)}
    for prim in Usd.PrimRange(env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot')):
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        sides = [joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()]
        if any(len(side) != 1 or side[0].name not in body_indices for side in sides):
            raise ValueError(f'Unresolved robot hinge body: {prim.GetPath()}')
        points, axes = [], []
        for side in (0, 1):
            points.append(list(getattr(joint, f'GetLocalPos{side}Attr')().Get()))
            local_rotation = getattr(joint, f'GetLocalRot{side}Attr')().Get()
            axis = Gf.Vec3d(*{'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[joint.GetAxisAttr().Get()])
            axes.append(list(Gf.Rotation(Gf.Quatd(local_rotation)).TransformDir(axis)))
        hinges.append(dict(name=prim.GetName(), bodies=[side[0].name for side in sides],
                           ids=[body_indices[side[0].name] for side in sides],
                           points=points, axes=axes,
                           excluded=bool(joint.GetExcludeFromArticulationAttr().Get())))
    env.hinge_spec = (
        [h['ids'][0] for h in hinges], [h['ids'][1] for h in hinges],
        *(torch.tensor([h[key][side] for h in hinges], device=env.device)
          for key in ('points', 'axes') for side in (0, 1)),
    )
    obs, _ = env.reset()
    if env.freefall_probe:
        env.freefall_probe.write_root_state_to_sim(env.freefall_probe.data.default_root_state.clone())
        env.freefall_probe.reset()
    initial = {
        'length_m': env._L0.cpu().tolist(),
        'angle_rad': env._theta0.cpu().tolist(),
        'joint_pos_rad': env._robot.data.joint_pos[:, env._leg_joint_ids].cpu().tolist(),
        'body_names': env._robot.body_names,
        'masses_kg': env._robot.root_physx_view.get_masses()[0].cpu().tolist(),
        'inertias_kg_m2': env._robot.root_physx_view.get_inertias()[0].cpu().tolist(),
        'leg_drive_kp': env._robot.root_physx_view.get_dof_stiffnesses()[0,env._leg_joint_ids].cpu().tolist(),
        'leg_drive_kd': env._robot.root_physx_view.get_dof_dampings()[0,env._leg_joint_ids].cpu().tolist(),
        'leg_effort_limits_nm': env._robot.root_physx_view.get_dof_max_forces()[0,env._leg_joint_ids].cpu().tolist(),
        'wheel_effort_limits_nm': env._robot.root_physx_view.get_dof_max_forces()[0,env._wheel_joint_ids].cpu().tolist(),
        'wheel_drive_kp': env._robot.root_physx_view.get_dof_stiffnesses()[0,env._wheel_joint_ids].cpu().tolist(),
        'wheel_drive_kd': env._robot.root_physx_view.get_dof_dampings()[0,env._wheel_joint_ids].cpu().tolist(),
        'joint_names': env._robot.joint_names,
        'leg_joint_ids': env._leg_joint_ids,
        'wheel_joint_ids': env._wheel_joint_ids,
        'joint_max_velocity_rad_s': env._robot.root_physx_view.get_dof_max_velocities()[0].cpu().tolist(),
        'gravity_compensation': env._robot.root_physx_view.get_gravity_compensation_forces()[0].cpu().tolist(),
        'generalized_mass_matrix': env._robot.root_physx_view.get_generalized_mass_matrices()[0].cpu().tolist(),
        'hinges': hinges,
        'hinge_gap_m': env.hinge_errors()[0].cpu().tolist(),
        'hinge_axis_error_rad': env.hinge_errors()[1].cpu().tolist(),
        'freefall_position_m': env.freefall_probe.data.root_pos_w.cpu().tolist() if env.freefall_probe else None,
        'freefall_velocity_m_s': env.freefall_probe.data.root_lin_vel_w.cpu().tolist() if env.freefall_probe else None,
    }
    records = []
    env.recording = True
    actions = torch.zeros(cfg.scene.num_envs, 6, device=env.device)
    with torch.inference_mode():
        for step_index in range(round(args.seconds/env.step_dt)):
            env.step(actions)
            env.snapshot['elapsed_s'] = (step_index + 1) * env.step_dt
            records.append(env.snapshot)
            if args.stop_on_failure and any(env.snapshot['died']):
                break
    first_failure = next((r['elapsed_s'] for r in records if any(r['died'])), None)
    report = dict(mode=args.mode, gains=gains, initial=initial,
                  hardware_case=args.hardware_case,
                  first_failure_s=first_failure,
                  survived_without_reset=first_failure is None,
                  minimum_pre_reset_height_m=min(min(r['height_m']) for r in records),
                  runtime_control={key:getattr(cfg,key) for key in (
                      'hardware_profile','leg_effort_limit','wheel_effort_limit',
                      'feedforward_force','feedforward_mode','feedforward_angle_bias',
                      'feedforward_force_limit','height_reference_blend','height_feedback_gain')},
                  pulse_controls=pulse_controls if args.mode == 'pulse' else None, hip_local_m=hip,
                  wheel_local_m=wheel, records=records, physics_records=env.physics_records,
                  physics_dt=cfg.sim.dt,
                  control_dt=1. / args.control_hz,
                  policy_dt=env.step_dt,
                  controller_updates=env.controller_updates,
                  reference_step=dict(length_m=args.length_reference, length_step_m=args.length_step,
                                      angle_step_rad=args.angle_step, step_time_s=args.step_time),
                  gravity_m_s2=cfg.sim.gravity,
                  position_iterations=cfg.robot.spawn.articulation_props.solver_position_iteration_count,
                  velocity_iterations=cfg.robot.spawn.articulation_props.solver_velocity_iteration_count,
                  robot_usd=cfg.robot.spawn.usd_path,
                  leg_control_mode=cfg.leg_control_mode,
                  diagnostic_execution=args.execution,
                  wheel_control_mode=cfg.wheel_control_mode,
                  wheel_ablation=args.wheel_mode,
                  virtual_force_override_n=args.virtual_force,
                  virtual_torque_override_nm=args.virtual_torque,
                  angle_reference_rad=args.angle_reference,
                  solver_type=cfg.sim.physx.solver_type,
                  velocity_iterations_override=args.velocity_iterations,
                  external_forces_every_iteration=PhysxSchema.PhysxSceneAPI(
                      env.sim.stage.GetPrimAtPath(cfg.sim.physics_prim_path)
                  ).GetEnableExternalForcesEveryIterationAttr().Get(),
                  joint_speed_limit_override_rad_s=args.joint_speed_limit,
                  policy_trained=False, training_modified=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print('SUPPORT_DIAG_COMPLETE '+str(args.output), flush=True)
    env.close()


try:
    main()
finally:
    app.close()
