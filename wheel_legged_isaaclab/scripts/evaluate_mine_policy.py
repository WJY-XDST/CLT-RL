"""Evaluate a frozen actor trained for the imported closed-five-bar vehicle.

All telemetry is captured before automatic resets. Each condition uses three
independent reset-velocity streams. No training, gain tuning or policy export.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import traceback

from isaaclab.app import AppLauncher
ROOT = Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint",type=Path,required=True)
parser.add_argument("--robot",choices=["mine"],default="mine")
parser.add_argument("--seconds",type=float,default=8.)
parser.add_argument("--settle-seconds",type=float,default=0.,help="Standing interval before each episode's motion ramp")
parser.add_argument("--heading-hold",action="store_true",help="Optional world-heading feedback; default replays the requested constant yaw-rate command")
parser.add_argument("--command-scale",type=float,default=1.,help="Diagnostic curriculum command amplitude; acceptance uses 1.0")
parser.add_argument("--curriculum-progress",type=float,default=None,
                    help="Test actual main-speed curriculum bounds, not uniform speed scaling; final acceptance uses 1.0")
parser.add_argument("--output",type=Path,required=True)
parser.add_argument("--seeds", type=int, nargs=3, default=[43,44,45])
parser.add_argument("--overrides_json",type=Path,default=None)
parser.add_argument("--export_dir",type=Path,default=None)
parser.add_argument("--record_policy_inputs",action="store_true",help="Save exact inference inputs and raw means for offline feedback diagnosis")
parser.add_argument("--diagnose_contacts", action="store_true",
                    help="Record base/ground filtered forces, collision-box clearance and joint tracking; no controller change")
parser.add_argument('--wheel-probe', action='store_true',
                    help='Diagnostic only: hold wheels at zero or opposite +/-0.05 normalized references; retain policy legs. Not policy acceptance.')
parser.add_argument('--bias-probe-json', type=Path, default=None,
                    help='Diagnostic only: paired small frozen-policy action offsets and actual reward telemetry. Cannot count as acceptance.')
parser.add_argument('--record-reward-terms', action='store_true',
                    help='Record actual per-step environment reward components without changing execution')
parser.add_argument('--tracking-goal-reward-probe', type=float, default=None,
                    help='Frozen direct-PD reward diagnosis only; cannot count as acceptance or export')
parser.add_argument('--joint-saturation-reward-probe', type=float, default=None,
                    help='Frozen direct-PD reward diagnosis only: coefficient in [-8,0], never acceptance or export')
parser.add_argument('--command-probe-json', type=Path, default=None,
                    help='Diagnostic only: commands within training ranges; cannot count as canonical acceptance')
parser.add_argument('--reset-velocity-amplitude', type=float, default=.05,
                    help='Diagnostic reset velocity amplitude; canonical acceptance uses .05')
parser.add_argument('--sample-policy-actions', action='store_true',
                    help='Command diagnosis only: use saved Gaussian exploration with paired noise seed 43')
parser.add_argument('--wheel-action-saturation-reward-probe', type=float, default=None,
                    help='Command diagnosis only: evaluate an opt-in raw wheel overshoot reward in [-2,0], with unchanged frozen actions')
parser.add_argument('--wheel-reference-limit-probe', type=float, default=None,
                    help='Command diagnosis only: test a software wheel-reference clamp in [1,2], keeping gains and torque limits')
parser.add_argument("--physics-hz",type=int,default=None,
                    help="Numerical replay only: refine physics while keeping policy at 50 Hz")
parser.add_argument("--control-hz",type=int,default=None,
                    help="Hold explicit VMC efforts between updates; must divide physics frequency")
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app


def main():
    import numpy as np
    import torch
    import yaml
    from rsl_rl.modules import ActorCritic
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env_cfg import (
        WheelLeggedVMCFlatEnvCfg, MANIFEST,
    )
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv
    from wheel_legged_gym_isaaclab.config_overrides import apply_config_overrides
    from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    from wheel_legged_gym_isaaclab.diagnostic_timing import control_stride, is_control_update

    saved_config = validate_checkpoint_robot(args.checkpoint)
    from wheel_legged_gym_isaaclab.physics_contract import (
        checkpoint_force_iteration_setting, checkpoint_robot_asset_setting,
    )
    saved_rewards = saved_config.get('rewards', {})
    cfg = WheelLeggedVMCFlatEnvCfg()
    # A replay must report the checkpoint's reward settings, not today's
    # defaults. This does not retune its actor or execution chain.
    apply_config_overrides(cfg, {f'rewards.{key}': value for key, value in saved_rewards.items()})
    # Old checkpoints keep their historical command range; new ones must be
    # tested at their actual full height extrema, not the old +/- 2 cm slots.
    apply_config_overrides(cfg, {f'commands.{key}': value
                                for key, value in saved_config['commands'].items()})
    overrides = json.loads(args.overrides_json.read_text()) if args.overrides_json else {}
    cfg.robot.spawn.usd_path = checkpoint_robot_asset_setting(saved_config, overrides)
    cfg.enable_external_forces_every_iteration = checkpoint_force_iteration_setting(saved_config, overrides)
    height=cfg.rewards.base_height_target
    cases=[("stand",0.,0.,height),("forward_0p3",.3,0.,height),
           ("forward_0p6",.6,0.,height),("reverse_0p3",-.3,0.,height),
           ("reverse_0p6",-.6,0.,height),("spin_left",0.,.3,height),
           ("spin_right",0.,-.3,height),("height_low",0.,0.,cfg.commands.ranges_height[0]),
           ("height_high",0.,0.,cfg.commands.ranges_height[1])]
    if not 0.<args.command_scale<=1.:
        raise ValueError('Command scale must be in (0, 1].')
    cases=[(name,speed*args.command_scale,yaw*args.command_scale,
            height+(target_height-height)*args.command_scale) for name,speed,yaw,target_height in cases]
    if args.curriculum_progress is not None:
        if args.command_scale != 1.:
            raise ValueError('Curriculum progress and uniform command scale cannot be combined')
        from wheel_legged_gym_isaaclab.yaw_training import curriculum_replay_cases
        cases=curriculum_replay_cases(height,cfg.commands,args.curriculum_progress)
    if args.wheel_probe:
        if args.export_dir or args.curriculum_progress is not None or args.command_scale != 1.:
            raise ValueError('Wheel probe cannot export a policy or use a curriculum-scaled acceptance test')
        cases=[('probe_neutral',0.,0.,height),('probe_left',0.,0.,height),('probe_right',0.,0.,height)]
    command_sequence, command_phase_seconds = None, None
    if args.command_probe_json:
        if (args.wheel_probe or args.bias_probe_json or args.export_dir or args.heading_hold
                or args.overrides_json or args.curriculum_progress is not None or args.command_scale != 1.
                or args.physics_hz is not None or args.control_hz is not None):
            raise ValueError('Command diagnosis must preserve the checkpoint control contract')
        from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_plan
        cases, command_sequence, command_phase_seconds = command_probe_plan(
            json.loads(args.command_probe_json.read_text()), cfg.commands)
    if args.sample_policy_actions and not args.command_probe_json:
        raise ValueError('Sampled policy actions are diagnostic only')
    if args.wheel_action_saturation_reward_probe is not None:
        if (not args.command_probe_json or saved_config['leg_control_mode'] != 'implicit_joint_position'
                or not -2. <= args.wheel_action_saturation_reward_probe <= 0.):
            raise ValueError('Wheel overshoot reward probe requires frozen direct-PD command diagnosis and a nonpositive coefficient bounded by minus two')
        cfg.rewards.wheel_action_saturation = args.wheel_action_saturation_reward_probe
    if args.tracking_goal_reward_probe is not None:
        import math
        if (saved_config['leg_control_mode'] != 'implicit_joint_position'
                or args.export_dir or args.overrides_json or args.bias_probe_json or args.wheel_probe
                or args.wheel_action_saturation_reward_probe is not None
                or args.wheel_reference_limit_probe is not None
                or not math.isfinite(args.tracking_goal_reward_probe)
                or not -2. <= args.tracking_goal_reward_probe <= 0.):
            raise ValueError('Tracking-goal reward probe requires frozen direct-PD diagnosis and a coefficient in [-2,0]')
        cfg.rewards.tracking_goal_violation = args.tracking_goal_reward_probe
    if args.joint_saturation_reward_probe is not None:
        import math
        if (saved_config['leg_control_mode'] != 'implicit_joint_position'
                or args.export_dir or args.overrides_json or args.bias_probe_json or args.wheel_probe
                or args.sample_policy_actions or args.heading_hold
                or args.physics_hz is not None or args.control_hz is not None
                or args.curriculum_progress is not None or args.command_scale != 1.
                or args.wheel_action_saturation_reward_probe is not None
                or args.tracking_goal_reward_probe is not None
                or args.wheel_reference_limit_probe is not None
                or not math.isfinite(args.joint_saturation_reward_probe)
                or not -8. <= args.joint_saturation_reward_probe <= 0.):
            raise ValueError('Joint saturation reward probe requires frozen direct-PD diagnosis and a coefficient in [-8,0]')
        cfg.rewards.joint_position_action_saturation = args.joint_saturation_reward_probe
    if args.wheel_reference_limit_probe is not None:
        if (not args.command_probe_json or saved_config['leg_control_mode'] != 'implicit_joint_position'
                or not 1. <= args.wheel_reference_limit_probe <= 2.):
            raise ValueError('Wheel-reference clamp test requires direct-PD command diagnosis and a limit in [1,2]')
    if not 0 <= args.reset_velocity_amplitude <= cfg.reset_velocity_final:
        raise ValueError('Diagnostic reset amplitude must stay inside the training range')
    if args.reset_velocity_amplitude != .05 and (not args.command_probe_json or args.export_dir):
        raise ValueError('Noncanonical reset amplitude is diagnostic only')
    bias_labels, case_biases = None, None
    if args.bias_probe_json:
        if (args.wheel_probe or args.export_dir or args.overrides_json or args.heading_hold
                or args.curriculum_progress is not None or args.command_scale != 1.
                or args.physics_hz is not None or args.control_hz is not None
                or saved_config['leg_control_mode'] != 'implicit_joint_position'):
            raise ValueError('Joint-action bias diagnosis requires the unchanged direct-PD physical contract')
        from wheel_legged_gym_isaaclab.action_bias_probe import build_bias_cases
        cases, case_biases, bias_labels = build_bias_cases(cases, json.loads(args.bias_probe_json.read_text()))
    seeds=args.seeds
    if args.overrides_json:
        apply_config_overrides(cfg,overrides)
        if cfg.wheel_control_mode == 'implicit_velocity':
            cfg.robot.actuators['wheels'].damping=cfg.wheel_damping
    from wheel_legged_gym_isaaclab.leg_control import require_matching_policy_contract
    require_matching_policy_contract(saved_config, cfg.to_dict())
    if args.physics_hz is not None:
        if args.physics_hz <= 0 or args.physics_hz % 50:
            raise ValueError('Physics frequency must be a positive multiple of 50 Hz')
        cfg.sim.dt=1./args.physics_hz
        cfg.decimation=args.physics_hz//50
    physics_hz=round(1./cfg.sim.dt)
    control_hz=args.control_hz or physics_hz
    stride=control_stride(physics_hz,control_hz,round(1./(cfg.sim.dt*cfg.decimation)))
    if stride>1 and cfg.leg_control_mode!='explicit_vmc':
        raise ValueError('Held-controller convergence replay requires explicit VMC')
    asset_chain, _ = verify_asset_chain(cfg.robot.spawn.usd_path)
    cfg.scene.num_envs=len(cases)*len(seeds)
    cfg.sim.device=args.device or "cuda:0"
    cfg.seed=43
    cfg.episode_length_s=args.seconds+2
    cfg.reset_velocity_initial=cfg.reset_velocity_final=0.
    cfg.commands.ranges_lin_vel_x=(0.,0.)
    cfg.commands.ranges_height=(height,height)
    cfg.commands.heading_command=False
    cfg.commands.grouped_training=False
    cfg.commands.resampling_time=1000.
    cfg.viewer.eye=(1.,1.,.6)
    cfg.viewer.lookat=(0.,0.,-.08)
    # Commands are per-environment; every seed uses identical initial-velocity
    # draws across all conditions.
    class EvaluationEnv(WheelLeggedVMCFlatEnv):
        def _setup_scene(self):
            super()._setup_scene()
            if args.diagnose_contacts:
                from isaaclab.sensors import ContactSensor, ContactSensorCfg
                from pxr import UsdGeom, Usd, UsdPhysics, Gf
                ground = self.sim.stage.GetPrimAtPath(self.cfg.terrain.prim_path)
                ground_shapes = [str(p.GetPath()) for p in Usd.PrimRange(ground)
                                 if p.HasAPI(UsdPhysics.CollisionAPI)
                                 and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False]
                if len(ground_shapes) != 1:
                    raise RuntimeError(f'Expected one flat-ground collider, found {ground_shapes}')
                # Filter one body per environment: the multi-body sensor's
                # net force alone cannot identify its collision partner.
                self._base_ground_sensor = ContactSensor(ContactSensorCfg(
                    prim_path="/World/envs/env_.*/Robot/base_link",
                    filter_prim_paths_expr=ground_shapes,
                    update_period=self.cfg.sim.dt,
                ))
                self.scene.sensors["base_ground_diagnostic"] = self._base_ground_sensor
                self._wheel_ground_sensor = ContactSensor(ContactSensorCfg(
                    prim_path="/World/envs/env_.*/Robot/LW_link",
                    filter_prim_paths_expr=ground_shapes, update_period=self.cfg.sim.dt,
                ))
                self.scene.sensors["wheel_ground_diagnostic"] = self._wheel_ground_sensor
                base = self.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/base_link')
                cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                        ['default', 'render', 'proxy', 'guide'], False, True)
                bounds = Gf.Range3d()
                base_shapes = []
                for prim in Usd.PrimRange(base, Usd.TraverseInstanceProxies()):
                    if (prim.HasAPI(UsdPhysics.CollisionAPI)
                            and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False):
                        current = cache.ComputeRelativeBound(prim, base).ComputeAlignedRange()
                        if not current.IsEmpty():
                            bounds.UnionWith(current)
                            base_shapes.append(str(prim.GetPath()))
                if bounds.IsEmpty():
                    raise RuntimeError('No active base collision geometry found; refuse an invalid clearance')
                lo, hi = bounds.GetMin(), bounds.GetMax()
                print(f'CONTACT_AUDIT ground={ground_shapes} base={base_shapes} bounds={lo}, {hi}', flush=True)
                self._base_collision_corners = torch.tensor([
                    [x, y, z] for x in (lo[0], hi[0])
                    for y in (lo[1], hi[1]) for z in (lo[2], hi[2])
                ], device=self.device)

        def __init__(self,cfg):
            self.recording=False
            self.control_substep=0
            self.controller_updates=0
            self.reset_counts=np.zeros(cfg.scene.num_envs,dtype=int)
            self.generators=[torch.Generator().manual_seed(seeds[i%3]) for i in range(cfg.scene.num_envs)]
            super().__init__(cfg)
            self.case_commands=torch.tensor([x[1:] for x in cases for _ in seeds],device=self.device)
            self.case_command_sequence = (torch.tensor(command_sequence, device=self.device).repeat_interleave(len(seeds), dim=1)
                                          if command_sequence is not None else None)
            self.snapshot=None

        def _reset_idx(self,env_ids):
            if env_ids is None:
                env_ids=self._robot._ALL_INDICES
            super()._reset_idx(env_ids)
            ids=env_ids.tolist() if hasattr(env_ids,"tolist") else list(env_ids)
            vel=torch.stack([(2*torch.rand(6,generator=self.generators[i])-1)*args.reset_velocity_amplitude for i in ids]).to(self.device)
            self._robot.write_root_velocity_to_sim(vel,env_ids)
            self._commands[env_ids,2]=self.case_commands[env_ids,2]
            if self.recording:
                self.reset_counts[ids]+=1

        def _update_commands(self):
            pass

        def _pre_physics_step(self, actions):
            super()._pre_physics_step(actions)
            if args.wheel_reference_limit_probe is not None:
                from wheel_legged_gym_isaaclab.action_bias_probe import wheel_reference_probe
                self._actions = wheel_reference_probe(self._actions, self._raw_actions,
                                                     args.wheel_reference_limit_probe)

        def _apply_action(self):
            # Scene writes held targets on every physics tick. Only recompute
            # J^T[F,T] at the original VMC clock, including the first tick.
            if is_control_update(self.control_substep,stride):
                super()._apply_action()
                self.controller_updates+=1
            self.control_substep+=1

        def _get_observations(self):
            if self.case_command_sequence is not None:
                phase = self.common_step_counter // round(command_phase_seconds / self.step_dt)
                self.case_commands.copy_(self.case_command_sequence[phase % len(self.case_command_sequence)])
                if phase != getattr(self, '_diagnostic_command_phase', None):
                    self._command_resample_step[:] = self.common_step_counter
                    self._diagnostic_command_phase = phase
            # Optional standing settling interval; restart the ramp
            # after failure, preserving the same requested test condition.
            moving=(self.episode_length_buf*self.step_dt>=args.settle_seconds)
            self._target_lin_vel_x[:]=torch.where(moving,self.case_commands[:,0],0.)
            self._commands[:,2]=self.case_commands[:,2]
            yaw_target=self.case_commands[:,1]
            if args.heading_hold:
                q=self._robot.data.root_quat_w
                yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
                straight=yaw_target==0
                yaw_target=torch.where(straight,(-self.cfg.commands.heading_kp*yaw).clamp(-.5,.5),yaw_target)
            self._commands[:,1]=torch.where(moving,yaw_target,0.)
            return super()._get_observations()

        def _get_dones(self):
            self._update_forward_kinematics()
            died,timeout=super()._get_dones()
            if self.recording:
                d=self._robot.data
                g=d.projected_gravity_b
                pitch=torch.asin(g[:,0].clamp(-1,1))
                roll=torch.atan2(-g[:,1],-g[:,2])
                q=d.root_quat_w
                yaw=torch.atan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]),1-2*(q[:,2]**2+q[:,3]**2))
                vel=self._get_heading_frame_horizontal_velocity()
                torques=d.applied_torque[:,self._torque_joint_ids]
                saturation=(torques.abs()>=self._torque_limits*.99).float().mean(-1)
                action_limits = self._actions.new_ones(6)
                if args.wheel_reference_limit_probe is not None:
                    action_limits[[2, 5]] = args.wheel_reference_limit_probe
                fields=[self.episode_length_buf*self.step_dt,self._commands[:,0],self._commands[:,1],self._commands[:,2],
                    vel[:,0],vel[:,1],d.root_ang_vel_b[:,2],d.root_pos_w[:,2],pitch,roll,yaw,
                    self._L0[:,0],self._L0[:,1],self._l0_ref_applied[:,0],self._l0_ref_applied[:,1],
                    self._theta0[:,0],self._theta0[:,1],self._theta0_ref_applied[:,0],self._theta0_ref_applied[:,1],
                    self._get_wheel_vel_forward_positive()[:,0],self._get_wheel_vel_forward_positive()[:,1],
                    self._actions[:,2]*self.cfg.action_scale_vel,self._actions[:,5]*self.cfg.action_scale_vel,
                    torques.abs().max(-1).values,saturation,(self._raw_actions.abs()>action_limits).float().mean(-1),
                    died.float(),timeout.float()]
                fields.extend(self._termination_reasons[key].float() for key in REASONS)
                if args.diagnose_contacts:
                    from isaaclab.utils.math import quat_apply
                    filtered = self._base_ground_sensor.data.force_matrix_w
                    if filtered is None or filtered.shape[2] != 1:
                        raise RuntimeError('Base/ground diagnostic sensor has no matched ground body')
                    base_ground = filtered[:, 0, 0].norm(dim=-1)
                    wheel_ground = self._wheel_ground_sensor.data.force_matrix_w[:, 0, 0].norm(dim=-1)
                    base_net = self._contact_sensor.data.net_forces_w[:, self._base_id].norm(dim=-1).max(-1).values
                    corners = quat_apply(
                        d.root_quat_w[:, None, :].expand(-1, 8, -1),
                        self._base_collision_corners[None].expand(self.num_envs, -1, -1),
                    ) + d.root_pos_w[:, None, :]
                    fields.extend([base_net, base_ground, wheel_ground, corners[:, :, 2].min(-1).values,
                                   (d.joint_pos[:, self._leg_joint_ids] - self._joint_position_ref_applied).abs().max(-1).values])
                self.snapshot=torch.stack(fields,-1).cpu().numpy()
                self.snapshot_episodes=self.reset_counts.copy()
            return died,timeout

    REASONS=["base_contact","fallen","pitch","roll","leg_contact","numerical","kinematic_singularity","low_height"]
    columns=["episode_time_s","speed_cmd_m_s","yaw_cmd_rad_s","height_cmd_m","speed_m_s","lateral_speed_m_s","yaw_rate_rad_s","height_m","pitch_rad","roll_rad","yaw_rad",
        "left_length_m","right_length_m","left_length_ref_m","right_length_ref_m","left_angle_rad","right_angle_rad","left_angle_ref_rad","right_angle_ref_rad",
        "left_wheel_rad_s","right_wheel_rad_s","left_wheel_ref_rad_s","right_wheel_ref_rad_s","peak_torque_nm","torque_saturation_fraction","action_clipping_fraction","terminated","timeout"]+["fail_"+r for r in REASONS]
    if args.diagnose_contacts:
        columns.extend(['base_net_force_n', 'base_ground_force_n', 'wheel_ground_force_n', 'base_collision_min_z_m',
                        'active_joint_tracking_max_abs_rad'])
    env=EvaluationEnv(cfg)
    obs,_=env.reset()
    record_rewards = args.record_reward_terms or case_biases is not None or args.tracking_goal_reward_probe is not None or args.joint_saturation_reward_probe is not None
    diagnostic_reward_names = list(env._episode_sums) if record_rewards else []
    if record_rewards:
        columns.extend(['reward_total'] + ['reward_' + key for key in diagnostic_reward_names])
    if case_biases is not None:
        from wheel_legged_gym_isaaclab.action_bias_probe import add_action_bias
        action_biases = torch.tensor(case_biases, device=env.device).repeat_interleave(len(seeds), dim=0)
    agent=yaml.safe_load((args.checkpoint.parent/"params/agent.yaml").read_text())
    policy_cfg=dict(agent["policy"]);policy_cfg.pop("class_name")
    actor=ActorCritic(obs,agent["obs_groups"],6,**policy_cfg).to(env.device)
    checkpoint=torch.load(args.checkpoint,map_location=env.device,weights_only=True)
    actor.load_state_dict(checkpoint["model_state_dict"],strict=True)
    actor.eval()
    if args.sample_policy_actions:
        policy_noise_generator = torch.Generator(device=env.device).manual_seed(43)
        policy_std = checkpoint['model_state_dict'].get('std')
        if policy_std is None:
            policy_std = checkpoint['model_state_dict']['log_std'].exp()
        if policy_std.shape != (6,) or not torch.isfinite(policy_std).all() or not (policy_std > 0).all():
            raise ValueError('Diagnostic Gaussian noise requires valid saved standard deviations')
    assert obs["policy"].shape==(len(cases)*3,27)
    assert actor.act_inference(obs).shape==(len(cases)*3,6)
    # Cross-check native RSL deterministic inference against the checkpoint actor.
    torch.testing.assert_close(actor.act_inference(obs),actor.actor(obs["policy"]))
    if args.export_dir:
        from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
        args.export_dir.mkdir(parents=True,exist_ok=True)
        export_policy_as_jit(actor,normalizer=actor.actor_obs_normalizer,path=str(args.export_dir),filename='policy.pt')
        export_policy_as_onnx(actor,path=str(args.export_dir),normalizer=actor.actor_obs_normalizer,filename='policy.onnx')
        exported=torch.jit.load(str(args.export_dir/'policy.pt'),map_location=env.device)
        torch.testing.assert_close(exported(obs['policy']),actor.act_inference(obs))
        contract={"observations":27,"actions":6,"action_order":["LB_angle","left_length","LW_speed","RB_angle","right_length","RW_speed"],"motor_order":["LB_joint","LL_joint","LW_joint","RB_joint","RL_joint","RW_joint"],"base_axes":"X forward, Y left, Z up","angle_reference":"base_link, CAD assembly q=0","policy_dt":env.step_dt,"physics_dt":cfg.sim.dt,"wheel_control_mode":cfg.wheel_control_mode,"wheel_damping":cfg.wheel_damping,"five_bar":cfg.five_bar_geometry,"runtime_config":cfg.to_dict(),"checkpoint":str(args.checkpoint.resolve()),"transfer_validation":"Isaac Lab replay only; new-model MuJoCo replay still required"}
        contract['observation_blocks']=[
            {'slice':[0,3],'name':'body angular velocity','scale':cfg.obs_scales.ang_vel},
            {'slice':[3,6],'name':'projected gravity','scale':1.},
            {'slice':[6,9],'name':'applied forward speed, yaw rate, root height commands','scale':[cfg.obs_scales.lin_vel,cfg.obs_scales.ang_vel,cfg.obs_scales.height_measurements]},
            {'slice':[9,11],'name':'left/right virtual leg angles','scale':cfg.obs_scales.dof_pos},
            {'slice':[11,13],'name':'left/right virtual leg angular velocities','scale':cfg.obs_scales.dof_vel},
            {'slice':[13,15],'name':'left/right virtual leg lengths','scale':cfg.obs_scales.l0},
            {'slice':[15,17],'name':'left/right virtual leg length velocities','scale':cfg.obs_scales.l0_dot},
            {'slice':[17,19],'name':'yaw-aligned horizontal body velocity X/Y','scale':cfg.obs_scales.lin_vel},
            {'slice':[19,21],'name':'forward-positive left/right wheel velocity','scale':cfg.obs_scales.dof_vel},
            {'slice':[21,27],'name':'previous physically applied normalized actions','scale':1.}]
        contract['policy_output']='Unclamped Gaussian mean; clamp and apply VMC reference blending in the simulator.'
        if cfg.leg_control_mode == 'implicit_joint_position':
            contract['action_order'] = ['LB_joint_position', 'LL_joint_position', 'LW_speed', 'RB_joint_position', 'RL_joint_position', 'RW_speed']
            contract['observation_blocks'] = [
                *contract['observation_blocks'][:3],
                {'slice':[9,13], 'name':'LB/LL/RB/RL motor position minus fixed nominal offset', 'scale':cfg.obs_scales.dof_pos},
                {'slice':[13,17], 'name':'LB/LL/RB/RL motor angular velocity', 'scale':cfg.obs_scales.dof_vel},
                *contract['observation_blocks'][7:],
            ]
            contract['policy_output'] = 'Unclamped Gaussian mean; clamp and map four motor-angle targets by offset+scale, bound at joint stops; no IK or virtual-coordinate PD. Optional gravity effort is separate.'
            contract['leg_joint_position_offset'] = list(cfg.leg_joint_position_offset)
            contract['leg_joint_position_scale'] = list(cfg.leg_joint_position_scale)
            contract['leg_joint_target_limit_margin'] = cfg.leg_joint_target_limit_margin
        contract['leg_control_mode']=cfg.leg_control_mode
        contract['enable_external_forces_every_iteration']=env._external_forces_every_iteration
        contract['leg_actuator_contract']=({'reference':'direct motor-angle targets in raw CAD axes',
            'drive_type':'force','stiffness_nm_per_rad':cfg.leg_joint_stiffness,
            'damping_nm_s_per_rad':cfg.leg_joint_damping,'effort_limit_nm':cfg.leg_effort_limit,
            'explicit_leg_effort':0.,'gravity_feedforward_applied':False}
            if cfg.leg_control_mode == 'implicit_joint_position' else
            {'reference':'J.transpose @ [radial PD + gravity feedforward, angular PD]',
             'effort_limit_nm':cfg.leg_effort_limit})
        if cfg.leg_control_mode == 'implicit_joint_position':
            from wheel_legged_gym_isaaclab.mcu_hardware import MCU_LIMIT_EVIDENCE
            contract['leg_actuator_contract']['reference'] = 'offset + clamp(action[0,1,3,4], -1, 1)*scale in raw CAD motor axes, bounded by joint stops'
            contract['leg_actuator_contract']['explicit_leg_effort'] = ('J.transpose @ [gravity feedforward, 0] only' if cfg.leg_joint_gravity_feedforward else 0.)
            contract['leg_actuator_contract']['gravity_feedforward_applied'] = cfg.leg_joint_gravity_feedforward
            contract['leg_actuator_contract']['total_effort_limit'] = 'PhysxDrivePerformanceEnvelopeAPI:angular bounds implicit drive + feedforward together'
            contract['hardware_limit_authority'] = MCU_LIMIT_EVIDENCE
            contract['joint_action_scale_is_hardware_limit'] = False
            contract['wheel_action_scale_is_hardware_limit'] = False
        contract['export_sha256']={name:hashlib.sha256((args.export_dir/name).read_bytes()).hexdigest() for name in ('policy.pt','policy.onnx')}
        contract['asset_chain_sha256'] = asset_chain
        (args.export_dir/'sim2sim_contract.json').write_text(json.dumps(contract,indent=2,default=str)+'\n')
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/"resolved_env.json").open("w") as f:
        json.dump(cfg.to_dict(),f,indent=2,default=str)
    manifest={"robot":args.robot,"checkpoint":str(args.checkpoint.resolve()),"checkpoint_iteration":checkpoint["iter"],
        "checkpoint_sha256":hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "asset_version":MANIFEST["version"],
        "seconds_per_case":args.seconds,"cases":[dict(name=x[0],speed=x[1],yaw=x[2],height=x[3]) for x in cases],
        "seeds":seeds,"reset_velocity_uniform_amplitude":args.reset_velocity_amplitude,"physics_dt":cfg.sim.dt,"policy_dt":env.step_dt,
        "vmc_dt":1./control_hz,"controller_stride":stride,
        "solver_position_iterations":cfg.robot.spawn.articulation_props.solver_position_iteration_count,
        "solver_velocity_iterations":cfg.robot.spawn.articulation_props.solver_velocity_iteration_count,
        "settle_seconds":args.settle_seconds,"speed_ramp_m_s2":.8,
        "command_scale":args.command_scale,
        "curriculum_progress":args.curriculum_progress,
        "case_names":"canonical acceptance slots; actual targets are in cases",
        "heading":("straight motion holds initial world yaw; spin uses constant yaw rate" if args.heading_hold
                   else "constant requested yaw rate; no added heading feedback"),
        "telemetry":"post-physics, BEFORE automatic reset; errors are actual minus applied reference",
        "weights_updated":False,"policy_input_remapping":False,"leg_control_mode":cfg.leg_control_mode,
        "wheel_probe":args.wheel_probe,"policy_acceptance_test":not (args.wheel_probe or args.bias_probe_json or args.command_probe_json or args.tracking_goal_reward_probe is not None or args.joint_saturation_reward_probe is not None),
        "joint_saturation_reward_probe":args.joint_saturation_reward_probe,
        "tracking_goal_reward_probe":args.tracking_goal_reward_probe,
        "tracking_goal_feedback_criteria":env._tracking_goal_evidence,
        "command_probe":bool(args.command_probe_json),
        "command_sequence":command_sequence,"command_phase_seconds":command_phase_seconds,
        "sampled_policy_actions":args.sample_policy_actions,
        "wheel_action_saturation_reward_probe":args.wheel_action_saturation_reward_probe,
        "policy_noise_seed":43 if args.sample_policy_actions else None,
        "wheel_reference_limit_probe":args.wheel_reference_limit_probe,
        "wheel_reference_max_rad_s":cfg.action_scale_vel * (args.wheel_reference_limit_probe or 1.),
        "action_bias_probes":bias_labels,
        "reward_telemetry":('Actual environment reward per control step; no weights changed'
                            if record_rewards else None),
        "enable_external_forces_every_iteration":env._external_forces_every_iteration,
        "torque_telemetry":"Isaac Lab actuator estimate; implicit Drive actual torques are not directly reported",
        "control_config":"model-specific virtual-leg references and selected actuator execution; no retuning during evaluation"}
    if args.robot=='mine':
        manifest['asset_sha256']=hashlib.sha256(Path(cfg.robot.spawn.usd_path).read_bytes()).hexdigest()
        manifest['asset_chain_sha256'] = asset_chain
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    env.recording=True
    policy_inputs=[]
    policy_means=[]
    with (args.output/"trace.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["time_s","case_id","seed","episode"]+columns)
        with torch.inference_mode():
            for step in range(round(args.seconds/env.step_dt)):
                actions=actor.act_inference(obs)
                if args.record_policy_inputs:
                    policy_inputs.append(obs['policy'].cpu().numpy().copy())
                    policy_means.append(actions.cpu().numpy().copy())
                if args.wheel_probe:
                    # The recorded mean remains the actor's original output.
                    # Only this diagnostic substitutes applied wheel references.
                    actions=actions.clone()
                    actions[:,(2,5)]=actions.new_tensor([(0.,0.),(-.05,.05),(.05,-.05)]).repeat_interleave(3,dim=0)
                if case_biases is not None:
                    actions = add_action_bias(actions, action_biases)
                if args.sample_policy_actions:
                    # Each seed receives the same noise across command cases;
                    # a dedicated generator is unaffected by environment resets.
                    noise = torch.randn((len(seeds), 6), generator=policy_noise_generator, device=env.device)
                    actions = actions + policy_std * noise.repeat(len(cases), 1)
                obs,reward,_,_,_=env.step(actions)
                if not np.isfinite(env.snapshot).all():
                    raise RuntimeError('Nonfinite replay telemetry; evaluation is incomplete')
                snapshot = env.snapshot
                if record_rewards:
                    actual_terms = torch.stack([reward] + [env._last_reward_terms[key]
                        for key in diagnostic_reward_names], dim=-1).cpu().numpy()
                    if not np.isfinite(actual_terms).all():
                        raise RuntimeError('Nonfinite diagnostic reward telemetry')
                    snapshot = np.concatenate((snapshot, actual_terms), axis=1)
                for eid,row in enumerate(snapshot):
                    writer.writerow([(step+1)*env.step_dt,eid//3,seeds[eid%3],int(env.snapshot_episodes[eid]),*row.tolist()])
                if step%100==0:
                    f.flush()
                    print(f"EVAL_PROGRESS robot={args.robot} t={(step+1)*env.step_dt:.2f} resets={env.reset_counts.sum()}",flush=True)
    if args.record_policy_inputs:
        np.savez_compressed(args.output/'policy_inputs.npz',observations=np.stack(policy_inputs),
                            means=np.stack(policy_means),seeds=seeds,policy_dt=env.step_dt)
    final={"completed":True,"samples":round(args.seconds/env.step_dt)*len(cases)*3,"resets":int(env.reset_counts.sum()),
           "controller_updates":env.controller_updates}
    (args.output/"completed.json").write_text(json.dumps(final,indent=2)+"\n")
    print("POLICY_EVALUATION_COMPLETE "+json.dumps(final),flush=True)
    env.close()


if __name__=="__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        app.close()
