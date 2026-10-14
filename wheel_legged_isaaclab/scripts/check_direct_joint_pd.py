"""Bounded direct-joint execution check, not a trained-policy acceptance test.

Hanging isolates four raw motor-angle step responses with gravity/feedforward
off; guide measures implicit PD + gravity compensation support under real
gravity with physical world-slider constraints. Neither is
free balancing or sim2real validation. No trained weights are loaded/updated.
"""
import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--mode', choices=('hanging', 'guide'), required=True)
parser.add_argument('--num_envs', type=int, choices=(1, 4), default=None)
parser.add_argument('--seconds', type=float, default=10.)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--wheel-steps', action='store_true', help='Unloaded fixed-root wheel velocity/sign isolation; no policy or ground contact')
parser.add_argument('--joint-sweep-json', type=Path,
                    help='Guide only: precomputed normalized joint-angle waypoints for height-envelope feasibility; no live IK or policy')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not math.isfinite(args.seconds) or args.seconds < 8 or args.seconds > 30:
    # The final neutral target starts at 6 s. Its last-1-s assessment must
    # begin after a settling interval, not include the 6.02 s transient.
    parser.error('seconds must be finite and between 8 and 30, including final-target settling time')
if args.mode == 'guide' and args.num_envs not in (None, 1):
    parser.error('Only the single-environment world-slider fixture is qualified; use hanging for four-joint isolation')
if args.wheel_steps and args.mode != 'hanging':
    parser.error('Wheel isolation requires the fixed-root hanging fixture')
sweep = None
if args.joint_sweep_json:
    if args.mode != 'guide' or args.wheel_steps:
        parser.error('A joint sweep requires the single-environment guide')
    sweep = json.loads(args.joint_sweep_json.read_text())
    if (not isinstance(sweep, list) or len(sweep) < 2
            or any(set(p) != {'time_s', 'action', 'height_m'} for p in sweep)
            or any(len(p['action']) != 4 or not all(math.isfinite(v) and abs(v) <= 1 for v in p['action'])
                   or not math.isfinite(p['height_m']) or p['height_m'] <= 0 for p in sweep)
            or sweep[0]['time_s'] != 0 or sweep[-1]['time_s'] != args.seconds
            or any(a['time_s'] >= b['time_s'] for a, b in zip(sweep, sweep[1:]))):
        parser.error('Sweep needs ordered finite waypoints, four bounded normalized angles, and the complete test duration')
app = AppLauncher(args).app


def main():
    import numpy as np
    import torch
    from pxr import Gf, UsdPhysics
    from rsl_rl.modules import ActorCritic
    from wheel_legged_gym_isaaclab.leg_control import LEG_ACTION_COLUMNS
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env_cfg import WheelLeggedVMCFlatEnvCfg
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.agents.rsl_rl_ppo_cfg import WheelLeggedVMCFlatPPORunnerCfg

    cfg = WheelLeggedVMCFlatEnvCfg()
    n = args.num_envs if args.num_envs is not None else (4 if args.mode == 'hanging' else 1)
    cfg.scene.num_envs = n
    cfg.sim.device = args.device or 'cuda:0'
    cfg.seed = 83
    cfg.episode_length_s = args.seconds + 2
    cfg.reset_velocity_initial = cfg.reset_velocity_final = 0.
    cfg.commands.ranges_lin_vel_x = (0., 0.)
    cfg.commands.ranges_ang_vel_yaw = (0., 0.)
    cfg.commands.ranges_height = (.30, .30)
    cfg.commands.grouped_training = False
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1000.
    if args.mode == 'hanging':
        cfg.robot.spawn.articulation_props.fix_root_link = True
        cfg.robot.init_state.pos = (0., 0., .9)
        cfg.sim.gravity = (0., 0., 0.)
        cfg.leg_joint_gravity_feedforward = False  # only this unloaded diagnostic
        # Projected gravity is zero in this diagnostic, so the normal
        # gravity-based fallen detector cannot describe body orientation.
        # Keep all other safety gates; defer only this detector beyond the
        # bounded test instead of changing production termination logic.
        cfg.fail_to_terminal_time_s = args.seconds + 1.

    class ProbeEnv(WheelLeggedVMCFlatEnv):
        def _setup_scene(self):
            super()._setup_scene()
            if args.mode == 'guide':
                for i in range(self.scene.cfg.num_envs):
                    root = f'/World/envs/env_{i}/Robot'
                    # World constraints must not live under env_0's inherited
                    # subtree: source edits would be inherited by sibling envs
                    # and interact with replicated world-anchor offsets.
                    guide = UsdPhysics.PrismaticJoint.Define(self.sim.stage, f'/World/DirectJointDiagnostics/slider_{i}')
                    guide.CreateBody1Rel().SetTargets([root + '/base_link'])
                    guide.CreateAxisAttr('Z')
                    origin = self._terrain.env_origins[i].cpu().tolist()
                    origin = [origin[k] + self.cfg.robot.init_state.pos[k] for k in range(3)]
                    guide.CreateLocalPos0Attr(Gf.Vec3f(*origin))
                    guide.CreateLocalPos1Attr(Gf.Vec3f(0., 0., 0.))
                    guide.CreateLocalRot0Attr(Gf.Quatf(1., Gf.Vec3f(0.)))
                    guide.CreateLocalRot1Attr(Gf.Quatf(1., Gf.Vec3f(0.)))
                    guide.CreateExcludeFromArticulationAttr(True)

        def _vmc(self, *unused):
            raise AssertionError('Direct-joint mode must never enter VMC')

    env = ProbeEnv(cfg)
    rows = []
    try:
        obs, _ = env.reset()
        runner_cfg = WheelLeggedVMCFlatPPORunnerCfg()
        policy_cfg = runner_cfg.policy.to_dict()
        policy_cfg.pop('class_name')
        actor = ActorCritic(obs, runner_cfg.obs_groups, 6, **policy_cfg).to(env.device)
        actor.eval()
        inferred = actor.act_inference(obs)
        assert obs['policy'].shape == (n, 27) and inferred.shape == (n, 6)
        assert torch.isfinite(obs['policy']).all() and torch.isfinite(inferred).all()
        drive = env._robot.root_physx_view
        kp = drive.get_dof_stiffnesses()[0, env._leg_joint_ids].cpu().tolist()
        kd = drive.get_dof_dampings()[0, env._leg_joint_ids].cpu().tolist()
        caps = drive.get_dof_max_forces()[0, env._leg_joint_ids].cpu().tolist()
        wheel_caps = drive.get_dof_max_forces()[0, env._wheel_joint_ids].cpu().tolist()
        assert all(abs(v - cfg.leg_joint_stiffness) < 1e-5 for v in kp)
        assert all(abs(v - cfg.leg_joint_damping) < 1e-5 for v in kd)
        assert all(abs(v - cfg.leg_effort_limit) < 1e-4 for v in caps)
        assert all(abs(v - cfg.wheel_effort_limit) < 1e-5 for v in wheel_caps)
        from omni.physx.bindings._physx import PERF_ENV_API
        combined_limit_enabled = all(
            env.sim.stage.GetPrimAtPath(f'/World/envs/env_{i}/Robot/joints/{name}').HasAPI(PERF_ENV_API, 'angular')
            for i in range(n) for name in cfg.leg_joint_names
        )
        assert combined_limit_enabled, 'All cloned active joints require a combined PD + feedforward cap'
        resets = 0
        explicit_peak = 0.
        obs_error = 0.
        feedforward_error = 0.
        termination_counts = {}
        steps = round(args.seconds / env.step_dt)
        for step in range(steps):
            t = step * env.step_dt
            action = torch.zeros(n, 6, device=env.device)
            if args.mode == 'hanging':
                # Each environment isolates one active hip. Positive/negative
                # steps are raw motor-angle changes, not mirrored virtual axes.
                value = .15 if 2 <= t < 4 else -.15 if 4 <= t < 6 else 0.
                for i, column in enumerate(LEG_ACTION_COLUMNS):
                    action[i % n, column] = value
                if args.wheel_steps:
                    action.zero_()
                    wheel_action = .05 if 2 <= t < 4 else -.05 if 4 <= t < 6 else 0.
                    action[:, (2, 5)] = wheel_action
            elif sweep is not None:
                left, right = next((a, b) for a, b in zip(sweep, sweep[1:]) if a['time_s'] <= t < b['time_s'])
                fraction = (t - left['time_s']) / (right['time_s'] - left['time_s'])
                start = action.new_tensor(left['action'])
                end = action.new_tensor(right['action'])
                action[0, LEG_ACTION_COLUMNS] = start + fraction * (end - start)
            elif 2 <= t < 6:
                if n == 1:
                    sign = 1. if t < 4 else -1.
                    action[0, LEG_ACTION_COLUMNS] = sign * action.new_tensor([-.1, .1, .1, -.1])
                else:
                    action[1, LEG_ACTION_COLUMNS] = action.new_tensor([-.1, .1, .1, -.1])
                    action[2, LEG_ACTION_COLUMNS] = action.new_tensor([.1, -.1, -.1, .1])
            with torch.inference_mode():
                obs, reward, terminated, timeout, _ = env.step(action)
            resets += int((terminated | timeout).sum())
            for name, mask in env._termination_reasons.items():
                termination_counts[name] = termination_counts.get(name, 0) + int(mask.sum())
            explicit = env._robot.data.joint_effort_target[:, env._leg_joint_ids]
            explicit_peak = max(explicit_peak, float(explicit.abs().max()))
            feedforward_error = max(feedforward_error, float((explicit - env._leg_gravity_effort_applied).abs().max()))
            q = env._robot.data.joint_pos[:, env._leg_joint_ids]
            dq = env._robot.data.joint_vel[:, env._leg_joint_ids]
            offset = q.new_tensor(cfg.leg_joint_position_offset)
            expected = torch.cat(((q - offset).clamp(-cfg.obs_clip.theta, cfg.obs_clip.theta) * cfg.obs_scales.dof_pos,
                                  dq.clamp(-cfg.obs_clip.theta_dot, cfg.obs_clip.theta_dot) * cfg.obs_scales.dof_vel), dim=-1)
            obs_error = max(obs_error, float((obs['policy'][:, 9:17] - expected).abs().max()))
            assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
            assert all(float(env._last_reward_terms[k].abs().max()) == 0. for k in
                       ('leg_length_action_saturation', 'leg_length_action_difference', 'leg_length_target_underreach'))
            torque_estimate = env._robot.data.applied_torque[:, env._leg_joint_ids]
            rows.append(dict(time_s=(step + 1) * env.step_dt,
                             q=q.cpu().tolist(), dq=dq.cpu().tolist(),
                             target=env._joint_position_ref_applied.cpu().tolist(),
                             leg_length=env._L0.cpu().tolist(),
                             leg_angle=env._theta0.cpu().tolist(),
                             height=env._robot.data.root_pos_w[:, 2].cpu().tolist(),
                             root_position=env._robot.data.root_pos_w.cpu().tolist(),
                             root_quaternion=env._robot.data.root_quat_w.cpu().tolist(),
                             projected_gravity=env._robot.data.projected_gravity_b.cpu().tolist(),
                             wheel_velocity_forward=env._get_wheel_vel_forward_positive().cpu().tolist(),
                             wheel_reference_forward=(env._actions[:, (2, 5)] * cfg.action_scale_vel).cpu().tolist(),
                             base_contact_force_n=env._contact_sensor.data.net_forces_w[:, env._base_id].cpu().tolist(),
                             force_drive_estimate_nm=torque_estimate.cpu().tolist(),
                             explicit_leg_effort_nm=explicit.cpu().tolist(),
                             gravity_feedforward_nm=env._leg_gravity_effort_applied.cpu().tolist(),
                             physical_finite=bool(torch.isfinite(env._robot.data.joint_pos).all()),
                             terminated=terminated.cpu().tolist()))
            if step % 100 == 0:
                print(f'[DIRECT JOINT PD PROGRESS] mode={args.mode} t={(step + 1)*env.step_dt:.2f} resets={resets}', flush=True)
        # Gate step accuracy only in the zero-gravity fixed-root test. Loaded
        # joint PD has static compliance, so guide support is not exact tracking.
        tail = rows[-round(1 / env.step_dt):]
        errors = [abs(a - b) for row in tail for qa, qt in zip(row['q'], row['target']) for a, b in zip(qa, qt)]
        max_tail_error = max(errors)
        min_height = min(h for row in rows for h in row['height'])
        max_estimate = max(abs(x) for row in rows for qa in row['force_drive_estimate_nm'] for x in qa)
        accuracy = args.mode != 'hanging' or max_tail_error < .003
        support = args.mode != 'guide' or min_height > cfg.min_root_height
        effort_path_ok = feedforward_error < 1e-6 and explicit_peak <= cfg.leg_effort_limit + 1e-5
        effort_path_ok = effort_path_ok and (explicit_peak > .1 if cfg.leg_joint_gravity_feedforward else explicit_peak == 0.)
        report = dict(mode=args.mode, seconds=args.seconds, num_envs=n,
                      action_order=['LB_q', 'LL_q', 'LW_speed', 'RB_q', 'RL_q', 'RW_speed'],
                      observation_shape=[n, 27], policy_output_shape=[n, 6],
                      leg_joint_ids=env._leg_joint_ids, joint_names=cfg.leg_joint_names,
                      joint_offset_rad=list(cfg.leg_joint_position_offset), joint_scale_rad=list(cfg.leg_joint_position_scale),
                      kp_nm_per_rad=kp, kd_nm_s_per_rad=kd, torque_limits_nm=caps,
                      wheel_torque_limits_nm=wheel_caps,
                      explicit_leg_effort_peak_nm=explicit_peak, max_actuator_estimate_nm=max_estimate,
                      torque_note='Implicit actuator estimate, not directly measured PhysX drive torque',
                      gravity_feedforward_enabled=cfg.leg_joint_gravity_feedforward,
                      gravity_feedforward_channel_max_error_nm=feedforward_error,
                      combined_pd_feedforward_limit_enabled=combined_limit_enabled,
                      obs_leg_block_max_error=obs_error, resets=resets,
                      termination_counts=termination_counts,
                      max_final_joint_error_rad=max_tail_error, minimum_height_m=min_height,
                      steady_height_m=np.mean([r['height'] for r in tail], axis=0).tolist(),
                      passed=resets == 0 and accuracy and support and effort_path_ok and combined_limit_enabled and obs_error < 1e-6,
                      limitations='Fixed-root/vertical-guide tests only; not trained free balancing or sim2real acceptance',
                      weights_updated=False, training_resumed=False,
                      resolved_config=cfg.to_dict())
        if args.mode == 'hanging':
            report['positive_step_max_error_rad'] = max(abs(a - b) for row in rows if 3.5 <= row['time_s'] < 4
                                                       for qa, qt in zip(row['q'], row['target']) for a, b in zip(qa, qt))
            report['negative_step_max_error_rad'] = max(abs(a - b) for row in rows if 5.5 <= row['time_s'] < 6
                                                       for qa, qt in zip(row['q'], row['target']) for a, b in zip(qa, qt))
            report['passed'] = report['passed'] and report['positive_step_max_error_rad'] < .003 and report['negative_step_max_error_rad'] < .003
        if args.wheel_steps:
            errors = [abs(actual - target) for row in rows
                      if 3.5 <= row['time_s'] < 4. or 5.5 <= row['time_s'] < 6.
                      for velocity, reference in zip(row['wheel_velocity_forward'], row['wheel_reference_forward'])
                      for actual, target in zip(velocity, reference)]
            report['wheel_step_max_error_rad_s'] = max(errors)
            report['passed'] = report['passed'] and max(errors) < .01
            report['wheel_isolation'] = True
            report['limitations'] = 'Unloaded, zero-gravity fixed-root wheel channel/sign test; not free balance or ground steering acceptance'
        if sweep is not None:
            from wheel_legged_gym_isaaclab.height_commands import MEASURED_LEG_STOP_RANGE_M
            steady = []
            for left, right in zip(sweep, sweep[1:]):
                if left['action'] != right['action'] or right['time_s'] - left['time_s'] < 2:
                    continue
                window = [r for r in rows if right['time_s'] - 1 <= r['time_s'] < right['time_s']]
                lengths = [v for r in window for pair in r['leg_length'] for v in pair]
                heights = [v for r in window for v in r['height']]
                steady.append(dict(target_height_m=left['height_m'],
                                   actual_height_mean_m=float(np.mean(heights)),
                                   height_mae_m=float(np.mean([abs(v-left['height_m']) for v in heights])),
                                   leg_length_min_m=min(lengths), leg_length_max_m=max(lengths),
                                   minimum_stop_margin_m=min(min(lengths)-MEASURED_LEG_STOP_RANGE_M[0],
                                                             MEASURED_LEG_STOP_RANGE_M[1]-max(lengths))))
            report['joint_sweep'] = sweep
            report['height_envelope_steady'] = steady
            report['passed'] = report['passed'] and len(steady) >= 3 and all(
                p['height_mae_m'] < .1*p['target_height_m'] and p['minimum_stop_margin_m'] > .005 for p in steady)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'assessment.json').write_text(json.dumps(report, indent=2, default=str) + '\n')
        (args.output / 'samples.json').write_text(json.dumps(rows) + '\n')
        print('[DIRECT JOINT PD RESULT]', json.dumps({k: report[k] for k in
              ('mode', 'passed', 'resets', 'minimum_height_m', 'max_final_joint_error_rad', 'explicit_leg_effort_peak_nm')}), flush=True)
        if not report['passed']:
            raise RuntimeError('Direct-joint actuator feasibility gate failed; inspect saved telemetry')
    finally:
        env.close()


try:
    main()
finally:
    app.close()
