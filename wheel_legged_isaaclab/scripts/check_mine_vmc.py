"""Verify actual J-transpose leg efforts, disabled joint PD and finite GPU dynamics.

This is a controller smoke test, not a learned standing/motion acceptance test.
No hardware ports, model weights or permanent USD assets are modified.
"""
import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--overrides_json', type=Path)
parser.add_argument('--num_envs', type=int, default=16)
parser.add_argument('--seconds', type=float, default=2.)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app


def main():
    import torch
    from wheel_legged_gym_isaaclab.config_overrides import apply_config_overrides
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env_cfg import WheelLeggedVMCFlatEnvCfg
    from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv

    cfg = WheelLeggedVMCFlatEnvCfg()
    if args.overrides_json:
        apply_config_overrides(cfg, json.loads(args.overrides_json.read_text()))
    cfg.leg_control_mode = 'explicit_vmc'
    # Deliberately request residual position gains to verify they are disabled.
    cfg.robot.actuators['legs'].stiffness = 300.
    cfg.robot.actuators['legs'].damping = 3.
    cfg.scene.num_envs = args.num_envs
    cfg.scene.clone_in_fabric = False
    cfg.sim.device = args.device or 'cuda:0'
    cfg.seed = 43
    cfg.reset_velocity_initial = cfg.reset_velocity_final = 0.
    cfg.commands.ranges_lin_vel_x = (0., 0.)
    cfg.commands.ranges_ang_vel_yaw = (0., 0.)
    cfg.commands.ranges_height = (.30, .30)
    cfg.commands.grouped_training = False
    cfg.commands.heading_command = False
    cfg.episode_length_s = args.seconds + 2.

    class AuditEnv(WheelLeggedVMCFlatEnv):
        effort_checks = 0
        effort_max = 0.
        effort_max_error = 0.

        def _vmc(self, force, torque):
            result = super()._vmc(force, torque)
            self.expected_effort = torch.stack(
                (result[0][:,0], result[1][:,0], result[0][:,1], result[1][:,1]), dim=-1
            ).clamp(-self.cfg.leg_effort_limit, self.cfg.leg_effort_limit)
            return result

        def _apply_action(self):
            super()._apply_action()
            actual = self._robot.data.joint_effort_target[:, self._leg_joint_ids]
            if not torch.isfinite(actual).all():
                raise RuntimeError('Non-finite VMC effort target')
            error = (actual-self.expected_effort).abs().max().item()
            self.effort_max_error = max(self.effort_max_error, error)
            self.effort_max = max(self.effort_max, actual.abs().max().item())
            self.effort_checks += 1
            if error > 1.e-5:
                raise RuntimeError(f'VMC effort was replaced or cleared: max error={error}')

    env = AuditEnv(cfg)
    obs, _ = env.reset()
    stiffness = env._robot.root_physx_view.get_dof_stiffnesses()[:, env._leg_joint_ids]
    damping = env._robot.root_physx_view.get_dof_dampings()[:, env._leg_joint_ids]
    if stiffness.abs().max().item() != 0. or damping.abs().max().item() != 0.:
        raise RuntimeError('Residual leg joint PD is active in explicit VMC mode')
    original_position_target = env._robot.set_joint_position_target
    def reject_position_target(*unused_args, **unused_kwargs):
        raise RuntimeError('Explicit VMC must not send a joint-position target')
    env._robot.set_joint_position_target = reject_position_target
    actions = torch.zeros(args.num_envs, 6, device=env.device)
    # Mirrored modest targets exercise both legs, not a zero-effort shortcut.
    actions[::2, (0,3)] = .1
    actions[1::2, (0,3)] = -.1
    counts = {}
    minimum_height = float('inf')
    with torch.inference_mode():
        for _ in range(round(args.seconds/env.step_dt)):
            obs, reward, terminated, timeout, extra = env.step(actions)
            if not torch.isfinite(obs['policy']).all() or not torch.isfinite(reward).all():
                raise RuntimeError('Non-finite observations or reward')
            minimum_height = min(minimum_height, env._robot.data.root_pos_w[:,2].min().item())
            for name, mask in env._termination_reasons.items():
                counts[name] = counts.get(name, 0) + int(mask.sum().item())
    env._robot.set_joint_position_target = original_position_target
    report = dict(
        controller_contract_passed=env.effort_checks>0 and env.effort_max>0,
        leg_control_mode=cfg.leg_control_mode, wheel_control_mode=cfg.wheel_control_mode,
        joint_drive_stiffness=stiffness.tolist(), joint_drive_damping=damping.tolist(),
        checked_physics_steps=env.effort_checks, maximum_leg_effort_nm=env.effort_max,
        maximum_jt_effort_target_error_nm=env.effort_max_error,
        minimum_post_reset_root_height_m=minimum_height, termination_counts=counts,
        learned_policy_tested=False, hardware_controller_calibrated=False,
        runtime_config=cfg.to_dict(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str)+'\n')
    print('VMC_CONTROLLER_AUDIT '+json.dumps({k:v for k,v in report.items()
          if k not in ('runtime_config','joint_drive_stiffness','joint_drive_damping')}), flush=True)
    env.close()
    if not report['controller_contract_passed']:
        raise RuntimeError('No nonzero VMC effort was exercised')


try:
    main()
except BaseException:
    # Kit shutdown can wait on worker threads; retain the original failure
    # before entering close(), including in a redirected headless log.
    import sys
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
