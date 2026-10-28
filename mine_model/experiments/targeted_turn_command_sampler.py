"""Provisional joint-command practice, requiring separate trial qualification.

No active runtime imports or installs this module. The original sampler runs
first; only a declared fraction of its spin/turn targets can be replaced.
Standing IDs, slew limiting, actions, rewards and physics are untouched.
Fixed-command replay and a zero fraction consume no additional RNG draws.
"""
import math
from types import MethodType

import torch

from wheel_legged_gym_isaaclab.yaw_training import grouped_task_ids

CONFIG_KEY = 'targeted_turn_command_fraction'


def apply_after_original(env, env_ids):
    """Replace selected requested targets, preserving the applied speed buffer."""
    fraction = float(getattr(env.cfg.commands, CONFIG_KEY, 0.))
    if not math.isfinite(fraction) or not 0. <= fraction <= 1.:
        raise ValueError('Targeted command fraction must be finite and in[0,1]')
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    empty = ids[:0]
    if not fraction or ids.numel() == 0:
        return empty
    if float(env._command_ranges[0, 0]) == float(env._command_ranges[0, 1]):
        return empty
    cfg = env.cfg.commands
    if (not cfg.grouped_training or not cfg.fixed_standing_groups
            or env._dedicated_standing_count is None or cfg.training_motion_scale != 1.):
        raise ValueError('Require full-envelope grouped training with fixed standing IDs')
    step = float(env.common_step_counter)
    mature = (step >= cfg.standing_only_steps + cfg.motion_ramp_steps
              and step >= cfg.reverse_ramp_start_steps + cfg.reverse_ramp_steps
              and step >= cfg.yaw_start_steps + cfg.yaw_ramp_steps)
    if not mature:
        return empty
    forward_max = float(cfg.ranges_lin_vel_x[1])
    reverse_max = -float(cfg.ranges_reverse_lin_vel_x[0])
    yaw_max = float(cfg.ranges_ang_vel_yaw[1])
    height_low, height_high = map(float, cfg.ranges_height)
    height_nominal = float(env.cfg.rewards.base_height_target)
    if (not math.isclose(forward_max, reverse_max, rel_tol=0., abs_tol=1.e-12)
            or not math.isclose(yaw_max, -float(cfg.ranges_ang_vel_yaw[0]), rel_tol=0., abs_tol=1.e-12)
            or not forward_max > 0. or not yaw_max > 0.
            or not 0. <= float(cfg.ranges_lin_vel_x[0]) <= forward_max * .25
            or not -reverse_max * .25 <= float(cfg.ranges_reverse_lin_vel_x[1]) <= 0.
            or float(cfg.ranges_transition_lin_vel_x[1]) < forward_max * .125
            or float(cfg.ranges_transition_lin_vel_x[0]) > -reverse_max * .125
            or not height_low < height_nominal < height_high):
        raise ValueError('Re-derive the target grid for different symmetric command bounds')
    period = max(1, int(cfg.resampling_time / env.step_dt))
    groups = grouped_task_ids(ids, phase=env.common_step_counter // period,
                              dedicated_standing_count=env._dedicated_standing_count)
    eligible = ((groups == 1) | (groups == 4) | (groups == 5)) & (ids >= env._dedicated_standing_count)
    positions = eligible.nonzero(as_tuple=False).flatten()
    if positions.numel() == 0:
        return empty
    positions = positions[torch.rand(positions.numel(), device=env.device) < fraction]
    chosen, roles = ids[positions], groups[positions]
    size = chosen.numel()
    if size == 0:
        return empty
    # Discrete targets supplement the original continuous distribution.
    speed_bins = env._commands.new_tensor([.125, .25, .5, .75, 1.]) * forward_max
    yaw_bins = env._commands.new_tensor([.25, .5, 1.]) * yaw_max
    height_bins = env._commands.new_tensor([height_low, height_nominal, height_high])
    speed = speed_bins[torch.randint(5, (size,), device=env.device)]
    speed = torch.where(roles == 5, -speed, speed)
    speed = torch.where(roles == 1, 0., speed)
    yaw = yaw_bins[torch.randint(3, (size,), device=env.device)]
    yaw = torch.where(torch.rand(size, device=env.device) < .5, yaw, -yaw)
    height = height_bins[torch.randint(3, (size,), device=env.device)]
    env._target_lin_vel_x[chosen] = speed
    env._commands[chosen, 1] = yaw
    env._commands[chosen, 2] = height
    return chosen


def install(env):
    """Bind one environment only; a future qualified runtime must call this."""
    if getattr(env, '_targeted_turn_sampler_installed', False):
        return
    original = env._resample_commands_for
    env._targeted_turn_sampler_original_resample = original
    env._targeted_turn_sampler_calls = 0
    env._targeted_turn_sampler_selected = 0

    def resample(self, env_ids):
        original(env_ids)
        selected = apply_after_original(self, env_ids)
        self._targeted_turn_sampler_calls += 1
        self._targeted_turn_sampler_selected += selected.numel()

    env._resample_commands_for = MethodType(resample, env)
    env._targeted_turn_sampler_installed = True
