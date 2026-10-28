"""Supplement source3001 spin practice at its historical failing commands.

Only half the existing spin-role requests are selected. Dedicated standing
IDs, the other four motion roles, speed slew, rewards and control are retained.
The remainder keeps the complete original continuous command distribution.
"""
import math
from types import MethodType

import torch

from wheel_legged_gym_isaaclab.yaw_training import grouped_task_ids


FRACTION = .5
YAW_VALUES = (.05, .0625, .075)
YAW_WEIGHTS = (.2, .2, .6)
HEIGHT_WEIGHTS = (.35, .25, .4)


def specification():
    return dict(fraction_of_existing_spin_role=FRACTION,
                yaw_abs_values=list(YAW_VALUES), yaw_weights=list(YAW_WEIGHTS),
                height_values=[.3, .35, .37534645531], height_weights=list(HEIGHT_WEIGHTS),
                both_yaw_signs_equal_probability=True,
                other_motion_roles_and_dedicated_standing_unchanged=True)


def apply_after_original(env, env_ids):
    ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    empty = ids[:0]
    if not ids.numel() or float(env._command_ranges[0, 0]) == float(env._command_ranges[0, 1]):
        return empty
    cfg = env.cfg.commands
    assert cfg.training_motion_scale == .25 and cfg.grouped_training and cfg.fixed_standing_groups
    assert env._dedicated_standing_count == env.num_envs // 4
    assert math.isclose(float(cfg.ranges_height[1]), .37534645531, abs_tol=1.e-10)
    step = float(env.common_step_counter)
    if (step < cfg.standing_only_steps + cfg.motion_ramp_steps or
            step < cfg.yaw_start_steps + cfg.yaw_ramp_steps or
            step < cfg.reverse_ramp_start_steps + cfg.reverse_ramp_steps):
        return empty
    period = max(1, int(cfg.resampling_time / env.step_dt))
    roles = grouped_task_ids(ids, phase=env.common_step_counter // period,
                             dedicated_standing_count=env._dedicated_standing_count)
    eligible = ids[(roles == 1) & (ids >= env._dedicated_standing_count)]
    if not eligible.numel():
        return empty
    selected = eligible[torch.rand(eligible.numel(), device=env.device) < FRACTION]
    size = selected.numel()
    if not size:
        return empty
    yaw = env._commands.new_tensor(YAW_VALUES)[torch.multinomial(
        env._commands.new_tensor(YAW_WEIGHTS), size, replacement=True)]
    yaw = torch.where(torch.rand(size, device=env.device) < .5, yaw, -yaw)
    heights = env._commands.new_tensor([.3, .35, float(cfg.ranges_height[1])])
    height = heights[torch.multinomial(env._commands.new_tensor(HEIGHT_WEIGHTS), size, replacement=True)]
    # Original resampling already assigned zero target speed to this spin role.
    assert torch.all(env._target_lin_vel_x[selected] == 0.)
    env._commands[selected, 1] = yaw
    env._commands[selected, 2] = height
    return selected


def install(env):
    if getattr(env, "_historical_yaw_focus_installed", False):
        return
    original = env._resample_commands_for
    env._historical_yaw_focus_calls = 0
    env._historical_yaw_focus_selected = 0

    def resample(self, ids):
        original(ids)
        selected = apply_after_original(self, ids)
        self._historical_yaw_focus_calls += 1
        self._historical_yaw_focus_selected += selected.numel()

    env._resample_commands_for = MethodType(resample, env)
    env._historical_yaw_focus_installed = True
