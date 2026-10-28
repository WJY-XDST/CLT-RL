# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause
"""新车的命令课程与抽样，仅由 env._resample_commands_for 调用。

这里没有第二个环境、车辆配置或控制器。env 中保留全部 LAB 回调，
把较长的抽样流程放在本文件，以免遮住观测、奖励和动作执行主线。
"""
import torch
from wheel_legged_gym_isaaclab.yaw_training import sample_grouped_motion
from wheel_legged_gym_isaaclab.height_commands import training_height_bounds
from wheel_legged_gym_isaaclab.motion_curriculum import scale_training_motion


def resample_commands(self, env_ids):
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
    if env_ids.numel() == 0:
        return
    previous_speed = self._commands[env_ids, 0].clone()

    lin_min = float(self._command_ranges[0, 0].item())
    lin_max = float(self._command_ranges[0, 1].item())

    # 固定速度范围用于回放；按原来的随机数顺序抽取 yaw 和高度。
    if lin_min == lin_max:
        rand = torch.rand(env_ids.numel(), 3, device=self.device)
        self._commands[env_ids, :3] = (
            self._command_ranges[:, 1] - self._command_ranges[:, 0]
        )[None, :] * rand + self._command_ranges[:, 0][None, :]
    else:
        step = float(self.common_step_counter)
        standing_steps = float(self.cfg.commands.standing_only_steps)
        ramp_steps = max(float(self.cfg.commands.motion_ramp_steps), 1.0)
        progress = min(max((step - standing_steps) / ramp_steps, 0.0), 1.0)
        reverse_progress = min(
            max(
                (step - self.cfg.commands.reverse_ramp_start_steps)
                / self.cfg.commands.reverse_ramp_steps,
                0.0,
            ),
            1.0,
        )
        # 课程先从零速和中心高度开始，随后扩展到配置的高度范围。
        self._commands[env_ids, :3] = 0.0
        height_center = self.cfg.rewards.base_height_target
        progressive_height_min, progressive_height_max = training_height_bounds(
            self.cfg.commands, height_center, step, progress)
        self._commands[env_ids, 2] = progressive_height_min + (
            progressive_height_max - progressive_height_min
        ) * torch.rand(env_ids.numel(), device=self.device)
        height_sample = torch.rand(env_ids.numel(), device=self.device)
        half_boundary_fraction = self.cfg.commands.height_boundary_fraction / 2.0
        low_height_ids = env_ids[height_sample < half_boundary_fraction]
        high_height_ids = env_ids[
            (height_sample >= half_boundary_fraction)
            & (height_sample < 2.0 * half_boundary_fraction)
        ]
        self._commands[low_height_ids, 2] = progressive_height_min
        self._commands[high_height_ids, 2] = progressive_height_max
        # 当前训练只保留任务分组：固定站立组，其余环境轮换五种运动任务。
        yaw_progress = (step - self.cfg.commands.yaw_start_steps) / self.cfg.commands.yaw_ramp_steps
        speed, yaw = sample_grouped_motion(
            env_ids, self.cfg.commands, progress, reverse_progress, yaw_progress,
            phase=self.common_step_counter // max(1, int(self.cfg.commands.resampling_time / self.step_dt)),
            dedicated_standing_count=self._dedicated_standing_count,
        )
        speed, yaw = scale_training_motion(speed, yaw, self.cfg.commands)
        self._commands[env_ids, 0] = speed
        self._commands[env_ids, 1] = yaw
    self._target_lin_vel_x[env_ids] = self._commands[env_ids, 0]
    self._commands[env_ids, 0] = previous_speed
