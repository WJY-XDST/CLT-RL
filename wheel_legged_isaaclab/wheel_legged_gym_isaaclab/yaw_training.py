"""Tensor helpers for mixed straight/turn commands and centered pivot turns."""

import torch


TASK_NAMES = ("standing", "spin", "forward", "reverse", "forward_turn", "reverse_turn")


def sample_grouped_motion(env_ids, config, progress, reverse_progress, yaw_progress, phase=0):
    """Assign equal task groups by global env ID and rotate roles each command phase.

    Groups are interleaved in the scene. Each group has 2048 members with 12288
    environments. All groups still share one policy and the height curriculum.
    """
    groups = (env_ids + phase) % len(TASK_NAMES)
    count, device = env_ids.numel(), env_ids.device
    progress = min(max(float(progress), 0.), 1.)
    reverse_progress = min(max(float(reverse_progress), 0.), 1.)
    low, high = config.ranges_lin_vel_x
    vmax = low + (high - low) * progress
    reverse_low, reverse_high = config.ranges_reverse_lin_vel_x
    vmin = reverse_high + (reverse_low - reverse_high) * reverse_progress
    forward = low + (vmax - low) * torch.rand(count, device=device)
    reverse = vmin + (reverse_high - vmin) * torch.rand(count, device=device)
    boundary = torch.rand(count, device=device) < config.speed_boundary_fraction
    forward = torch.where(boundary, vmax, forward)
    reverse = torch.where(boundary, vmin, reverse)
    slow = torch.rand(count, device=device) < config.grouped_low_speed_fraction
    transition_low, transition_high = config.ranges_transition_lin_vel_x
    # Keep low-speed practice in the group's assigned direction, avoiding accidental zero commands.
    positive_max = transition_high * progress
    negative_max = -transition_low * reverse_progress
    positive_min, negative_min = min(.02, positive_max), min(.02, negative_max)
    forward_slow = positive_min + (positive_max - positive_min) * torch.rand(count, device=device)
    reverse_slow = -(negative_min + (negative_max - negative_min) * torch.rand(count, device=device))
    forward = torch.where(slow, forward_slow, forward)
    reverse = torch.where(slow, reverse_slow, reverse)
    speed = torch.zeros(count, device=device)
    speed = torch.where((groups == 2) | (groups == 4), forward, speed)
    speed = torch.where((groups == 3) | (groups == 5), reverse, speed)
    # Original startup curriculum is respected when training from scratch.
    enabled = torch.rand(count, device=device) < progress
    enabled &= ~(((groups == 3) | (groups == 5)) & (reverse_progress == 0.))
    speed = torch.where(enabled, speed, 0.)
    yaw = sample_yaw_rates(count, device=device, low=config.ranges_ang_vel_yaw[0],
                           high=config.ranges_ang_vel_yaw[1], fraction=1., minimum=config.yaw_min_abs_rate,
                           boundary_fraction=config.yaw_boundary_fraction, progress=min(progress, yaw_progress))
    yaw = torch.where((groups == 1) | (groups == 4) | (groups == 5), yaw, 0.)
    return speed, yaw


def sample_yaw_rates(count, *, device, low, high, fraction, minimum, boundary_fraction, progress):
    """Sample both turn directions independently of the forward-speed population."""
    rates = torch.zeros(count, device=device)
    progress = min(max(float(progress), 0.0), 1.0)
    if progress == 0.0 or fraction == 0.0 or low == high == 0.0:
        return rates
    if not (low < 0 < high and 0 <= minimum <= min(-low, high)):
        raise ValueError("Yaw training requires a range spanning zero and a valid minimum magnitude")
    if not (0 <= fraction <= 1 and 0 <= boundary_fraction <= 1):
        raise ValueError("Yaw fractions must be between zero and one")
    active = torch.rand(count, device=device) < fraction
    positive = torch.rand(count, device=device) < 0.5
    maximum = torch.where(positive, high, -low)
    magnitude = minimum + (maximum - minimum) * torch.rand(count, device=device)
    boundary = torch.rand(count, device=device) < boundary_fraction
    magnitude = torch.where(boundary, maximum, magnitude) * progress
    return torch.where(active, torch.where(positive, magnitude, -magnitude), rates)


def spin_center_velocity_penalty(wheel_velocities, wheel_radius, command_speed, target_speed, yaw_rate):
    """Penalize wheel common-mode motion only when a stationary spin is requested.

    Wheel coordinates must both be forward-positive. Equal opposite wheel speeds
    are valid for a centered spin; unequal speeds remain allowed during travel.
    """
    spinning = (command_speed.abs() < 0.01) & (target_speed.abs() < 0.01) & (yaw_rate.abs() > 0.05)
    center_velocity = wheel_radius * wheel_velocities.mean(dim=1)
    return center_velocity.square() * spinning
