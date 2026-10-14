"""Tensor helpers for mixed straight/turn commands and centered pivot turns."""

import torch


TASK_NAMES = ("standing", "spin", "forward", "reverse", "forward_turn", "reverse_turn")


def curriculum_replay_cases(height, config, progress):
    """Diagnostic targets, not a change to the training sample distribution.

    Main speeds start at +/-0.2 m/s while their enabled population ramps.
    Height and yaw amplitudes, unlike main minimum speed, do ramp from zero.
    Canonical slot names are retained; the manifest records exact targets.
    """
    if not 0. <= progress <= 1.:
        raise ValueError('Curriculum replay progress must be in [0, 1]')
    start_forward = config.ranges_lin_vel_x[0]
    start_reverse = config.ranges_reverse_lin_vel_x[1]
    forward = [start_forward + (target - start_forward) * progress for target in (.3, .6)]
    reverse = [start_reverse + (target - start_reverse) * progress for target in (-.3, -.6)]
    height_low, height_high = config.ranges_height
    return [('stand', 0., 0., height),
            ('forward_0p3', forward[0], 0., height), ('forward_0p6', forward[1], 0., height),
            ('reverse_0p3', reverse[0], 0., height), ('reverse_0p6', reverse[1], 0., height),
            ('spin_left', 0., .3 * progress, height), ('spin_right', 0., -.3 * progress, height),
            ('height_low', 0., 0., height + (height_low - height) * progress),
            ('height_high', 0., 0., height + (height_high - height) * progress)]


def grouped_task_ids(env_ids, phase=0, dedicated_standing_count=None):
    """Keep dedicated global IDs standing and rotate only the remaining motion roles.

    None preserves the historical six-role rotation. A nonnegative count reserves
    IDs [0, count) for standing; the other IDs rotate through five motion roles.
    Passing global rather than batch-relative IDs also preserves partial resets.
    """
    if dedicated_standing_count is None:
        return (env_ids + phase) % len(TASK_NAMES)
    if (isinstance(dedicated_standing_count, bool)
            or not isinstance(dedicated_standing_count, int)
            or dedicated_standing_count < 0):
        raise ValueError("Dedicated standing count must be a nonnegative integer")
    motion_groups = 1 + (env_ids - dedicated_standing_count + phase) % (len(TASK_NAMES) - 1)
    return torch.where(env_ids < dedicated_standing_count, 0, motion_groups)


def sample_grouped_motion(env_ids, config, progress, reverse_progress, yaw_progress, phase=0,
                          *, dedicated_standing_count=None):
    """Sample assigned task roles using one shared policy and height curriculum."""
    groups = grouped_task_ids(env_ids, phase, dedicated_standing_count)
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
