"""Tensor helpers for mixed straight/turn commands and centered pivot turns."""

import torch


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
