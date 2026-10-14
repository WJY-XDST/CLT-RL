"""Pointwise learning feedback scaled by the existing full tracking gate.

The independent gate still uses MAE/P95 over complete replay windows. This
instantaneous cost is a training signal, never a replacement acceptance test.
"""
import hashlib
import json
import math
from pathlib import Path

import torch


def load_tracking_goal_criteria(path=None):
    path = Path(path) if path is not None else (
        Path(__file__).resolve().parents[2] / 'mine_model/config/tracking_acceptance.json')
    content = path.read_bytes()
    criteria = json.loads(content)
    required = ('relative_error', 'speed_absolute_m_s', 'yaw_absolute_rad_s',
                'wheel_absolute_rad_s', 'leg_angle_design_range_rad', 'body_p95_deg',
                'minimum_height_fraction')
    for key in required:
        value = criteria.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid tracking-goal criterion: ' + key)
    if criteria['relative_error'] > .1 or criteria['minimum_height_fraction'] > 1:
        raise ValueError('Tracking feedback requires the original ten-percent gate')
    return criteria, dict(path=str(path.resolve()), sha256=hashlib.sha256(content).hexdigest(),
                          criteria=criteria)


def _tracking_goal_error_ratios(commands, speed, yaw, height, gravity,
                                    length, length_reference, angle, angle_reference,
                                    wheels, wheel_reference, criteria):
    """Shared pointwise tolerances; the independent window gate is unchanged."""
    count = commands.shape[0]
    if (commands.shape != (count, 3) or gravity.shape != (count, 3)
            or any(value.shape != (count,) for value in (speed, yaw, height))
            or any(value.shape != (count, 2) for value in
                   (length, length_reference, angle, angle_reference, wheels, wheel_reference))):
        raise ValueError('Expected matched physical tracking batches')
    fraction = criteria['relative_error']
    speed_limit = (fraction * commands[:, 0].abs()).clamp_min(criteria['speed_absolute_m_s'])
    yaw_limit = (fraction * commands[:, 1].abs()).clamp_min(criteria['yaw_absolute_rad_s'])
    height_limit = fraction * commands[:, 2].abs().clamp_min(.001)
    length_limit = fraction * length_reference.abs().amin(-1).clamp_min(.001)
    wheel_limit = (fraction * wheel_reference.abs().amin(-1)).clamp_min(criteria['wheel_absolute_rad_s'])
    pitch = torch.asin(gravity[:, 0].clamp(-1., 1.))
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    body = torch.maximum(pitch.abs(), roll.abs()) * (180. / math.pi)
    ratios = torch.stack((
        (speed - commands[:, 0]).abs() / speed_limit,
        (yaw - commands[:, 1]).abs() / yaw_limit,
        (height - commands[:, 2]).abs() / height_limit,
        (length - length_reference).abs().amax(-1) / length_limit,
        (angle - angle_reference).abs().amax(-1) / (fraction * criteria['leg_angle_design_range_rad']),
        (wheels - wheel_reference).abs().amax(-1) / wheel_limit,
        body / criteria['body_p95_deg'],
    ), dim=-1)
    return ratios


def tracking_goal_violation_penalty(commands, speed, yaw, height, gravity,
                                    length, length_reference, angle, angle_reference,
                                    wheels, wheel_reference, criteria):
    """Keep the original Huber cost of the largest instantaneous excess."""
    ratios = _tracking_goal_error_ratios(
        commands, speed, yaw, height, gravity, length, length_reference,
        angle, angle_reference, wheels, wheel_reference, criteria)
    excess = (ratios.amax(-1) - 1.).clamp_min(0.)
    near = excess.clamp(max=1.)
    return .5 * near.square() + (excess - 1.).clamp_min(0.)


def tracking_goal_component_violation_penalty(commands, speed, yaw, height, gravity,
                                              length, length_reference, angle, angle_reference,
                                              wheels, wheel_reference, criteria):
    """Sum nonnegative Huber excesses so concurrent failures retain feedback.

    Passing components contribute zero and cannot cancel a failed component.
    This opt-in term supplements the worst-error feedback; neither term proves
    the independent MAE/P95 window acceptance or sustained verification.
    """
    ratios = _tracking_goal_error_ratios(
        commands, speed, yaw, height, gravity, length, length_reference,
        angle, angle_reference, wheels, wheel_reference, criteria)
    excess = (ratios - 1.).clamp_min(0.)
    near = excess.clamp(max=1.)
    return (.5 * near.square() + (excess - 1.).clamp_min(0.)).sum(-1)


def validate_tracking_goal_reward_config(rewards, leg_control_mode):
    """Validate opt-in bounds without coupling disabled legacy rewards to PD."""
    goal_scale = getattr(rewards, 'tracking_goal_violation', 0.)
    goal_cap = getattr(rewards, 'tracking_goal_penalty_clip', 40.)
    if not math.isfinite(goal_scale) or not -2. <= goal_scale <= 0.:
        raise ValueError('Tracking-goal feedback must be finite, nonpositive and bounded by minus two')
    if goal_scale and leg_control_mode != 'implicit_joint_position':
        raise ValueError('Tracking-goal feedback requires direct joint-position PD')
    if not math.isfinite(goal_cap) or goal_cap <= 0 or goal_cap < 24.5 * abs(goal_scale):
        raise ValueError('Tracking-goal cap must retain feedback up to 26 times the original tolerances')
    component_scale = getattr(rewards, 'tracking_goal_component_violation', 0.)
    component_cap = getattr(rewards, 'tracking_goal_component_penalty_clip', 200.)
    if not math.isfinite(component_scale) or not -1. <= component_scale <= 0.:
        raise ValueError('Component tracking feedback must be finite and in [-1,0]')
    if component_scale and leg_control_mode != 'implicit_joint_position':
        raise ValueError('Component tracking feedback requires direct joint-position PD')
    if (not math.isfinite(component_cap) or component_cap <= 0
            or component_cap < 7. * 24.5 * abs(component_scale)):
        raise ValueError('Component cap must retain all seven concurrent errors up to 26 times tolerance')
