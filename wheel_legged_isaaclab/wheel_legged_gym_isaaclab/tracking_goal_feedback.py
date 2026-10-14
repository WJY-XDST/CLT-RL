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


def tracking_goal_violation_penalty(commands, speed, yaw, height, gravity,
                                    length, length_reference, angle, angle_reference,
                                    wheels, wheel_reference, criteria):
    """Huber cost of the largest instantaneous normalized excess over one.

    A good height cannot compensate a failed speed, posture or wheel reference.
    Outer errors retain linear feedback. Reference tolerances use instantaneous
    references here; the original evaluator uses window means and is unchanged.
    """
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
    excess = (ratios.amax(-1) - 1.).clamp_min(0.)
    near = excess.clamp(max=1.)
    return .5 * near.square() + (excess - 1.).clamp_min(0.)
