"""Direct motor-angle and wheel-speed requests; local PD/gravity stay on the MCU."""

import math

from .schema import ACTION_DIM, finite_vector


def clip(value, lower, upper):
    return min(upper, max(lower, value))


def wheel_action_limit(cfg):
    value = cfg.get("wheel_action_limit", 1.)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 1. <= value <= 2.):
        raise ValueError("wheel_action_limit must be a finite number in [1, 2]")
    return float(value)


def clip_action(action, cfg):
    values = finite_vector(action, ACTION_DIM, "action")
    wheel_bound = wheel_action_limit(cfg)
    bounds = (1., 1., wheel_bound, 1., 1., wheel_bound)
    return [clip(value, -bound, bound) for value, bound in zip(values, bounds)]


def action_to_targets(action, observation, cfg):
    if cfg.get('leg_control_mode') != 'implicit_joint_position':
        raise ValueError('Retired virtual-leg/IK action contracts are not supported')
    action = clip_action(action, cfg)
    s = cfg["obs_scales"]
    offset = finite_vector(cfg['leg_joint_position_offset'], 4, 'motor offsets')
    scale = finite_vector(cfg['leg_joint_position_scale'], 4, 'motor action scales')
    if any(v <= 0 for v in scale):
        raise ValueError('Motor action scales must be positive')
    return dict(
        joint_position_ref_rad=[center + action[i] * amplitude
                                for i, center, amplitude in zip((0, 1, 3, 4), offset, scale)],
        joint_order=['LB', 'LL', 'RB', 'RL'],
        wheel_speed_ref_rad_s=[action[i] * cfg["action_scale_vel"] for i in (2, 5)],
        height_command_m=observation[8] / s["height_measurements"],
        vx_command_m_s=observation[6] / s["lin_vel"],
        target_stage="DIRECT_MOTOR_ANGLE_REQUEST; MCU must enforce verified joint stops and re-encode applied references",
        gravity_feedforward_enabled=bool(cfg['leg_joint_gravity_feedforward']),
        wheel_convention="left/right forward-positive; MCU applies raw axes [+1,-1] once",
    )
