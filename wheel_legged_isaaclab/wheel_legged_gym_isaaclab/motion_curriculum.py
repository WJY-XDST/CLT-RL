"""Review-controlled motion amplitude, independent of the saved PPO clock."""
import math


def training_motion_scale(commands):
    # Missing metadata belongs to historical, unrestricted motion curricula.
    value = getattr(commands, 'training_motion_scale', 1.)
    if (type(value) not in (float, int) or not math.isfinite(value)
            or not 0. <= value <= 1.):
        raise ValueError('training_motion_scale must be finite and in [0, 1]')
    return float(value)


def scale_training_motion(speed, yaw, commands):
    scale = training_motion_scale(commands)
    if scale == 1.:
        return speed, yaw
    return speed * scale, yaw * scale
