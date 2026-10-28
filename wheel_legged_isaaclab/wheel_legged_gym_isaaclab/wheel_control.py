"""Saved wheel reference limits, separate from policy scaling and motor ratings."""
import math


def validated_wheel_action_limit(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('wheel_action_limit must be a finite number in [1, 2]')
    value = float(value)
    if not math.isfinite(value) or not 1. <= value <= 2.:
        raise ValueError('wheel_action_limit must be a finite number in [1, 2]')
    return value


def apply_wheel_action_limit(applied, raw, limit):
    """Keep all four leg columns intact; limit only wheel columns 2 and 5.

    The unit limit returns the historical action tensor without extra arithmetic.
    The wider limit matches the frozen wheel-reference intervention exactly.
    Neither the scale nor a motor's physical speed/torque limit changes here.
    """
    limit = validated_wheel_action_limit(limit)
    if applied.shape != raw.shape or raw.ndim != 2 or raw.shape[-1] != 6:
        raise ValueError('Expected paired six-dimensional action tensors')
    if limit == 1.:
        return applied
    result = applied.clone()
    result[:, (2, 5)] = raw[:, (2, 5)].clamp(-limit, limit)
    return result


def checkpoint_wheel_action_limit(saved_config, overrides=None):
    """Historical checkpoints omitted this field and used a unit wheel clamp."""
    value = (overrides or {}).get('wheel_action_limit', saved_config.get('wheel_action_limit', 1.))
    return validated_wheel_action_limit(value)
