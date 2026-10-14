"""Height commands derived from the protected CAD leg stops, in metres.

This is a command envelope and curriculum, not a leg controller. The policy
still supplies four motor-angle targets. The upright conversion is not a
world-height estimator for a tilted body.
"""
import math

from wheel_legged_gym_isaaclab.robot_model import MODEL

# Same asset chain as protected_stops_v2_live_cad/report.json. Conservative
# intersection of both legs and both overstroke tests, fixed root/zero gravity.
MEASURED_LEG_STOP_RANGE_M = (0.17886618077754973, 0.35423238188028333)
LEG_STOP_MARGIN_M = 0.01
# Rounded inward after applying the 10 mm margin, not firmware command extrema.
SAFE_LEG_LENGTH_RANGE_M = (0.19, 0.34)
UPRIGHT_HEIGHT_OFFSET_M = MODEL['wheel_radius'] - MODEL['hip_z']
SAFE_BASE_HEIGHT_RANGE_M = tuple(v + UPRIGHT_HEIGHT_OFFSET_M for v in SAFE_LEG_LENGTH_RANGE_M)


def training_height_bounds(commands, center, step, motion_progress):
    """Expand only height from a saved range; never restart speed/yaw curricula."""
    final = tuple(commands.ranges_height)
    initial = getattr(commands, 'height_expansion_from', None)
    p = min(max(float(motion_progress), 0.), 1.)
    if initial is None:
        return tuple(center + (v - center) * p for v in final)
    if (len(initial) != 2 or not all(math.isfinite(v) for v in (*initial, *final))
            or not final[0] <= initial[0] <= center <= initial[1] <= final[1]):
        raise ValueError('Height expansion must start inside the final range and contain the nominal height')
    start = commands.height_expansion_start_steps
    duration = commands.height_expansion_ramp_steps
    if type(start) is not int or start < 0 or type(duration) is not int or duration <= 0:
        raise ValueError('Height expansion requires a nonnegative step and positive duration')
    expansion = min(max((float(step) - start) / duration, 0.), 1.)
    envelope = tuple(a + (b - a) * expansion for a, b in zip(initial, final))
    return tuple(center + (v - center) * p for v in envelope)
