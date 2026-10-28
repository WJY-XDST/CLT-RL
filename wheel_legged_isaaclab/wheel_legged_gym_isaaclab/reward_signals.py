"""Simulator-independent penalties for applied six-dimensional references."""
import math

DIRECT_JOINT_REWARD_PROFILE = 'joint_pd_balance_v1'


def task_tracking_quality(height, command_height, gravity, criteria, floor):
    """Motion bonus quality, not acceptance or a command/action override.

    Full bonuses remain inside half of the ORIGINAL height/body tolerances.
    Smooth degradation outside does not zero motion learning: floor >= .05.
    """
    if height.ndim != 1 or command_height.shape != height.shape or gravity.shape != (*height.shape,3):
        raise ValueError('Expected one height/command and three gravity coordinates per environment')
    if type(floor) not in (int,float) or not math.isfinite(floor) or not .05 <= floor <= 1.:
        raise ValueError('Motion quality floor must remain finite in [.05,1]')
    relative, body_deg = criteria['relative_error'], criteria['body_p95_deg']
    if not all(math.isfinite(v) and v>0 for v in (relative,body_deg)):
        raise ValueError('Quality guard requires the original positive tracking criteria')
    height_ratio = (height-command_height).abs()/(relative*command_height.abs().clamp_min(.001))
    pitch = gravity[:,0].clamp(-1.,1.).asin().abs()
    roll = (-gravity[:,1]).atan2(-gravity[:,2]).abs()
    body_ratio = pitch.maximum(roll)/math.radians(body_deg)
    height_excess = (height_ratio-.5).clamp_min(0.)
    body_excess = (body_ratio-.5).clamp_min(0.)
    return floor+(1.-floor)/((1.+height_excess.square())*(1.+body_excess.square()))


def guard_tracking_bonuses(signals, height, command_height, gravity, criteria, floor):
    if floor == 1.:
        return signals  # exact historical reduction/order, not even an extra multiply
    quality = task_tracking_quality(height,command_height,gravity,criteria,floor)
    return {key:value*quality if key in ('track_lin_vel','track_ang_vel') else value
            for key,value in signals.items()}


def validate_task_quality_gate(floor, has_criteria):
    if (type(floor) not in (int,float) or not math.isfinite(floor) or not .05<=floor<=1.
            or (floor<1. and not has_criteria)):
        raise ValueError('A nonzero-floor motion quality guard requires original tracking criteria')


def joint_target_length_error_squared(length_reference, height_command, height_offset):
    """Score direct motor-target FK against the requested upright leg length.

    This is a training reward, not IK or an action/height controller. The two
    sides are scored separately before averaging: an overlong leg cannot
    cancel an underlength leg. Do not apply the legacy VMC 0.32 m cap.
    """
    if (length_reference.ndim != 2 or length_reference.shape[-1] != 2
            or height_command.shape != length_reference.shape[:1]
            or not math.isfinite(height_offset)):
        raise ValueError('Expected two applied FK lengths, one height command and a finite offset')
    target = height_command[:, None] + height_offset
    return (length_reference-target).square().mean(-1)


def validate_joint_target_length_reward(mode, coefficient):
    if (type(coefficient) not in (int, float) or not math.isfinite(coefficient)
            or not -1000. <= coefficient <= 0.
            or (coefficient != 0. and mode != 'implicit_joint_position')):
        raise ValueError('Joint-target length reward requires direct joint PD and a finite scale in [-1000,0]')


def height_tracking_kernel(height, target, sigma, relative_deadband=0.0):
    """Do not trade failed motion tracking for already adequate height precision.

    Zero preserves historical checkpoints exactly. A nonzero inner band removes
    incentives to polish millimetres there; errors outside still receive a dense
    restoring reward. This is reward shaping, never an acceptance tolerance.
    """
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError('Height reward width must be finite and positive')
    if not math.isfinite(relative_deadband) or not 0 <= relative_deadband < .1:
        raise ValueError('Height reward deadband must be smaller than ten percent')
    error = height - target
    if relative_deadband == 0:
        return (-error.square() / sigma).exp()
    excess = (error.abs() - relative_deadband * target.abs()).clamp_min(0.)
    return (-excess.square() / sigma).exp()




def wheel_action_overshoot_penalty(raw_actions):
    """Huber cost on discarded wheel outputs; zero throughout valid references."""
    if raw_actions.ndim != 2 or raw_actions.shape[-1] != 6:
        raise ValueError('Expected a batch of six raw policy actions')
    excess = (raw_actions[:, (2, 5)].clamp(-10., 10.).abs() - 1.).clamp_min(0.)
    near = excess.clamp(max=1.)
    return (.5 * near.square() + (excess - 1.).clamp_min(0.)).sum(-1)


def horizontal_velocity_error_squared(velocity, forward_command, lateral_multiplier=0.0):
    """Heading-frame vx tracking plus optional vy drift, for every command.

    Zero multiplier reproduces the legacy forward-only squared penalty.
    This is not a standing-only reward or a left/right wheel equality rule.
    """
    if velocity.ndim != 2 or velocity.shape[-1] != 2 or forward_command.shape != velocity.shape[:1]:
        raise ValueError('Expected horizontal velocity [N,2] and forward command [N]')
    if not math.isfinite(lateral_multiplier) or lateral_multiplier < 0:
        raise ValueError('Lateral velocity multiplier must be finite and nonnegative')
    return (forward_command - velocity[:, 0]).square() + lateral_multiplier * velocity[:, 1].square()


def merge_step_diagnostics(episode_log, diagnostics):
    """Publish a fresh log after resets without aliasing the cached scalars.

    DirectRLEnv resets after rewards and replaces its episode log. RSL-RL also
    converts log values into tensors in place, so a shared dictionary is unsafe.
    """
    return {**episode_log, **diagnostics}


def action_second_difference_penalty(current, previous, older, wheel_multiplier=0.0):
    """Penalize reference chatter, not a constant wheel speed or balancing torque.

    Columns 0/1 and 3/4 are leg references in the selected control mode;
    2/5 are wheel references.
    A zero wheel multiplier reproduces the historical leg-only penalty.
    """
    if current.shape != previous.shape or current.shape != older.shape:
        raise ValueError("Action histories must have identical shapes")
    if current.ndim != 2 or current.shape[-1] != 6:
        raise ValueError("Expected a batch of six-dimensional applied actions")
    if not math.isfinite(wheel_multiplier) or wheel_multiplier < 0:
        raise ValueError("Wheel smoothness multiplier must be finite and nonnegative")
    difference = current - 2.0 * previous + older
    return (
        difference[:, (0, 1, 3, 4)].square().sum(-1)
        + wheel_multiplier * difference[:, (2, 5)].square().sum(-1)
    )
