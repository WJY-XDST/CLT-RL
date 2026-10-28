"""Explicit reset-noise curriculum for static and later gradual-motion stages.

Configuration preparation only. Never changes a running environment, policy,
optimizer or acceptance file. The original full-motion reset bound is kept as
an explicit argument so a saved static configuration cannot erase it.
"""
import copy
import math

from joint_pd_continuation import validate_continuation_config


def reset_final_for_motion(initial, full_motion_final, motion_scale):
    if (not all(type(value) in (int, float) and math.isfinite(value)
                for value in (initial, full_motion_final, motion_scale))
            or not 0. <= initial <= full_motion_final or not 0. <= motion_scale <= 1.):
        raise ValueError('Use finite ordered reset bounds and a motion scale in[0,1]')
    return float(initial + motion_scale * (full_motion_final - initial))


def prepare_stationary_reset(saved_cfg, *, full_motion_final):
    """Static commands retain initial reset noise, without changing the PPO clock."""
    if (saved_cfg['leg_control_mode'] != 'implicit_joint_position'
            or saved_cfg['commands'].get('training_motion_scale') != 0.):
        raise ValueError('Prepare only the declared direct-joint stationary stage')
    initial = saved_cfg['reset_velocity_initial']
    target = reset_final_for_motion(initial, full_motion_final, 0.)
    if not initial <= saved_cfg['reset_velocity_final'] <= full_motion_final:
        raise ValueError('The source reset noise exceeds the explicitly retained full-motion bound')
    result = copy.deepcopy(saved_cfg)
    result['reset_velocity_final'] = target
    validate_stationary_reset_revision(saved_cfg, result, full_motion_final=full_motion_final)
    return result


def validate_stationary_reset_revision(previous, current, *, full_motion_final):
    if previous['commands'].get('training_motion_scale') != 0. or current['commands'].get('training_motion_scale') != 0.:
        raise ValueError('A motion-stage change needs its own explicit review')
    expected = copy.deepcopy(previous)
    target = reset_final_for_motion(previous['reset_velocity_initial'], full_motion_final, 0.)
    expected['reset_velocity_final'] = target
    if current['reset_velocity_final'] != target:
        raise ValueError('Keep initial reset noise during the static stage')
    # Only this sampling field is substituted before the existing complete
    # configuration guard; mass, gains, action map, rewards and commands stay strict.
    validate_continuation_config(expected, current, previous['commands'])
    return dict(only_sampling_change={'reset_velocity_final':
        [previous['reset_velocity_final'], current['reset_velocity_final']]},
        reset_velocity_initial=previous['reset_velocity_initial'],
        full_motion_reset_velocity_final_retained=float(full_motion_final),
        training_motion_scale=0., original_PPO_clock_preserved=True,
        network_and_optimizer_untouched=True, acceptance_unchanged=True)
