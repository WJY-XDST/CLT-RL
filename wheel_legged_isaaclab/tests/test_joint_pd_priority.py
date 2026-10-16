import copy

import pytest

from joint_pd_priority import (CASE_NAMES, REWARD_KEY, compare_priority,
    next_shaping_cost, priority_summary, revised_rewards)


def report():
    conditions = []
    for name in CASE_NAMES:
        conditions.append(dict(name=name, speed=0., yaw=0., height=.3,
            trials=[dict(seed=s, survived=True, resets=0, first_failure_s=None,
                metrics=dict(body_p95_deg=1., speed_mae_m_s=.01, yaw_mae_rad_s=.01, height_mae_m=.01),
                limits=dict(body_p95_deg=2.5, speed_mae_m_s=.025, yaw_mae_rad_s=.025, height_mae_m=.03))
                for s in (83,84,85)]))
    return dict(conditions=conditions, total_trials=27, survivors=27, resets=0, passed=False)


def trial(value, name='stand'):
    return next(c for c in value['conditions'] if c['name'] == name)['trials'][0]


@pytest.mark.parametrize('key', ['body_p95_deg', 'speed_mae_m_s', 'yaw_mae_rad_s'])
def test_height_improvement_cannot_hide_lost_standing_stability(key):
    before = report(); after = copy.deepcopy(before)
    trial(after)['metrics'][key] = trial(after)['limits'][key]*2
    for case in after['conditions']:
        for row in case['trials']: row['metrics']['height_mae_m'] = .001
    result = compare_priority(after, before)
    assert result['verdict'] == 'stability_regression'
    assert not result['stability_retained'] and not result['full_original_gate_passed']


def test_lost_motion_survival_is_stability_loss_despite_better_standing():
    before = report(); after = copy.deepcopy(before)
    row = trial(after, 'reverse_0p3'); row.update(survived=False, resets=1, metrics=None, first_failure_s=4.)
    after.update(survivors=26, resets=1)
    trial(after)['metrics']['body_p95_deg'] = .1
    assert compare_priority(after, before)['verdict'] == 'stability_regression'
    assert priority_summary(after)['missing_steady_trials'] == 1


def test_height_regression_is_not_hidden_by_speed_improvement():
    before = report(); after = copy.deepcopy(before)
    trial(after, 'forward_0p3')['metrics'].update(height_mae_m=.05, speed_mae_m_s=0.)
    assert compare_priority(after, before)['verdict'] == 'height_regression'


def test_missing_endpoint_height_is_not_zero_error():
    value = report(); row = trial(value, 'height_high')
    row.update(survived=False, resets=1, metrics=None, first_failure_s=2.)
    value.update(survivors=26, resets=1)
    result = priority_summary(value)
    assert result['worst_endpoint_height_relative_mae'] is None
    assert not result['stationary_pose_passed']


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1.])
def test_invalid_telemetry_is_an_error(bad):
    value = report(); trial(value)['metrics']['body_p95_deg'] = bad
    with pytest.raises(ValueError): priority_summary(value)


def test_changed_original_limits_cannot_be_compared():
    before = report(); after = copy.deepcopy(before)
    trial(after)['limits']['body_p95_deg'] = 3.
    with pytest.raises(ValueError): compare_priority(after, before)


def test_policy_dependent_reference_limits_are_not_mistaken_for_relaxed_task_limits():
    before = report(); after = copy.deepcopy(before)
    trial(before)['limits']['wheel_speed_mae_rad_s'] = .15
    trial(after)['limits']['wheel_speed_mae_rad_s'] = .2
    assert compare_priority(after, before)['verdict'] == 'no_measured_priority_change'


def test_secondary_height_shaping_change_preserves_primary_rewards():
    source = dict(orientation=-400., base_height_error_sq=-500.,
                  joint_target_leg_length_error_sq=-500., tracking_lin_vel=2., tracking_ang_vel=1.)
    revised = revised_rewards(source, -250.)
    assert {k for k in source if source[k] != revised[k]} == {REWARD_KEY}
    assert source[REWARD_KEY] == -500.
    assert revised['orientation'] == -400. and revised['base_height_error_sq'] == -500.
    assert revised['tracking_lin_vel'] > 0 and revised['tracking_ang_vel'] > 0


def test_stability_loss_changes_reward_without_exclusive_task_phases():
    assert next_shaping_cost(-250., []) == -250.
    assert next_shaping_cost(-250., [dict(stability_retained=True)]) == -250.
    assert next_shaping_cost(-250., [dict(stability_retained=False)]) == -100.
    assert next_shaping_cost(0., [dict(stability_retained=False)]) == 0.
