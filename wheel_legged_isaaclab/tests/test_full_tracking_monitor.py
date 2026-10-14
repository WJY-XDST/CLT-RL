"""Full gate failures cannot disappear from the learning-stagnation monitor."""
import copy

import pytest

from joint_pd_continuation import tracking_monitoring_scope, stagnant_monitoring_metrics
from joint_pd_optimization import TASK_GOAL_METRICS, REFERENCE_TRACKING_METRICS

KEYS = (*TASK_GOAL_METRICS, *REFERENCE_TRACKING_METRICS)


def fixture():
    cfg = dict(commands=dict(ranges_height=[.22534645531, .37534645531],
               height_expansion_from=[.28, .32], height_expansion_start_steps=117024,
               height_expansion_ramp_steps=48000, standing_only_steps=48000, motion_ramp_steps=48000),
               rewards=dict(base_height_target=.3))
    report = dict(conditions=[dict(name=name, trials=[dict(metrics=dict.fromkeys(KEYS, .8),
                                                   limits=dict.fromkeys(KEYS, 1.))])
                              for name in ('stand', 'forward_0p6', 'height_low', 'height_high')])
    return report, cfg


def test_completed_height_course_monitors_both_endpoints_and_all_seven_metrics():
    report, cfg = fixture()
    report['conditions'][-1]['trials'][0]['metrics']['height_mae_m'] = 3.
    result = tracking_monitoring_scope(report, cfg, 3438, 48)
    assert result['height_course_complete'] and result['scope'] == 'all_conditions'
    assert result['excluded_conditions'] == []
    assert set(result['metric_ratios']) == set(KEYS) and len(KEYS) == 7
    assert result['worst_ratio'] == 3.
    assert result['restored_curriculum_control_steps'] == 165024
    assert result['training_height_bounds'] == pytest.approx(cfg['commands']['ranges_height'])


def test_reference_failure_is_monitored_even_when_all_task_errors_pass():
    report, cfg = fixture()
    report['conditions'][0]['trials'][0]['metrics']['wheel_speed_mae_rad_s'] = 4.
    result = tracking_monitoring_scope(report, cfg, 4059, 48)
    assert result['worst_ratio'] == 4.
    assert result['metric_ratios']['wheel_speed_mae_rad_s'] == 4.


def test_unreached_endpoints_are_explicitly_scoped_without_changing_acceptance():
    report, cfg = fixture()
    saved = copy.deepcopy(report)
    report['conditions'][-1]['trials'][0]['metrics']['height_mae_m'] = 10.
    result = tracking_monitoring_scope(report, cfg, 3000, 48)
    assert result['scope'] == 'central_until_height_expanded'
    assert result['excluded_conditions'] == ['height_low', 'height_high']
    assert result['worst_ratio'] == .8 and result['acceptance_unchanged']
    assert report['conditions'][-1]['trials'][0]['metrics']['height_mae_m'] == 10.
    assert cfg['commands']['ranges_height'] == [.22534645531, .37534645531]
    assert saved['conditions'][0] == report['conditions'][0]


def histories(height, speed, wheel=None):
    return [dict(scope='all_conditions', conditions=['stand','height_low','height_high'],
                 metric_ratios=dict(height_mae_m=h, speed_mae_m_s=s,
                                    wheel_speed_mae_rad_s=(wheel[i] if wheel else .8)))
            for i, (h, s) in enumerate(zip(height, speed))]


def test_improving_speed_cannot_hide_failed_height_and_wheel_plateaus():
    values = histories([2.5,2.49,2.45], [2.,1.3,.8], [4.,4.05,4.04])
    assert stagnant_monitoring_metrics(values[:2]) == []
    assert stagnant_monitoring_metrics(values) == ['height_mae_m', 'wheel_speed_mae_rad_s']


def test_real_progress_and_passing_metrics_do_not_trigger_stagnation():
    assert stagnant_monitoring_metrics(histories([2.5,2.3,2.0], [2.,1.3,.8])) == []
    assert stagnant_monitoring_metrics(histories([.8,.8,.8], [.8,.8,.8])) == []


def test_changed_scope_needs_three_comparable_milestones():
    values=histories([2.5,2.49,2.45], [2.,2.1,2.2])
    values[0]['scope']='central_until_height_expanded'
    assert stagnant_monitoring_metrics(values) == []


@pytest.mark.parametrize('iteration,steps', [(4059,0),(-1,48),(True,48)])
def test_invalid_curriculum_clock_is_rejected(iteration,steps):
    report,cfg=fixture()
    with pytest.raises(ValueError):
        tracking_monitoring_scope(report,cfg,iteration,steps)
