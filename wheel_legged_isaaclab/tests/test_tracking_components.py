"""Body goals and policy references must not overwrite the strict full gate."""
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'mine_model/scripts'
sys.path.insert(0, str(SCRIPTS))
from summarize_mine_trial import (REFERENCE_TRACKING_METRICS, TASK_GOAL_METRICS,
                                  tracking_components)
from watch_joint_pd_scratch import restore_standing_history, standing_summary


def sample():
    keys = TASK_GOAL_METRICS + REFERENCE_TRACKING_METRICS
    return (dict.fromkeys(keys, .01) | dict(command_reached=True, upright_support=True),
            dict.fromkeys(keys, .1))


def assessment(wheel_error=.56):
    metrics, limits = sample()
    metrics['wheel_speed_mae_rad_s'] = wheel_error
    trial = dict(seed=83, survived=True, first_failure_s=None, metrics=metrics,
                 limits=limits, **tracking_components(True, metrics, limits),
                 passed=all(metrics[k] <= v for k, v in limits.items()))
    case = dict(name='stand', survivors=1, trials=[trial], passed=trial['passed'],
                task_goal_passed=trial['task_goal_passed'],
                reference_tracking_passed=trial['reference_tracking_passed'])
    return dict(iteration=200, survivors=1, conditions=[case])


def test_body_pass_with_wheel_reference_gap_keeps_full_failure():
    report = assessment()
    summary = standing_summary(report)
    assert summary['task_goal_passed'] is True
    assert summary['reference_tracking_passed'] is False
    assert summary['passed'] is False
    assert summary['failed_metrics_by_seed'] == {'83': ['wheel_speed_mae_rad_s']}


def test_all_metrics_pass_without_altering_limits():
    metrics, limits = sample()
    original = dict(limits)
    components = tracking_components(True, metrics, limits)
    assert components['task_goal_passed'] and components['reference_tracking_passed']
    assert not components['failed_metrics']
    assert limits == original


@pytest.mark.parametrize('prerequisite', ['survived', 'command_reached', 'upright_support'])
def test_missing_prerequisite_cannot_pass_either_component(prerequisite):
    metrics, limits = sample()
    if prerequisite != 'survived':
        metrics[prerequisite] = False
    components = tracking_components(prerequisite != 'survived', metrics, limits)
    assert not components['task_goal_passed']
    assert not components['reference_tracking_passed']
    assert components['failed_prerequisites'] == [prerequisite]


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -.01])
def test_invalid_metric_not_counted_as_pass(invalid):
    metrics, limits = sample()
    metrics['speed_mae_m_s'] = invalid
    components = tracking_components(True, metrics, limits)
    assert not components['task_goal_passed']
    assert components['failed_metrics'] == ['speed_mae_m_s']


def test_failed_episode_has_no_steady_state_pass():
    _, limits = sample()
    components = tracking_components(False, None, limits)
    assert not components['task_goal_passed']
    assert not components['reference_tracking_passed']
    assert len(components['failed_metrics']) == 7


def test_missing_metric_cannot_pass():
    metrics, limits = sample()
    del metrics['height_mae_m']
    assert tracking_components(True, metrics, limits)['failed_metrics'] == ['height_mae_m']


@pytest.mark.parametrize('invalid', [0., -1., float('nan'), float('inf'), None])
def test_invalid_limit_rejected(invalid):
    metrics, limits = sample()
    if invalid is None:
        del limits['speed_mae_m_s']
    else:
        limits['speed_mae_m_s'] = invalid
    with pytest.raises(ValueError, match='acceptance limit'):
        tracking_components(True, metrics, limits)


def test_history_enriched_without_overwriting_old_assessment(tmp_path):
    report = assessment()
    directory = tmp_path / 'checkpoint_200'
    directory.mkdir()
    original = directory / 'assessment.json'
    original.write_text('{"historical_evidence": true}')
    (directory / 'assessment_components.json').write_text(json.dumps(report))
    history = [dict(iteration=200, survivors=1, trials=1, passed=False)]
    restored = restore_standing_history(tmp_path, history)
    assert restored[0]['task_goal_passed'] is True
    assert restored[0]['passed'] is False
    assert history == [dict(iteration=200, survivors=1, trials=1, passed=False)]
    assert original.read_text() == '{"historical_evidence": true}'


@pytest.mark.parametrize('mismatch', ['iteration', 'passed', 'survivors', 'trials'])
def test_history_mismatch_rejected(tmp_path, mismatch):
    directory = tmp_path / 'checkpoint_200'
    directory.mkdir()
    (directory / 'assessment.json').write_text(json.dumps(assessment()))
    history = [dict(iteration=200, survivors=1, trials=1, passed=False)]
    history[0][mismatch] = {'iteration':400, 'passed':True, 'survivors':0, 'trials':3}[mismatch]
    if mismatch == 'iteration':
        (tmp_path / 'checkpoint_400').mkdir()
        (tmp_path / 'checkpoint_400' / 'assessment.json').write_text(json.dumps(assessment()))
    with pytest.raises(ValueError, match='differs|changes'):
        restore_standing_history(tmp_path, history)


def test_legacy_history_without_report_kept(tmp_path):
    history = [dict(iteration=200, survivors=3, trials=3, passed=False)]
    assert restore_standing_history(tmp_path, history) == history
