"""Drift diagnostics cannot turn an early fall into an apparent improvement."""
import csv
import json
import math

import pytest

from train_joint_pd_fresh500 import drift_summary, comparison, METRICS


def trace(directory, *, fail=False):
    cases = [dict(name=name, speed=speed, yaw=yaw, height=height) for name, speed, yaw, height in (
        ('stand', 0., 0., .3), ('height_low', 0., 0., .22534645531),
        ('height_high', 0., 0., .37534645531), ('forward', .3, 0., .3), ('spin', 0., .3, .3))]
    (directory / 'manifest.json').write_text(json.dumps(dict(cases=cases, seeds=[83, 84, 85], seconds_per_case=4)))
    columns = ['time_s', 'case_id', 'seed', 'episode', 'root_x_w_m', 'root_y_w_m',
               'terminated', 'timeout', 'speed_m_s', 'lateral_speed_m_s']
    with (directory / 'trace.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        for cid, case in enumerate(cases):
            for seed in (83, 84, 85):
                for step in range(4):
                    failed_trial = fail and cid == 0 and seed == 83
                    writer.writerow([step, cid, seed, int(failed_trial and step >= 2),
                        100 * cid + seed + step * .02, -100 * cid - seed + step * .03,
                        int(failed_trial and step == 1), 0, .02, .03])


def test_world_xy_drift_subtracts_each_origin_and_excludes_commanded_motion(tmp_path):
    trace(tmp_path)
    result = drift_summary(tmp_path)
    assert result['all_stationary_trials_survived'] and len(result['trials']) == 9
    assert result['mean_displacement_xy_m'] == pytest.approx(math.hypot(.06, .09))
    assert result['max_displacement_xy_m'] == pytest.approx(math.hypot(.06, .09))
    for trial in result['trials']:
        assert trial['path_length_xy_m'] == pytest.approx(math.hypot(.06, .09))
        assert trial['steady_speed_xy_mae_m_s'] == pytest.approx(math.hypot(.02, .03))


def test_shorter_displacement_before_a_fall_is_not_scored_as_less_drift(tmp_path):
    trace(tmp_path, fail=True)
    result = drift_summary(tmp_path)
    assert not result['all_stationary_trials_survived']
    assert result['mean_displacement_xy_m'] is result['max_displacement_xy_m'] is None
    failed = result['trials'][0]
    assert not failed['survived'] and failed['steady_speed_xy_mae_m_s'] is None


def test_partial_survival_does_not_claim_global_height_improvement_or_acceptance():
    def report(survivors, value):
        return dict(survivors=survivors, total_trials=27, resets=27 - survivors, passed=False,
            conditions=[dict(trials=[dict(metrics={key: value for key in METRICS},
                                          limits={key: 1. for key in METRICS})])])
    result = comparison(report(27, 2.), report(26, .01), {}, {})
    assert not result['comparable_all_trials_survived']
    assert result['height_worst_relative_error_percent'] is result['height_worst_error_reduction_percent'] is None
    assert not result['full_original_gate_passed'] and not result['policy_accepted']
    assert not result['independent_sustained_acceptance_verified']
