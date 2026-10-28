"""User-authorized 20% assessment, retaining immutable historical 10% evidence.

Only evaluation criteria change. Native reward normalization, physics,
policy and checkpoint metadata retain their qualified definitions.
"""
from contextlib import contextmanager
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import summarize_mine_trial as assessor
import assess_height_leg_command_tracking as leg_assessor

CONFIG = ROOT / 'mine_model/config'
POLICY = CONFIG / 'joint_pd_training_review.json'
HISTORICAL = CONFIG / 'tracking_acceptance.json'
CURRENT = CONFIG / 'tracking_acceptance_20pct.json'
HISTORICAL_SHA = '08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d'


def profile():
    policy = json.loads(POLICY.read_text())
    old, current = (json.loads(path.read_text()) for path in (HISTORICAL, CURRENT))
    expected = dict(old, relative_error=.2)
    if (sha(HISTORICAL) != HISTORICAL_SHA or old['relative_error'] != .1
            or current != expected or policy['actual_ppo_updates_per_block'] != 200
            or policy['acceptance_file'] != CURRENT.name
            or policy['historical_acceptance_file'] != HISTORICAL.name
            or policy['authority'] != 'explicit_user_instruction'):
        raise ValueError('Require the explicit user20% revision and all unchanged nonrelative limits')
    return dict(profile='user_relative_error20pct_review200_v1', criteria=current,
        acceptance_file=str(CURRENT), acceptance_sha256=sha(CURRENT),
        historical10pct_file=str(HISTORICAL), historical10pct_sha256=HISTORICAL_SHA,
        review_policy_file=str(POLICY), review_policy_sha256=sha(POLICY),
        actual_ppo_updates_per_block=200, authority='explicit_user_instruction',
        only_criterion_change={'relative_error': [.1, .2]},
        native_reward_and_control_unchanged=True)


@contextmanager
def use_current():
    """Process-local assessor selection; never edit frozen source/gate files."""
    profile()
    old_main, old_leg = assessor.ACCEPTANCE_PATH, leg_assessor.ACCEPTANCE_PATH
    assessor.ACCEPTANCE_PATH = leg_assessor.ACCEPTANCE_PATH = CURRENT
    try:
        yield
    finally:
        assessor.ACCEPTANCE_PATH, leg_assessor.ACCEPTANCE_PATH = old_main, old_leg


def assess_current(directory, report_name='assessment.json'):
    with use_current():
        report = assessor.assess(directory, report_name=report_name)
    # Keep the requested strict less-than semantics, including roundoff margin.
    for case in report['conditions']:
        for trial in case['trials']:
            strict = {k: limit - 1.e-12 for k, limit in trial['limits'].items()}
            parts = assessor.tracking_components(trial['survived'], trial['metrics'], strict)
            trial.update(parts)
            trial['passed'] = bool(parts['task_goal_passed'] and parts['reference_tracking_passed'])
        for flag in ('passed', 'task_goal_passed', 'reference_tracking_passed'):
            case[flag] = all(t[flag] for t in case['trials'])
    for flag in ('passed', 'task_goal_passed', 'reference_tracking_passed'):
        report[flag] = all(c[flag] for c in report['conditions'])
    report.update(acceptance_revision=profile(), historical10pct_gate_passed_not_implied=True)
    write_json(Path(directory) / report_name, report)
    return report


def assess_current_legs(directory):
    with use_current():
        report = leg_assessor.assess_directory(directory)
    report['acceptance_sha256'] = report.pop('original_acceptance_sha256')
    report.update(acceptance_revision=profile(), historical10pct_gate_passed_not_implied=True)
    return report


def assess_current_stability(directory, stable_seconds, window_seconds=3.):
    with use_current():
        report = assessor.assess_stability(directory, stable_seconds, window_seconds)
    report.update(acceptance_revision=profile(), historical10pct_gate_passed_not_implied=True)
    write_json(Path(directory) / 'assessment_stability.json', report)
    return report
