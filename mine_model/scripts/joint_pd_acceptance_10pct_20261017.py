"""Return to the user's final10% target without rewriting prior20% evidence."""
from contextlib import contextmanager
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import summarize_mine_trial as assessor
import assess_height_leg_command_tracking as leg_assessor

CURRENT = ROOT / 'mine_model/config/tracking_acceptance.json'
POLICY = ROOT / 'mine_model/config/joint_pd_training_review_10pct_20261017.json'
ORIGINAL_SHA = '08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d'


def profile():
    criteria = json.loads(CURRENT.read_text())
    policy = json.loads(POLICY.read_text())
    if (sha(CURRENT) != ORIGINAL_SHA or criteria['relative_error'] != .1
            or policy['acceptance_file'] != CURRENT.name
            or policy['actual_ppo_updates_per_block'] != 200
            or policy['authority'] != 'explicit_user_instruction'
            or not policy['motion_requires_static_all_metrics_and_independent_sustained_pass']):
        raise ValueError('Require the unchanged original10% criteria and latest explicit staged instruction')
    return dict(profile='user_final10pct_review200_20261017_v1', criteria=criteria,
        acceptance_file=str(CURRENT), acceptance_sha256=ORIGINAL_SHA,
        review_policy_file=str(POLICY), review_policy_sha256=sha(POLICY),
        actual_ppo_updates_per_block=200, authority='explicit_user_instruction',
        previous20pct_reports_preserved=True, original_nonrelative_limits_unchanged=True,
        motion_requires_static_and_independent_sustained_pass=True,
        native_reward_and_control_unchanged=True)


@contextmanager
def use_current():
    profile()
    previous = assessor.ACCEPTANCE_PATH, leg_assessor.ACCEPTANCE_PATH
    assessor.ACCEPTANCE_PATH = leg_assessor.ACCEPTANCE_PATH = CURRENT
    try:
        yield
    finally:
        assessor.ACCEPTANCE_PATH, leg_assessor.ACCEPTANCE_PATH = previous


def assess_current(directory, report_name='assessment.json'):
    with use_current():
        report = assessor.assess(directory, report_name=report_name)
    for case in report['conditions']:
        for trial in case['trials']:
            strict = {key: limit - 1.e-12 for key, limit in trial['limits'].items()}
            parts = assessor.tracking_components(trial['survived'], trial['metrics'], strict)
            trial.update(parts, passed=bool(parts['task_goal_passed'] and parts['reference_tracking_passed']))
        for flag in ('passed', 'task_goal_passed', 'reference_tracking_passed'):
            case[flag] = all(trial[flag] for trial in case['trials'])
    for flag in ('passed', 'task_goal_passed', 'reference_tracking_passed'):
        report[flag] = all(case[flag] for case in report['conditions'])
    report.update(acceptance_revision=profile(), independent_sustained_verified=False)
    write_json(Path(directory) / report_name, report)
    return report


def assess_current_legs(directory):
    with use_current():
        report = leg_assessor.assess_directory(directory)
    report['acceptance_sha256'] = report.pop('original_acceptance_sha256')
    report.update(acceptance_revision=profile())
    return report


def assess_current_stability(directory, stable_seconds, window_seconds=3.):
    with use_current():
        report = assessor.assess_stability(directory, stable_seconds, window_seconds)
    report.update(acceptance_revision=profile())
    write_json(Path(directory) / 'assessment_stability.json', report)
    return report
