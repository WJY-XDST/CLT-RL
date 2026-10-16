"""Use the latest failed replay when choosing one existing repair reward.

The retained training source and protected reference stay separate. All
original policy acceptance gates and the ordinary PPO update are unchanged.
"""
import argparse
import json
from pathlib import Path

from auto_train_mine import write_json
import continue_joint_pd_repair100 as legacy
from joint_pd_priority import STATIC_CASES, compare_priority

ROOT, core = legacy.ROOT, legacy.core
ORIGINAL_PREPARE = legacy.prepare
ORIGINAL_CHOOSE = legacy.choose
TEST = ROOT/'wheel_legged_isaaclab/tests/test_priority_repair100_evidence.py'
LEVELS = dict(orientation=(80., 160., 320.), base_height_error_sq=(250., 500., 1000.),
              standing_velocity=(25., 50., 100., 200.), yaw_rate_error_sq=(2., 4., 8., 16.))


def choose(cfg, source_report, used, history, evidence):
    if not legacy.stationary_safe(source_report):
        raise ValueError('Keep a reviewed stationary-safe source and its complete optimizer')
    compare_priority(evidence, source_report)  # Identical cases, seeds, commands and original limits.
    failures = legacy.observed.drift.deficits(evidence)
    static_body = [row for row in failures['body'] if row['case'] in STATIC_CASES]
    key = ('orientation' if failures['survival'] or static_body else
           'standing_velocity' if failures['speed'] else
           'yaw_rate_error_sq' if failures['yaw'] else None)
    if key is None:
        return ORIGINAL_CHOOSE(cfg, evidence, used, history)
    for magnitude in LEVELS[key]:
        value = -magnitude
        signature = (43, ((key, value),))
        if value < cfg['rewards'][key] and signature not in used:
            return 43, {key: value}, signature
    raise RuntimeError('Latest primary stability failure exhausted bounded costs; diagnose before more training')


def prepare(previous, output, delta, coefficient, *, pins=(), evidence_path=None):
    if len(delta) > 1 or any(key not in LEVELS for key in delta):
        raise ValueError('Revise at most one diagnosed existing reward cost')
    for key, value in delta.items():
        if type(value) not in (int, float) or value not in tuple(-v for v in LEVELS[key]):
            raise ValueError('Require a bounded existing cost revision')
    watched = [Path(__file__).resolve(), TEST, *pins]
    if evidence_path is not None:
        watched.append(Path(evidence_path))
    plan, expected = ORIGINAL_PREPARE(previous, output, {}, coefficient, pins=watched)
    changes = {}
    for key, value in delta.items():
        before = expected['rewards'][key]
        if not value < before <= 0.:
            raise ValueError('Strengthen only the cost identified by the latest failed replay')
        changes[key] = [before, value]
    expected['rewards'].update(delta)
    key, value = next(iter(delta.items())) if delta else ('base_height_error_sq', expected['rewards']['base_height_error_sq'])
    plan.update(reward_key=key, reward_value=value, only_reward_change=changes,
        only_training_change='One existing cost selected from latest replay; retained complete weights/std/Adam/adaptiveLR/entropy',
        latest_completed_replay_used=True, priority_evidence=str(evidence_path) if evidence_path else None,
        priority_order=['survival_and_stationary_posture', 'stationary_drift', 'moving_posture', 'height', 'velocity'],
        final_acceptance_unchanged=True)
    write_json(Path(output)/'experiment_plan.json', plan)
    return plan, expected


def run(previous_cursor, repair_seed, base, unit, parity_path):
    previous_cursor, base = Path(previous_cursor).resolve(), Path(base).resolve()
    selected_evidence = None
    old_choose, old_prepare = legacy.choose, legacy.prepare
    def proposal(cfg, source_report, used, history):
        nonlocal selected_evidence
        inherited = json.loads(previous_cursor.read_text())['history']
        reviews = [*inherited, *history]
        source_sha = source_report['checkpoint_sha256']
        for row in reviews:
            plan = json.loads((Path(row['output'])/'experiment_plan.json').read_text())
            if plan['source_sha256'] == source_sha:
                values = tuple(sorted((key, pair[1]) for key, pair in plan['only_reward_change'].items()))
                used.add((plan.get('training_seed', 43), values))
        latest = reviews[-1]
        directory = Path(latest['output'])
        latest_plan = json.loads((directory/'experiment_plan.json').read_text())
        if not latest['promote'] and latest_plan['source_sha256'] != source_sha:
            cursor = json.loads((base/'coordinator.json').read_text())
            directory = Path(cursor['incumbent'])
        evidence = legacy.observed.drift.completed_report(directory)
        selected_evidence = directory/'replay/assessment.json'
        return choose(cfg, source_report, used, reviews, evidence)
    def prepared(*args, **kwargs):
        return prepare(*args, **kwargs, evidence_path=selected_evidence)
    try:
        legacy.choose, legacy.prepare = proposal, prepared
        legacy.run(previous_cursor, repair_seed, base, unit, parity_path)
    finally:
        legacy.choose, legacy.prepare = old_choose, old_prepare


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator', type=Path, required=True)
    parser.add_argument('--repair-seed', type=Path, required=True)
    parser.add_argument('--coordinator', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--observer-parity', type=Path, required=True)
    args = parser.parse_args()
    run(args.previous_coordinator, args.repair_seed, args.coordinator, args.unit, args.observer_parity)
