"""Preserve reviewed learning while repairing posture every 100 PPO updates.

A stationary-safe experimental candidate is separate from the protected
stability reference. Neither a training continuation nor a reference update
means policy acceptance: all original canonical and sustained gates remain.
"""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import time
from types import SimpleNamespace

import yaml

from auto_train_mine import write_json
import continue_joint_pd_entropy100 as entropy
from joint_pd_priority import compare_priority, priority_summary
from joint_pd_rollback import admit as original_admit, quality

observed, guard = entropy.observed, entropy.guard
core, ROOT = guard.core, guard.ROOT
TEST = ROOT/'wheel_legged_isaaclab/tests/test_repair100_training_selection.py'


def stationary_safe(report):
    summary = priority_summary(report)
    return (summary['survivors'] == 27 and summary['resets'] == 0
            and summary['stationary_pose_passed'])


def review_for_learning(after, before, protected, original_protected):
    local = compare_priority(after, before)
    stable = original_admit(after, protected, original_protected)
    progress = (stationary_safe(after) and local['stability_retained']
                and quality(after) < quality(before))
    return dict(promote=bool(progress or stable['promote']), comparison=local,
        against_restored_anchor=stable['against_restored_anchor'],
        candidate_quality=quality(after), incumbent_quality=quality(before),
        protected_reference_promoted=stable['promote'], protected_reference_review=stable,
        retain_experimental_learning_only=bool(progress and not stable['promote']),
        promote_is_training_selection_only=True,
        decision=('retain_reviewed_learning_for_next_100_update_repair' if progress
                  else stable['decision']), policy_accepted=False,
        full_original_gate_passed=bool(after['passed']), final_acceptance_unchanged=True)


def select_seed(reports, protected_report):
    eligible = [directory for directory, report in reports.items()
                if stationary_safe(report)
                and quality(report)[6] <= quality(protected_report)[6]]
    if not eligible:
        raise ValueError('No reviewed stationary-safe height-preserving learning source')
    return min(eligible, key=lambda directory: quality(reports[directory]))


def choose(cfg, report, used, history):
    if not stationary_safe(report):
        raise ValueError('Never continue a source with lost survival or stationary stability')
    failures = observed.drift.deficits(report)
    rewards = cfg['rewards']
    progressing = bool(history and history[-1]['promote'])
    metric = 5 if failures['body'] else 6
    recent = [row['candidate_quality'][metric] for row in history[-3:]]
    stagnant = len(recent) == 3 and min(recent) >= .98*recent[0]
    choices = []
    if progressing and not stagnant:
        choices.append({})
    if failures['body']:
        choices.extend({'orientation': -value} for value in (80., 160., 320.)
                       if -value < rewards['orientation'])
    elif failures['height']:
        choices.extend({'base_height_error_sq': -value} for value in (250., 500., 1000.)
                       if -value < rewards['base_height_error_sq'])
    for delta in choices:
        signature = (43, tuple(sorted(delta.items())))
        if signature not in used:
            return 43, delta, signature
    raise RuntimeError('No measured repair step remains; inspect complete native/replay evidence before extending')


def prepare(previous, output, delta, coefficient, *, pins=()):
    if len(delta) > 1 or any(key not in ('orientation', 'base_height_error_sq') for key in delta):
        raise ValueError('Change at most one existing posture or actual-height cost')
    for key, value in delta.items():
        allowed = (-80., -160., -320.) if key == 'orientation' else (-250., -500., -1000.)
        if type(value) not in (int, float) or value not in allowed:
            raise ValueError('Require a bounded existing reward cost')
    original_interval = core.INTERVAL
    try:
        core.INTERVAL = 100
        plan, expected = core.prepare(previous, output, coefficient)
    finally:
        core.INTERVAL = original_interval
    changes = {key: [expected['rewards'][key], value] for key, value in delta.items()}
    for key, value in delta.items():
        before = expected['rewards'][key]
        limit = -320. if key == 'orientation' else -1000.
        if (value not in ((-80., -160., -320.) if key == 'orientation' else (-250., -500., -1000.))
                or not limit <= value < before < 0.):
            raise ValueError('Require a bounded strengthening of the diagnosed existing cost')
    expected['rewards'].update(delta)
    key, value = next(iter(delta.items())) if delta else ('base_height_error_sq', expected['rewards']['base_height_error_sq'])
    plan.update(reward_key=key, reward_value=value, only_reward_change=changes,
        only_training_change='One existing reward cost; preserve reviewed candidate weights/std/all17Adam/adaptiveLR and entropy',
        training_seed=43, native_num_envs=8192, ordinary_full_policy_PPO=True,
        read_only_learning_signal=True, auxiliary_losses_enabled=False,
        optimizer_reset=False, learned_std_reset=False, no_training_phases=True,
        experimental_continuation_is_not_acceptance=True)
    for path in [Path(__file__).resolve(), TEST, observed.ENTRY, observed.OBSERVER,
                 Path(observed.__file__), Path(entropy.__file__), *pins]:
        plan['source_files'][str(path)] = core.sha(path)
    write_json(Path(output)/'experiment_plan.json', plan)
    return plan, expected


def train(previous, output, delta, base, pins):
    source = Path(json.loads((Path(previous)/'status.json').read_text())['checkpoint'])
    coefficient = yaml.safe_load((source.parent/'params/agent.yaml').read_text())['algorithm']['entropy_coef']
    original_command = core.block.continuation_command
    def command(*args, **kwargs):
        return entropy.observed_command(*args, coefficient=coefficient, seed=43, **kwargs)
    def prepared(*unused):
        plan, expected = prepare(previous, output, delta, coefficient, pins=pins)
        observed.record_resource_wait(output, plan)
        observed.drift.wait_memory(base)
        return plan, expected
    args = SimpleNamespace(previous=previous, output=output, reward_key='unused', value=0., updates=100, prepare_only=False)
    try:
        core.block.continuation_command = command
        guard.NATIVE(args, prepare_function=prepared)
        observed.finalize_observations(output)
        checkpoint = Path(json.loads((Path(output)/'status.json').read_text())['checkpoint'])
        algorithm = yaml.safe_load((checkpoint.parent/'params/agent.yaml').read_text())['algorithm']
        if algorithm['class_name'] != 'PPO' or algorithm['entropy_coef'] != coefficient:
            raise ValueError('Do not change ordinary PPO entropy during a reward repair')
    finally:
        core.block.continuation_command = original_command


def run(previous_cursor, repair_seed, base, unit, parity_path):
    previous_cursor, repair_seed, base, parity_path = [Path(p).resolve()
        for p in (previous_cursor, repair_seed, base, parity_path)]
    if base.parent != ROOT/'mine_model/results':
        raise ValueError('Use an owned fresh coordinator')
    base.mkdir(exist_ok=False)
    with (previous_cursor.parent/'.lock').open('a') as owner:
        while True:
            if (base/'STOP').exists():
                return
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                write_json(base/'handoff_status.json', dict(state='waiting_for_completed_100_update_review',
                    previous_coordinator=str(previous_cursor), updated_at=datetime.now().astimezone().isoformat()))
                time.sleep(5)
        prior = json.loads(previous_cursor.read_text())
        previous_source = observed.drift.handoff.selected_incumbent(prior)
        protected = Path(prior.get('protected_stability_reference', previous_source)).resolve()
        protected_report = observed.drift.completed_report(protected)
        original_protected = protected_report
        reports = {repair_seed: observed.drift.completed_report(repair_seed),
                   previous_source: observed.drift.completed_report(previous_source)}
        latest = Path(prior['history'][-1]['output'])
        reports[latest] = observed.drift.completed_report(latest)
        if prior['history'][-1]['promote']:
            selected = previous_source
        else:
            reports.pop(protected, None)
            selected = select_seed(reports, protected_report)
        proofs = {str(path): observed.verify_source(path, parity_path) for path in {selected, protected}}
        selected_report = observed.drift.completed_report(selected)
        if not stationary_safe(selected_report):
            raise ValueError('Every selected learning source must retain all stationary stability and survival')
        if not stationary_safe(protected_report) or quality(protected_report)[4] != 0:
            raise ValueError('Keep a protected reference with all original stationary and moving posture limits')
        write_json(base/'source_selection.json', dict(selected_for_training=str(selected),
            protected_stability_reference=str(protected), proofs=proofs,
            selected_training_quality=quality(selected_report), protected_stability_quality=quality(protected_report),
            selected_full_original_gate_passed=bool(selected_report['passed']),
            reason='Keep stationary-safe experimental learning separate from the original final acceptance gate',
            policy_accepted=False, final_acceptance_unchanged=True))
        write_json(base/'inherited_reviews.json', prior['history'])
        original_train, original_updates = guard.train_trial, guard.UPDATES
        original_review, original_write = guard.admit, guard.write_json
        decision = None
        def record(path, value):
            if Path(path).name in ('coordinator.json', 'joint_pd_active_joint_tracking.json', 'joint_pd_active_standing_seed.json'):
                value = dict(value, protected_stability_reference=str(protected),
                    incumbent_is_experimental_training_candidate=True,
                    controller_kind='reviewed_stationary_safe_repair_candidate', policy_accepted=False)
            original_write(path, value)
        def proposal(cfg, report, used):
            nonlocal decision
            history = json.loads((base/'coordinator.json').read_text()).get('history', [])
            seed, delta, signature = choose(cfg, report, used, history)
            decision = base/f'repair_decision_{len(history):04d}.json'
            write_json(decision, dict(only_reward_change=delta, seed=seed, actual_PPO_updates=100,
                reason='Repair original moving-posture failures first, then strengthen dense actual-height learning',
                tasks_trained_simultaneously=True, policy_accepted=False))
            return seed, delta, signature
        def training(previous, output, delta, seed):
            if seed != 43:
                raise ValueError('Do not repeat failed profiles with arbitrary new seeds')
            paths = [parity_path, previous_cursor, base/'source_selection.json', decision,
                     protected/'replay/assessment.json', protected/'status.json']
            train(previous, output, delta, base, paths)
        def review(after, before, unused_anchor):
            nonlocal protected, protected_report
            result = review_for_learning(after, before, protected_report, original_protected)
            if result['protected_reference_promoted']:
                cursor = json.loads((base/'coordinator.json').read_text())
                protected, protected_report = Path(cursor['current_output']), after
            result['protected_stability_reference'] = str(protected)
            return result
        try:
            guard.train_trial, guard.UPDATES = training, 100
            guard.admit, guard.write_json = review, record
            guard.run(selected, protected, base, unit, bootstrap_function=lambda *unused: selected,
                      proposal_function=proposal)
        finally:
            guard.train_trial, guard.UPDATES = original_train, original_updates
            guard.admit, guard.write_json = original_review, original_write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator', type=Path, required=True)
    parser.add_argument('--repair-seed', type=Path, required=True)
    parser.add_argument('--coordinator', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--observer-parity', type=Path, required=True)
    args = parser.parse_args()
    run(args.previous_coordinator, args.repair_seed, args.coordinator, args.unit, args.observer_parity)
