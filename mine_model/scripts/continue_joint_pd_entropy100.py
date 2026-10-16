"""Compare ordinary PPO entropy after measured weak height exploration.

Every block retains weights, learned std, all Adam states and the original
environment/rewards. Only the existing PPO entropy coefficient can change.
"""
import argparse
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import yaml

from auto_train_mine import write_json
import continue_joint_pd_observed100 as observed
from joint_pd_priority import priority_summary

guard, ROOT = observed.guard, observed.ROOT
core = guard.core
TEST = ROOT/'wheel_legged_isaaclab/tests/test_entropy100_comparison.py'


def next_coefficient(current, used, *, initial_promoted=False, reviews=()):
    """A bounded measured comparison; never restart std or repeat a seed."""
    if not reviews:
        choices = [current] if initial_promoted else [.003]
    elif reviews[-1]['promote']:
        recent = reviews[-3:]
        values = [row['candidate_quality'][6] for row in recent]
        stagnant = len(values) == 3 and min(values) >= .98*values[0] and min(values) > .1
        if stagnant and current >= .005:
            raise RuntimeError('Height remains stagnant at the bounded entropy limit; inspect learning and control diagnostics')
        choices = [min(.005, current*1.5)] if stagnant else [current]
    else:
        last = reviews[-1]
        survival_or_pose_loss = not last['comparison']['stability_retained']
        choices = [.002] if survival_or_pose_loss else [.005]
    for value in choices:
        signature = (43, ('entropy_coef', value))
        if signature not in used:
            return value, signature
    raise RuntimeError('Entropy comparison has no untried evidence-backed continuation; inspect native learning diagnostics')


def observed_command(source, cfg, label, updates, commands, stop_file, *, coefficient, seed=43, **kwargs):
    kwargs['minimum_wheel_std'] = None
    argv, final, saved = core.ORIGINAL_COMMAND(source, cfg, label, updates, commands, stop_file, **kwargs)
    argv = core.entropy_command(argv, coefficient)
    argv[argv.index('--seed')+1] = str(seed)
    argv.append('agent.policy.class_name=ActorCritic')
    return [argv[0], str(observed.ENTRY), '--observation-output',
            str(Path(stop_file).parent/'native_learning_signal'), '--', *argv[2:]], final, saved


def evidence_paths(directory):
    directory = Path(directory)
    names = ('complete_learning_signal_summary.json', 'early_native_learning_signal_diagnosis_0007.json',
             'native_inference_precision_diagnosis.json', 'replay/assessment.json',
             'native_learning_signal/completed_external.json')
    paths = [directory/name for name in names]
    summary = json.loads(paths[0].read_text())
    early = json.loads(paths[1].read_text())
    replay = observed.drift.completed_report(directory)
    completed = json.loads(paths[4].read_text())
    failures = [t['metrics']['height_mae_m']/case['height'] for case in replay['conditions']
                if case['name'] in ('height_low', 'height_high') for t in case['trials'] if t['survived']]
    if (completed['actual_observed_updates'] != 100 or completed['optimizer_minibatches'] != 2000
            or not completed['all_probe_RNG_unchanged']
            or max(completed['all_environment_alignment_max_errors'].values()) != 0.
            or completed['checkpoint_sha256'] != replay['checkpoint_sha256']
            or summary['checkpoint_sha256'] != replay['checkpoint_sha256']
            or len(failures) != 6 or max(failures) <= .1
            or not early['rows'] or any(not math.isfinite(row['all_sampled_reference_height_noise_std_mean_m'])
                                      or row['all_sampled_reference_height_noise_std_mean_m'] <= 0.
                                      for row in early['rows'])):
        raise ValueError('Require a full checkpoint-matched failed height review and intact native learning evidence')
    return paths


def run(previous_cursor, base, unit, parity_path, evidence_directory):
    previous_cursor, base, parity_path = map(lambda p: Path(p).resolve(), (previous_cursor, base, parity_path))
    if base.parent != ROOT/'mine_model/results':
        raise ValueError('Use an owned fresh review coordinator')
    evidence = evidence_paths(evidence_directory)
    base.mkdir(exist_ok=False)
    with (previous_cursor.parent/'.lock').open('a') as owner:
        while True:
            if (base/'STOP').exists(): return
            try:
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                write_json(base/'handoff_status.json', dict(state='waiting_for_completed_100_update_review',
                    previous_coordinator=str(previous_cursor), updated_at=datetime.now().astimezone().isoformat()))
                time.sleep(5)
        prior = json.loads(previous_cursor.read_text())
        previous = observed.drift.handoff.selected_incumbent(prior)
        proof = observed.verify_source(previous, parity_path)
        if not priority_summary(observed.drift.completed_report(previous))['stationary_pose_passed']:
            raise ValueError('Keep the independently stable incumbent as the continuation source')
        write_json(base/'source_verification.json', proof)
        write_json(base/'inherited_reviews.json', prior['history'])
        original_train, original_updates = guard.train_trial, guard.UPDATES
        decision_path = None
        coefficient = None

        def propose(cfg, report, used):
            nonlocal coefficient, decision_path
            cursor = json.loads((base/'coordinator.json').read_text())
            selected = Path(cursor.get('incumbent', previous))
            source = Path(json.loads((selected/'status.json').read_text())['checkpoint'])
            agent = yaml.safe_load((source.parent/'params/agent.yaml').read_text())
            current = agent['algorithm']['entropy_coef']
            reviews = cursor.get('history', [])
            coefficient, signature = next_coefficient(current, used,
                initial_promoted=prior['history'][-1]['promote'], reviews=reviews)
            decision_path = base/f'entropy_decision_{len(reviews):04d}.json'
            write_json(decision_path, dict(previous_entropy_coefficient=current,
                entropy_coefficient=coefficient, rewards_changed=False, control_changed=False,
                learned_std_reset=False, optimizer_reset=False, actual_PPO_updates=100,
                reason='Retain measured improvements; otherwise compare existing entropy against weak height-reference exploration.',
                hypothesis_only=True, policy_accepted=False))
            return 43, {}, signature

        def prepare(previous_output, output):
            old_interval = core.INTERVAL
            try:
                core.INTERVAL = 100
                plan, expected = core.prepare(previous_output, output, coefficient)
            finally: core.INTERVAL = old_interval
            observed.record_resource_wait(output, plan)
            observed.drift.wait_memory(base)
            for path in [Path(__file__).resolve(), TEST, observed.ENTRY, observed.OBSERVER,
                         Path(observed.__file__), parity_path, previous_cursor, decision_path, *evidence]:
                plan['source_files'][str(path)] = core.sha(path)
            plan.update(only_reward_change={}, only_training_change='Existing ordinary PPO entropy coefficient only; retained std/weights/Adam/adaptive LR',
                entropy_decision=json.loads(decision_path.read_text()), native_num_envs=8192,
                ordinary_full_policy_PPO=True, read_only_learning_signal=True, observer_parity=str(parity_path),
                auxiliary_losses_enabled=False, physical_action_mapping_unchanged=True,
                all_commands_and_rewards_unchanged=True, optimizer_reset=False, learned_std_reset=False)
            write_json(Path(output)/'experiment_plan.json', plan)
            return plan, expected

        def train(previous_output, output, delta, seed):
            if delta: raise ValueError('Entropy comparison cannot also modify reward scales')
            original_command = core.block.continuation_command
            def command(*args, **kwargs):
                return observed_command(*args, coefficient=coefficient, seed=seed, **kwargs)
            args = SimpleNamespace(previous=previous_output, output=output, reward_key='unused', value=0., updates=100, prepare_only=False)
            try:
                core.block.continuation_command = command
                guard.NATIVE(args, prepare_function=lambda *unused: prepare(previous_output, output))
                observed.finalize_observations(output)
                state = json.loads((Path(output)/'status.json').read_text())
                algorithm = yaml.safe_load((Path(state['checkpoint']).parent/'params/agent.yaml').read_text())['algorithm']
                if algorithm['class_name'] != 'PPO' or algorithm['entropy_coef'] != coefficient:
                    raise ValueError('Native PPO did not apply the declared entropy comparison')
            finally: core.block.continuation_command = original_command

        try:
            guard.train_trial, guard.UPDATES = train, 100
            guard.run(previous, previous, base, unit, bootstrap_function=lambda *unused: previous,
                      proposal_function=propose)
        finally: guard.train_trial, guard.UPDATES = original_train, original_updates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator', required=True, type=Path)
    parser.add_argument('--coordinator', required=True, type=Path)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--observer-parity', required=True, type=Path)
    parser.add_argument('--learning-evidence', required=True, type=Path)
    args = parser.parse_args()
    run(args.previous_coordinator, args.coordinator, args.unit, args.observer_parity, args.learning_evidence)
