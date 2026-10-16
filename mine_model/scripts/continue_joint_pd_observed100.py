"""Continue reviewed direct-joint PPO with qualified read-only learning data."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import time

import torch
import yaml

from auto_train_mine import write_json
import continue_joint_pd_stationary_drift100 as drift
from diagnose_plain_joint_pd_height import require_plain_ppo
from joint_pd_continuation import validate_continuation_config

guard, ROOT = drift.guard, drift.ROOT
ENTRY = ROOT/'mine_model/scripts/train_joint_pd_observed.py'
OBSERVER = ROOT/'mine_model/scripts/joint_pd_training_observer.py'


def lineage(source):
    """Trace paired resume metadata to the exact owner-selected model_200."""
    result = []
    while True:
        source = Path(source).resolve()
        if source in result:
            raise ValueError('Cyclic checkpoint lineage')
        result.append(source)
        if guard.core.sha(source) == drift.handoff.experiment.SOURCE_SHA:
            return result
        precision = yaml.safe_load((source.parent/'params/precision_mode.yaml').read_text())
        if not precision.get('source_checkpoint'):
            raise ValueError('The selected policy is not a retained-state model_200 descendant')
        source = precision['source_checkpoint']


def verify_source(previous, parity_path):
    report = drift.completed_report(previous)
    state = json.loads((previous/'status.json').read_text())
    source = Path(state['checkpoint'])
    parent = json.loads((previous/'experiment_plan.json').read_text())
    if any(guard.core.sha(Path(p)) != digest for p, digest in parent['source_files'].items()):
        raise ValueError('Frozen implementation or source evidence changed before handoff')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    actual = guard.core.saved_updates(source)
    if (not guard.core.tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
            or guard.core.optimizer_step(saved) != 20*actual):
        raise ValueError('Keep the selected full finite policy and corresponding 17 Adam states')
    cfg = guard.core.validate_checkpoint_robot(source)
    experiment = drift.handoff.experiment
    asset, _ = guard.core.qualified_wheel_mass_asset(experiment.ASSETS/experiment.MANIFEST['wheel_mass_calibration'])
    baseline = experiment.starting_config(guard.core.validate_checkpoint_robot(experiment.SOURCE), asset)
    for key in ('base_height_error_sq', 'standing_velocity', 'yaw_rate_error_sq', 'orientation'):
        baseline['rewards'][key] = cfg['rewards'][key]
    validate_continuation_config(baseline, cfg, baseline['commands'])
    require_plain_ppo(yaml.safe_load((source.parent/'params/agent.yaml').read_text()))
    ancestors = lineage(source)
    parity = json.loads(parity_path.read_text())
    required = ('weights_bitwise_equal', 'Adam_bitwise_equal', 'gradients_bitwise_equal',
                'losses_equal', 'RNG_bitwise_equal')
    if (parity['observer_sha256'] != guard.core.sha(OBSERVER)
            or parity['entrypoint_sha256'] != guard.core.sha(ENTRY)
            or parity['source_sha256'] not in {guard.core.sha(p) for p in ancestors}
            or {row['device'] for row in parity['results']} != {'cpu', 'cuda:0'}
            or any(not row[key] for row in parity['results'] for key in required)):
        raise ValueError('The unchanged observer needs current CPU/GPU retained-state parity evidence')
    if guard.core.sha(ROOT/'mine_model/config/tracking_acceptance.json') != guard.GATE_SHA:
        raise ValueError('Original acceptance changed')
    return dict(checkpoint=str(source), checkpoint_sha256=guard.core.sha(source),
        actual_ppo_updates=actual, Adam_step=guard.core.optimizer_step(saved), optimizer_states=17,
        checkpoint_lineage=list(map(str, ancestors)), canonical_survivors=report['survivors'],
        canonical_resets=report['resets'], full_original_gate_passed=report['passed'], policy_accepted=False)


def run(previous_cursor, base, unit, parity_path):
    previous_cursor, base, parity_path = map(lambda p: Path(p).resolve(), (previous_cursor, base, parity_path))
    if base.parent != ROOT/'mine_model/results':
        raise ValueError('Use an owned fresh review coordinator')
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
        previous = drift.handoff.selected_incumbent(prior)
        proof = verify_source(previous, parity_path)
        write_json(base/'source_verification.json', proof)
        write_json(base/'inherited_reviews.json', prior['history'])
        original_prepare, original_train, original_updates = guard.prepare, guard.train_trial, guard.UPDATES
        decision_path = None

        def propose(cfg, report, used):
            nonlocal decision_path
            current = json.loads((base/'coordinator.json').read_text())
            reviews = [*prior['history'], *current.get('history', [])]
            latest = reviews[-1]
            source = Path(json.loads((Path(current.get('incumbent', previous))/'status.json').read_text())['checkpoint'])
            for row in reviews:
                plan = json.loads((Path(row['output'])/'experiment_plan.json').read_text())
                if plan['source_sha256'] == guard.core.sha(source):
                    values = tuple(sorted((k, v[1]) for k, v in plan['only_reward_change'].items()))
                    used.add((plan['training_seed'], values))
            evidence_path = Path(latest['output'])/'replay/assessment.json'
            evidence = drift.completed_report(latest['output'])
            seed, delta, signature, diagnosis = drift.choose(cfg, evidence, used,
                continue_improvement=latest['promote'])
            decision_path = base/f'reward_decision_{len(reviews):04d}.json'
            write_json(decision_path, dict(diagnosis, evidence=str(evidence_path),
                evidence_sha256=guard.core.sha(evidence_path), seed=seed,
                read_only_native_observation=True, original_control_and_PPO_loss_unchanged=True))
            return seed, delta, signature

        def prepare(previous_output, output, delta, seed):
            plan, expected = original_prepare(previous_output, output, delta, seed)
            write_json(Path(output)/'status.json', dict(state='waiting_for_host_memory', training_active=False,
                actual_ppo_updates=plan['source_actual_ppo_updates'],
                live_ppo_updates=plan['source_actual_ppo_updates'], checkpoint=plan['source_checkpoint'],
                checkpoint_sha256=plan['source_sha256'], target_actual_ppo_updates=plan['target_actual_ppo_updates'],
                goal_remains_active=True, policy_accepted=False, updated_at=datetime.now().astimezone().isoformat()))
            drift.wait_memory(base)
            decision = json.loads(decision_path.read_text())
            for path in [Path(__file__).resolve(), ENTRY, OBSERVER, parity_path, previous_cursor,
                         ROOT/'mine_model/scripts/diagnose_plain_joint_pd_height.py',
                         ROOT/'mine_model/scripts/audit_joint_pd_native_training.py',
                         ROOT/'wheel_legged_isaaclab/tests/test_observed_joint_pd_entrypoint.py',
                         ROOT/'wheel_legged_isaaclab/tests/test_training_observer.py',
                         ROOT/'wheel_legged_isaaclab/tests/test_observed100_handoff.py',
                         decision_path, Path(decision['evidence'])]:
                plan['source_files'][str(path)] = guard.core.sha(path)
            plan.update(diagnosed_reward_decision=decision, native_num_envs=8192,
                read_only_learning_signal=True, observer_parity=str(parity_path),
                auxiliary_losses_enabled=False, original_control_and_PPO_loss_unchanged=True)
            write_json(Path(output)/'experiment_plan.json', plan)
            return plan, expected

        def train(previous_output, output, delta, seed):
            original_command = guard.old.split.ORIGINAL_COMMAND
            def observed_command(*args, **kwargs):
                argv, final, saved = original_command(*args, **kwargs)
                return [argv[0], str(ENTRY), '--observation-output', str(Path(output)/'native_learning_signal'),
                        '--', *argv[2:]], final, saved
            try:
                guard.old.split.ORIGINAL_COMMAND = observed_command
                original_train(previous_output, output, delta, seed)
            finally:
                guard.old.split.ORIGINAL_COMMAND = original_command

        try:
            guard.prepare, guard.train_trial, guard.UPDATES = prepare, train, 100
            guard.run(previous, previous, base, unit, bootstrap_function=lambda *unused: previous,
                      proposal_function=propose)
        finally:
            guard.prepare, guard.train_trial, guard.UPDATES = original_prepare, original_train, original_updates


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator', required=True, type=Path)
    parser.add_argument('--coordinator', required=True, type=Path)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--observer-parity', required=True, type=Path)
    args = parser.parse_args()
    run(args.previous_coordinator, args.coordinator, args.unit, args.observer_parity)
