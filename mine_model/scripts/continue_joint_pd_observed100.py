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
from finish_joint_pd_native_audit import summarize_records as summarize_pair
from joint_pd_continuation import validate_continuation_config

guard, ROOT = drift.guard, drift.ROOT
ENTRY = ROOT/'mine_model/scripts/train_joint_pd_observed.py'
OBSERVER = ROOT/'mine_model/scripts/joint_pd_training_observer.py'


def choose(cfg, report, used, *, continue_improvement=False):
    failures = drift.deficits(report)
    if failures['survival']:
        for value in (40., 80., 160., 320.):
            delta = {'orientation': -value}
            signature = (43, tuple(delta.items()))
            if -value < cfg['rewards']['orientation'] and signature not in used:
                return 43, delta, signature, dict(
                    reason='Canonical survival regressed; isolate existing posture cost before drift or height',
                    failures=failures, only_reward_change=delta, actual_PPO_updates=100,
                    hypothesis_only=True, policy_accepted=False)
        raise RuntimeError('Survival still fails after bounded posture comparisons; diagnose control before more training')
    return drift.choose(cfg, report, used, continue_improvement=continue_improvement)


def record_resource_wait(output, plan):
    # The native block deliberately accepts only a prepared status before it
    # starts. Keep resource waiting as a substate, never an already-started run.
    write_json(Path(output)/'status.json', dict(state='prepared', resource_state='waiting_for_host_memory',
        training_active=False, actual_ppo_updates=plan['source_actual_ppo_updates'],
        live_ppo_updates=plan['source_actual_ppo_updates'], checkpoint=plan['source_checkpoint'],
        checkpoint_sha256=plan['source_sha256'], target_actual_ppo_updates=plan['target_actual_ppo_updates'],
        goal_remains_active=True, policy_accepted=False, updated_at=datetime.now().astimezone().isoformat()))


def summarize_observations(records):
    if len(records) != 100 or [row['update'] for row in records] != list(range(1, 101)):
        raise ValueError('Require every one of the 100 completed native updates')
    pairs = [summarize_pair([dict(records[i], update=1), dict(records[i+1], update=2)])
             for i in range(0, 100, 2)]
    return dict(optimizer_minibatches=sum(row['optimizer_minibatches'] for row in pairs),
        all_environment_alignment_max_errors={key: max(row['all_environment_alignment_max_errors'][key]
                                                       for row in pairs)
                                              for key in pairs[0]['all_environment_alignment_max_errors']},
        all_probe_RNG_unchanged=True, original_control_and_PPO_loss_unchanged=True,
        policy_accepted=False)


def finalize_observations(output):
    """Kit close can exit before runpy returns; verify persisted data in parent."""
    import numpy as np
    output = Path(output)
    directory = output/'native_learning_signal'
    plan = json.loads((output/'experiment_plan.json').read_text())
    native = json.loads((directory/'plan.json').read_text())
    proof = json.loads((output/'resume_verification.json').read_text())
    if (native['source_sha256'] != plan['source_sha256']
            or native['target_actual_ppo_updates'] != plan['target_actual_ppo_updates']
            or proof['actual_ppo_updates'] != plan['target_actual_ppo_updates']
            or proof['optimizer_states'] != 17 or not proof['all_tensors_finite']
            or any(guard.core.sha(Path(p)) != value for p, value in native['source_files_sha256'].items())):
        raise ValueError('Observation source, code or saved native optimizer changed')
    paths = sorted(directory.glob('audit_update_*.json'))
    records = [json.loads(path.read_text()) for path in paths]
    summary = summarize_observations(records)
    for record in records:
        with np.load(directory/f"rollout_{record['update']:04d}.npz", allow_pickle=False) as arrays:
            if (arrays['observations'].shape != (48, 32, 27)
                    or arrays['actions'].shape != (48, 32, 6)
                    or any(not np.isfinite(arrays[key]).all() for key in arrays.files)
                    or not np.array_equal(arrays['actions'], arrays['native_raw_actions'])):
                raise ValueError('Missing, nonfinite or mismatched native observation record')
    write_json(directory/'completed_external.json', dict(summary, checkpoint=proof['checkpoint'],
        checkpoint_sha256=proof['checkpoint_sha256'], actual_observed_updates=100,
        actual_ppo_updates=proof['actual_ppo_updates'], PPO_updates_repeated=0,
        in_process_completion_returned=(directory/'completed.json').exists(),
        verified_outside_Kit=True, records_sha256={str(path): guard.core.sha(path) for path in paths},
        updated_at=datetime.now().astimezone().isoformat(),
        limitation='Transition and optimizer diagnosis, not closed-loop tracking acceptance'))


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
            seed, delta, signature, diagnosis = choose(cfg, evidence, used,
                continue_improvement=latest['promote'])
            decision_path = base/f'reward_decision_{len(reviews):04d}.json'
            write_json(decision_path, dict(diagnosis, evidence=str(evidence_path),
                evidence_sha256=guard.core.sha(evidence_path), seed=seed,
                read_only_native_observation=True, original_control_and_PPO_loss_unchanged=True))
            return seed, delta, signature

        def prepare(previous_output, output, delta, seed):
            plan, expected = original_prepare(previous_output, output, delta, seed)
            record_resource_wait(output, plan)
            drift.wait_memory(base)
            decision = json.loads(decision_path.read_text())
            for path in [Path(__file__).resolve(), ENTRY, OBSERVER, parity_path, previous_cursor,
                         ROOT/'mine_model/scripts/diagnose_plain_joint_pd_height.py',
                         ROOT/'mine_model/scripts/audit_joint_pd_native_training.py',
                         ROOT/'mine_model/scripts/finish_joint_pd_native_audit.py',
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
                finalize_observations(output)
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
