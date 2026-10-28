"""Compare committed capacity and production updates without running simulation.

This is a diagnostic of the first five updates, not capacity-weight promotion
or independent policy acceptance. Saved rollouts contain 32 sampled environments;
the native alignment audits cover all 8192 environments.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml


SCRIPT = Path(__file__).resolve()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def differences(before, after, prefix=''):
    if isinstance(before, dict) and isinstance(after, dict):
        result = []
        for key in sorted(set(before) | set(after)):
            result.extend(differences(before.get(key), after.get(key), prefix + '/' + str(key)))
        return result
    return [(prefix, before, after)] if before != after else []


def compare_state(before, after):
    result = dict(tensor_count=0, scalar_count=0, mismatches=[])

    def visit(a, z, path):
        if isinstance(a, torch.Tensor) and isinstance(z, torch.Tensor):
            result['tensor_count'] += 1
            equal = a.dtype == z.dtype and a.shape == z.shape
            if equal:
                equal = torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                    z.contiguous().reshape(-1).view(torch.uint8))
            if not equal:
                result['mismatches'].append(path)
            if a.is_floating_point() and (not torch.isfinite(a).all() or not torch.isfinite(z).all()):
                result['mismatches'].append(path + '/nonfinite')
        elif isinstance(a, dict) and isinstance(z, dict) and set(a) == set(z):
            for key in a:
                visit(a[key], z[key], path + '/' + str(key))
        elif isinstance(a, (list, tuple)) and type(a) is type(z) and len(a) == len(z):
            for index, (left, right) in enumerate(zip(a, z)):
                visit(left, right, path + '/' + str(index))
        else:
            result['scalar_count'] += 1
            if type(a) is not type(z) or a != z:
                result['mismatches'].append(path)

    visit(before, after, '')
    result['bitwise_equal'] = not result['mismatches']
    return result


def audit(production, capacity, production_run, capacity_run, destination):
    production, capacity, production_run, capacity_run, destination = (
        Path(p).resolve() for p in (production, capacity, production_run, capacity_run, destination))
    assert destination.parent == production and not destination.exists()
    pins = {str(SCRIPT): sha(SCRIPT)}

    def pin(path):
        path = Path(path)
        pins[str(path)] = sha(path)
        return path

    def read(path):
        return json.loads(pin(path).read_text())

    plan = read(production / 'experiment_plan.json')
    contract = read(capacity / 'capacity_runtime_contract.json')
    assert plan['native_num_envs'] == contract['num_envs'] == 8192
    assert plan['bounded_training_updates'] == 200 and contract['probe_updates'] == 5
    assert contract['capacity_only'] and not contract['capacity_weights_may_be_promoted']
    assert plan['ordinary_full_policy_PPO'] and not plan['auxiliary_losses_enabled']
    env_configs, agent_configs = [], []
    for run in (capacity_run, production_run):
        env_configs.append(yaml.load(pin(run / 'params/env.yaml').read_text(), Loader=yaml.BaseLoader))
        agent_configs.append(yaml.load(pin(run / 'params/agent.yaml').read_text(), Loader=yaml.BaseLoader))
    env_delta = differences(*env_configs)
    agent_delta = differences(*agent_configs)
    assert env_delta == [('/log_dir', str(capacity_run), str(production_run))], env_delta
    assert [row[0] for row in agent_delta] == ['/max_iterations', '/run_name'], agent_delta
    assert agent_delta[0][1:] == ('5', '200')
    assert env_configs[0]['scene']['num_envs'] == '8192'
    assert env_configs[0]['wheel_damping'] == '0.75'
    loads, drives = [], []
    for folder in (capacity, production):
        loads.append(read(folder / 'native_learning_signal/source200_runtime_load.json'))
        drives.append(read(folder / 'native_learning_signal/wheelD075_native_drive_evidence.json'))
    for load, drive in zip(loads, drives):
        assert load['checkpoint_sha256'] == plan['source_sha256']
        assert load['source_actual_updates'] == plan['source_actual_ppo_updates']
        assert load['all17_Adam_step'] == plan['source_Adam_step']
        assert load['all_model_and_optimizer_tensors_bitwise_preserved']
        assert not load['optimizer_reset'] and not load['architecture_migrated']
        assert drive['actual_environments'] == 8192
        assert drive['actual_wheel_damping_min'] == drive['actual_wheel_damping_max'] == .75
        assert drive['actual_USD_drive_damping'] == [.75, .75]
    load_delta = differences({k: v for k, v in loads[0].items() if k != 'profile'},
                             {k: v for k, v in loads[1].items() if k != 'profile'})
    assert not load_delta
    rows = []
    for update in range(1, 6):
        native = [folder / 'native_learning_signal' for folder in (capacity, production)]
        audits = [read(folder / f'audit_update_{update:04}.json') for folder in native]
        assert audits[0] == audits[1]
        for entry in audits:
            assert entry['update'] == update and entry['total_environments'] == 8192
            assert len(entry['optimizer_steps']) == 20 and entry['probe_RNG_unchanged']
            assert len(entry['all_environment_alignment_max_errors']) == 9
            assert not any(entry['all_environment_alignment_max_errors'].values())
            assert not entry['added_loss_or_backward']
        arrays = []
        with np.load(pin(native[0] / f'rollout_{update:04}.npz'), allow_pickle=False) as before, \
                np.load(pin(native[1] / f'rollout_{update:04}.npz'), allow_pickle=False) as after:
            assert set(before.files) == set(after.files)
            for key in before.files:
                a, z = before[key], after[key]
                assert a.dtype == z.dtype and a.shape == z.shape and a.tobytes() == z.tobytes(), (update, key)
                arrays.append(dict(name=key, shape=list(a.shape), dtype=str(a.dtype), bitwise_equal=True))
        iteration = plan['source_actual_ppo_updates'] + update - 1
        states = [torch.load(pin(run / f'model_{iteration}.pt'), map_location='cpu', weights_only=False)
                  for run in (capacity_run, production_run)]
        model = compare_state(states[0]['model_state_dict'], states[1]['model_state_dict'])
        optimizer = compare_state(states[0]['optimizer_state_dict'], states[1]['optimizer_state_dict'])
        assert model['bitwise_equal'] and model['tensor_count'] == 17
        assert optimizer['bitwise_equal'] and optimizer['tensor_count'] == 51
        expected_step = plan['source_Adam_step'] + update * 20
        for state in states:
            assert state['iter'] == iteration and len(state['optimizer_state_dict']['state']) == 17
            assert all(value['step'].item() == expected_step
                       for value in state['optimizer_state_dict']['state'].values())
        rows.append(dict(update=update, actual_updates=iteration + 1, all17_Adam_step=expected_step,
            native_all8192_alignment_errors_zero=True, all20_Adam_audits_equal=True,
            sampled32_rollout_arrays=arrays, model=model, optimizer=optimizer,
            saved_std=states[1]['model_state_dict']['std'].tolist(),
            saved_optimizer_LR=[group['lr'] for group in states[1]['optimizer_state_dict']['param_groups']]))
        del states
    assert all(sha(path) == digest for path, digest in pins.items())
    value = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        source_checkpoint=plan['source_checkpoint'], source_sha256=plan['source_sha256'],
        capacity=str(capacity), production=str(production), compared_actual_updates=5,
        all17_model_and_all17_Adam_states_bitwise_equal_after_each_update=True,
        saved_std_and_actual_adaptive_LR_equal=True, all100_original_Adam_audits_equal=True,
        all9_alignment_errors_zero_for_all8192_each_update=True,
        sampled32_rollout_arrays_bitwise_equal=True, configuration_differences=dict(env=env_delta, agent=agent_delta),
        rows=rows, source_pins=pins, simulator_started=False, optimizer_executed=False,
        training_modified=False, capacity_weights_promoted=False, policy_accepted=False,
        limitation='First five committed updates only. Saved rollouts contain32 sampled environments; '
            'alignment audits cover all8192. This is not independent policy acceptance or proof of later convergence.')
    with destination.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(verified=True, compared_updates=5, all100_Adam_audits_equal=True,
        all17_model_and_Adam_states_bitwise_equal=True, destination=str(destination))), flush=True)
    return value


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('production', 'capacity', 'production-run', 'capacity-run', 'destination'):
        cli.add_argument('--' + name, type=Path, required=True)
    args = cli.parse_args()
    audit(args.production, args.capacity, args.production_run, args.capacity_run, args.destination)
