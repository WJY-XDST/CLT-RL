"""Compare saved critic residuals by known command-clock phase; no fitting."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(values, returns, mask):
    if not mask.any():
        return dict(samples=0)
    target, predicted = returns[mask], values[mask]
    residual = target - predicted
    variance = float(target.var())
    return dict(samples=int(mask.sum()),
        preupdate_critic_explained_variance=1. - float(residual.var()) / variance if variance > 1.e-15 else None,
        critic_return_residual_mean=float(residual.mean()),
        critic_return_residual_RMS=float(np.sqrt(np.mean(residual**2))),
        GAE_return_std=float(target.std()), predicted_value_std=float(predicted.std()))


def audit(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'critic_fit_by_resampling_phase_prefix_{through:04d}_readonly.json'
    if destination.exists() or through < 40 or through > 200:
        raise ValueError('Use a fresh committed40to200 prefix')
    plan_path = directory / 'experiment_plan.json'
    reference_path = directory / f'critic_and_command_coverage_prefix_{through:04d}_readonly.json'
    plan, reference = [json.loads(p.read_text()) for p in (plan_path, reference_path)]
    cfg = plan['training_config']
    if (cfg['observation_space'] != 27 or cfg['leg_control_mode'] != 'implicit_joint_position'
            or not cfg['commands']['grouped_training'] or reference['committed_updates'] != through
            or plan['native_num_envs'] != 4096):
        raise ValueError('Require qualified native4096/plain27 grouped-command records')
    fields = ('values', 'returns', 'native_commands', 'dones')
    chunks = {key: [] for key in fields}
    pins = {**reference['source_pins'], **{str(p): sha(p) for p in
        [Path(__file__), plan_path, reference_path]}}
    ids = reference['sampled_environment_ids']
    for index in range(1, through + 1):
        proof_path = directory / f'native_learning_signal/audit_update_{index:04d}.json'
        data_path = proof_path.with_name(f'rollout_{index:04d}.npz')
        proof = json.loads(proof_path.read_text())
        if (proof['sampled_environment_ids'] != ids or proof['total_environments'] != 4096
                or len(proof['all_environment_alignment_max_errors']) != 9
                or any(proof['all_environment_alignment_max_errors'].values())
                or not proof['probe_RNG_unchanged'] or proof['added_loss_or_backward']):
            raise ValueError('Resolve source alignment before comparing value residuals')
        with np.load(data_path, allow_pickle=False) as arrays:
            for key in fields:
                value = arrays[key]
                if value.shape[:2] != (48, len(ids)) or not np.isfinite(value).all():
                    raise ValueError('Incomplete native array: ' + key)
                chunks[key].append(value.copy())
    data = {key: np.concatenate(parts) for key, parts in chunks.items()}
    values, returns = [data[key][..., 0].astype(np.float64) for key in ('values', 'returns')]
    commands, dones = data['native_commands'], data['dones'][..., 0].astype(bool)
    dt = cfg['sim']['dt'] * cfg['decimation']
    period = int(cfg['commands']['resampling_time'] / dt)
    initial = plan['source_actual_ppo_updates'] * 48
    health = json.loads((directory / f'policy_update_health_prefix_{through:04d}_readonly.json').read_text())
    start_path = Path(health['snapshots'][-1]['checkpoint_after']).parent / 'params/curriculum_start.yaml'
    start = yaml.safe_load(start_path.read_text())
    if start['initial_common_step_counter'] != initial or period <= 0:
        raise ValueError('Requalify the actual native command clock')
    pins[str(start_path)] = sha(start_path)
    counter = initial + np.arange(len(commands))
    changed_height = abs(commands[1:, :, 2] - commands[:-1, :, 2]) > 1.e-7
    changed_height &= ~dones[:-1]
    remainders = np.broadcast_to((counter[1:] % period)[:, None], changed_height.shape)[changed_height]
    if not len(remainders) or np.any(remainders):
        raise ValueError('Observed non-reset height changes do not match the declared global phase')
    shape = values.shape
    age, command_age = np.zeros(shape), np.zeros(shape)
    age[0] = dt
    for step in range(1, shape[0]):
        age[step] = np.where(dones[step - 1], dt, age[step - 1] + dt)
        changed = np.any(abs(commands[step] - commands[step - 1]) > 1.e-7, axis=-1)
        command_age[step] = np.where(changed | dones[step - 1], 0., command_age[step - 1] + dt)
    settled = (age > 2.) & (command_age >= 1.) & ~dones
    static = np.broadcast_to(np.asarray(ids)[None] < plan['fixed_standing_environments'], shape)
    low, high = cfg['commands']['ranges_height']
    populations = dict(all=np.ones(shape, bool), all_settled=settled,
        dedicated_static_settled=static & settled, motion_population_settled=~static & settled,
        low_height_settled=settled & np.isclose(commands[..., 2], low, atol=1.e-6, rtol=0.),
        high_height_settled=settled & np.isclose(commands[..., 2], high, atol=1.e-6, rtol=0.))
    step_index = np.arange(shape[0])[:, None]
    windows = dict(entire_prefix=np.ones(shape, bool),
        first20=np.broadcast_to(step_index < 20 * 48, shape),
        last20=np.broadcast_to(step_index >= (through - 20) * 48, shape))
    until = np.broadcast_to(((period - counter % period) * dt)[:, None], shape)
    phases = dict(next_resample_le0p4s=until <= .4,
        next_resample_0p4to1s=(until > .4) & (until <= 1.), next_resample_above1s=until > 1.)
    if not np.all(sum(value.astype(int) for value in phases.values()) == 1):
        raise ValueError('Phase groups do not partition the recorded transitions')
    baseline_error, rows = 0., {}
    for window, selection in windows.items():
        rows[window] = {}
        for name, population in populations.items():
            mask = selection & population
            baseline = statistics(values, returns, mask)
            original = reference['groups'][window].get(name)
            if original is None:
                if baseline['samples']:
                    raise ValueError('Original conditional population differs')
                continue
            for key in ('samples', 'preupdate_critic_explained_variance',
                        'critic_return_residual_mean', 'critic_return_residual_RMS', 'GAE_return_std'):
                if baseline[key] is None or original[key] is None:
                    if baseline[key] != original[key]:
                        raise ValueError('Original conditional statistic differs')
                else:
                    baseline_error = max(baseline_error, abs(baseline[key] - original[key]))
            rows[window][name] = dict(original_population=baseline,
                by_next_global_resample={key: statistics(values, returns, mask & phase)
                                        for key, phase in phases.items()})
    if baseline_error > 1.e-12 or any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('Baseline metric reproduction or immutable inputs changed')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_prefix=through,
        initial_control_clock=initial, resampling_period_control_steps=period, policy_dt=dt,
        nonreset_height_changes_verified_at_exact_global_boundary=len(remainders),
        reference_metrics_reproduced_max_error=baseline_error, groups=rows, source_pins=pins,
        phase_used_only_for_diagnosis_not_added_to_policy=True,
        RNG_draws_added=0, optimizer_executed=False, simulator_started=False,
        training_modified=False, acceptance_changed=False, policy_accepted=False,
        limitations=['The same32 native environments, evolving policies and stored bootstrapped GAE targets are used; these are not independent critic targets or acceptance.',
            'The diagnostic phase is verified against non-reset height changes and the saved native clock. Correlation with residuals does not establish a causal reward, clock or exploration defect.',
            'Small conditional return variance can make explained variance unstable; sample counts, absolute residuals and prediction/return spread must be read together.',
            'The explicit clock is not added to observations. This analysis does not justify changing the retained27-dimensional policy or optimizer.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return destination, result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    path, result = audit(args.directory, args.through)
    print(json.dumps(dict(report=str(path), reference_metric_error=result['reference_metrics_reproduced_max_error'],
        verified_nonreset_height_change_count=result['nonreset_height_changes_verified_at_exact_global_boundary'],
        training_modified=False)))
