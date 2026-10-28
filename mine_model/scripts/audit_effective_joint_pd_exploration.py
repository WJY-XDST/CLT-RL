"""Read raw Gaussian coverage and clipping-reduced exploration, without RNG."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'effective_exploration_prefix_{through:04d}_readonly.json'
    if destination.exists():
        raise ValueError('Preserve the existing immutable exploration report')
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    cfg = plan['training_config']
    fields = ('native_raw_actions', 'mu', 'sigma', 'native_commands', 'dones')
    chunks, pins, ids = {k: [] for k in fields}, {str(plan_path): sha(plan_path)}, None
    for index in range(1, through + 1):
        proof_path = directory / 'native_learning_signal' / f'audit_update_{index:04d}.json'
        data_path = proof_path.with_name(f'rollout_{index:04d}.npz')
        proof = json.loads(proof_path.read_text())
        if (proof['total_environments'] != plan['native_num_envs']
                or proof['added_loss_or_backward'] or not proof['probe_RNG_unchanged']
                or len(proof['all_environment_alignment_max_errors']) != 9
                or any(proof['all_environment_alignment_max_errors'].values())):
            raise ValueError('Resolve native alignment before interpreting exploration')
        if ids is None:
            ids = proof['sampled_environment_ids']
        if proof['sampled_environment_ids'] != ids:
            raise ValueError('Sample identity changed')
        with np.load(data_path, allow_pickle=False) as data:
            for key in fields:
                value = data[key]
                if value.shape[:2] != (48, len(ids)) or not np.isfinite(value).all():
                    raise ValueError('Invalid immutable native array: ' + key)
                chunks[key].append(value.copy())
        pins.update({str(p): sha(p) for p in (proof_path, data_path)})
    data = {key: np.concatenate(value) for key, value in chunks.items()}
    raw, mean, sigma = (data[k] for k in ('native_raw_actions', 'mu', 'sigma'))
    if np.any(sigma <= 0.):
        raise ValueError('Nonpositive policy exploration')
    normalized = (raw - mean) / sigma
    applied_noise = np.clip(raw, -1., 1.) - np.clip(mean, -1., 1.)
    done, commands = data['dones'][..., 0].astype(bool), data['native_commands']
    shape, dt = done.shape, cfg['sim']['dt'] * cfg['decimation']
    age, command_age = np.zeros(shape), np.zeros(shape)
    age[0] = dt
    for t in range(1, shape[0]):
        age[t] = np.where(done[t - 1], dt, age[t - 1] + dt)
        changed = np.any(abs(commands[t] - commands[t - 1]) > 1.e-7, axis=-1)
        command_age[t] = np.where(changed | done[t - 1], 0., command_age[t - 1] + dt)
    settled = (age > 2.) & (command_age >= 1.) & ~done
    static = np.broadcast_to(np.array(ids)[None] < plan['fixed_standing_environments'], shape)
    low, high = cfg['commands']['ranges_height']
    populations = dict(all_settled=settled, dedicated_static=settled & static,
        static_low=settled & static & np.isclose(commands[..., 2], low, atol=1.e-6, rtol=0.),
        static_high=settled & static & np.isclose(commands[..., 2], high, atol=1.e-6, rtol=0.),
        motion_forward=settled & ~static & (commands[..., 0] > .05),
        motion_reverse=settled & ~static & (commands[..., 0] < -.05))
    windows = dict(entire_prefix=np.ones(shape, dtype=bool))
    if through >= 40:
        windows.update(first20=np.broadcast_to(np.arange(shape[0])[:, None] < 20 * 48, shape),
            last20=np.broadcast_to(np.arange(shape[0])[:, None] >= (through - 20) * 48, shape))
    groups = {}
    for window, selection in windows.items():
        groups[window] = {}
        for label, population in populations.items():
            mask = selection & population
            if not mask.any():
                continue
            eps, noise, sampled, mu, std = [x[mask] for x in (normalized, applied_noise, raw, mean, sigma)]
            wheel = noise[:, (2, 5)] * cfg['action_scale_vel']
            groups[window][label] = dict(samples=int(mask.sum()),
                raw_gaussian_standardized_noise_mean=eps.mean(0).tolist(),
                raw_gaussian_standardized_noise_std=eps.std(0).tolist(),
                raw_standardized_noise_head_correlation=np.corrcoef(eps, rowvar=False).tolist(),
                saved_distribution_std_min=std.min(0).tolist(), saved_distribution_std_max=std.max(0).tolist(),
                raw_action_clipping_fraction_by_head=(abs(sampled) > 1.).mean(0).tolist(),
                mean_outside_action_envelope_fraction_by_head=(abs(mu) > 1.).mean(0).tolist(),
                clipping_reduced_action_noise_RMS_by_head=np.sqrt((noise**2).mean(0)).tolist(),
                exactly_zero_clipped_noise_fraction_by_head=(abs(noise) < 1.e-8).mean(0).tolist(),
                wheel_reference_common_noise_RMS_rad_s=float(np.sqrt((wheel.mean(-1)**2).mean())),
                wheel_reference_differential_noise_RMS_rad_s=float(np.sqrt(((wheel[:, 1] - wheel[:, 0])**2).mean())))
    report = dict(observed_at=datetime.now().astimezone().isoformat(), committed_updates=through,
        native_num_envs=plan['native_num_envs'], sampled_environment_ids=ids,
        head_order=['left_hip_back', 'left_hip_front', 'left_wheel', 'right_hip_back', 'right_hip_front', 'right_wheel'],
        groups=groups, source_pins=pins, auditor_sha256=sha(__file__),
        RNG_draws_added=0, simulator_started=False, optimizer_loaded=False, training_modified=False,
        policy_accepted=False, limitations=[
            '32 sampled native environments, with changing policies and command distributions; not independent acceptance.',
            'Clipped-action noise is an execution-envelope diagnostic. Hard joint limits and nonlinear FK can reduce actual geometric exploration further.',
            'Wheel-reference noise is not actual body-speed or yaw noise. Correlations do not prove a local optimum or a policy gradient.'])
    with destination.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return destination, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--through', type=int, required=True)
    args = parser.parse_args()
    path, report = audit(args.directory, args.through)
    print(json.dumps(dict(report=str(path), latest_window=report['groups'].get('last20', report['groups']['entire_prefix'])), ensure_ascii=False))
