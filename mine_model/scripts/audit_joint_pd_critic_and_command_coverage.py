"""Read saved critic fit, entropy and command coverage without changing PPO.

The retained 32 environments are a diagnostic sample, not the complete4096
training population. No metric in this file replaces independent acceptance.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'critic_and_command_coverage_prefix_{through:04d}_readonly.json'
    if destination.exists():
        raise ValueError('Preserve the previous immutable diagnostic')
    plan_path = directory / 'experiment_plan.json'
    health_path = directory / f'policy_update_health_prefix_{through:04d}_readonly.json'
    plan, health = [json.loads(p.read_text()) for p in (plan_path, health_path)]
    cfg = plan['training_config']
    if (not plan['ordinary_full_policy_PPO'] or cfg['observation_space'] != 27
            or cfg['leg_control_mode'] != 'implicit_joint_position'
            or health['committed_updates'] != through
            or len(health['snapshots']) != through
            or any(s['source_mean_parity_max_error'] or s['source_std_parity_max_error']
                   for s in health['snapshots'])):
        raise ValueError('Require contiguous qualified native plain27 PPO evidence')
    pins = {str(p): sha(p) for p in (Path(__file__), plan_path, health_path)}
    fields = ('values', 'returns', 'advantages', 'rewards', 'dones', 'sigma',
              'native_reward', 'native_timeouts', 'native_commands')
    chunks, ids = {key: [] for key in fields}, health['sampled_environment_ids']
    for index in range(1, through + 1):
        proof_path = directory / 'native_learning_signal' / f'audit_update_{index:04d}.json'
        data_path = proof_path.with_name(f'rollout_{index:04d}.npz')
        proof = json.loads(proof_path.read_text())
        if (proof['update'] != index or proof['sampled_environment_ids'] != ids
                or proof['total_environments'] != plan['native_num_envs']
                or len(proof['all_environment_alignment_max_errors']) != 9
                or any(proof['all_environment_alignment_max_errors'].values())
                or not proof['probe_RNG_unchanged'] or proof['added_loss_or_backward']):
            raise ValueError('Resolve native alignment before interpreting critic fit')
        with np.load(data_path, allow_pickle=False) as data:
            for key in fields:
                value = data[key]
                if value.shape[:2] != (48, len(ids)) or not np.isfinite(value).all():
                    raise ValueError('Incomplete or nonfinite original native array: ' + key)
                chunks[key].append(value.copy())
        pins.update({str(p): sha(p) for p in (proof_path, data_path)})
    data = {key: np.concatenate(value) for key, value in chunks.items()}
    values, returns, advantages = [data[key][..., 0].astype(np.float64)
                                  for key in ('values', 'returns', 'advantages')]
    commands, dones = data['native_commands'], data['dones'][..., 0].astype(bool)
    shape = values.shape
    dt = cfg['sim']['dt'] * cfg['decimation']
    age, command_age = np.zeros(shape), np.zeros(shape)
    age[0] = dt
    for step in range(1, shape[0]):
        age[step] = np.where(dones[step - 1], dt, age[step - 1] + dt)
        changed = np.any(abs(commands[step] - commands[step - 1]) > 1.e-7, axis=-1)
        command_age[step] = np.where(changed | dones[step - 1], 0., command_age[step - 1] + dt)
    settled = (age > 2.) & (command_age >= 1.) & ~dones
    static = np.broadcast_to(np.array(ids)[None] < plan['fixed_standing_environments'], shape)
    low, high = cfg['commands']['ranges_height']
    populations = dict(all=np.ones(shape, bool), all_settled=settled,
        dedicated_static_settled=static & settled, motion_population_settled=~static & settled,
        low_height_settled=settled & np.isclose(commands[..., 2], low, atol=1.e-6, rtol=0.),
        high_height_settled=settled & np.isclose(commands[..., 2], high, atol=1.e-6, rtol=0.))
    windows = dict(entire_prefix=np.ones(shape, bool))
    if through >= 40:
        row_index = np.arange(shape[0])[:, None]
        windows.update(first20=np.broadcast_to(row_index < 20 * 48, shape),
                       last20=np.broadcast_to(row_index >= (through - 20) * 48, shape))
    sigma = data['sigma'].astype(np.float64)
    if np.any(sigma <= 0.):
        raise ValueError('Nonpositive saved Gaussian std')
    raw_entropy = np.sum(np.log(sigma) + .5 * np.log(2. * np.pi * np.e), axis=-1)
    groups = {}
    for window, selection in windows.items():
        groups[window] = {}
        for name, population in populations.items():
            mask = selection & population
            if not mask.any():
                continue
            target, residual, advantage = returns[mask], (returns - values)[mask], advantages[mask]
            variance = float(target.var())
            groups[window][name] = dict(samples=int(mask.sum()),
                preupdate_critic_explained_variance=1. - float(residual.var()) / variance
                    if variance > 1.e-15 else None,
                critic_return_residual_mean=float(residual.mean()),
                critic_return_residual_RMS=float(np.sqrt(np.mean(residual**2))),
                GAE_return_std=float(target.std()), normalized_advantage_mean=float(advantage.mean()),
                normalized_advantage_std=float(advantage.std()),
                raw_Gaussian_differential_entropy_mean=float(raw_entropy[mask].mean()),
                actual_reward_mean=float(data['native_reward'][mask].mean()),
                sampled_terminal_count=int(dones[mask].sum()))
    scale = plan['training_motion_scale']
    command_cfg = cfg['commands']
    vmax = scale * max(abs(v) for key in ('ranges_lin_vel_x', 'ranges_reverse_lin_vel_x',
        'ranges_transition_lin_vel_x') for v in command_cfg[key])
    ymax = scale * max(abs(v) for v in command_cfg['ranges_ang_vel_yaw'])
    bounds = [(low, high), (-vmax, vmax), (-ymax, ymax)]
    bins = []
    for lo, hi in bounds:
        if hi <= lo:
            raise ValueError('Use the moving-stage command coverage diagnostic after static qualification')
        edges = np.linspace(lo, hi, 7)
        edges[[0, -1]] += [-1.e-6, 1.e-6]  # Include float32 endpoint representations.
        bins.append(edges)
    coverage = {}
    ordered = commands[..., (2, 0, 1)]
    for name, mask in populations.items():
        selected = ordered[mask]
        if not len(selected):
            continue
        hist, _ = np.histogramdd(selected, bins=bins)
        if int(hist.sum()) != len(selected):
            raise ValueError('Commands outside declared stage bounds require inspection')
        coverage[name] = dict(samples=len(selected),
            observed_height_vx_yaw_min=selected.min(0).tolist(),
            observed_height_vx_yaw_max=selected.max(0).tolist(),
            height_bin_counts=hist.sum((1, 2)).astype(int).tolist(),
            vx_bin_counts=hist.sum((0, 2)).astype(int).tolist(),
            yaw_bin_counts=hist.sum((0, 1)).astype(int).tolist(),
            height_vx_bin_counts=hist.sum(2).astype(int).tolist(),
            exact_zero_vx_and_yaw_fraction=float(np.all(abs(selected[:, 1:]) < 1.e-8, axis=-1).mean()))
    # RSL adds timeout bootstrapping to its storage reward. The larger stored
    # reward at a timeout must not be diagnosed as an environment reward spike.
    # The original native auditor independently checked every4096-row timeout
    # bootstrap. Confirm the recorded gamma from the owned agent configuration.
    run = Path(health['snapshots'][-1]['checkpoint_after']).parent
    agent_path = run / 'params/agent.yaml'
    agent = yaml.safe_load(agent_path.read_text())
    gamma = agent['algorithm']['gamma']
    expected = data['native_reward'] + gamma * data['values'][..., 0] * data['native_timeouts']
    bootstrap_error = float(abs(expected - data['rewards'][..., 0]).max())
    if bootstrap_error > 2.e-6:
        raise ValueError('Resolve the independent float32 timeout-bootstrap reconstruction')
    pins[str(agent_path)] = sha(agent_path)
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('Immutable diagnostic inputs changed')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_updates=through,
        native_num_envs=plan['native_num_envs'], sampled_environment_ids=ids,
        sampled_transitions=int(np.prod(shape)), groups=groups, command_coverage=coverage,
        coverage_axis_order=['height_m', 'vx_m_s', 'yaw_rad_s'],
        histogram_edges_with_float32_outer_padding=[edge.tolist() for edge in bins],
        independent_timeout_bootstrap_max_error=bootstrap_error,
        entropy_coefficient=agent['algorithm']['entropy_coef'],
        normalized_advantage_per_minibatch=agent['algorithm']['normalize_advantage_per_mini_batch'],
        source_pins=pins, RNG_draws_added=0, simulator_started=False, optimizer_loaded=False,
        training_modified=False, acceptance_changed=False, local_optimum_proven=False, policy_accepted=False,
        limitations=[
            'These are32 sampled environments from evolving policies; no fixed-checkpoint acceptance or causal before/after improvement is established.',
            'Critic explained variance compares stored preupdate predictions with bootstrapped GAE targets; it is not an independent value-function test.',
            'Normalized advantages need not have unit variance in this32-environment subset of the original4096-environment batch.',
            'Gaussian differential entropy depends on action units and can be negative. It does not measure clipped physical exploration.',
            'Empty command bins can follow the intentional curriculum and grouped command design; coverage alone does not prove a local optimum.'])
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return destination, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--through', type=int, required=True)
    args = parser.parse_args()
    path, result = audit(args.directory, args.through)
    print(json.dumps(dict(report=str(path), sampled_transitions=result['sampled_transitions'],
        bootstrap_error=result['independent_timeout_bootstrap_max_error'],
        latest_groups=result['groups'].get('last20', result['groups']['entire_prefix']),
        training_modified=False), ensure_ascii=False))
