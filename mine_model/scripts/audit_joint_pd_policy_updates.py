"""Read saved PPO update sizes and optimizer health without changing training.

Raw Gaussian densities match PPO's unclipped action storage. Physical action
clipping is a separate diagnostic in audit_effective_joint_pd_exploration.py.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diagnose_height_response import actor
from joint_pd_continuation import sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size or not np.isfinite(values).all():
        raise ValueError('Require nonempty finite original optimizer evidence')
    return dict(min=float(values.min()), median=float(np.median(values)),
                mean=float(values.mean()), max=float(values.max()))


def gaussian_update(old_mean, old_std, new_mean, new_std, actions):
    """Return exact KL(old||new) and the actual raw-action density ratio."""
    if np.any(old_std <= 0.) or np.any(new_std <= 0.):
        raise ValueError('Nonpositive Gaussian exploration')
    kl = (np.log(new_std / old_std)
          + (old_std**2 + (old_mean - new_mean)**2) / (2. * new_std**2) - .5)
    log_ratio = (np.log(old_std / new_std)
                 + .5 * ((actions - old_mean) / old_std)**2
                 - .5 * ((actions - new_mean) / new_std)**2).sum(-1)
    return kl, log_ratio


def audit(directory, through, interval, device='cpu'):
    directory = Path(directory).resolve()
    destination = directory / f'policy_update_health_prefix_{through:04d}_readonly.json'
    if destination.exists():
        raise ValueError('Preserve previous immutable policy-update evidence')
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    root = Path(__file__).resolve().parents[2]
    matches = list((root / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat').glob(
        '*_' + plan['run_name']))
    if len(matches) != 1:
        raise ValueError('Require exactly the owned native run')
    run = matches[0]
    agent_path = run / 'params/agent.yaml'
    agent = yaml.safe_load(agent_path.read_text())
    policy, algorithm = agent['policy'], agent['algorithm']
    cfg = plan['training_config']
    if (policy['class_name'] != 'ActorCritic' or policy['activation'] != 'elu'
            or policy.get('actor_obs_normalization')
            or policy['actor_hidden_dims'] != [256, 128, 64]
            or cfg['leg_control_mode'] != 'implicit_joint_position'
            or cfg['observation_space'] != 27 or algorithm['class_name'] != 'PPO'):
        raise ValueError('Require the original unnormalized plain27 direct-joint PPO')
    minibatches = algorithm['num_learning_epochs'] * algorithm['num_mini_batches']
    if minibatches != plan['optimizer_minibatches_per_actual_update']:
        raise ValueError('Optimizer accounting differs from the owned trial')
    pins = {str(p): sha(p) for p in (plan_path, agent_path, Path(__file__).resolve())}
    device = torch.device(device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Only qualified CPU or native CUDA inference is supported')
    if device.type == 'cuda':
        free, _ = torch.cuda.mem_get_info(device)
        if free < 2 * 1024**3:
            raise ValueError('Defer read-only inference to protect native training memory')
        train_path = root / 'wheel_legged_isaaclab/scripts/rsl_rl/train.py'
        if 'torch.backends.cuda.matmul.allow_tf32 = True' not in train_path.read_text():
            raise ValueError('Requalify native inference precision after the trainer changes')
        pins[str(train_path)] = sha(train_path)
        torch.backends.cuda.matmul.allow_tf32 = True
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone() if device.type == 'cuda' else None
    updates, native_ids = [], None
    for index in range(1, through + 1):
        path = directory / 'native_learning_signal' / f'audit_update_{index:04d}.json'
        record = json.loads(path.read_text())
        errors = record['all_environment_alignment_max_errors']
        if (record['update'] != index or record['total_environments'] != plan['native_num_envs']
                or len(errors) != 9 or any(errors.values())
                or not record['probe_RNG_unchanged'] or record['added_loss_or_backward']
                or len(record['optimizer_steps']) != minibatches):
            raise ValueError('Resolve native alignment and complete optimizer records first')
        if native_ids is None:
            native_ids = record['sampled_environment_ids']
        if native_ids != record['sampled_environment_ids']:
            raise ValueError('Native sample identity changed')
        updates.append(record)
        pins[str(path)] = sha(path)
    fields = ('actor_gradient_norm_after_global_clip', 'critic_gradient_norm_after_global_clip',
              'all_parameter_gradient_norm_after_global_clip', 'adaptive_learning_rate')
    windows = dict(entire_prefix=updates, first20=updates[:20], last20=updates[-20:])
    optimizer = {name: {key: describe([step[key] for update in rows
                    for step in update['optimizer_steps']]) for key in fields}
                 for name, rows in windows.items()}
    for update in updates:
        for step in update['optimizer_steps']:
            if (step['actor_gradient_norm_after_global_clip'] <= 0.
                    or step['adaptive_learning_rate'] <= 0.):
                raise ValueError('Actor gradient or actual adaptive learning rate is zero')

    def load(path, expected):
        digest = sha(path)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        states = saved['optimizer_state_dict']['state']
        if (not tensor_finite(saved) or len(states) != 17 or saved_updates(path) != expected
                or any(float(value['step']) != expected * minibatches for value in states.values())):
            raise ValueError('Require every original Adam state at the exact saved update clock')
        pins[str(path)] = digest
        return saved['model_state_dict']

    def infer(inputs, state):
        with torch.inference_mode():
            if device.type == 'cpu':
                return actor(torch.from_numpy(inputs), state).numpy()
            # CUDA TF32 kernels depend on matrix shape. Reproduce the native
            #4096-row call with sampled rows at their actual positions. The
            #unobserved rows are zeros and do not mix across the linear layers.
            weights = {key: value.to(device) for key, value in state.items()}
            ids = torch.tensor(native_ids, device=device)
            full = torch.zeros((plan['native_num_envs'], 27), device=device)
            result = []
            for step in inputs:
                full[ids] = torch.as_tensor(step, device=device)
                result.append(actor(full, weights)[ids].cpu().numpy())
            return np.stack(result)

    snapshots, max_formula_error = [], 0.
    source = plan['source_actual_ppo_updates']
    for index in sorted(set([1, through, *range(interval, through + 1, interval)])):
        data_path = directory / 'native_learning_signal' / f'rollout_{index:04d}.npz'
        with np.load(data_path, allow_pickle=False) as arrays:
            data = {key: arrays[key].copy() for key in ('observations', 'mu', 'sigma', 'actions', 'advantages')}
        for key, value in data.items():
            if value.shape[:2] != (48, len(native_ids)) or not np.isfinite(value).all():
                raise ValueError('Invalid immutable native policy sample: ' + key)
        before_path = (Path(plan['source_checkpoint']) if index == 1
                       else run / f'model_{source + index - 2}.pt')
        after_path = run / f'model_{source + index - 1}.pt'
        before = load(before_path, source + index - 1)
        after = load(after_path, source + index)
        old_recomputed = infer(data['observations'], before)
        new_mean = infer(data['observations'], after).astype(np.float64)
        parity = float(np.abs(old_recomputed - data['mu']).max())
        std_parity = float(np.abs(before['std'].numpy() - data['sigma']).max())
        if parity > 1.e-4 or std_parity > 1.e-7:
            raise ValueError('Pre-update checkpoint does not reproduce the recorded old distribution')
        old_mean, old_std, actions = [data[key].astype(np.float64) for key in ('mu', 'sigma', 'actions')]
        new_std = after['std'].numpy().astype(np.float64)
        kl, log_ratio = gaussian_update(old_mean, old_std, new_mean, new_std, actions)
        # Independent Torch distribution API checks the analytic KL and log-ratio.
        old = torch.distributions.Normal(torch.from_numpy(old_mean), torch.from_numpy(old_std))
        new = torch.distributions.Normal(torch.from_numpy(new_mean), torch.from_numpy(new_std))
        reference_kl = torch.distributions.kl_divergence(old, new).numpy()
        reference_ratio = (new.log_prob(torch.from_numpy(actions))
                           - old.log_prob(torch.from_numpy(actions))).sum(-1).numpy()
        error = max(float(np.abs(kl - reference_kl).max()),
                    float(np.abs(log_ratio - reference_ratio).max()))
        max_formula_error = max(max_formula_error, error)
        if error > 1.e-9:
            raise ValueError('Independent distribution API differs from analytic update diagnostics')
        ratio = np.exp(log_ratio)
        if not np.isfinite(ratio).all():
            raise ValueError('Nonfinite saved-policy density ratio')
        clip = algorithm['clip_param']
        advantage = data['advantages'][..., 0]
        snapshots.append(dict(committed_update=index, actual_updates_before_after=[source+index-1, source+index],
            checkpoint_after=str(after_path), checkpoint_after_sha256=sha(after_path),
            all17_Adam_step_after=(source+index)*minibatches,
            sampled_transitions=int(np.prod(log_ratio.shape)), source_mean_parity_max_error=parity,
            source_std_parity_max_error=std_parity, old_std=before['std'].tolist(), new_std=after['std'].tolist(),
            exact_gaussian_KL_old_to_new=describe(kl.sum(-1)),
            exact_KL_mean_by_head=kl.mean((0, 1)).tolist(),
            raw_mean_step_RMS_by_head=np.sqrt(((new_mean-old_mean)**2).mean((0, 1))).tolist(),
            mean_step_in_old_std_RMS_by_head=np.sqrt((((new_mean-old_mean)/old_std)**2).mean((0, 1))).tolist(),
            applied_mean_step_RMS_by_head=np.sqrt(((np.clip(new_mean,-1.,1.)-np.clip(old_mean,-1.,1.))**2).mean((0,1))).tolist(),
            raw_PPO_ratio_outside_clip_fraction=float(((ratio < 1.-clip)|(ratio > 1.+clip)).mean()),
            raw_PPO_surrogate_clipped_fraction=float((((advantage > 0.) & (ratio > 1.+clip))
                | ((advantage < 0.) & (ratio < 1.-clip))).mean()),
            desired_kl=algorithm['desired_kl'], clip_param=clip))
        pins[str(data_path)] = sha(data_path)
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('Immutable diagnostic inputs changed while reading')
    if (not torch.equal(cpu_rng, torch.random.get_rng_state())
            or (cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device)))):
        raise ValueError('The read-only diagnostic unexpectedly drew random numbers')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_updates=through,
        original_Adam_minibatches=through*minibatches, native_num_envs=plan['native_num_envs'],
        sampled_environment_ids=native_ids, optimizer=optimizer, snapshots=snapshots,
        independent_distribution_formula_max_error=max_formula_error, source_pins=pins,
        inference_device=str(device), native_TF32_and_full_batch_shape=device.type == 'cuda',
        diagnostic_RNG_states_unchanged=True,
        simulator_started=False, optimizer_executed=False, RNG_draws_added=0,
        training_modified=False, policy_accepted=False,
        limitations=['KL and PPO ratios use32 immutable sampled environments at selected saved updates, not the full native minibatches.',
            'Unclipped Gaussian update size does not establish physical exploration, convergence, a local optimum, or independent acceptance.',
            'The desired KL is the existing adaptive-LR target, not a newly imposed acceptance or stop threshold.'])
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return destination, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--through', type=int, required=True)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if args.through < 1 or args.interval < 1:
        parser.error('through and interval must be positive')
    torch.set_num_threads(1)
    path, result = audit(args.directory, args.through, args.interval, args.device)
    print(json.dumps(dict(report=str(path), committed_updates=result['committed_updates'],
        optimizer_last20=result['optimizer']['last20'], snapshots=result['snapshots']), ensure_ascii=False))
