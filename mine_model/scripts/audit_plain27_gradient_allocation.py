"""Partition recorded ordinary PPO gradient norms; never run backward or an optimizer."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
import yaml
import rsl_rl.modules.actor_critic as actor_critic_source

from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
import joint_pd_training_observer as observer

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'


def audit(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'gradient_allocation_prefix_{through:04d}_readonly.json'
    if directory.parent != RESULTS or destination.exists() or not 1 <= through <= 200:
        raise ValueError('Preserve one fresh diagnostic of an owned committed200-update block')
    plan_path = directory / 'experiment_plan.json'
    health_path = directory / f'policy_update_health_prefix_{through:04d}_readonly.json'
    plan, health = [json.loads(p.read_text()) for p in (plan_path, health_path)]
    if (not plan['ordinary_full_policy_PPO'] or plan['auxiliary_losses_enabled']
            or plan['training_config']['observation_space'] != 27
            or plan['native_num_envs'] != 4096 or health['committed_updates'] != through
            or len(health['snapshots']) != through
            or any(row['source_mean_parity_max_error'] or row['source_std_parity_max_error']
                   for row in health['snapshots'])):
        raise ValueError('Require the original contiguous plain27 ordinary PPO evidence')
    checkpoint = Path(health['snapshots'][-1]['checkpoint_after'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model, states = saved['model_state_dict'], saved['optimizer_state_dict']['state']
    actual = plan['source_actual_ppo_updates'] + through
    agent_path = checkpoint.parent / 'params/agent.yaml'
    agent = yaml.safe_load(agent_path.read_text())
    actor = [k for k in model if k.startswith('actor.')]
    critic = [k for k in model if k.startswith('critic.')]
    other = set(model) - set(actor) - set(critic)
    if (not tensor_finite(saved) or saved_updates(checkpoint) != actual or len(states) != 17
            or any(float(state['step']) != actual * 20 for state in states.values())
            or len(actor) != 8 or len(critic) != 8 or other != {'std'}
            or list(model['std'].shape) != [6] or not bool((model['std'] > 0.).all())
            or agent['policy']['class_name'] != 'ActorCritic'
            or agent['policy'].get('actor_obs_normalization')
            or agent['policy'].get('critic_obs_normalization')
            or agent['policy']['noise_std_type'] != 'scalar'):
        raise ValueError('Requalify the exact disjoint actor/critic/std parameter partition')
    fields = ('actor_gradient_norm_after_global_clip', 'critic_gradient_norm_after_global_clip',
              'all_parameter_gradient_norm_after_global_clip')
    pins = {**health['source_pins'], **{str(p): sha(p) for p in
        (SCRIPT, plan_path, health_path, checkpoint, agent_path, Path(observer.__file__),
         Path(actor_critic_source.__file__))}}
    records, fractions, minimum_residual = [], [], float('inf')
    rounding_corrections = 0
    for index in range(1, through + 1):
        path = directory / f'native_learning_signal/audit_update_{index:04d}.json'
        record = json.loads(path.read_text())
        if (record['update'] != index or record['total_environments'] != 4096
                or len(record['all_environment_alignment_max_errors']) != 9
                or any(record['all_environment_alignment_max_errors'].values())
                or not record['probe_RNG_unchanged'] or record['added_loss_or_backward']
                or len(record['optimizer_steps']) != 20):
            raise ValueError('Resolve native gradient and transition accounting first')
        pins[str(path)] = sha(path)
        for minibatch, step in enumerate(record['optimizer_steps'], 1):
            a, c, total = [float(step[key]) for key in fields]
            if not np.isfinite([a, c, total]).all() or min(a, c, total) < 0. or total <= 0.:
                raise ValueError('Invalid recorded Euclidean gradient norm')
            residual = total**2 - a**2 - c**2
            minimum_residual = min(minimum_residual, residual)
            tolerance = 2.e-6 * max(total**2, 1.e-12)
            if residual < -tolerance:
                raise ValueError('Recorded disjoint gradient norms fail squared-norm additivity')
            if residual < 0.:
                rounding_corrections += 1
            energy = np.asarray([a*a, c*c, max(residual, 0.)], dtype=np.float64)
            fraction = energy / total**2
            fractions.append(fraction)
            records.append(dict(update=index, minibatch=minibatch,
                actor_critic_std_squared_gradient_fractions=fraction.tolist(),
                std_norm_inferred_from_disjoint_parameter_partition=float(np.sqrt(max(residual, 0.)))))
    values = np.asarray(fractions)
    windows = dict(entire_prefix=values, first20=values[:400], last20=values[-400:])
    summaries = {name: dict(original_minibatches=len(rows),
        mean_actor_critic_std_squared_gradient_fraction=rows.mean(0).tolist(),
        median_actor_critic_std_squared_gradient_fraction=np.median(rows, axis=0).tolist(),
        min_actor_critic_std_squared_gradient_fraction=rows.min(0).tolist(),
        max_actor_critic_std_squared_gradient_fraction=rows.max(0).tolist(),
        actor_majority_squared_norm_minibatches=int((rows[:, 0] > .5).sum()),
        critic_majority_squared_norm_minibatches=int((rows[:, 1] > .5).sum()),
        std_majority_squared_norm_minibatches=int((rows[:, 2] > .5).sum()))
        for name, rows in windows.items()}
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('Immutable native gradient sources changed')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_updates=through,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), actual_ppo_updates=actual,
        all17_Adam_step=actual * 20, parameter_partition=dict(actor=actor, critic=critic, other=['std']),
        original_Euclidean_gradient_norm_reader_pinned=True,
        fraction_order=['actor_mean_network', 'critic_network', 'six_Gaussian_std_parameters'],
        groups=summaries, minibatches=records, inferred_std_norm_uses_recorded_float32_norms=True,
        minimum_unrounded_squared_residual=minimum_residual,
        tolerance_limited_negative_rounding_corrections=rounding_corrections, source_pins=pins,
        simulator_started=False, backward_executed=False, optimizer_executed=False,
        training_modified=False, acceptance_changed=False, policy_accepted=False,
        limitations=['The std gradient norm is inferred from disjoint squared norms recorded after the original global clipping; its direction and individual heads are not recorded.',
            'Float32 norm rounding is explicitly bounded. Fractions measure squared gradient norms, not Adam parameter displacement or policy influence.',
            'Neither a positive actor gradient nor small std gradient share proves convergence, adequate physical exploration, or absence of local optima.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return destination, result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    path, result = audit(args.directory, args.through)
    print(json.dumps(dict(report=str(path), groups=result['groups'], training_modified=False)))
