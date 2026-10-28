"""Read saved Adam moments at exploration floors; never step an optimizer.

Moment direction is historical optimization pressure, not a future gradient,
variance optimum, causal learning benefit, or an independent acceptance result.
"""
import argparse
import ast
import contextlib
from datetime import datetime
import inspect
import io
import json
from pathlib import Path
import textwrap

import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()


def audit(directory, destination):
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    if destination.exists() or directory.parent != ROOT / 'mine_model/results':
        raise ValueError('Keep earlier diagnostics and use a retained native block')
    plan_path = directory / 'experiment_plan.json'
    health_path = directory / 'policy_update_health_prefix_0200_readonly.json'
    final_path = directory / 'final_checkpoint_verification_readonly_20261019.json'
    plan, health, final = [json.loads(p.read_text()) for p in (plan_path, health_path, final_path)]
    source = Path(plan['source_checkpoint'])
    if (not plan['ordinary_full_policy_PPO'] or plan['auxiliary_losses_enabled']
            or plan['policy_class'] != 'ActorCritic' or plan['external_observation_size'] != 27
            or health['committed_updates'] != 200 or len(health['snapshots']) != 200
            or not final['verified'] or final['exact_actual_update_delta'] != 200
            or final['checkpoint_sha256'] != health['snapshots'][-1]['checkpoint_after_sha256']
            or sha(source) != plan['source_sha256']):
        raise ValueError('Require the intact plain27 ordinary-PPO completed200 source')
    optimizer_expression = ast.parse('optim.Adam(self.policy.parameters(), lr=learning_rate)', mode='eval').body
    constructor = ast.parse(textwrap.dedent(inspect.getsource(PPO.__init__)))
    assignments = [node for node in ast.walk(constructor) if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == 'self.optimizer' for target in node.targets)]
    if len(assignments) != 1 or ast.dump(assignments[0].value) != ast.dump(optimizer_expression):
        raise ValueError('Recheck the original PPO Adam parameter-order contract')
    rng = torch.random.get_rng_state().clone()
    state = torch.load(source, map_location='cpu', weights_only=True)
    agent_path = source.parent / 'params/agent.yaml'
    agent = yaml.safe_load(agent_path.read_text())
    options = dict(agent['policy'])
    if options.pop('class_name') != 'ActorCritic':
        raise ValueError('Require the original policy class')
    observation = TensorDict({'policy': torch.zeros(1, 27)}, batch_size=[1])
    with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
        policy = ActorCritic(observation, agent['obs_groups'], 6, **options)
        policy.load_state_dict(state['model_state_dict'], strict=True)
    parameters = list(policy.named_parameters())
    if len(parameters) != 17 or sum(name == 'std' for name, _ in parameters) != 1:
        raise ValueError('Require the exact original seventeen parameters')
    order = [name for name, _ in parameters]
    profile = plan['profile']
    floor = np.full(6, max(profile['global_minimum_std'], profile['minimum_leg_std']))
    floor[[2, 5]] = max(profile['global_minimum_std'], profile['minimum_wheel_std'])
    rows, pins = [], {str(p): sha(p) for p in
        [SCRIPT, plan_path, health_path, final_path, source, agent_path, Path(inspect.getsourcefile(PPO))]}
    for snapshot in health['snapshots']:
        checkpoint = Path(snapshot['checkpoint_after'])
        digest = sha(checkpoint)
        if digest != snapshot['checkpoint_after_sha256']:
            raise ValueError('A saved checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        actual = saved_updates(checkpoint)
        optimizer = saved['optimizer_state_dict']
        if len(optimizer['param_groups']) != 1 or len(optimizer['state']) != 17:
            raise ValueError('Keep the original complete Adam group')
        group = optimizer['param_groups'][0]
        ids = group['params']
        if (len(ids) != 17 or group['weight_decay'] or group.get('maximize', False)
                or group.get('amsgrad', False) or actual != plan['source_actual_ppo_updates'] + snapshot['committed_update']):
            raise ValueError('Unexpected Adam formula or native clock')
        for pid, (name, parameter) in zip(ids, parameters):
            moment = optimizer['state'][pid]
            if (float(moment['step']) != 20 * actual
                    or moment['exp_avg'].shape != parameter.shape
                    or moment['exp_avg_sq'].shape != parameter.shape
                    or not torch.isfinite(moment['exp_avg']).all()
                    or not torch.isfinite(moment['exp_avg_sq']).all()):
                raise ValueError('All original parameter moments and clocks must match')
        moment = optimizer['state'][ids[order.index('std')]]
        m, v = [moment[k].double().numpy() for k in ('exp_avg', 'exp_avg_sq')]
        sigma = saved['model_state_dict']['std'].double().numpy()
        beta1, beta2 = group['betas']
        t = float(moment['step'])
        normalized_momentum = (m / (1. - beta1**t)) / (np.sqrt(v / (1. - beta2**t)) + group['eps'])
        direction = -group['lr'] * normalized_momentum
        bound = abs(sigma - floor) <= 1.e-8
        if np.any(sigma < floor - 1.e-8) or not np.isfinite(direction).all():
            raise ValueError('Invalid saved noise or moment arithmetic')
        rows.append(dict(committed_update=snapshot['committed_update'], actual_updates=actual,
            all17_Adam_step=int(t), saved_std=sigma.tolist(), floor_binding=bound.tolist(),
            std_first_moment=m.tolist(), std_second_moment=v.tolist(),
            historical_momentum_direction_at_saved_LR=direction.tolist(),
            binding_and_momentum_toward_lower_std=(bound & (direction < 0.)).tolist(),
            saved_actual_LR=group['lr']))
        pins[str(checkpoint)] = digest
    if not torch.equal(rng, torch.random.get_rng_state()):
        raise ValueError('Read-only reconstruction must preserve the CPU RNG')
    groups = {}
    for name, selected in [('all200', rows), ('first20', rows[:20]), ('last20', rows[-20:])]:
        groups[name] = dict(checkpoints=len(selected),
            floor_binding_fraction=np.mean([x['floor_binding'] for x in selected], axis=0).tolist(),
            binding_and_momentum_toward_lower_std_fraction=np.mean(
                [x['binding_and_momentum_toward_lower_std'] for x in selected], axis=0).tolist(),
            mean_historical_momentum_direction=np.mean(
                [x['historical_momentum_direction_at_saved_LR'] for x in selected], axis=0).tolist())
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_updates=200,
        source_actual_updates=plan['source_actual_ppo_updates'], candidate_actual_updates=final['actual_ppo_updates'],
        candidate_checkpoint_sha256=final['checkpoint_sha256'], original_parameter_order=order,
        native_optimizer_parameter_order_verified_from_original_PPO=True, floors=floor.tolist(),
        action_order=['left_back', 'left_front', 'left_wheel', 'right_back', 'right_front', 'right_wheel'],
        groups=groups, checkpoints=rows, source_pins=pins,
        model_constructed_readonly_under_preserved_RNG=True, optimizer_constructed=False,
        optimizer_executed=False, RNG_draws_added=0, RNG_state_unchanged=True,
        simulator_started=False, training_modified=False, saved_std_updated=False,
        parameter_change_selected=False, local_optimum_proven=False, policy_accepted=False,
        limitations=['Saved per-update Adam moments summarize earlier gradients, not the next minibatch gradient or exact blocked optimizer displacement.',
            'A binding floor and downward momentum justify testing the constraint, but do not prove that lower exploration will improve body tracking or stability.',
            'These are checkpoint statistics, not independent policy-performance samples.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = audit(args.directory, args.output)
    print(json.dumps(dict(report=str(args.output), groups=result['groups'], training_modified=False)))
