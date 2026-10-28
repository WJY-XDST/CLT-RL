"""Read saved std and Adam moments; execute no optimizer or simulation.

The moment-implied update below holds stored moments fixed. It describes
the direction encoded in those moments, not the next unknown gradient.
"""
import argparse
from contextlib import redirect_stdout
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path

import torch
from rsl_rl.modules import ActorCritic


SCRIPT = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, choices=[100, 200], required=True)
    args = cli.parse_args()
    folder = args.directory.resolve()
    destination = folder / f'saved_std_optimizer_pressure_prefix_{args.through:04d}_readonly.json'
    if destination.exists():
        raise FileExistsError('Preserve the completed diagnostic')
    plan_file = folder / 'experiment_plan.json'
    plan = read(plan_file)
    assert plan['source_actual_ppo_updates'] == 7001
    checkpoints = [(0, Path(plan['source_checkpoint']))]
    inputs = [SCRIPT, plan_file, Path(ActorCritic.__init__.__code__.co_filename)]
    for prefix in (50, 100, 200):
        if prefix > args.through:
            continue
        path = folder / f'policy_update_health_prefix_{prefix:04d}_readonly.json'
        health = read(path)
        assert health['committed_updates'] == prefix and health['native_num_envs'] == 8192
        snapshot = health['snapshots'][-1]
        checkpoint = Path(snapshot['checkpoint_after'])
        assert sha(checkpoint) == snapshot['checkpoint_after_sha256']
        checkpoints.append((prefix, checkpoint))
        inputs.append(path)
    inputs += [p for _, p in checkpoints]
    pins = {str(p): sha(p) for p in inputs}
    torch.set_num_threads(1)
    rng = torch.random.get_rng_state().clone()
    rows = []
    for prefix, checkpoint in checkpoints:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = saved['model_state_dict']
        assert list(state['std'].shape) == [6]
        # Obtain the exact parameter order from the installed ordinary RSL
        # policy instead of guessing a state ID from an ambiguous [6] shape.
        with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
            model = ActorCritic({'policy': torch.zeros(1, 27)},
                {'policy': ['policy'], 'critic': ['policy']}, 6,
                actor_hidden_dims=[256, 128, 64], critic_hidden_dims=[256, 128, 64])
            model.load_state_dict(state, strict=True)
        named = list(model.named_parameters())
        optim = saved['optimizer_state_dict']
        assert len(optim['param_groups']) == 1 and len(optim['state']) == len(named) == 17
        group = optim['param_groups'][0]
        ids = group['params']
        assert len(ids) == 17
        actual = 7001 + prefix
        for (name, parameter), pid in zip(named, ids):
            moment = optim['state'][pid]
            assert float(moment['step']) == actual * 20
            assert moment['exp_avg'].shape == moment['exp_avg_sq'].shape == parameter.shape
            assert torch.isfinite(moment['exp_avg']).all() and torch.isfinite(moment['exp_avg_sq']).all()
        index = [name for name, _ in named].index('std')
        m = optim['state'][ids[index]]
        beta1, beta2 = group['betas']
        assert group['weight_decay'] == 0. and not group.get('amsgrad', False)
        step = float(m['step'])
        first = m['exp_avg'].double() / (1. - beta1**step)
        second = m['exp_avg_sq'].double() / (1. - beta2**step)
        direction = -group['lr'] * first / (second.sqrt() + group['eps'])
        rows.append(dict(prefix=prefix, actual_ppo_updates=actual,
            checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            all17_Adam_step=int(step), saved_actual_adaptive_LR=group['lr'],
            named_parameter_order=[name for name, _ in named],
            std_optimizer_parameter_id=ids[index], saved_six_std=state['std'].tolist(),
            saved_std_at_unchanged_floor=(state['std'] == torch.tensor(.01)).tolist(),
            saved_std_Adam_first_moment=m['exp_avg'].tolist(),
            saved_std_Adam_second_moment=m['exp_avg_sq'].tolist(),
            stored_moment_implied_unprojected_Adam_direction=direction.tolist(),
            stored_moment_direction_would_reduce_std=(direction < 0).tolist()))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert {str(p): sha(p) for p in inputs} == pins
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        through=args.through, head_order=['left_hip_back', 'left_hip_front', 'left_wheel',
                                        'right_hip_back', 'right_hip_front', 'right_wheel'],
        checkpoints=rows, source_pins=pins, all_inspected_files_unchanged=True,
        private_model_initialization_CPU_RNG_restored=True,
        simulator_started=False, optimizer_constructed=False, optimizer_executed=False,
        training_modified=False, std_reset=False, next_parameter_change_selected=False,
        policy_accepted=False,
        limitations=[
            'Adam moments encode recent normalized gradients, not the unknown next gradient '
            'or a causal decomposition of reward terms.',
            'The moment-implied unprojected direction is algebra only. It does not execute '
            'an Adam step, predict postprojection std, or establish a better floor.',
            'A boundary-constrained std is still trainable. A smaller hypothetical future '
            'floor requires a separate qualified trial and complete independent reviews.',
        ])
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(verified=True, receipt=str(destination),
                         summaries=[{k: row[k] for k in ('prefix', 'saved_six_std',
                            'stored_moment_implied_unprojected_Adam_direction')} for row in rows])))


if __name__ == '__main__':
    main()
