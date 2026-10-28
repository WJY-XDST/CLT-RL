"""Verify and archive a completed prefix after a native process interruption.

This is a CPU-only recovery tool. It never repeats PPO, calls optimizer.step,
labels a partial trial as a completed200 block, or accepts a policy.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace, FunctionType

import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
import rsl_rl.runners as runners

import joint_pd_uniform_control_runtime_v3 as runtime
import train_joint_pd_standing_first as boundary
from auto_train_mine import write_json
from finalize_uniform_control_completed200_readonly import declared_contract_validator
from joint_pd_continuation import RUNS, sha
from preserve_uniform_joint_pd_candidate import REQUIRED


def preserve(output, recipe, destination):
    runtime.configure(recipe)
    plan = json.loads((output / 'experiment_plan.json').read_text())
    assert all(sha(path) == digest for path, digest in plan['source_files'].items())
    source = Path(plan['source_checkpoint'])
    runs = list(RUNS.glob('*_' + plan['run_name']))
    assert len(runs) == 1 and not destination.exists()
    checkpoints = list(runs[0].glob('model_*.pt'))
    checkpoint = max(checkpoints, key=runtime.saved_updates)
    actual = runtime.saved_updates(checkpoint)
    completed = actual - plan['source_actual_ppo_updates']
    assert 0 < completed < plan['bounded_training_updates']
    saved, cfg = runtime.require_source(checkpoint)
    records = sorted((output / 'native_learning_signal').glob('audit_update_*.json'))
    assert len(records) == completed
    for index, path in enumerate(records, 1):
        record = json.loads(path.read_text())
        assert record['update'] == index
        assert record['total_environments'] == plan['native_num_envs']
        assert not any(record['all_environment_alignment_max_errors'].values())
        assert record['probe_RNG_unchanged'] and not record['added_loss_or_backward']
        assert len(record['optimizer_steps']) == 20
        assert (path.parent / f'rollout_{index:04d}.npz').is_file()
    validator, admission = declared_contract_validator(plan, runtime)
    original = boundary.verify_update
    verify = FunctionType(original.__code__, dict(original.__globals__, validate_contract=validator),
                          original.__name__, original.__defaults__, original.__closure__)
    proof = verify(source, checkpoint, actual, plan['training_config'], source)
    assert REQUIRED <= {p.name for p in (checkpoint.parent / 'params').iterdir() if p.is_file()}
    destination.mkdir(parents=True)
    archived = destination / checkpoint.name
    shutil.copy2(checkpoint, archived)
    shutil.copytree(checkpoint.parent / 'params', destination / 'params')
    shutil.copy2(recipe, destination / 'recipe.json')
    assert sha(archived) == sha(checkpoint)
    for path in (checkpoint.parent / 'params').rglob('*'):
        if path.is_file():
            assert sha(path) == sha(destination / 'params' / path.relative_to(checkpoint.parent / 'params'))
    copied, copied_cfg = runtime.require_source(archived)
    assert copied_cfg == cfg
    runtime.base.equal_state(saved, copied, 'archived_full_checkpoint')
    agent = yaml.safe_load((destination / 'params/agent.yaml').read_text())
    options = dict(agent['policy'])
    assert options.pop('class_name') == 'ActorCritic' and agent['algorithm']['class_name'] == 'PPO'
    observations = TensorDict({'policy': torch.zeros(4, 27)}, batch_size=[4])
    restore = destination / 'CPU_restore_verification'
    restore.mkdir()
    rng = torch.get_rng_state().clone()
    upstream, cls = runtime.private_for_count(cfg['scene']['num_envs']).install_runner(restore)
    try:
        with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(observations, agent['obs_groups'], 6, **options)
            runner = object.__new__(cls)
            runner.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
            runner.current_learning_iteration = 0
            runner.logger_type, runner.disable_logs = 'tensorboard', True
            runner.load(str(archived), load_optimizer=True, map_location='cpu')
        runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'loaded_NN_and_six_std')
        runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'loaded17_Adam')
        assert runner.alg.learning_rate == saved['optimizer_state_dict']['param_groups'][0]['lr']
        assert all(p.requires_grad for p in runner.alg.policy.parameters())
    finally:
        runners.OnPolicyRunner = upstream
    assert torch.equal(rng, torch.get_rng_state())
    proof.update(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        source_checkpoint_sha256=sha(source), completed_native_updates=completed,
        partial_trial_not_completed200=True, all_completed_native_alignment_verified=True,
        all17_Adam_step=actual * 20, saved_std=saved['model_state_dict']['std'].tolist(),
        saved_actual_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        archived_checkpoint=str(archived), physical_control_contract_unchanged=False,
        only_declared_global_wheel_damping_changed=True, declared_contract_admission=admission,
        actual_CPU_runner_load_optimizer_true_full_state_bitwise_verified=True,
        optimizer_step_or_backward_executed=False, simulator_started=False,
        policy_accepted=False, original_target_not_reached=True,
        source_pins={str(p): sha(p) for p in [Path(__file__), recipe, checkpoint,
                     output / 'experiment_plan.json', *records]})
    write_json(destination / 'preservation_manifest.json', proof)
    write_json(output / 'partial_checkpoint_verification_readonly.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = preserve(args.output.resolve(), args.recipe.resolve(), args.archive.resolve())
    print(json.dumps({k: result[k] for k in ('verified', 'actual_ppo_updates',
          'completed_native_updates', 'checkpoint_sha256', 'all17_Adam_step', 'saved_actual_LR')}))
