"""Qualify a sampler-only expansion with actual retained runner load/save; no PPO."""
import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
import rsl_rl.runners as runners

from auto_train_mine import write_json
from joint_pd_continuation import sha, optimizer_step, validate_continuation_config
from qualify_source801_drift50_transfer import differences
from train_joint_pd_standing_seed import saved_updates
import joint_pd_source3601_motion0375_runtime as runtime

SCRIPT = Path(__file__).resolve()


def qualify(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    decision = runtime.repair_ready()
    saved, cfg = runtime.require_source(source, transition=True)
    expected = copy.deepcopy(cfg)
    expected['commands']['training_motion_scale'] = runtime.MOTION_SCALE
    validate_continuation_config(cfg, expected, expected['commands'])
    if (differences(cfg, expected) != ['/commands/training_motion_scale']
            or saved_updates(source) != 3601 or optimizer_step(saved) != 72020):
        raise ValueError('Require exact source3601 and one sampler field')
    agent = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    options = dict(agent['policy'])
    if (options.pop('class_name') != 'ActorCritic' or agent['algorithm']['class_name'] != 'PPO'
            or agent['algorithm'].get('joint_height_feedback_coef', 0.)
            or agent['algorithm'].get('joint_mean_guard_coef', 0.)):
        raise ValueError('Keep the original ordinary PPO ActorCritic')
    obs = TensorDict({'policy': torch.linspace(-.2, .2, 32 * 27).reshape(32, 27)}, batch_size=[32])

    def bare(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(obs, agent['obs_groups'], 6, **options)
        runner = object.__new__(cls)
        runner.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
        runner.current_learning_iteration = 0
        runner.logger_type, runner.disable_logs = 'tensorboard', True
        return runner

    output.mkdir(exist_ok=False)
    first = output / 'initial_load'
    first.mkdir()
    upstream, cls = runtime.private_runtime(source).install_runner(first)
    try:
        runner = bare(cls)
        runner.load(str(source), load_optimizer=True, map_location='cpu')
        runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'initial_model')
        runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'initial_all17_Adam')
        prediction = runner.alg.policy.act_inference(obs).detach().clone()
        original = runtime.install_leg_exploration(first)
        import wheel_legged_gym_isaaclab.exploration as exploration
        handle = exploration.enforce_positive_exploration(runner.alg.policy, runner.alg.optimizer,
            runtime.profile()['global_minimum_std'], minimum_wheel_std=.03)
        try:
            runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'after_guard_model')
            runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'after_guard_all17')
            fixture = torch.tensor([-.02, .02, .015, .025, .04, .01])
            with torch.no_grad():
                runner.alg.policy.std.copy_(fixture)
            for hook in runner.alg.optimizer._optimizer_step_post_hooks.values():
                hook(runner.alg.optimizer, (), {})
            projected = fixture.clamp_min(.02)
            projected[[2, 5]] = projected[[2, 5]].clamp_min(.03)
            if not torch.equal(projected, runner.alg.policy.std):
                raise ValueError('Registered unchanged per-head projection differs')
            runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'fixture_Adam')
            runner.alg.policy.load_state_dict(saved['model_state_dict'], strict=True)
        finally:
            handle.remove()
            exploration.enforce_positive_exploration = original
        folder = output / 'untrained_profile_roundtrip'
        folder.mkdir()
        shutil.copytree(source.parent / 'params', folder / 'params')
        (folder / 'params/env.yaml').write_text(yaml.safe_dump(expected, sort_keys=False))
        checkpoint = folder / 'offline_untrained_motion0375.pt'
        runner.save(str(checkpoint))
        runners.OnPolicyRunner = upstream
        second = output / 'roundtrip_load'
        second.mkdir()
        upstream2, cls2 = runtime.private_runtime().install_runner(second)
        try:
            restored = bare(cls2)
            restored.load(str(checkpoint), load_optimizer=True, map_location='cpu')
            runtime.base.equal_state(saved['model_state_dict'], restored.alg.policy.state_dict(), 'roundtrip_model')
            runtime.base.equal_state(saved['optimizer_state_dict'], restored.alg.optimizer.state_dict(), 'roundtrip_all17')
            if not torch.equal(prediction, restored.alg.policy.act_inference(obs)):
                raise ValueError('Metadata roundtrip changed inference')
        finally:
            runners.OnPolicyRunner = upstream2
        for rejected, transition in ((source, False), (runtime.base.SOURCE, True)):
            try:
                runtime.require_source(rejected, transition=transition)
            except ValueError:
                pass
            else:
                raise ValueError('Undeclared profile transfer or original200 reload was allowed')
        tail = ('offline_motion0375', 200, cfg['commands'], output / 'not_a_native_stop_request')
        before, old_final, _ = runtime.previous.command(source, cfg, *tail)
        after, final, _ = runtime.command(source, cfg, tail[0], 200, expected['commands'], tail[3])
        predicted = list(before)
        predicted[predicted.index(str(runtime.previous.SCRIPT))] = str(runtime.SCRIPT)
        predicted[predicted.index('env.commands.training_motion_scale=0.25')] = 'env.commands.training_motion_scale=0.375'
        predicted.append(runtime.FLAG)
        if (after != predicted or old_final != final or final != 3800
                or after[after.index('--target_iteration') + 1] != '3801'
                or after[after.index('--num_envs') + 1] != '4096'):
            raise ValueError('Command changed outside entry, transfer flag, and the single sampler field')
        pins = {str(p): sha(p) for p in (SCRIPT, runtime.SCRIPT, runtime.DECISION, source,
            source.parent / 'params/agent.yaml', source.parent / 'params/env.yaml')}
        write_json(output / 'motion_metadata_override.json', {'commands.training_motion_scale': runtime.MOTION_SCALE})
        result = dict(qualified=True, source_checkpoint=str(source), source_sha256=sha(source),
            source_actual_updates=3601, source_all17_Adam_step=72020,
            profile=runtime.profile(), implementation_sha256=sha(runtime.SCRIPT), qualifier_sha256=sha(SCRIPT),
            decision_sha256=sha(runtime.DECISION), all_model_and_Adam_tensors_bitwise_preserved=True,
            initial_saved_std_bitwise_preserved=True, actual_saved_adaptive_LR_preserved=True,
            exact_registered_projection_test_passed=True, projection_fixture=fixture.tolist(),
            expected_projected_fixture=projected.tolist(), untrained_metadata_roundtrip_inference_bitwise_equal=True,
            untrained_roundtrip_checkpoint=str(checkpoint), saved_env_diff=['/commands/training_motion_scale'],
            saved_agent_configuration_unchanged=True, original200_and_undeclared_transfer_rejected=True,
            next_configuration_changes=runtime.DELTA, continuation_command=after, source_files=pins,
            actual_PPO_updates_executed=0, optimizer_steps_executed=0, simulator_started=False,
            policy_accepted=False)
        write_json(output / 'qualification.json', result)
        return result
    finally:
        runners.OnPolicyRunner = upstream


def frozen(previous, output):
    before, after = Path(previous) / 'replay', Path(output) / 'native_command_transfer'
    read = lambda root, name: json.loads((root / name).read_text())
    a, b = read(before, 'manifest.json'), read(after, 'manifest.json')
    ca, cb = read(before, 'completed.json'), read(after, 'completed.json')
    if (a['checkpoint_sha256'] != runtime.ANCHOR_SHA or b['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not ca['completed'] or not cb['completed'] or ca['samples'] != 27000
            or cb['samples'] != 27000 or ca['resets'] != 15 or cb['resets'] != ca['resets']):
        raise ValueError('Require the original complete27 replay including all source15 resets')
    if differences(read(before, 'resolved_env.json'), read(after, 'resolved_env.json')) != ['/commands/training_motion_scale']:
        raise ValueError('Frozen native configuration changed outside the sampler field')
    x, y = [np.genfromtxt(root / 'trace.csv', delimiter=',', names=True) for root in (before, after)]
    if (x.dtype.names != y.dtype.names or len(y) != 27000
            or any(not np.array_equal(x[n], y[n]) for n in x.dtype.names)):
        raise ValueError('Frozen physics, actions, references or rewards changed')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as x, np.load(after / 'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[n], y[n]) for n in x.files):
            raise ValueError('Frozen observations or actor means changed')
    result = dict(qualified=True, source_sha256=runtime.ANCHOR_SHA,
        implementation_sha256=sha(runtime.SCRIPT), qualifier_sha256=sha(SCRIPT),
        only_command_change=runtime.DELTA['commands'], full27_samples=27000, source_resets_preserved=15,
        physics_actions_observations_means_and_all_rewards_bitwise_equal=True,
        source_data_sha256={str(root / n): sha(root / n) for root in (before, after) for n in
            ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz')},
        PPO_updates_performed=0, policy_accepted=False,
        limitation='Fixed-command transfer equivalence only; source failures remain failures.')
    write_json(Path(output) / 'native_command_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-previous', type=Path)
    args = parser.parse_args()
    result = frozen(args.frozen_previous, args.output) if args.frozen_previous else qualify(args.source, args.output)
    print(json.dumps(dict(qualified=result['qualified'], policy_accepted=False)))
