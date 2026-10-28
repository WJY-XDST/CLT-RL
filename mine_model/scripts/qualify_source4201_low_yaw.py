"""Retained CPU runner and full27 frozen qualification of one command-field edit."""
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
import source4201_low_yaw_plan as plan

SCRIPT = Path(__file__).resolve()
runtime = plan.runtime


def qualify(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    decision = plan.ready()
    saved, cfg = plan.require_source(source, transition=True)
    expected = copy.deepcopy(cfg)
    expected['commands']['yaw_min_abs_rate'] = 0.
    validate_continuation_config(cfg, expected, expected['commands'])
    if (differences(cfg, expected) != ['/commands/yaw_min_abs_rate']
            or saved_updates(source) != 4201 or optimizer_step(saved) != 84020
            or len(saved['optimizer_state_dict']['state']) != 17):
        raise ValueError('Require all17 continuous4201 states and only one command field')
    agent = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    options = dict(agent['policy'])
    if (options.pop('class_name') != 'ActorCritic' or agent['algorithm']['class_name'] != 'PPO'
            or agent['algorithm'].get('joint_height_feedback_coef', 0.)
            or agent['algorithm'].get('joint_mean_guard_coef', 0.)):
        raise ValueError('Keep ordinary full-policy PPO and the original27-input ActorCritic')
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
    upstream, cls = runtime.private_runtime().install_runner(first)
    try:
        runner = bare(cls)
        runner.load(str(source), load_optimizer=True, map_location='cpu')
        runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'loaded_model')
        runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'loaded_all17_Adam')
        prediction = runner.alg.policy.act_inference(obs).detach().clone()
        original = runtime.install_leg_exploration(first)
        import wheel_legged_gym_isaaclab.exploration as exploration
        handle = exploration.enforce_positive_exploration(runner.alg.policy, runner.alg.optimizer,
            runtime.profile()['global_minimum_std'], minimum_wheel_std=.03)
        try:
            runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'guard_model')
            runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'guard_all17_Adam')
            fixture = torch.tensor([-.02, .02, .015, .025, .04, .01])
            with torch.no_grad():
                runner.alg.policy.std.copy_(fixture)
            for hook in runner.alg.optimizer._optimizer_step_post_hooks.values():
                hook(runner.alg.optimizer, (), {})
            projected = fixture.clamp_min(.02)
            projected[[2, 5]] = projected[[2, 5]].clamp_min(.03)
            if not torch.equal(projected, runner.alg.policy.std):
                raise ValueError('Unchanged per-head projection differs')
            runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'fixture_Adam')
            runner.alg.policy.load_state_dict(saved['model_state_dict'], strict=True)
        finally:
            handle.remove()
            exploration.enforce_positive_exploration = original
        folder = output / 'untrained_profile_roundtrip'
        folder.mkdir()
        shutil.copytree(source.parent / 'params', folder / 'params')
        (folder / 'params/env.yaml').write_text(yaml.safe_dump(expected, sort_keys=False))
        checkpoint = folder / 'offline_untrained_low_yaw_metadata.pt'
        runner.save(str(checkpoint))
        runners.OnPolicyRunner = upstream
        second = output / 'roundtrip_load'
        second.mkdir()
        upstream2, cls2 = runtime.private_runtime().install_runner(second)
        try:
            restored = bare(cls2)
            restored.load(str(checkpoint), load_optimizer=True, map_location='cpu')
            runtime.base.equal_state(saved['model_state_dict'], restored.alg.policy.state_dict(), 'roundtrip_model')
            runtime.base.equal_state(saved['optimizer_state_dict'], restored.alg.optimizer.state_dict(), 'roundtrip_all17_Adam')
            if not torch.equal(prediction, restored.alg.policy.act_inference(obs)):
                raise ValueError('Untrained metadata roundtrip changed inference')
        finally:
            runners.OnPolicyRunner = upstream2
        try:
            runtime.require_source(runtime.base.SOURCE)
        except ValueError:
            pass
        else:
            raise ValueError('Original model200 was allowed as a continuation source')
        tail = ('offline_low_yaw', 200, cfg['commands'], output / 'not_a_native_stop_request')
        before, old_final, _ = runtime.command(source, cfg, *tail)
        after, final, _ = runtime.command(source, cfg, tail[0], 200, expected['commands'], tail[3])
        predicted = list(before)
        predicted[predicted.index('env.commands.yaw_min_abs_rate=0.15')] = 'env.commands.yaw_min_abs_rate=0.0'
        if (after != predicted or final != old_final or final != 4400
                or after[after.index('--target_iteration') + 1] != '4401'
                or after[after.index('--num_envs') + 1] != '4096'
                or runtime.FLAG in after or any(arg.startswith('--repair_') for arg in after)):
            raise ValueError('Only yaw minimum may change; no transfer flag or reset')
        pins = {str(p): sha(p) for p in (SCRIPT, plan.SCRIPT, runtime.SCRIPT, plan.DECISION, source,
            source.parent / 'params/agent.yaml', source.parent / 'params/env.yaml',
            checkpoint, folder / 'params/agent.yaml', folder / 'params/env.yaml')}
        result = dict(qualified=True, source_checkpoint=str(source), source_sha256=sha(source),
            source_actual_updates=4201, source_all17_Adam_step=84020, profile=runtime.profile(),
            implementation_sha256=sha(runtime.SCRIPT), qualifier_sha256=sha(SCRIPT),
            decision_sha256=sha(plan.DECISION), all_model_and_Adam_tensors_bitwise_preserved=True,
            initial_saved_std_bitwise_preserved=True, actual_saved_adaptive_LR_preserved=True,
            exact_registered_projection_test_passed=True, projection_fixture=fixture.tolist(),
            expected_projected_fixture=projected.tolist(), untrained_metadata_roundtrip_inference_bitwise_equal=True,
            untrained_roundtrip_checkpoint=str(checkpoint), untrained_roundtrip_checkpoint_sha256=sha(checkpoint),
            saved_env_diff=['/commands/yaw_min_abs_rate'], saved_agent_configuration_unchanged=True,
            original200_rejected=True, existing_profile_transfer_flag_absent=True,
            next_configuration_changes=plan.DELTA, continuation_command=after, source_files=pins,
            actual_PPO_updates_executed=0, optimizer_steps_executed=0, simulator_started=False,
            old_runtime_family_initial_delta_is_historical=True, policy_accepted=False)
        write_json(output / 'qualification.json', result)
        return result
    finally:
        runners.OnPolicyRunner = upstream


def frozen(previous, output):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    cpu = json.loads((output / 'qualification.json').read_text())
    before, after = previous / 'replay', output / 'native_command_transfer'
    read = lambda root, name: json.loads((root / name).read_text())
    a, b = read(before, 'manifest.json'), read(after, 'manifest.json')
    ca, cb = read(before, 'completed.json'), read(after, 'completed.json')
    if (a['checkpoint_sha256'] != plan.ANCHOR_SHA
            or b['checkpoint_sha256'] != cpu['untrained_roundtrip_checkpoint_sha256']
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not ca['completed'] or not cb['completed'] or ca['samples'] != 27000
            or cb['samples'] != 27000 or ca['resets'] or cb['resets']):
        raise ValueError('Require all27 original fixed-command trajectories and both exact hashes')
    if differences(read(before, 'resolved_env.json'), read(after, 'resolved_env.json')) != ['/commands/yaw_min_abs_rate']:
        raise ValueError('Frozen configuration differs outside yaw minimum')
    x, y = [np.genfromtxt(root / 'trace.csv', delimiter=',', names=True) for root in (before, after)]
    if x.dtype.names != y.dtype.names or len(y) != 27000 or any(not np.array_equal(x[n], y[n]) for n in x.dtype.names):
        raise ValueError('Frozen physics/actions/rewards changed')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as x, np.load(after / 'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[n], y[n]) for n in x.files):
            raise ValueError('Frozen observations or actor means changed')
    pins = {str(root / name): sha(root / name) for root in (before, after) for name in
        ('manifest.json', 'completed.json', 'trace.csv', 'resolved_env.json', 'policy_inputs.npz')}
    pins.update(cpu['source_files'])
    pins[str(output / 'qualification.json')] = sha(output / 'qualification.json')
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Immutable qualification evidence changed')
    result = dict(qualified=True, source_sha256=plan.ANCHOR_SHA,
        untrained_metadata_checkpoint_sha256=cpu['untrained_roundtrip_checkpoint_sha256'],
        hashes_differ_because_untrained_metadata_save=True, all_model_and_Adam_tensors_bitwise_preserved=True,
        implementation_sha256=sha(runtime.SCRIPT), qualifier_sha256=sha(SCRIPT),
        only_command_change=plan.DELTA['commands'], full27_samples=27000, source_resets_preserved=0,
        physics_actions_observations_means_and_all_rewards_bitwise_equal=True,
        source_data_sha256=pins, PPO_updates_performed=0, policy_accepted=False)
    write_json(output / 'native_command_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--frozen-previous', type=Path)
    args = cli.parse_args()
    result = frozen(args.frozen_previous, args.output) if args.frozen_previous else qualify(args.source, args.output)
    print(json.dumps(dict(qualified=result['qualified'], policy_accepted=False)))
