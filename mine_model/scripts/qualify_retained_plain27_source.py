"""CPU qualification of a saved plain27 continuation, without changing its recipe."""
import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import torch
import yaml
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
import rsl_rl.runners as runners

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, optimizer_step
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()


def qualify(source, runtime_path, decision_path, output):
    source, runtime_path, decision_path, output = [Path(p).resolve() for p in
        (source, runtime_path, decision_path, output)]
    decision = json.loads(decision_path.read_text())
    empty = dict(rewards={}, commands={}, exploration={}, controller={}, physics={})
    if (runtime_path.parent != ROOT / 'mine_model/experiments'
            or decision['source_checkpoint_sha256'] != sha(source)
            or decision['next_configuration_changes'] != empty
            or decision['planned_actual_updates'] != 200
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require an intact, source-bound, unchanged-recipe200 decision')
    spec = importlib.util.spec_from_file_location('_qualified_retained_plain27_runtime', runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    saved, cfg = runtime.require_source(source)
    actual = saved_updates(source)
    profile = runtime.profile()
    if (actual != decision['source_actual_updates'] or actual <= 201
            or optimizer_step(saved) != 20 * actual
            or len(saved['optimizer_state_dict']['state']) != 17
            or cfg['leg_control_mode'] != 'implicit_joint_position'
            or cfg['observation_space'] != 27 or cfg['scene']['num_envs'] != 4096):
        raise ValueError('Require the current plain27 direct-joint state and all17 Adam clocks')
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
        result = object.__new__(cls)
        result.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
        result.current_learning_iteration = 0
        result.logger_type, result.disable_logs = 'tensorboard', True
        return result

    output.mkdir(exist_ok=False)
    initial = output / 'initial_load'
    initial.mkdir()
    upstream, cls = runtime.private_runtime().install_runner(initial)
    try:
        runner = bare(cls)
        runner.load(str(source), load_optimizer=True, map_location='cpu')
        runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'initial_model')
        runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'initial_all17_Adam')
        prediction = runner.alg.policy.act_inference(obs).detach().clone()
        original = runtime.install_leg_exploration(initial)
        import wheel_legged_gym_isaaclab.exploration as exploration
        handle = exploration.enforce_positive_exploration(runner.alg.policy, runner.alg.optimizer,
            profile['global_minimum_std'], minimum_wheel_std=profile['minimum_wheel_std'])
        try:
            runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'after_guard_model')
            runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'after_guard_all17')
            fixture = torch.tensor([-.02, .02, .015, .025, .04, .01])
            with torch.no_grad():
                runner.alg.policy.std.copy_(fixture)
            for hook in runner.alg.optimizer._optimizer_step_post_hooks.values():
                hook(runner.alg.optimizer, (), {})
            expected = fixture.clamp_min(profile['global_minimum_std'])
            expected[[0, 1, 3, 4]] = expected[[0, 1, 3, 4]].clamp_min(profile['minimum_leg_std'])
            expected[[2, 5]] = expected[[2, 5]].clamp_min(profile['minimum_wheel_std'])
            if not torch.equal(runner.alg.policy.std, expected):
                raise ValueError('The exact registered per-head projections differ')
            runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'fixture_Adam')
            runner.alg.policy.load_state_dict(saved['model_state_dict'], strict=True)
        finally:
            handle.remove()
            exploration.enforce_positive_exploration = original
        roundtrip = output / 'untrained_profile_roundtrip'
        roundtrip.mkdir()
        shutil.copytree(source.parent / 'params', roundtrip / 'params')
        offline = roundtrip / 'offline_untrained_retained_source.pt'
        runner.save(str(offline))
        runners.OnPolicyRunner = upstream
        restored_dir = output / 'roundtrip_load'
        restored_dir.mkdir()
        upstream2, cls2 = runtime.private_runtime().install_runner(restored_dir)
        try:
            restored = bare(cls2)
            restored.load(str(offline), load_optimizer=True, map_location='cpu')
            runtime.base.equal_state(saved['model_state_dict'], restored.alg.policy.state_dict(), 'roundtrip_model')
            runtime.base.equal_state(saved['optimizer_state_dict'], restored.alg.optimizer.state_dict(), 'roundtrip_all17')
            if not torch.equal(prediction, restored.alg.policy.act_inference(obs)):
                raise ValueError('Untrained save/reload changed inference')
        finally:
            runners.OnPolicyRunner = upstream2
        argv, final, _ = runtime.command(source, cfg, 'offline_retained_plain27_probe', 200,
            cfg['commands'], output / 'not_a_native_stop_request')
        if (final != actual + 199 or argv[argv.index('--target_iteration') + 1] != str(actual + 200)
                or argv[argv.index('--num_envs') + 1] != '4096'
                or float(argv[argv.index('--min_exploration_std') + 1]) != profile['global_minimum_std']
                or float(argv[argv.index('--min_wheel_exploration_std') + 1]) != profile['minimum_wheel_std']
                or argv.count('agent.policy.class_name=ActorCritic') != 1
                or any(arg.startswith('--repair_') for arg in argv)):
            raise ValueError('Changed200-update/no-transfer/plain27/exploration command')
        pins = {str(p): sha(p) for p in (SCRIPT, runtime_path, decision_path, source,
            source.parent / 'params/agent.yaml', source.parent / 'params/env.yaml')}
        result = dict(qualified=True, source_checkpoint=str(source), source_sha256=sha(source),
            source_actual_updates=actual, source_all17_Adam_step=optimizer_step(saved),
            profile=profile, implementation_sha256=sha(runtime_path), qualifier_sha256=sha(SCRIPT),
            decision_sha256=sha(decision_path), all_model_and_Adam_tensors_bitwise_preserved=True,
            initial_saved_std_bitwise_preserved=True, actual_saved_adaptive_LR_preserved=True,
            exact_registered_projection_test_passed=True, projection_fixture=fixture.tolist(),
            expected_projected_fixture=expected.tolist(), untrained_metadata_roundtrip_inference_bitwise_equal=True,
            untrained_roundtrip_checkpoint=str(offline), saved_env_and_agent_configurations_unchanged=True,
            next_configuration_changes=empty, continuation_command=argv, source_files=pins,
            actual_PPO_updates_executed=0, optimizer_steps_executed=0, simulator_started=False,
            policy_accepted=False)
        write_json(output / 'qualification.json', result)
        return result
    finally:
        runners.OnPolicyRunner = upstream


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'runtime', 'decision', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.source, args.runtime, args.decision, args.output)
    print(json.dumps(dict(qualified=result['qualified'], actual_PPO_updates=0,
        optimizer_steps=0, simulator_started=False)))
