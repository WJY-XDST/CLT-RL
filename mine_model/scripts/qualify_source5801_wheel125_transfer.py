"""Qualify the real retained-state wheel-clamp entry without PPO or simulation."""
import argparse
import contextlib
import copy
from datetime import datetime
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

from joint_pd_continuation import ROOT, sha, optimizer_step
import joint_pd_source5801_wheel125_runtime as runtime
from wheel_legged_gym_isaaclab.wheel_control import apply_wheel_action_limit
from wheel_legged_gym_isaaclab.action_bias_probe import wheel_reference_probe

SCRIPT = Path(__file__).resolve()
OUTPUT = ROOT / 'mine_model/results/joint_pd_source5801_wheel125_transfer_qualification_v2_20261020'


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def qualify():
    decision = runtime.repair_ready()
    source, saved, cfg = runtime.anchor()
    if OUTPUT.exists():
        raise ValueError('Preserve every prior CPU qualification and untrained roundtrip')
    agent = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    options = dict(agent['policy'])
    if (options.pop('class_name') != 'ActorCritic' or agent['algorithm']['class_name'] != 'PPO'
            or agent['clip_actions'] is not None
            or agent['algorithm'].get('joint_height_feedback_coef', 0.)
            or agent['algorithm'].get('joint_mean_guard_coef', 0.)):
        raise ValueError('Keep ordinary PPO and direct unclipped raw Gaussian policy outputs')
    obs = TensorDict({'policy': torch.linspace(-.2, .2, 32 * 27).reshape(32, 27)}, batch_size=[32])

    def bare(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(obs, agent['obs_groups'], 6, **options)
        result = object.__new__(cls)
        result.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
        result.current_learning_iteration = 0
        result.logger_type, result.disable_logs = 'tensorboard', True
        return result

    OUTPUT.mkdir()
    initial = OUTPUT / 'initial_load'
    initial.mkdir()
    upstream, cls = runtime.private_runtime(source).install_runner(initial)
    try:
        runner = bare(cls)
        runner.load(str(source), load_optimizer=True, map_location='cpu')
        runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'initial_model')
        runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'initial_all17_Adam')
        prediction = runner.alg.policy.act_inference(obs).detach().clone()
        original = runtime.install_leg_exploration(initial)
        import wheel_legged_gym_isaaclab.exploration as exploration
        profile = runtime.profile()
        handle = exploration.enforce_positive_exploration(runner.alg.policy, runner.alg.optimizer,
            profile['global_minimum_std'], minimum_wheel_std=profile['minimum_wheel_std'])
        try:
            runtime.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'after_guard_model')
            runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'after_guard_all17_Adam')
            fixture = torch.tensor([-.02, .02, .015, .025, .04, .01])
            with torch.no_grad():
                runner.alg.policy.std.copy_(fixture)
            for hook in runner.alg.optimizer._optimizer_step_post_hooks.values():
                hook(runner.alg.optimizer, (), {})
            expected_std = fixture.clamp_min(profile['global_minimum_std'])
            expected_std[[0, 1, 3, 4]] = expected_std[[0, 1, 3, 4]].clamp_min(profile['minimum_leg_std'])
            expected_std[[2, 5]] = expected_std[[2, 5]].clamp_min(profile['minimum_wheel_std'])
            if not torch.equal(runner.alg.policy.std, expected_std):
                raise ValueError('Original registered per-head std projections differ')
            runtime.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'fixture_Adam')
            runner.alg.policy.load_state_dict(saved['model_state_dict'], strict=True)
        finally:
            handle.remove()
            exploration.enforce_positive_exploration = original
        roundtrip = OUTPUT / 'untrained_controller_roundtrip'
        roundtrip.mkdir()
        shutil.copytree(source.parent / 'params', roundtrip / 'params')
        env_path = roundtrip / 'params/env.yaml'
        new_cfg = copy.deepcopy(cfg)
        new_cfg['wheel_action_limit'] = 1.25
        env_path.write_text(yaml.safe_dump(new_cfg, allow_unicode=True, sort_keys=False))
        offline = roundtrip / 'offline_untrained_retained_source.pt'
        runner.save(str(offline))
        runners.OnPolicyRunner = upstream
        restored_dir = OUTPUT / 'roundtrip_load'
        restored_dir.mkdir()
        upstream2, cls2 = runtime.private_runtime().install_runner(restored_dir)
        try:
            restored = bare(cls2)
            restored.load(str(offline), load_optimizer=True, map_location='cpu')
            runtime.base.equal_state(saved['model_state_dict'], restored.alg.policy.state_dict(), 'roundtrip_model')
            runtime.base.equal_state(saved['optimizer_state_dict'], restored.alg.optimizer.state_dict(), 'roundtrip_all17_Adam')
            if not torch.equal(prediction, restored.alg.policy.act_inference(obs)):
                raise ValueError('Saving the controller contract changed policy inference')
        finally:
            runners.OnPolicyRunner = upstream2
    finally:
        runners.OnPolicyRunner = upstream
    # Check the exact normal helper on every recorded mean from the324-case
    # native intervention. Unit-clamp behavior and four leg columns are kept.
    rng = torch.random.get_rng_state().clone()
    mapping_rows = 0
    for level in ('low', 'nominal', 'high'):
        directory = ROOT / ('mine_model/results/joint_pd_source5801_combined_headroom125_diagnostic_20261020/'
            + level + '/candidate')
        with np.load(directory / 'policy_inputs.npz') as z:
            raw = torch.from_numpy(z['means'].copy()).reshape(-1, 6)
        applied = raw.clamp(-1., 1.)
        normal = apply_wheel_action_limit(applied, raw, 1.25)
        if (not torch.equal(normal, wheel_reference_probe(applied, raw, 1.25))
                or apply_wheel_action_limit(applied, raw, 1.) is not applied
                or not torch.equal(normal[:, (0, 1, 3, 4)], applied[:, (0, 1, 3, 4)])):
            raise ValueError('Normal control differs from the native counterfactual or historical leg map')
        mapping_rows += len(raw)
    if mapping_rows != 324000 or not torch.equal(rng, torch.random.get_rng_state()):
        raise ValueError('Keep every recorded mean and add no RNG draw')
    argv, final, _ = runtime.command(source, cfg, 'offline_wheel125_entry_preflight', 200,
        cfg['commands'], OUTPUT / 'not_a_native_stop_request',
        reward_overrides={'rewards.tracking_ang_vel': cfg['rewards']['tracking_ang_vel']},
        minimum_wheel_std=.03, joint_mean_guard_coef=0., joint_height_feedback_coef=0.)
    if (final != 6000 or argv[argv.index('--target_iteration') + 1] != '6001'
            or argv[argv.index('--num_envs') + 1] != '4096'
            or argv.count('env.wheel_action_limit=1.25') != 1 or argv.count(runtime.FLAG) != 1
            or argv.count('agent.policy.class_name=ActorCritic') != 1
            or any(f in argv for f in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint'))):
        raise ValueError('Original full-state200 entry differs beyond the declared wheel cap')
    pins = dict(decision['source_pins'])
    for path in (SCRIPT, runtime.SCRIPT, runtime.DECISION, source,
            source.parent / 'params/env.yaml', source.parent / 'params/agent.yaml', offline, env_path,
            ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/wheel_control.py'):
        pins[str(path)] = sha(path)
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('Qualified current implementation or archived historical evidence changed')
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_updates=5801,
        source_all17_Adam_step=116020, target_actual_updates=6001, target_all17_Adam_step=120020,
        profile=profile, implementation_sha256=sha(runtime.SCRIPT), qualifier_sha256=sha(SCRIPT),
        decision_sha256=sha(runtime.DECISION), next_configuration_changes=runtime.DELTA,
        all_model_and_Adam_tensors_bitwise_preserved=True, initial_saved_std_bitwise_preserved=True,
        actual_saved_adaptive_LR_preserved=True, exact_registered_projection_test_passed=True,
        untrained_metadata_roundtrip_inference_bitwise_equal=True,
        untrained_controller_fixture=str(offline), untrained_fixture_not_a_PPO_checkpoint=True,
        all324000_recorded_means_match_diagnostic_bitwise=True, historical_unit_clamp_and_leg_columns_preserved=True,
        normal_helper_RNG_unchanged=True, actual_entry_command=argv,
        source_std=saved['model_state_dict']['std'].tolist(),
        source_adaptive_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_files=pins, actual_PPO_updates_executed=0, optimizer_steps_executed=0,
        simulator_started=False, policy_accepted=False)
    save(OUTPUT / 'qualification.json', result)
    return result


if __name__ == '__main__':
    result = qualify()
    print(json.dumps({k: result[k] for k in ('qualified', 'source_actual_updates', 'target_actual_updates',
        'all_model_and_Adam_tensors_bitwise_preserved', 'all324000_recorded_means_match_diagnostic_bitwise')}))
