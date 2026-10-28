"""Qualify the retained full PPO state and one orientation reward delta; no PPO."""
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
from qualify_source1001_reference_smooth10 import event_observer_check
import source1201_orientation40 as repair
import wheel_legged_gym_isaaclab.exploration as exploration


def qualify(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.mkdir(exist_ok=False)
    saved, cfg = repair.require_source(source, transition=True)
    agent = yaml.safe_load((source.parent/'params/agent.yaml').read_text())
    options = copy.deepcopy(agent['policy'])
    if options.pop('class_name') != 'ActorCritic':
        raise ValueError('Keep plain27 ActorCritic')
    observation = TensorDict({'policy':torch.zeros(32,27)}, batch_size=[32])
    observation['policy'][:,5] = -1.
    observation['policy'][:,8] = 1.5

    def bare(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(observation, agent['obs_groups'], 6, **options)
        runner = object.__new__(cls)
        runner.alg = SimpleNamespace(policy=policy, optimizer=torch.optim.Adam(policy.parameters()), rnd=None)
        runner.current_learning_iteration = 0
        runner.logger_type, runner.disable_logs = 'tensorboard', True
        return runner

    first = output/'first_load'
    first.mkdir()
    upstream, cls = repair.private_runtime(source).install_runner(first)
    original_enforce = exploration.enforce_positive_exploration
    try:
        runner = bare(cls)
        runner.load(str(source), load_optimizer=True, map_location='cpu')
        repair.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'loaded_model')
        repair.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'loaded_Adam')
        with torch.no_grad():
            before = runner.alg.policy.act_inference(observation).clone()
        repair.install_leg_exploration(first)
        handle = exploration.enforce_positive_exploration(runner.alg.policy, runner.alg.optimizer,
            .02, minimum_wheel_std=.03)
        repair.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'unchanged_std')
        with torch.no_grad():
            runner.alg.policy.std[[0,1,3,4]] = .001
        runner.alg.optimizer.zero_grad(set_to_none=True)
        runner.alg.optimizer.step()
        repair.base.equal_state(saved['model_state_dict'], runner.alg.policy.state_dict(), 'persistent_floor')
        repair.base.equal_state(saved['optimizer_state_dict'], runner.alg.optimizer.state_dict(), 'no_gradient_Adam')
        handle.remove()
        expected = copy.deepcopy(cfg)
        expected['rewards']['orientation'] = repair.ORIENTATION
        validate_continuation_config(cfg, expected, cfg['commands'],
            reward_overrides={'rewards.orientation':repair.ORIENTATION})
        if differences(cfg, expected) != ['/rewards/orientation']:
            raise ValueError('Only the existing orientation coefficient may change')
        roundtrip = output/'roundtrip'
        roundtrip.mkdir()
        shutil.copytree(source.parent/'params', roundtrip/'params')
        (roundtrip/'params/env.yaml').write_text(yaml.safe_dump(expected, sort_keys=False))
        offline = roundtrip/'offline_source1201_orientation40_roundtrip.pt'
        runner.save(str(offline))
        runners.OnPolicyRunner = upstream
        second = output/'second_load'
        second.mkdir()
        upstream2, cls2 = repair.private_runtime().install_runner(second)
        try:
            restored = bare(cls2)
            restored.load(str(offline), load_optimizer=True, map_location='cpu')
            repair.base.equal_state(saved['model_state_dict'], restored.alg.policy.state_dict(), 'roundtrip_model')
            repair.base.equal_state(saved['optimizer_state_dict'], restored.alg.optimizer.state_dict(), 'roundtrip_Adam')
            with torch.no_grad():
                if not torch.equal(before, restored.alg.policy.act_inference(observation)):
                    raise ValueError('Roundtrip changed deterministic means')
        finally:
            runners.OnPolicyRunner = upstream2
        for rejected, transition in [(source,False), (repair.base.SOURCE,True)]:
            try:
                repair.require_source(rejected, transition=transition)
            except ValueError:
                pass
            else:
                raise ValueError('Unqualified transition or original200 reload admitted')
        arguments = (source, cfg, 'offline_source1201_command_probe', 200,
                     cfg['commands'], output/'not_a_native_stop_request')
        keywords = dict(reward_overrides={'rewards.orientation':repair.ORIENTATION})
        old, old_final, _ = repair.previous.command(*arguments, **keywords)
        new, new_final, _ = repair.command(*arguments, **keywords)
        expected_command = list(old)
        expected_command[expected_command.index(str(repair.previous.SCRIPT))] = str(repair.SCRIPT)
        expected_command.append(repair.FLAG)
        if new != expected_command or old_final != new_final:
            raise ValueError('Native CLI changed outside the declared entry/flag')
        if old_final != 1400 or new[new.index('--target_iteration')+1] != '1401':
            raise ValueError('Preserve actual1401 updates and zero-based model1400 filename')
        event_dir = output/'synthetic_event_branch'
        event_dir.mkdir()
        event = event_observer_check(event_dir)
        if sha(source) != repair.ANCHOR_SHA:
            raise ValueError('Source checkpoint changed')
        repair.previous.evaluator_tree()
        result = dict(qualified=True, profile=repair.profile(), source_checkpoint=str(source),
            source_sha256=sha(source), source_actual_updates=repair.saved_updates(source),
            source_all17_Adam_step=optimizer_step(saved), all17_Adam_bitwise_preserved=True,
            actor_and_critic_bitwise_preserved=True, deterministic_inference_bitwise_equal=True,
            saved_std_bitwise_preserved=True, persistent_leg05_wheel03_floor_verified=True,
            actual_saved_adaptive_LR_preserved=True, native_command_only_entry_flag_changed=True,
            unqualified_and_original200_reload_rejected=True, event_observer=event,
            only_reward_change={'orientation':[-20.,repair.ORIENTATION]},
            implementation_sha256=sha(repair.SCRIPT), qualifier_sha256=sha(__file__),
            actual_PPO_updates_executed=0, simulator_started=False, offline_checkpoint_not_trained=True,
            policy_accepted=False)
        write_json(output/'qualification.json', result)
        return result
    finally:
        exploration.enforce_positive_exploration = original_enforce
        runners.OnPolicyRunner = upstream


def frozen(previous, directory):
    before, after = Path(previous)/'replay', Path(directory)/'native_reward_transfer'
    read = lambda root,name:json.loads((root/name).read_text())
    a, b = read(before,'manifest.json'), read(after,'manifest.json')
    complete = read(after,'completed.json')
    if (a['checkpoint_sha256'] != repair.ANCHOR_SHA or b['checkpoint_sha256'] != repair.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not complete['completed'] or complete['samples'] != 27000 or complete['resets']):
        raise ValueError('Require exact frozen source1201 full27')
    ca, cb = read(before,'resolved_env.json'), read(after,'resolved_env.json')
    if (differences(ca, cb) != ['/rewards/orientation']
            or ca['rewards']['orientation'] != -20. or cb['rewards']['orientation'] != repair.ORIENTATION):
        raise ValueError('Frozen native configuration changed outside orientation')
    old, new = [np.genfromtxt(root/'trace.csv', delimiter=',', names=True) for root in (before, after)]
    excluded = {'reward_total','reward_orientation'}
    if (old.dtype.names != new.dtype.names or len(new) != 27000
            or any(not np.array_equal(old[key], new[key]) for key in old.dtype.names if key not in excluded)):
        raise ValueError('Physical outputs or another reward changed')
    with np.load(before/'policy_inputs.npz', allow_pickle=False) as x, np.load(after/'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[key], y[key]) for key in x.files):
            raise ValueError('Frozen observations or deterministic means changed')
    # CSV roll/pitch and reward refer to the same post-physics state. Recorded
    # policy observations precede the step, so cannot reconstruct this reward.
    raw = np.sin(new['pitch_rad'])**2 + cb['rewards']['orientation_roll_multiplier'] * (
        np.sin(new['roll_rad'])*np.cos(new['pitch_rad']))**2
    dt = b['policy_dt']
    expected_old = -np.minimum(20.*raw, ca['rewards']['orientation_penalty_clip'])*dt
    expected_new = -np.minimum(abs(repair.ORIENTATION)*raw, cb['rewards']['orientation_penalty_clip'])*dt
    formula_error = float(max(abs(expected_old-old['reward_orientation']).max(),
                              abs(expected_new-new['reward_orientation']).max()))
    delta_error = float(abs(new['reward_total']-old['reward_total']
        -new['reward_orientation']+old['reward_orientation']).max())
    if formula_error > 1.e-7 or delta_error > 1.e-7:
        raise ValueError('Independently reconstructed orientation reward delta differs')
    result = dict(qualified=True, source_sha256=repair.ANCHOR_SHA,
        only_reward_change={'orientation':[-20.,repair.ORIENTATION]}, full27_samples=27000, resets=0,
        physical_outputs_observations_actions_and_other_rewards_bitwise_equal=True,
        independent_orientation_formula_max_error=formula_error, total_reward_delta_max_error=delta_error,
        source_data_sha256={str(root/name):sha(root/name) for root in (before, after)
            for name in ('manifest.json','completed.json','resolved_env.json','trace.csv','policy_inputs.npz')},
        implementation_sha256=sha(repair.SCRIPT), qualifier_sha256=sha(__file__), PPO_updates_performed=0,
        policy_accepted=False, limitation='Frozen reward equivalence, not a convergence claim.')
    write_json(Path(directory)/'native_reward_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-previous', type=Path)
    args = parser.parse_args()
    result = frozen(args.frozen_previous, args.output) if args.frozen_previous else qualify(args.source,args.output)
    print(json.dumps({key:value for key,value in result.items() if key != 'source_data_sha256'}, indent=2))
