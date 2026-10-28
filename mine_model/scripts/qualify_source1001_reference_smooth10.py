"""Retained-state, event-observer and frozen reward checks; never train PPO."""
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
import source1001_reference_smooth10 as repair
import wheel_legged_gym_isaaclab.exploration as exploration


def event_observer_check(directory):
    data = SimpleNamespace(root_pos_w=torch.zeros(2,3), root_quat_w=torch.tensor([[1.,0.,0.,0.]]*2),
        projected_gravity_b=torch.tensor([[0.,0.,-1.]]*2), root_ang_vel_b=torch.zeros(2,3),
        root_lin_vel_b=torch.zeros(2,3), joint_pos=torch.zeros(2,4), joint_vel=torch.zeros(2,4))
    data.root_pos_w[:,2] = .3
    result = torch.tensor([.125,.25])
    env = SimpleNamespace(_robot=SimpleNamespace(data=data),
        reset_terminated=torch.tensor([True,False]), _commands=torch.tensor([[0.,0.,.3]]*2),
        episode_length_buf=torch.tensor([50,50]), step_dt=.02, common_step_counter=50,
        _command_resample_step=torch.tensor([25,0]), _raw_actions=torch.zeros(2,6),
        _actions=torch.zeros(2,6), _last_actions=torch.zeros(2,6,2),
        _leg_joint_ids=[0,1,2,3], _joint_position_ref_applied=torch.zeros(2,4),
        _L0=torch.full((2,2),.27), _theta0=torch.zeros(2,2), _base_id=[0],
        _contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.tensor([[[12.,0.,0.]],[[0.,0.,0.]]]))),
        _termination_reasons={'base_contact':torch.tensor([True,False]),'numerical':torch.tensor([False,False])},
        _get_rewards=lambda:result)
    tensors = [v for v in vars(env).values() if torch.is_tensor(v)]
    tensors += list(vars(data).values())
    tensors += list(env._termination_reasons.values())
    tensors += [env._contact_sensor.data.net_forces_w, result]
    before = [value.clone() for value in tensors]
    rng = torch.random.get_rng_state().clone()
    repair.install_failure_capture(env,directory)
    returned = env._get_rewards()
    if returned is not result or any(not torch.equal(a,b) for a,b in zip(tensors,before)):
        raise ValueError('The failure observer changed physical/reward tensor data')
    if not torch.equal(rng,torch.random.get_rng_state()):
        raise ValueError('The failure observer changed RNG')
    rows = [json.loads(line) for line in (directory/'terminal_events_before_reset.jsonl').read_text().splitlines()]
    if (len(rows) != 1 or rows[0]['environment_id'] != 0 or rows[0]['reasons'] != ['base_contact']
            or rows[0]['state']['command_age_s'] != .5 or rows[0]['state']['base_net_force_n'] != 12.):
        raise ValueError('The pre-reset event identity/state capture is incorrect')
    return dict(synthetic_branch_software_check_only=True, exact_original_reward_object_returned=True,
        all_physical_tensors_and_RNG_bitwise_preserved=True, correct_event_identity_and_state=True)


def qualify(source, output):
    source, output = Path(source).resolve(),Path(output).resolve()
    output.mkdir(exist_ok=False)
    saved,cfg = repair.require_source(source,transition=True)
    agent = yaml.safe_load((source.parent/'params/agent.yaml').read_text())
    options = copy.deepcopy(agent['policy'])
    if options.pop('class_name') != 'ActorCritic':
        raise ValueError('Keep plain27 ActorCritic')
    observation = TensorDict({'policy':torch.zeros(32,27)},batch_size=[32])
    observation['policy'][:,5] = -1.
    observation['policy'][:,8] = 1.5
    def bare(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            policy = ActorCritic(observation,agent['obs_groups'],6,**options)
        runner = object.__new__(cls)
        runner.alg = SimpleNamespace(policy=policy,optimizer=torch.optim.Adam(policy.parameters()),rnd=None)
        runner.current_learning_iteration = 0
        runner.logger_type,runner.disable_logs = 'tensorboard',True
        return runner
    first = output/'first_load'
    first.mkdir()
    upstream,cls = repair.private_runtime(source).install_runner(first)
    original_enforce = exploration.enforce_positive_exploration
    try:
        runner = bare(cls)
        runner.load(str(source),load_optimizer=True,map_location='cpu')
        repair.base.equal_state(saved['model_state_dict'],runner.alg.policy.state_dict(),'loaded_model')
        repair.base.equal_state(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict(),'loaded_Adam')
        with torch.no_grad():
            before = runner.alg.policy.act_inference(observation).clone()
        repair.install_leg_exploration(first)
        handle = exploration.enforce_positive_exploration(runner.alg.policy,runner.alg.optimizer,.02,minimum_wheel_std=.03)
        repair.base.equal_state(saved['model_state_dict'],runner.alg.policy.state_dict(),'floor_did_not_reset_std')
        with torch.no_grad():
            runner.alg.policy.std[[0,1,3,4]] = .001
        runner.alg.optimizer.zero_grad(set_to_none=True)
        runner.alg.optimizer.step()
        repair.base.equal_state(saved['model_state_dict'],runner.alg.policy.state_dict(),'persistent_floor')
        repair.base.equal_state(saved['optimizer_state_dict'],runner.alg.optimizer.state_dict(),'no_gradient_Adam')
        handle.remove()
        expected = copy.deepcopy(cfg)
        expected['rewards']['action_smooth'] = repair.ACTION_SMOOTH
        validate_continuation_config(cfg,expected,cfg['commands'],reward_overrides={'rewards.action_smooth':repair.ACTION_SMOOTH})
        roundtrip = output/'roundtrip'
        roundtrip.mkdir()
        shutil.copytree(source.parent/'params',roundtrip/'params')
        (roundtrip/'params/env.yaml').write_text(yaml.safe_dump(expected,sort_keys=False))
        offline = roundtrip/'offline_source1001_smooth10_roundtrip.pt'
        runner.save(str(offline))
        runners.OnPolicyRunner = upstream
        second = output/'second_load'
        second.mkdir()
        upstream2,cls2 = repair.private_runtime().install_runner(second)
        try:
            restored = bare(cls2)
            restored.load(str(offline),load_optimizer=True,map_location='cpu')
            repair.base.equal_state(saved['model_state_dict'],restored.alg.policy.state_dict(),'roundtrip_model')
            repair.base.equal_state(saved['optimizer_state_dict'],restored.alg.optimizer.state_dict(),'roundtrip_Adam')
            with torch.no_grad():
                if not torch.equal(before,restored.alg.policy.act_inference(observation)):
                    raise ValueError('Roundtrip changed deterministic means')
        finally:
            runners.OnPolicyRunner = upstream2
        for rejected,transition in [(source,False),(repair.base.SOURCE,True)]:
            try:repair.require_source(rejected,transition=transition)
            except ValueError:pass
            else:raise ValueError('An unqualified transition or original200 reload was admitted')
        arguments = (source,cfg,'offline_source1001_command_probe',200,cfg['commands'],output/'not_a_native_stop_request')
        keywords = dict(reward_overrides={'rewards.action_smooth':repair.ACTION_SMOOTH})
        old,old_final,_ = repair.previous_controller.command(*arguments,**keywords)
        new,new_final,_ = repair.command(*arguments,**keywords)
        index = old.index(str(repair.previous_controller.ENTRY))
        expected_command = list(old)
        expected_command[index:index+1] = [str(repair.SCRIPT),'native']
        expected_command.append(repair.FLAG)
        if new != expected_command or old_final != new_final:
            raise ValueError('Native CLI changed outside the declared entry/mode/flag')
        event_dir = output/'synthetic_event_branch'
        event_dir.mkdir()
        event = event_observer_check(event_dir)
        if sha(source) != repair.ANCHOR_SHA:
            raise ValueError('Source checkpoint changed')
        repair.evaluator_tree()
        result = dict(qualified=True,profile=repair.profile(),source_checkpoint=str(source),
            source_sha256=sha(source),source_actual_updates=repair.saved_updates(source),
            source_all17_Adam_step=optimizer_step(saved),all17_Adam_bitwise_preserved=True,
            actor_and_critic_bitwise_preserved=True,deterministic_inference_bitwise_equal=True,
            saved_std_bitwise_preserved=True,persistent_leg05_wheel03_floor_verified=True,
            actual_saved_adaptive_LR_preserved=True,native_command_only_entry_mode_flag_changed=True,
            unqualified_and_original200_reload_rejected=True,event_observer=event,
            only_reward_change={'action_smooth':[0.,repair.ACTION_SMOOTH]},
            implementation_sha256=sha(repair.SCRIPT),qualifier_sha256=sha(__file__),
            actual_PPO_updates_executed=0,simulator_started=False,offline_checkpoint_not_trained=True,
            policy_accepted=False)
        write_json(output/'qualification.json',result)
        return result
    finally:
        exploration.enforce_positive_exploration = original_enforce
        runners.OnPolicyRunner = upstream


def frozen(previous, directory):
    before,after = Path(previous)/'replay',Path(directory)/'native_reward_transfer'
    read = lambda root,name:json.loads((root/name).read_text())
    a,b = read(before,'manifest.json'),read(after,'manifest.json')
    complete = read(after,'completed.json')
    if (a['checkpoint_sha256'] != repair.ANCHOR_SHA or b['checkpoint_sha256'] != repair.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not complete['completed'] or complete['samples'] != 27000 or complete['resets']):
        raise ValueError('Require exact frozen source1001 full27')
    ca,cb = read(before,'resolved_env.json'),read(after,'resolved_env.json')
    if differences(ca,cb) != ['/rewards/action_smooth'] or ca['rewards']['action_smooth'] != 0. or cb['rewards']['action_smooth'] != repair.ACTION_SMOOTH:
        raise ValueError('Frozen native configuration changed outside action_smooth')
    old,new = [np.genfromtxt(root/'trace.csv',delimiter=',',names=True) for root in (before,after)]
    excluded = {'reward_total','reward_action_smooth'}
    if old.dtype.names != new.dtype.names or len(new) != 27000 or any(not np.array_equal(old[key],new[key]) for key in old.dtype.names if key not in excluded):
        raise ValueError('Physical outputs or another reward changed')
    with np.load(before/'policy_inputs.npz',allow_pickle=False) as x,np.load(after/'policy_inputs.npz',allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[key],y[key]) for key in x.files):
            raise ValueError('Frozen observations or deterministic means changed')
        current = np.clip(y['means'],-1.,1.)
        previous_actions = y['observations'][:,:,21:27]
        older = np.concatenate([np.zeros_like(previous_actions[:1]),previous_actions[:-1]])
        second = current-2.*previous_actions+older
        raw = (second[:,:,[0,1,3,4]]**2).sum(-1)+5.*(second[:,:,[2,5]]**2).sum(-1)
    expected = -np.minimum(abs(repair.ACTION_SMOOTH)*raw,cb['rewards']['action_smooth_penalty_clip'])*b['policy_dt']
    formula_error = float(abs(expected.reshape(-1)-new['reward_action_smooth']).max())
    delta_error = float(abs(new['reward_total']-old['reward_total']-new['reward_action_smooth']+old['reward_action_smooth']).max())
    if formula_error > 1.e-7 or delta_error > 1.e-7:
        raise ValueError('Independently reconstructed reward delta differs')
    result = dict(qualified=True,source_sha256=repair.ANCHOR_SHA,
        only_reward_change={'action_smooth':[0.,repair.ACTION_SMOOTH]},full27_samples=27000,resets=0,
        physical_outputs_observations_actions_and_other_rewards_bitwise_equal=True,
        independent_second_difference_formula_max_error=formula_error,total_reward_delta_max_error=delta_error,
        source_data_sha256={str(root/name):sha(root/name) for root in (before,after)
            for name in ('manifest.json','completed.json','resolved_env.json','trace.csv','policy_inputs.npz')},
        implementation_sha256=sha(repair.SCRIPT),qualifier_sha256=sha(__file__),PPO_updates_performed=0,
        policy_accepted=False,limitation='Reward and observation-only observer equivalence; no convergence claim.')
    write_json(Path(directory)/'native_reward_transfer_qualification.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frozen-previous',type=Path)
    args = parser.parse_args()
    result = frozen(args.frozen_previous,args.output) if args.frozen_previous else qualify(args.source,args.output)
    print(json.dumps({key:value for key,value in result.items() if key != 'source_data_sha256'},indent=2))
