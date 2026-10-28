import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
from joint_pd_training_observer import TrainingObserver
from test_joint_mean_guard import make_policy,fill_storage
from test_joint_height_feedback_ppo import height_model


@pytest.mark.parametrize('schedule',['fixed','adaptive'])
def test_observer_preserves_real_ppo_weights_adam_gradients_losses_and_rng(schedule):
    policy,obs=make_policy()
    obs['policy'].zero_()
    obs['policy'][:,5]=-1.
    obs['policy'][:,8]=.22534645531*5.
    settings=dict(num_learning_epochs=2,num_mini_batches=2,learning_rate=3.e-5,
                  schedule=schedule,entropy_coef=.001)
    original=PPO(copy.deepcopy(policy),**settings)
    measured=PPO(copy.deepcopy(policy),**settings)
    observer=TrainingObserver(measured,height_model())
    for _ in range(2):
        fill_storage(original,obs)
        fill_storage(measured,obs)
        torch.manual_seed(85)
        expected=original.update()
        rng=torch.random.get_rng_state().clone()
        torch.manual_seed(85)
        actual=measured.update()
        assert expected==actual
        assert torch.equal(rng,torch.random.get_rng_state())
        for key,value in original.policy.state_dict().items():
            torch.testing.assert_close(value,measured.policy.state_dict()[key],rtol=0.,atol=0.)
        for left,right in zip(original.policy.parameters(),measured.policy.parameters()):
            torch.testing.assert_close(left.grad,right.grad,rtol=0.,atol=0.)
        assert original.optimizer.param_groups[0]['lr']==measured.optimizer.param_groups[0]['lr']
        for key,states in original.optimizer.state_dict()['state'].items():
            for name,value in states.items():
                torch.testing.assert_close(value,measured.optimizer.state_dict()['state'][key][name],rtol=0.,atol=0.)
    assert len(observer.updates)==2 and len(observer.gradients)==8
    assert observer.updates[-1]['probe_RNG_unchanged']
    assert not observer.updates[-1]['added_loss_or_backward']
    assert not observer.updates[-1]['policy_accepted']
    assert len(measured.optimizer.state_dict()['state'])==17
    observer.close()


def test_probe_does_not_change_the_actual_sample_distribution_or_inputs():
    policy,obs=make_policy()
    obs['policy'].zero_();obs['policy'][:,5]=-1.;obs['policy'][:,8]=1.5
    algorithm=PPO(policy)
    observer=TrainingObserver(algorithm,height_model())
    actions=policy.act(obs)
    old_mu=policy.action_mean.clone();old_std=policy.action_std.clone()
    old_log=policy.get_actions_log_prob(actions).clone()
    before=obs.clone()
    observer.minibatch=obs['policy'].clone()
    observer._probe()
    torch.testing.assert_close(policy.action_mean,old_mu,rtol=0.,atol=0.)
    torch.testing.assert_close(policy.action_std,old_std,rtol=0.,atol=0.)
    torch.testing.assert_close(policy.get_actions_log_prob(actions),old_log,rtol=0.,atol=0.)
    torch.testing.assert_close(obs['policy'],before['policy'],rtol=0.,atol=0.)
    observer.close()


def test_observer_rejects_random_or_stateful_probe_layers():
    policy,_=make_policy()
    policy.actor=torch.nn.Sequential(policy.actor,torch.nn.Dropout(.1))
    with pytest.raises(ValueError,match='RNG or running state'):
        TrainingObserver(PPO(policy),height_model())


def native_fixture():
    data=SimpleNamespace(root_pos_w=torch.tensor([[0.,0.,.3]]).repeat(8,1),
        projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(8,1),
        root_ang_vel_b=torch.zeros(8,3),joint_pos=torch.zeros(8,6))
    env=SimpleNamespace(_robot=SimpleNamespace(data=data),_commands=torch.tensor([[0.,0.,.3]]).repeat(8,1),
        _raw_actions=torch.zeros(8,6),_joint_position_ref_applied=torch.zeros(8,4),_leg_joint_ids=[0,1,3,4],
        _L0=torch.full((8,2),.26465),_l0_ref_applied=torch.full((8,2),.26465))
    env._get_heading_frame_horizontal_velocity=lambda:torch.zeros(8,2)
    env._get_rewards=lambda:-env._raw_actions.square().sum(-1)
    cfg=dict(obs_scales=dict(lin_vel=2.,ang_vel=.25,height_measurements=5.))
    return env,cfg


def native_rollout(algorithm,obs,env,*,mutate_observation=False,wrong_native_action=False,post_reset=False):
    torch.manual_seed(84)
    algorithm.init_storage('rl',8,4,obs,[6])
    with torch.no_grad():
        for index in range(4):
            value=obs.clone()
            actions=algorithm.act(value)
            env._raw_actions=actions.clone()+(.01 if wrong_native_action else 0.)
            env._robot.data.root_pos_w[:,2]=.3
            rewards=env._get_rewards()
            if post_reset:
                env._raw_actions.zero_()
                env._robot.data.root_pos_w[:,2]=999.
            if mutate_observation:
                value['policy'][:,8]+=.1
            timeouts=torch.zeros(8,dtype=torch.bool)
            timeouts[0]=index==1
            algorithm.process_env_step(value,rewards,timeouts,dict(time_outs=timeouts))
        algorithm.compute_returns(obs)


def native_policy():
    policy,obs=make_policy()
    obs['policy'].zero_();obs['policy'][:,5]=-1.;obs['policy'][:,8]=1.5
    return policy,obs


def test_real_transition_storage_timeout_math_and_before_reset_snapshot():
    policy,obs=native_policy()
    algorithm=PPO(policy,num_learning_epochs=1,num_mini_batches=1)
    env,cfg=native_fixture()
    observer=TrainingObserver(algorithm,height_model(),env=env,cfg=cfg)
    native_rollout(algorithm,obs,env,post_reset=True)
    algorithm.update()
    errors=observer.updates[0]['all_environment_alignment_max_errors']
    assert errors and max(errors.values())==0.
    assert (observer.last_arrays['native_height']==torch.tensor(.3).item()).all()
    assert observer.last_arrays['native_timeouts'].sum()==1
    assert observer.last_arrays['native_raw_actions'].any()
    observer.close()


@pytest.mark.parametrize(('option','metric'),[
    ('mutate_observation','actor_observation_storage_max_error'),
    ('wrong_native_action','action_native_max_error'),
])
def test_transition_audit_detects_real_mismatch_instead_of_reporting_assumed_zeros(option,metric):
    policy,obs=native_policy()
    algorithm=PPO(policy,num_learning_epochs=1,num_mini_batches=1)
    env,cfg=native_fixture()
    observer=TrainingObserver(algorithm,height_model(),env=env,cfg=cfg)
    native_rollout(algorithm,obs,env,**{option:True})
    algorithm.update()
    assert observer.updates[0]['all_environment_alignment_max_errors'][metric]>.005
    observer.close()
