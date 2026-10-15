"""Counterfactual command learning preserves the PPO and inference contracts."""
import copy
import subprocess
import types
from pathlib import Path

import torch

from test_joint_mean_guard import make_policy,fill_storage
from test_joint_height_feedback_ppo import height_model
from wheel_legged_gym_isaaclab.joint_height_feedback import height_training_loss,height_proxy_loss,predict_support_heights
from wheel_legged_gym_isaaclab.joint_mean_guard import JointMeanGuardPPO


def standing_policy():
    policy,obs=make_policy([0.,0.,0.,0.,0.,0.])
    obs['policy'].zero_()
    obs['policy'][:,5]=-1.
    obs['policy'][:,8]=1.5
    return policy,obs


def test_auxiliary_commands_preserve_inputs_rng_and_the_actual_ppo_distribution():
    policy,obs=standing_policy()
    model=height_model()
    before=obs.clone()
    actions=policy.act(obs)
    means=policy.action_mean.clone()
    std=policy.action_std.clone()
    logp=policy.get_actions_log_prob(actions).clone()
    rng=torch.random.get_rng_state().clone()
    captured=[]
    original=policy.act_inference
    def recording(value):
        captured.append(value.clone())
        return original(value)
    policy.act_inference=recording
    loss=height_training_loss(policy,means,obs,model)
    assert loss>0. and len(captured)==1
    auxiliary=captured[0]['policy'];unchanged=list(range(8))+list(range(9,27))
    torch.testing.assert_close(auxiliary[:,unchanged],obs['policy'].repeat(3,1)[:,unchanged],atol=0.,rtol=0.)
    torch.testing.assert_close(auxiliary[:,8]/5.,torch.tensor(model['command_counterfactuals']['commands_m']).repeat_interleave(8))
    torch.testing.assert_close(obs['policy'],before['policy'],atol=0.,rtol=0.)
    torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0.,rtol=0.)
    torch.testing.assert_close(policy.action_mean,means,atol=0.,rtol=0.)
    torch.testing.assert_close(policy.action_std,std,atol=0.,rtol=0.)
    torch.testing.assert_close(policy.get_actions_log_prob(actions),logp,atol=0.,rtol=0.)
    loss.backward()
    assert torch.isfinite(policy.actor[-1].weight.grad).all()
    assert policy.actor[-1].weight.grad[[0,1,3,4]].abs().sum()>0.
    assert policy.actor[-1].weight.grad[[2,5]].abs().sum()==0.


def test_nonstanding_or_tilted_states_keep_only_the_original_height_loss():
    for column,value in [(6,.2),(7,.1),(17,.2),(5,-.9),(0,.5)]:
        policy,obs=standing_policy();obs['policy'][:,column]=value
        model=height_model();means=policy.act_inference(obs)
        expected=height_proxy_loss(means,obs['policy'],model)
        policy.act_inference=lambda _:(_ for _ in ()).throw(AssertionError('Ineligible state used for an auxiliary command'))
        torch.testing.assert_close(height_training_loss(policy,means,obs,model),expected,rtol=0.,atol=0.)


def test_old_saved_height_model_preserves_actual_ppo_updates_bitwise():
    source=Path('/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_mean_guard.py')
    module=types.ModuleType('production_height_reference')
    code=subprocess.check_output(['git','show','c4a45d6:wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_mean_guard.py'],
                                 cwd='/home/aaa/studyRL/src/CLT-RL',text=True)
    exec(compile(code,str(source),'exec'),module.__dict__)
    policy,obs=standing_policy();obs['policy'][:,8]=.22534645531*5.
    model=height_model();model.pop('command_counterfactuals');model.pop('velocity_scale')
    kwargs=dict(joint_mean_guard_coef=10.,joint_height_feedback_coef=.15,joint_height_feedback_model=model,
                num_learning_epochs=2,num_mini_batches=2,schedule='adaptive',learning_rate=3.e-5)
    old=module.JointMeanGuardPPO(copy.deepcopy(policy),**kwargs)
    candidate=JointMeanGuardPPO(copy.deepcopy(policy),**kwargs)
    fill_storage(old,obs);fill_storage(candidate,obs)
    torch.manual_seed(85);baseline=old.update()
    torch.manual_seed(85);measured=candidate.update()
    assert baseline==measured
    for k,v in old.policy.state_dict().items():
        torch.testing.assert_close(v,candidate.policy.state_dict()[k],rtol=0.,atol=0.)
    for k,values in old.optimizer.state_dict()['state'].items():
        for name,v in values.items():
            torch.testing.assert_close(v,candidate.optimizer.state_dict()['state'][k][name],rtol=0.,atol=0.)


def test_counterfactual_height_commands_teach_a_constant_actor_to_respond():
    policy,obs=standing_policy();model=height_model()
    low,_,high=model['command_counterfactuals']['commands_m']
    def heights():
        batch=obs.clone().repeat(2)
        batch['policy'][:8,8]=low*5.;batch['policy'][8:,8]=high*5.
        height=predict_support_heights(policy.act_inference(batch),batch['policy'],model)[0].mean(1)
        return height[:8].mean(),height[8:].mean()
    initial_low,initial_high=heights()
    assert abs(float(initial_high-initial_low))<1.e-8
    optimizer=torch.optim.Adam(policy.actor.parameters(),lr=.002)
    initial_loss=float(height_training_loss(policy,policy.act_inference(obs),obs,model))
    for _ in range(100):
        optimizer.zero_grad()
        loss=height_training_loss(policy,policy.act_inference(obs),obs,model)
        loss.backward();optimizer.step()
    final_low,final_high=heights()
    assert float(final_high-final_low)>.01
    assert float((final_low-low).abs()+(final_high-high).abs())<float((initial_low-low).abs()+(initial_high-high).abs())
    assert float(height_training_loss(policy,policy.act_inference(obs),obs,model))<initial_loss
