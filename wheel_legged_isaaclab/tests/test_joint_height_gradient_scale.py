"""Scale a diagnosed weak gradient without changing old saved PPO updates."""
import copy
from pathlib import Path
import types
import subprocess
from unittest.mock import patch

import pytest
import torch

from test_joint_height_commands import standing_policy
from test_joint_height_feedback_ppo import height_model
from test_joint_mean_guard import fill_storage
import wheel_legged_gym_isaaclab.joint_height_feedback as height_module
from wheel_legged_gym_isaaclab.joint_mean_guard import JointMeanGuardPPO


@pytest.mark.parametrize('value',[0.,-1.,65.,float('nan'),float('inf'),True])
def test_invalid_gradient_scale_is_rejected(value):
    with pytest.raises(ValueError,match='gradient scale'):
        height_module.validate_counterfactual_gradient_scale(value)


@pytest.mark.parametrize('value',[1.,32.,64.])
def test_qualified_scale_bounds(value):
    assert height_module.validate_counterfactual_gradient_scale(value)==value


def test_scale_restores_the_direct_counterfactual_gradient_strength():
    policy,obs=standing_policy();original=height_model();original['command_counterfactuals']['gradient_scale']=1.
    enhanced=copy.deepcopy(original);enhanced['command_counterfactuals']['gradient_scale']=32.
    means=policy.act_inference(obs)
    old=height_module.height_training_loss(policy,means,obs,original)
    new=height_module.height_training_loss(policy,means,obs,enhanced)
    assert old>0.
    parameters=list(policy.actor.parameters())
    old_grad=torch.autograd.grad(old,parameters,retain_graph=True)
    new_grad=torch.autograd.grad(new,parameters)
    torch.testing.assert_close(new,32.*old,atol=1.e-6,rtol=1.e-6)
    for a,b in zip(old_grad,new_grad):torch.testing.assert_close(b,32.*a,atol=1.e-6,rtol=1.e-6)


def test_saved_v1_profile_keeps_actual_ppo_and_adam_updates_bitwise_equal():
    source=Path('/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_height_feedback.py')
    reference=types.ModuleType('unscaled_height_reference')
    code=subprocess.check_output(['git','show','2cc42e8:wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_height_feedback.py'],
                                 cwd='/home/aaa/studyRL/src/CLT-RL',text=True)
    exec(compile(code,str(source),'exec'),reference.__dict__)
    policy,obs=standing_policy();model=height_model()
    model['command_counterfactuals'].pop('gradient_scale')
    model['command_counterfactuals']['profile']='standing_height_command_v1'
    kwargs=dict(joint_mean_guard_coef=10.,joint_height_feedback_coef=.15,joint_height_feedback_model=model,
                num_learning_epochs=2,num_mini_batches=2,schedule='adaptive',learning_rate=.0001)
    baseline=JointMeanGuardPPO(copy.deepcopy(policy),**kwargs)
    candidate=JointMeanGuardPPO(copy.deepcopy(policy),**kwargs)
    fill_storage(baseline,obs);fill_storage(candidate,obs)
    with patch.object(height_module,'height_training_loss',reference.height_training_loss):
        torch.manual_seed(85);before=baseline.update()
    torch.manual_seed(85);after=candidate.update()
    assert before==after
    for k,v in baseline.policy.state_dict().items():torch.testing.assert_close(v,candidate.policy.state_dict()[k],rtol=0.,atol=0.)
    for k,values in baseline.optimizer.state_dict()['state'].items():
        for name,v in values.items():torch.testing.assert_close(v,candidate.optimizer.state_dict()['state'][k][name],rtol=0.,atol=0.)
