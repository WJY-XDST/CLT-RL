"""Real PPO equivalence and isolated forward-height training gradients."""
import copy
import importlib.util
import json
import subprocess
import types
from pathlib import Path

import torch

from test_joint_mean_guard import make_policy,fill_storage
from wheel_legged_gym_isaaclab.joint_mean_guard import JointMeanGuardPPO
from wheel_legged_gym_isaaclab.joint_height_feedback import model_from_config


def height_model():
    path=Path('/home/aaa/studyRL/src/CLT-RL/mine_model/results/joint_pd_optimization_20261015_004022_041465_mass_recovery_cumulative/early_4830_replay_verified/resolved_env.json')
    return model_from_config(json.loads(path.read_text()),[[-1.56999993,1.56999993]]*4)


def test_zero_height_variant_is_bitwise_equal_to_production_guard():
    source=Path('/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_mean_guard.py')
    code=subprocess.check_output(['git','show','ecbab07:wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_mean_guard.py'],cwd='/home/aaa/studyRL/src/CLT-RL',text=True)
    module=types.ModuleType('production_guard_reference')
    exec(compile(code,str(source),'exec'),module.__dict__)
    policy,obs=make_policy([1.2,-.4,.3,.4,-1.3,-.3])
    kwargs=dict(joint_mean_guard_coef=10.,num_learning_epochs=2,num_mini_batches=2,schedule='adaptive',learning_rate=3.e-5)
    original=module.JointMeanGuardPPO(copy.deepcopy(policy),**kwargs)
    candidate=JointMeanGuardPPO(copy.deepcopy(policy),joint_height_feedback_coef=0.,**kwargs)
    fill_storage(original,obs)
    fill_storage(candidate,obs)
    torch.manual_seed(85);baseline=original.update()
    torch.manual_seed(85);measured=candidate.update()
    assert baseline==measured
    for key,value in original.policy.state_dict().items():
        torch.testing.assert_close(value,candidate.policy.state_dict()[key],rtol=0.,atol=0.)
    for key,values in original.optimizer.state_dict()['state'].items():
        for name,value in values.items():
            torch.testing.assert_close(value,candidate.optimizer.state_dict()['state'][key][name],rtol=0.,atol=0.)


def test_positive_height_changes_leg_means_without_changing_action_distribution_before_update():
    policy,obs=make_policy([0.,0.,0.,0.,0.,0.])
    obs['policy'][:,0:6]=0.
    obs['policy'][:,5]=-1.
    obs['policy'][:,8]=.22534645531*5.
    kwargs=dict(joint_mean_guard_coef=10.,num_learning_epochs=1,num_mini_batches=1,schedule='fixed',
                value_loss_coef=0.,entropy_coef=0.,learning_rate=1.e-4)
    original=JointMeanGuardPPO(copy.deepcopy(policy),**kwargs)
    candidate=JointMeanGuardPPO(copy.deepcopy(policy),joint_height_feedback_coef=.1,
                               joint_height_feedback_model=height_model(),**kwargs)
    fill_storage(original,obs);fill_storage(candidate,obs)
    original.storage.advantages.zero_();candidate.storage.advantages.zero_()
    for name in ('actions','actions_log_prob','mu','sigma'):
        torch.testing.assert_close(getattr(original.storage,name),getattr(candidate.storage,name),rtol=0.,atol=0.)
    before=candidate.policy.actor[-1].bias.detach().clone()
    torch.manual_seed(85);original.update()
    torch.manual_seed(85);result=candidate.update()
    after=candidate.policy.actor[-1].bias.detach()
    assert result['joint_height_feedback']>0.
    assert after[0]>before[0] and after[1]<before[1]
    assert after[3]<before[3] and after[4]>before[4]
    torch.testing.assert_close(after[(2,5),],before[(2,5),],rtol=0.,atol=0.)
    assert len(candidate.optimizer.state_dict()['state'])==17
