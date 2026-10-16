import copy

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from tensordict import TensorDict

from wheel_legged_gym_isaaclab.joint_pd_exploration import (
    JointPdExplorationActorCritic, entropy_multipliers,
)


def policies(coefficient=.005):
    torch.manual_seed(918)
    obs = TensorDict({'policy': torch.randn(8,27)}, batch_size=[8])
    arguments = (obs, {'policy':['policy'], 'critic':['policy']}, 6)
    config = dict(actor_hidden_dims=[8,8,8], critic_hidden_dims=[8,8,8], activation='elu', init_noise_std=.04)
    original = ActorCritic(*arguments, **config)
    revised = JointPdExplorationActorCritic(*arguments, leg_entropy_coefficient=coefficient, **config)
    revised.load_state_dict(original.state_dict(), strict=True)
    return original, revised, obs


def test_sampling_likelihood_inference_and_checkpoint_parameters_are_unchanged():
    original, revised, obs = policies()
    torch.manual_seed(919); a = original.act(obs)
    torch.manual_seed(919); b = revised.act(obs)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(original.get_actions_log_prob(a),revised.get_actions_log_prob(a),rtol=0,atol=0)
    torch.testing.assert_close(original.act_inference(obs),revised.act_inference(obs),rtol=0,atol=0)
    assert list(original.state_dict()) == list(revised.state_dict())
    assert len(list(original.parameters())) == len(list(revised.parameters())) == 17
    assert all(p.requires_grad for p in revised.parameters())


def test_only_leg_entropy_is_increased_wheel_entropy_keeps_original_coefficient():
    _, revised, obs = policies()
    revised.act(obs)
    (-.005*revised.entropy.mean()).backward()
    expected = -torch.tensor([.005,.005,.001,.005,.005,.001])/revised.std.detach()
    torch.testing.assert_close(revised.std.grad,expected)


def fill(algorithm, obs):
    torch.manual_seed(920)
    algorithm.init_storage('rl',8,4,obs,[6])
    with torch.no_grad():
        for _ in range(4):
            actions=algorithm.act(obs)
            algorithm.process_env_step(obs,-actions.square().sum(-1),torch.zeros(8,dtype=torch.bool),{})
        algorithm.compute_returns(obs)


def test_equal_weights_match_upstream_ppo_and_resume_complete_adam_bitwise():
    original, revised, obs = policies(.001)
    config=dict(num_learning_epochs=2,num_mini_batches=2,learning_rate=3.e-5,
                schedule='fixed',entropy_coef=.001)
    before=PPO(original,**config)
    fill(before,obs);torch.manual_seed(921);before.update()
    revised.load_state_dict(original.state_dict(),strict=True)
    after=PPO(revised,**config)
    after.optimizer.load_state_dict(copy.deepcopy(before.optimizer.state_dict()))
    assert len(after.optimizer.state_dict()['state'])==17
    fill(before,obs);fill(after,obs)
    torch.manual_seed(922);a=before.update()
    torch.manual_seed(922);b=after.update()
    assert a==b
    for name,value in original.state_dict().items():
        torch.testing.assert_close(value,revised.state_dict()[name],rtol=0,atol=0)
    for key,state in before.optimizer.state_dict()['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(value,after.optimizer.state_dict()['state'][key][name],rtol=0,atol=0)


@pytest.mark.parametrize('value',[0.,-.001,.021,float('nan'),float('inf'),True])
def test_invalid_entropy_coefficients_rejected(value):
    with pytest.raises(ValueError):entropy_multipliers(value)
