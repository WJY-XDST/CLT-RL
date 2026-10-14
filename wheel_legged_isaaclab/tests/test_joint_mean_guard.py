"""Real PPO update and resume checks for the optional joint mean loss."""
import copy
import json

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic

from wheel_legged_gym_isaaclab.joint_mean_guard import (
    JointMeanGuardPPO, configured_runner_dict, joint_mean_overshoot_loss,
    register_runner_algorithm, resolve_coefficient, validate_coefficient,
)

torch.set_num_threads(1)


def make_policy(bias=None):
    torch.manual_seed(83)
    observations = TensorDict({'policy': torch.randn(8, 27)}, batch_size=[8])
    policy = ActorCritic(observations, {'policy': ['policy'], 'critic': ['policy']}, 6,
                         actor_hidden_dims=[8, 8, 8], critic_hidden_dims=[8, 8, 8],
                         activation='elu', init_noise_std=.05)
    if bias is not None:
        with torch.no_grad():
            policy.actor[-1].weight.zero_()
            policy.actor[-1].bias.copy_(torch.tensor(bias))
    return policy, observations


def fill_storage(algorithm, observations):
    torch.manual_seed(84)
    algorithm.init_storage('rl', 8, 4, observations, [6])
    with torch.no_grad():
        for index in range(4):
            obs = TensorDict({'policy': observations['policy'] + index * .01}, batch_size=[8])
            actions = algorithm.act(obs)
            rewards = -actions.square().sum(-1)
            algorithm.process_env_step(obs, rewards, torch.zeros(8, dtype=torch.bool), {})
        algorithm.compute_returns(observations)


def test_guard_gradient_restores_only_infeasible_leg_means():
    mean = torch.tensor([[1.2, -1.4, 20., .8, -1., -20.]], requires_grad=True)
    loss = joint_mean_overshoot_loss(mean)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(.05))
    torch.testing.assert_close(mean.grad, torch.tensor([[.1, -.2, 0., 0., 0., 0.]]))


@pytest.mark.parametrize('coefficient', [0., 1., 10.])
@pytest.mark.parametrize('schedule', ['adaptive', 'fixed'])
def test_zero_or_feasible_guard_matches_actual_upstream_ppo(coefficient, schedule):
    policy, obs = make_policy()
    original = PPO(copy.deepcopy(policy), num_learning_epochs=2, num_mini_batches=2,
                   learning_rate=3.e-5, schedule=schedule, entropy_coef=.001)
    guarded = JointMeanGuardPPO(copy.deepcopy(policy), joint_mean_guard_coef=coefficient,
                                num_learning_epochs=2, num_mini_batches=2,
                                learning_rate=3.e-5, schedule=schedule, entropy_coef=.001)
    fill_storage(original, obs)
    fill_storage(guarded, obs)
    for name in ('actions', 'actions_log_prob', 'mu', 'sigma'):
        torch.testing.assert_close(getattr(original.storage, name), getattr(guarded.storage, name), rtol=0, atol=0)
    torch.manual_seed(85)
    reference = original.update()
    torch.manual_seed(85)
    candidate = guarded.update()
    assert {key: candidate[key] for key in reference} == reference
    if coefficient:
        assert candidate['joint_mean_guard'] == 0.
    else:
        assert candidate == reference
    for name, value in original.policy.state_dict().items():
        torch.testing.assert_close(value, guarded.policy.state_dict()[name], rtol=0, atol=0)
    for key, state in original.optimizer.state_dict()['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, guarded.optimizer.state_dict()['state'][key][name], rtol=0, atol=0)


def test_positive_guard_has_extra_actor_gradient_without_changing_probabilities():
    policy, obs = make_policy([1.2, -.4, 3., .4, -1.3, -3.])
    original = PPO(copy.deepcopy(policy), num_learning_epochs=1, num_mini_batches=1,
                   schedule='fixed', learning_rate=1.e-4, value_loss_coef=0., entropy_coef=0.)
    guarded = JointMeanGuardPPO(copy.deepcopy(policy), joint_mean_guard_coef=1.,
                                num_learning_epochs=1, num_mini_batches=1,
                                schedule='fixed', learning_rate=1.e-4, value_loss_coef=0., entropy_coef=0.)
    fill_storage(original, obs)
    fill_storage(guarded, obs)
    # Zero advantage isolates the mean loss from task/value/entropy gradients.
    original.storage.advantages.zero_()
    guarded.storage.advantages.zero_()
    torch.testing.assert_close(original.storage.actions_log_prob, guarded.storage.actions_log_prob, rtol=0, atol=0)
    for alg in (original, guarded):
        alg.policy.act(obs)
    torch.testing.assert_close(original.policy.entropy, guarded.policy.entropy, rtol=0, atol=0)
    before = guarded.policy.actor[-1].bias.detach().clone()
    torch.manual_seed(85); original.update()
    torch.manual_seed(85); result = guarded.update()
    after = guarded.policy.actor[-1].bias.detach()
    assert result['joint_mean_guard'] > 0
    assert after[0] < before[0] and after[4] > before[4]
    torch.testing.assert_close(after[[1, 2, 3, 5]], before[[1, 2, 3, 5]], rtol=0, atol=0)
    assert all(param.requires_grad for param in guarded.policy.parameters())


def test_guard_resume_retains_all_parameter_identities_and_adam_moments():
    policy, obs = make_policy([1.2, -.4, .3, .4, -1.3, -.3])
    original = PPO(policy, num_learning_epochs=1, num_mini_batches=1, schedule='fixed')
    fill_storage(original, obs); original.update()
    saved = copy.deepcopy(original.optimizer.state_dict())
    restored_policy, _ = make_policy()
    restored_policy.load_state_dict(original.policy.state_dict())
    guarded = JointMeanGuardPPO(restored_policy, joint_mean_guard_coef=1.,
                                num_learning_epochs=1, num_mini_batches=1, schedule='fixed')
    identities = [id(param) for param in guarded.policy.parameters()]
    guarded.optimizer.load_state_dict(saved)
    assert len(saved['state']) == 17
    for key, state in saved['state'].items():
        for name, value in state.items():
            torch.testing.assert_close(value, guarded.optimizer.state_dict()['state'][key][name], rtol=0, atol=0)
    fill_storage(guarded, obs); guarded.update()
    assert identities == [id(param) for param in guarded.policy.parameters()]
    assert len(guarded.optimizer.state_dict()['state']) == 17
    assert all(state['step'].item() == 2. for state in guarded.optimizer.state_dict()['state'].values())


@pytest.mark.parametrize('invalid', [-.1, 10.01, float('nan'), float('inf'), True, None])
def test_invalid_guard_is_rejected(invalid):
    with pytest.raises(ValueError, match='coefficient'):
        validate_coefficient(invalid)


def test_configuration_is_opt_in_and_inherited_from_paired_agent(tmp_path):
    config = {'algorithm': {'class_name': 'PPO', 'gamma': .99}, 'policy': {'actor_hidden_dims': [256, 128, 64]}}
    assert configured_runner_dict(config, 0.) == config
    guarded = configured_runner_dict(config, 1.)
    assert config['algorithm']['class_name'] == 'PPO'
    checkpoint = tmp_path/'model_4356.pt'
    (tmp_path/'params').mkdir()
    (tmp_path/'params/agent.yaml').write_text(json.dumps(guarded))
    assert resolve_coefficient(checkpoint) == 1.
    assert resolve_coefficient(checkpoint, 0.) == 0.
    assert configured_runner_dict(guarded, 0.) == config
    register_runner_algorithm()
    import rsl_rl.runners.on_policy_runner as runner
    assert runner.JointMeanGuardPPO is JointMeanGuardPPO and runner.PPO is PPO


def test_cumulative_command_inherits_guard_without_resetting_checkpoint(tmp_path):
    import hashlib
    from joint_pd_continuation import continuation_command
    checkpoint = tmp_path/'model_4413_interrupted.pt'
    torch.save({'iter': 4413, 'model_state_dict': {'actor.6.weight': torch.ones(6, 64)},
                'optimizer_state_dict': {'state': {0: {'step': torch.tensor(68900.)}},
                                         'param_groups': [{'lr': .00011390625}]}}, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (tmp_path/'params').mkdir()
    (tmp_path/'params/agent.yaml').write_text(json.dumps({
        'algorithm': {'class_name': 'JointMeanGuardPPO', 'joint_mean_guard_coef': 10.}}))
    command, final, _ = continuation_command(checkpoint, {'rewards': {'orientation': -60.}},
        'continued', 100, {}, tmp_path/'STOP', minimum_wheel_std=.03)
    assert command[command.index('--joint_mean_guard_coef') + 1] == '10.0'
    assert command[command.index('--num_envs') + 1] == '8192'
    assert final == 4512
    assert '--reset_optimizer' not in command and '--reset_exploration_std' not in command
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
