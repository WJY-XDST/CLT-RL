"""Discarded-action feedback, through actual reward equations."""
import torch

from test_joint_pd_balance_rewards import reward_fixture
from wheel_legged_gym_isaaclab.reward_signals import wheel_action_overshoot_penalty


def test_whole_executed_range_remains_unpenalized_and_legs_are_independent():
    raw=torch.tensor([[9.,-9.,-1.,9.,-9.,1.],[0.,0.,.8,0.,0.,-.3]])
    original=raw.clone()
    torch.testing.assert_close(wheel_action_overshoot_penalty(raw),torch.zeros(2))
    torch.testing.assert_close(raw,original)


def test_both_wheels_and_both_directions_supply_distinct_feedback():
    raw=torch.tensor([[0.,0.,1.2,0.,0.,-1.2],[0.,0.,1.8,0.,0.,-1.8],
                      [0.,0.,2.5,0.,0.,0.]],requires_grad=True)
    cost=wheel_action_overshoot_penalty(raw)
    torch.testing.assert_close(cost,torch.tensor([.04,.64,1.]))
    cost.sum().backward()
    assert raw.grad[0,2]>0 and raw.grad[0,5]<0
    assert raw.grad[:,(0,1,3,4)].abs().max()==0


def test_production_reward_distinguishes_overshoot_only_when_enabled():
    env,_,calculate=reward_fixture(3)
    env._raw_actions[:,(2,5)]=torch.tensor([[1.,1.],[1.2,1.2],[1.8,1.8]])
    env._actions[:,(2,5)]=1.
    env._last_actions[:,(2,5),:]=1.
    legacy=calculate().clone()
    torch.testing.assert_close(legacy,legacy[0].expand(3))
    env.cfg.rewards.wheel_action_saturation=-2.
    changed=calculate()
    torch.testing.assert_close(changed-legacy,torch.tensor([0.,-.0016,-.0256]),atol=1e-7,rtol=0)
    assert changed[0]>changed[1]>changed[2]
    assert env._last_reward_terms['wheel_action_saturation'][2]<0


def test_reward_cap_keeps_feedback_at_large_finite_outputs():
    env,_,calculate=reward_fixture(2)
    env.cfg.rewards.wheel_action_saturation=-2.
    env._raw_actions[:,(2,5)]=torch.tensor([[8.,8.],[9.,9.]])
    env._actions[:,(2,5)]=1.
    env._last_actions[:,(2,5),:]=1.
    calculate()
    term=env._last_reward_terms['wheel_action_saturation']
    assert torch.isfinite(term).all() and term[1]<term[0]
    assert term.abs().max()<env.cfg.rewards.wheel_action_saturation_penalty_clip*env.step_dt
