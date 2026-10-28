import copy

import pytest
import torch

from test_joint_pd_balance_rewards import reward_fixture
from wheel_legged_gym_isaaclab.reward_signals import (
    joint_target_length_error_squared, validate_joint_target_length_reward,
)
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state
from wheel_legged_gym_isaaclab.joint_reset import DEFAULT_JOINT_OFFSETS
from wheel_legged_gym_isaaclab.robot_model import MAPPED_FIVE_BAR


def test_disabled_term_is_absent_from_legacy_sum_not_an_extra_zero_reduction():
    env, _, calculate = reward_fixture()
    before = calculate().clone()
    assert env.cfg.rewards.joint_target_leg_length_error_sq == 0.
    assert 'joint_target_leg_length_error_sq' not in env._last_reward_terms
    torch.testing.assert_close(calculate(), before, rtol=0, atol=0)


def test_two_sides_cannot_cancel_and_under_over_length_are_symmetric():
    command = torch.full((3,), .22534645531, dtype=torch.float64)
    refs = torch.tensor([[.19,.19],[.18,.20],[.20,.18]], dtype=torch.float64)
    result = joint_target_length_error_squared(refs,command,-.03534645531)
    assert result[0] < 1.e-28
    torch.testing.assert_close(result[1:],torch.full((2,),.0001,dtype=torch.float64))


def test_upper_endpoint_is_034_not_legacy_virtual_mode_032_cap():
    command = torch.full((2,),.37534645531,dtype=torch.float64)
    refs = torch.tensor([[.34,.34],[.32,.32]],dtype=torch.float64)
    result = joint_target_length_error_squared(refs,command,-.03534645531)
    assert result[0] < 1.e-28
    torch.testing.assert_close(result[1],torch.tensor(.0004,dtype=torch.float64))


def test_production_term_uses_target_FK_not_measured_lengths_or_raw_length_columns():
    env, _, calculate = reward_fixture(2)
    calculate(); old = env._last_reward_terms.copy()
    env.cfg.rewards.joint_target_leg_length_error_sq = -500.
    env._commands[:,2] = .37534645531
    env._l0_ref_applied = torch.tensor([[.34,.34],[.32,.32]])
    env._L0 = torch.full((2,2),.10)  # deliberately different: this cost scores decisions
    env._raw_actions[:,(1,4)] = -.9  # these are joint angles, NOT length commands
    env._joint_position_ref_applied = torch.randn(2,4)
    preserved = {name:getattr(env,name).clone() for name in
                 ('_raw_actions','_actions','_joint_position_ref_applied','_l0_ref_applied','_L0')}
    calculate()
    expected = torch.tensor([0.,-.0004*500*.02])
    torch.testing.assert_close(env._last_reward_terms['joint_target_leg_length_error_sq'],expected,atol=1.e-8,rtol=1.e-5)
    for name,value in preserved.items():
        torch.testing.assert_close(getattr(env,name),value,rtol=0,atol=0)
    for name in ('orientation','track_lin_vel','track_ang_vel','termination'):
        torch.testing.assert_close(env._last_reward_terms[name],old[name],rtol=0,atol=0)


def test_production_cost_has_dt_and_height_cap_not_flat_generic_five_reward_cap():
    env, _, calculate = reward_fixture(2)
    env.cfg.rewards.joint_target_leg_length_error_sq = -500.
    env._commands[:,2] = .22534645531
    env._l0_ref_applied = torch.tensor([[.34,.34],[1.,1.]])
    calculate()
    torch.testing.assert_close(env._last_reward_terms['joint_target_leg_length_error_sq'],
                               torch.tensor([-11.25*.02,-30.*.02]),rtol=1.e-5,atol=1.e-7)


def test_local_raw_joint_reward_gradient_moves_both_heights_correctly_and_not_wheels():
    raw = torch.zeros(2,6,dtype=torch.float64,requires_grad=True)
    commands = torch.tensor([.22534645531,.37534645531],dtype=torch.float64)
    def reward(actions):
        q = actions[:,(0,1,3,4)].clamp(-1,1)*.3+actions.new_tensor(DEFAULT_JOINT_OFFSETS)
        length,_,_,valid = five_bar_state(q[:,::2],q[:,1::2],**MAPPED_FIVE_BAR)
        assert valid.all()
        return -500.*joint_target_length_error_squared(length,commands,-.03534645531)
    original = reward(raw)
    gradient = torch.autograd.grad(original.sum(),raw)[0]
    assert torch.equal(gradient[:,(2,5)],torch.zeros(2,2,dtype=raw.dtype))
    assert torch.all(gradient[:,(0,1,3,4)].abs()>0.)
    improved = reward(raw+.001*gradient/gradient.norm(dim=1,keepdim=True))
    assert (improved>original).all()


@pytest.mark.parametrize(('mode','coefficient'),[
    ('implicit_joint_position',1.),('implicit_joint_position',-1001.),
    ('implicit_joint_position',float('nan')),('implicit_joint_position',float('inf')),
    ('implicit_joint_position',True),('explicit_vmc',-500.),
])
def test_invalid_scales_or_other_control_modes_cannot_enable_cost(mode,coefficient):
    with pytest.raises(ValueError):
        validate_joint_target_length_reward(mode,coefficient)


def test_zero_coefficient_is_backward_compatible_for_both_control_modes():
    validate_joint_target_length_reward('implicit_joint_position',0.)
    validate_joint_target_length_reward('explicit_vmc',0.)
    validate_joint_target_length_reward('implicit_joint_position',-500.)


def test_explicit_reward_override_required_when_resuming_legacy_missing_field():
    from joint_pd_continuation import HEIGHT_DEFAULTS, validate_continuation_config
    source = dict(commands=dict(HEIGHT_DEFAULTS),rewards={})
    current = copy.deepcopy(source)
    current['rewards']['joint_target_leg_length_error_sq'] = 0.
    validate_continuation_config(source,current,current['commands'])
    current['rewards']['joint_target_leg_length_error_sq'] = -500.
    with pytest.raises(ValueError):
        validate_continuation_config(source,current,current['commands'])
    validate_continuation_config(source,current,current['commands'],
                                {'rewards.joint_target_leg_length_error_sq':-500.})
