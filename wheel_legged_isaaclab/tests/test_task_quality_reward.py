import copy
import math

import pytest
import torch

from test_joint_pd_balance_rewards import reward_fixture
from wheel_legged_gym_isaaclab.reward_signals import (
    task_tracking_quality, guard_tracking_bonuses, validate_task_quality_gate,
)

CRITERIA = dict(relative_error=.1,body_p95_deg=2.5)


def gravity(pitch_deg, roll_deg):
    p,r = torch.deg2rad(torch.tensor(pitch_deg)),torch.deg2rad(torch.tensor(roll_deg))
    return torch.stack((p.sin(),-p.cos()*r.sin(),-p.cos()*r.cos()),-1)


def test_disabled_guard_preserves_same_signal_objects_and_original_reduction():
    signals = {'track_lin_vel':torch.tensor([2.]),'track_ang_vel':torch.tensor([1.])}
    assert guard_tracking_bonuses(signals,None,None,None,None,1.) is signals
    env,_,calculate = reward_fixture();first = calculate().clone()
    assert env.cfg.rewards.tracking_task_quality_gate_floor == 1.
    torch.testing.assert_close(calculate(),first,rtol=0,atol=0)


def test_full_bonuses_inside_half_original_height_and_body_tolerances():
    h = torch.tensor([.3,.31,.29]);cmd = torch.full((3,),.3)
    value = task_tracking_quality(h,cmd,gravity([0.,1.,-1.],[0.,.5,-.5]),CRITERIA,.1)
    torch.testing.assert_close(value,torch.ones(3),rtol=0,atol=0)


def test_quality_decreases_with_height_or_pitch_or_roll_but_never_removes_motion():
    h = torch.tensor([.3,.34,.3,.3,.4]);cmd = torch.full((5,),.3)
    value = task_tracking_quality(h,cmd,gravity([0.,0.,5.,0.,20.],[0.,0.,0.,5.,20.]),CRITERIA,.1)
    assert value[0] == 1.
    assert (value[1:]<value[0]).all()
    torch.testing.assert_close(value[2],value[3])
    assert (value>=.1).all() and (value<=1.).all()


def test_only_positive_motion_kernels_are_scaled_not_height_pose_or_speed_costs():
    signals = {key:torch.ones(2) for key in ['track_lin_vel','track_ang_vel','base_height','orientation','lin_vel_error_sq','yaw_rate_error_sq','joint_target_leg_length_error_sq']}
    result = guard_tracking_bonuses(signals,torch.full((2,),.3),torch.full((2,),.3),
                                   gravity([0.,5.],[0.,0.]),CRITERIA,.1)
    assert result['track_lin_vel'][1]<result['track_lin_vel'][0]
    for key in set(signals)-{'track_lin_vel','track_ang_vel'}:
        assert result[key] is signals[key]


def test_velocity_learning_remains_positive_at_poor_height_and_posture():
    quality = task_tracking_quality(torch.tensor([.4]),torch.tensor([.3]),gravity([10.],[0.]),CRITERIA,.1)
    slow = 2.*torch.exp(torch.tensor([-.3**2/.25]))*quality
    matched = 2.*quality
    assert matched>slow and slow>0.


def test_production_guard_changes_only_motion_bonuses_and_not_actions():
    env,_,calculate = reward_fixture(2)
    env._tracking_goal_criteria = CRITERIA
    env._robot.data.projected_gravity_b[:] = gravity([0.,5.],[0.,0.])
    original = calculate().clone();terms = copy.deepcopy(env._last_reward_terms)
    actions = env._actions.clone();raw = env._raw_actions.clone()
    env.cfg.rewards.tracking_task_quality_gate_floor = .1
    adjusted = calculate()
    assert adjusted[0] == original[0] and adjusted[1]<original[1]
    for key in terms:
        if key not in ['track_lin_vel','track_ang_vel']:
            torch.testing.assert_close(env._last_reward_terms[key],terms[key],rtol=0,atol=0)
    torch.testing.assert_close(env._actions,actions,rtol=0,atol=0)
    torch.testing.assert_close(env._raw_actions,raw,rtol=0,atol=0)


@pytest.mark.parametrize('floor',[0.,-.1,1.01,float('nan'),float('inf'),True])
def test_invalid_or_zero_floors_are_rejected(floor):
    with pytest.raises(ValueError):validate_task_quality_gate(floor,True)


def test_guard_requires_original_tracking_criteria_not_an_invented_tolerance():
    validate_task_quality_gate(1.,False)
    validate_task_quality_gate(.1,True)
    with pytest.raises(ValueError):validate_task_quality_gate(.1,False)
