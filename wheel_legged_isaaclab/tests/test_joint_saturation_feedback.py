"""Recover useful feedback on clipped motor requests without changing actions."""
import copy
import hashlib
import json

import pytest
import torch

from joint_pd_continuation import continuation_command, optimizer_step, validate_continuation_config
from joint_pd_optimization import prepared_reward_trial, validate_changes


def test_source_and_adam_are_retained_for_one_joint_feedback_change(tmp_path):
    path=tmp_path/'change.json';path.write_text(json.dumps({'rewards.joint_position_action_saturation':-8.}))
    change=prepared_reward_trial(path)
    source=tmp_path/'model_3553.pt'
    saved=dict(iter=3553,model_state_dict={'actor.6.weight':torch.ones(6,64)},
               optimizer_state_dict={'state':{0:{'step':torch.tensor(51480.),'exp_avg':torch.ones(6,64)}},
                                     'param_groups':[{'lr':.0000759375}]})
    torch.save(saved,source);digest=hashlib.sha256(source.read_bytes()).hexdigest()
    commands=dict(height_expansion_from=[.28,.32],height_expansion_start_steps=117024,
                  height_expansion_ramp_steps=48000,ranges_height=[.22534645531,.37534645531])
    previous=dict(commands=commands,rewards=dict(joint_position_action_saturation=-.5,
                   tracking_goal_violation=-.5,tracking_goal_penalty_clip=40.),leg_joint_stiffness=300.)
    argv,final,before=continuation_command(source,previous,'joint_feedback',100,commands,tmp_path/'STOP',
                                         reward_overrides=change,minimum_wheel_std=.03)
    assert optimizer_step(before)==51480 and final==3652
    assert '--reset_optimizer' not in argv and '--reset_exploration_std' not in argv
    assert 'env.rewards.joint_position_action_saturation=-8.0' in argv
    current=copy.deepcopy(previous);current['rewards']['joint_position_action_saturation']=-8.
    validate_changes(previous,current,change)
    validate_continuation_config(previous,current,commands,reward_overrides=change)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('value',[0.,1.,-8.01,float('nan'),float('inf')])
def test_invalid_joint_feedback_is_rejected_before_launch(tmp_path,value):
    path=tmp_path/'change.json';path.write_text(json.dumps({'rewards.joint_position_action_saturation':value}))
    with pytest.raises(ValueError):prepared_reward_trial(path)


def test_guide_endpoints_remain_below_a_small_reward_cost_and_no_action_is_remapped():
    waypoints=torch.tensor([[.8650308246,-.8650308209,-.8650308246,.8650308209],
                            [-.8850198937,.8850198800,.8850198937,-.8850198800]])
    penalty=8.*(waypoints.abs()-.85).clamp_min(0.).square().sum(-1)
    assert (penalty<.05).all()
    # Two raw requests formerly delivered as the same upper bound become
    # distinguishable, while the existing clamp and target scale are unchanged.
    raw=torch.tensor([1.05,1.10]);applied=raw.clamp(-1.,1.)
    assert torch.equal(applied,torch.ones_like(applied))
    costs=-8.*(raw-.85).square()
    assert costs[1]<costs[0] and (costs>-5.).all()
