"""Compare learning feedback with the independent physical error definitions."""
import hashlib
import json
import math

import numpy as np
import pytest
import torch

from joint_pd_continuation import continuation_command, optimizer_step, validate_continuation_config
from joint_pd_optimization import prepared_reward_trial
from joint_pd_optimization import validate_changes
from summarize_mine_trial import metrics_for_window
from wheel_legged_gym_isaaclab.tracking_goal_feedback import (
    load_tracking_goal_criteria, tracking_goal_violation_penalty,
)


def physical_batch():
    commands = torch.tensor([[.6, .3, .3]], dtype=torch.float64)
    return [commands, commands[:, 0].clone(), commands[:, 1].clone(), commands[:, 2].clone(),
            torch.tensor([[0., 0., -1.]], dtype=torch.float64),
            torch.tensor([[.2, .25]], dtype=torch.float64),
            torch.tensor([[.2, .25]], dtype=torch.float64),
            torch.zeros(1, 2, dtype=torch.float64), torch.zeros(1, 2, dtype=torch.float64),
            torch.tensor([[10., 12.]], dtype=torch.float64),
            torch.tensor([[10., 12.]], dtype=torch.float64)]


@pytest.mark.parametrize('metric', ['speed', 'yaw', 'height', 'length', 'angle', 'wheels', 'body'])
def test_each_existing_gate_component_has_feedback(metric):
    criteria, _ = load_tracking_goal_criteria()
    values = physical_batch()
    index, delta = {'speed':(1,.12), 'yaw':(2,.06), 'height':(3,.06),
                    'length':(5,.04), 'angle':(7,.04), 'wheels':(9,2.)}.get(metric, (None,None))
    if index is not None:
        values[index].flatten()[0] += delta
    else:
        values[4][0] = torch.tensor([math.sin(math.radians(5.)), 0., -math.cos(math.radians(5.))])
    cost = tracking_goal_violation_penalty(*values, criteria)
    assert cost.item() == pytest.approx(.5, abs=1.e-6)
    # A constant window has the same MAE/P95 definitions as the independent gate.
    c,s,y,h,g,l,lr,a,ar,w,wr = [v.numpy() for v in values]
    trace = dict(speed_m_s=s, speed_cmd_m_s=c[:,0], yaw_rate_rad_s=y, yaw_cmd_rad_s=c[:,1],
                 height_m=h, height_cmd_m=c[:,2], pitch_rad=np.arcsin(g[:,0]),
                 roll_rad=np.arctan2(-g[:,1],-g[:,2]))
    for i, side in enumerate(('left','right')):
        trace.update({side+'_length_m':l[:,i], side+'_length_ref_m':lr[:,i],
                      side+'_angle_rad':a[:,i], side+'_angle_ref_rad':ar[:,i],
                      side+'_wheel_rad_s':w[:,i], side+'_wheel_ref_rad_s':wr[:,i]})
    measured = metrics_for_window(trace, dict(speed=.6, yaw=.3, height=.3), criteria)
    key, limit = {'speed':('speed_mae_m_s',.06), 'yaw':('yaw_mae_rad_s',.03),
                  'height':('height_mae_m',.03), 'length':('leg_length_mae_m',.02),
                  'angle':('leg_angle_mae_rad',.02), 'wheels':('wheel_speed_mae_rad_s',1.),
                  'body':('body_p95_deg',2.5)}[metric]
    assert measured[key] / limit == pytest.approx(2., abs=1.e-6)


def test_zero_commands_are_finite_and_large_errors_keep_restoring_signal():
    criteria, _ = load_tracking_goal_criteria()
    values = physical_batch(); values[0][:,:2]=0.; values[1].zero_();values[2].zero_()
    values[9].zero_(); values[10].zero_()
    assert tracking_goal_violation_penalty(*values, criteria).item() == 0.
    values[1] = torch.tensor([.2], dtype=torch.float64, requires_grad=True)
    cost = tracking_goal_violation_penalty(*values, criteria)
    cost.backward()
    assert cost.item() == pytest.approx(6.5)
    assert values[1].grad.item() == pytest.approx(40.)


def test_other_good_metrics_cannot_compensate_a_failed_one():
    criteria, _ = load_tracking_goal_criteria();values=physical_batch()
    values[3] += .15
    failed = tracking_goal_violation_penalty(*values, criteria)
    values[1] -= .03; values[2] -= .01; values[9] -= .1
    assert torch.equal(failed, tracking_goal_violation_penalty(*values, criteria))


@pytest.mark.parametrize('key,value', [('relative_error',.11),('speed_absolute_m_s',0.),
                                     ('yaw_absolute_rad_s',True),('body_p95_deg',float('nan'))])
def test_invalid_or_relaxed_criteria_are_rejected_before_use(tmp_path,key,value):
    criteria,_=load_tracking_goal_criteria();criteria[key]=value
    path=tmp_path/'criteria.json';path.write_text(json.dumps(criteria))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):load_tracking_goal_criteria(path)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==digest


def test_prepared_reward_keeps_source_weights_optimizer_and_physics(tmp_path):
    change=tmp_path/'change.json';change.write_text(json.dumps({'rewards.tracking_goal_violation':-.5}))
    override=prepared_reward_trial(change)
    source=tmp_path/'model_3250.pt'
    torch.save(dict(iter=3250,model_state_dict={'actor.6.weight':torch.ones(6,64)},
                   optimizer_state_dict={'state':{0:{'step':torch.tensor(45340.),'exp_avg':torch.ones(6,64)}},
                                         'param_groups':[{'lr':.00011390625}]}),source)
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    cfg={'robot':{'spawn':{'usd_path':'/confirmed_mass.usda'}},'commands':{},
         'rewards':{'orientation':-60.}}
    argv, final, saved=continuation_command(source,cfg,'goal',100,{},tmp_path/'STOP',
                                          reward_overrides=override,minimum_wheel_std=.03)
    assert final==3349 and optimizer_step(saved)==45340
    assert '--allow_physics_contract_change' not in argv
    assert '--reset_optimizer' not in argv and '--reset_exploration_std' not in argv
    assert 'env.rewards.tracking_goal_violation=-0.5' in argv
    current=json.loads(json.dumps(cfg));current['commands']=dict(
        height_expansion_from=None,height_expansion_start_steps=0,height_expansion_ramp_steps=48000)
    current['rewards'].update(tracking_goal_violation=-.5,tracking_goal_penalty_clip=40.)
    validate_continuation_config(cfg,current,{},reward_overrides=override)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest


@pytest.mark.parametrize('value',[.1,-3.,float('nan')])
def test_unqualified_reward_changes_are_rejected(tmp_path,value):
    path=tmp_path/'change.json';path.write_text(json.dumps({'rewards.tracking_goal_violation':value}))
    with pytest.raises(ValueError):prepared_reward_trial(path)


@pytest.mark.parametrize('check', [validate_changes, validate_continuation_config])
def test_old_checkpoint_accepts_neutral_fields_but_not_an_unrequested_reward(check):
    before=dict(commands=dict(height_expansion_from=None,height_expansion_start_steps=0,
                              height_expansion_ramp_steps=48000),
                rewards=dict(orientation=-60.),leg_joint_stiffness=300.)
    after=json.loads(json.dumps(before))
    after['rewards'].update(tracking_goal_violation=0.,tracking_goal_penalty_clip=40.)
    if check is validate_changes:
        check(before,after,{})
    else:
        check(before,after,before['commands'])
    after['rewards']['tracking_goal_violation']=-.5
    with pytest.raises(ValueError):
        if check is validate_changes:
            check(before,after,{})
        else:
            check(before,after,before['commands'])


def test_explicit_zero_worst_cost_is_an_admitted_single_term_trial(tmp_path):
    path=tmp_path/"change.json"
    path.write_text(json.dumps({"rewards.tracking_goal_violation":0.}))
    assert prepared_reward_trial(path)=={"rewards.tracking_goal_violation":0.}
