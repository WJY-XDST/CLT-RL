import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import train_joint_pd_standing_seed as training
from wheel_legged_gym_isaaclab.yaw_training import sample_grouped_motion


@pytest.fixture
def report():
    names=('stand','forward_0p3','forward_0p6','reverse_0p3','reverse_0p6',
           'spin_left','spin_right','height_low','height_high')
    conditions=[]
    for name in names:
        metrics=dict.fromkeys(training.METRICS,0.)
        metrics.update(upright_support=True,command_reached=True)
        trials=[dict(seed=seed,survived=True,metrics=copy.deepcopy(metrics),
                     limits=dict.fromkeys(training.METRICS,1.)) for seed in (83,84,85)]
        conditions.append(dict(name=name,height=.3,trials=trials,survivors=3))
    return dict(conditions=conditions,survivors=27,total_trials=27,resets=0,passed=True)


def test_only_requested_tracking_reinforcements_change_historical_rewards():
    original=dict(orientation=-20.,yaw_rate_error_sq=-1.,base_height=2.,
                  base_height_error_sq=0.,action_rate=-.02,
                  joint_position_action_saturation=-.5)
    before=copy.deepcopy(original)
    base=training.simple_rewards(original)
    result=training.reinforced_rewards(original)
    assert original==before
    assert {k:v for k,v in result.items() if base.get(k)!=v} == {
        'base_height_error_sq':-50.,'tracking_goal_component_violation':-.1}
    assert result['orientation']==-20 and result['yaw_rate_error_sq']==-1


@pytest.mark.parametrize(('actual','target'),[(201,206),(202,206),(206,301),(250,301),(301,401),(350,401)])
def test_resume_targets_count_real_updates_instead_of_legacy_label(actual,target):
    assert training.next_target(actual,201)==target


def test_target_does_not_allow_source_rollback():
    with pytest.raises(ValueError,match='roll back'):
        training.next_target(200,201)


def test_nominal_standing_requires_all_seeds_and_original_strict_limits(report):
    assert training.nominal_standing_intact(report)
    report['conditions'][0]['trials'][0]['metrics']['body_p95_deg']=1.
    assert not training.nominal_standing_intact(report)


def test_nominal_standing_cannot_be_replaced_by_survival_only(report):
    report['conditions'][0]['trials'][0]['metrics']['upright_support']=False
    assert not training.nominal_standing_intact(report)


def test_high_height_task_must_pass_before_motion_is_released(report):
    assert training.stationary_task_ready(report)
    report['conditions'][-1]['trials'][0]['metrics']['height_mae_m']=1.1
    assert training.standing_readiness(report)['ready']
    assert not training.stationary_task_ready(report)


def test_monitoring_keeps_both_full_height_conditions(report):
    report['conditions'][-1]['trials'][0]['metrics']['height_mae_m']=2.
    assert training.monitoring_ratios(report)['height_mae_m']==2.


def test_untrained_motion_is_not_hidden_from_full_gate(report):
    report['conditions'][1]['trials'][0]['metrics']['speed_mae_m_s']=3.
    assert training.monitoring_ratios(report)['speed_mae_m_s']==0
    assert training.monitoring_ratios(report,True)['speed_mae_m_s']==3
    assert not training.strict_full_gate(report)


def test_full_goal_is_strict_and_not_the_cached_passed_flag(report):
    assert training.strict_full_gate(report)
    report['conditions'][0]['trials'][0]['metrics']['wheel_speed_mae_rad_s']=1.
    assert report['passed']
    assert not training.strict_full_gate(report)


def test_full_goal_requires_the_complete_27_trials(report):
    report['conditions'].pop()
    assert not training.strict_full_gate(report)


def history(values,scope='all_three_full_height_stationary_conditions'):
    return [dict(additional_ppo_updates=extra,monitoring_scope=scope,
                 monitoring_ratios=dict.fromkeys(training.METRICS,value))
            for extra,value in zip((5,100,200),values)]


def test_surviving_but_stagnant_failed_axes_request_diagnosis():
    assert set(training.stagnant_axes(history((3.,3.,3.))))==set(training.METRICS)


def test_learning_and_already_passing_axes_do_not_trigger_plateau():
    assert training.stagnant_axes(history((3.,2.,1.5)))==[]
    assert training.stagnant_axes(history((.5,.5,.5)))==[]


def test_a_passing_axis_does_not_conceal_a_failed_one():
    records=history((.5,.4,.3))
    for entry in records: entry['monitoring_ratios']['height_mae_m']=3.
    assert training.stagnant_axes(records)==['height_mae_m']


def test_scope_change_does_not_compare_incomparable_plateau():
    records=history((3.,3.,3.))
    records[-1]['monitoring_scope']='all_original_conditions'
    assert training.stagnant_axes(records)==[]


def test_saved_updates_uses_Adam_not_model_200_label(monkeypatch):
    saved=dict(iter=200,model_state_dict={'std':torch.ones(6)},
               optimizer_state_dict={'state':{i:{'step':torch.tensor(4020.)} for i in range(17)}})
    monkeypatch.setattr(training.torch,'load',lambda *args,**kwargs:saved)
    monkeypatch.setattr(training,'load_curriculum_clock',lambda path:dict(
        offset_steps=0,steps_per_update=48,adam_steps_per_update=20))
    assert training.saved_updates('model_200.pt')==201


def test_clockless_or_incomplete_state_is_not_silently_reinitialized(monkeypatch):
    monkeypatch.setattr(training.torch,'load',lambda *args,**kwargs:dict(
        optimizer_state_dict={'state':{}},model_state_dict={}))
    monkeypatch.setattr(training,'load_curriculum_clock',lambda path:None)
    with pytest.raises(ValueError,match='intact explicit-clock'):
        training.saved_updates('model_200.pt')


def test_normal_joint_training_keeps_speed_yaw_and_full_height_from_saved_clock():
    original=dict(ranges_lin_vel_x=[.2,.8], ranges_reverse_lin_vel_x=[-.8,-.2],
        ranges_transition_lin_vel_x=[-.2,.2], ranges_ang_vel_yaw=[-.3,.3],
        motion_ramp_steps=48000, reverse_ramp_steps=48000, yaw_ramp_steps=48000,
        speed_boundary_fraction=.3, grouped_low_speed_fraction=1/3,
        yaw_min_abs_rate=.15, yaw_boundary_fraction=.2)
    before=copy.deepcopy(original)
    commands=training.normal_joint_commands(original)
    assert original==before
    assert commands['full_height_range_from_start']
    assert tuple(commands['ranges_height'])==training.SAFE_BASE_HEIGHT_RANGE_M
    assert all(commands[key]==0 for key in ('standing_only_steps','reverse_ramp_start_steps','yaw_start_steps'))
    progress=321*48/48000
    speed,yaw=sample_grouped_motion(torch.arange(8192),SimpleNamespace(**commands),
        progress,progress,progress,dedicated_standing_count=2048)
    assert speed.min()<0.<speed.max()
    assert yaw.min()<0.<yaw.max()
    assert not speed[:2048].any() and not yaw[:2048].any()


def test_current_serialized_reward_defaults_are_retained_without_becoming_new_changes():
    original=dict(orientation=-20.,base_height=2.)
    saved=training.reinforced_rewards(original)
    saved.update(tracking_goal_penalty_clip=40.,tracking_goal_component_penalty_clip=200.,
                 wheel_action_saturation_penalty_clip=40.)
    before=copy.deepcopy(saved)
    result=training.retained_joint_rewards(original,saved)
    assert result==saved==before
    result['orientation']=-60.
    assert saved['orientation']==-20.


def test_joint_correction_rejects_unrequested_reward_retuning():
    original=dict(orientation=-20.,base_height=2.)
    saved=training.reinforced_rewards(original)
    saved['orientation']=-60.
    with pytest.raises(ValueError,match='retain the current rewards'):
        training.retained_joint_rewards(original,saved)


@pytest.fixture
def protected_report(report):
    for case in report['conditions']:
        for trial in case['trials']:
            trial['metrics'].update(height_mae_m=.01,body_p95_deg=.5,speed_mae_m_s=.02)
            trial['limits'].update(height_mae_m=.03,body_p95_deg=2.5)
    return report


def test_speed_improvement_does_not_cover_height_loss(protected_report):
    candidate=copy.deepcopy(protected_report)
    trial=candidate['conditions'][1]['trials'][0]
    trial['metrics'].update(speed_mae_m_s=.001,height_mae_m=.02)
    failed=training.height_posture_regressions(candidate,protected_report)
    assert len(failed)==1 and failed[0]['metric']=='height_mae_m'


def test_speed_improvement_does_not_cover_body_tilt_loss(protected_report):
    candidate=copy.deepcopy(protected_report)
    candidate['conditions'][1]['trials'][0]['metrics'].update(speed_mae_m_s=.001,body_p95_deg=1.)
    failed=training.height_posture_regressions(candidate,protected_report)
    assert len(failed)==1 and failed[0]['metric']=='body_p95_deg'


def test_retention_noise_slack_is_not_a_relaxed_original_acceptance_limit(protected_report):
    candidate=copy.deepcopy(protected_report)
    for case in protected_report['conditions']:
        for trial in case['trials']: trial['metrics']['body_p95_deg']=2.49
    for case in candidate['conditions']:
        for trial in case['trials']: trial['metrics']['body_p95_deg']=2.5
    assert training.height_posture_regressions(candidate,protected_report)


def test_retention_allows_small_noise_but_never_changes_the_gate(protected_report):
    before=copy.deepcopy(protected_report)
    candidate=copy.deepcopy(protected_report)
    for case in candidate['conditions']:
        for trial in case['trials']: trial['metrics'].update(height_mae_m=.012,body_p95_deg=.6)
    assert training.height_posture_regressions(candidate,protected_report)==[]
    assert protected_report==before
    assert candidate['conditions'][0]['trials'][0]['limits']==before['conditions'][0]['trials'][0]['limits']


def test_best_height_and_posture_are_protected_independently(protected_report):
    candidate=copy.deepcopy(protected_report)
    candidate['conditions'][0]['trials'][0]['metrics'].update(height_mae_m=.005,body_p95_deg=.7)
    result=training.update_protected_height_posture(protected_report,candidate)
    trial=result['conditions'][0]['trials'][0]
    assert trial['metrics']['height_mae_m']==.005
    assert trial['metrics']['body_p95_deg']==.5
    assert protected_report['conditions'][0]['trials'][0]['metrics']['height_mae_m']==.01
    assert result['reference_only_not_an_accepted_policy'] and not result['policy_accepted']


@pytest.mark.parametrize('field,value',[('survived',False),('metrics',None)])
def test_retention_rejects_first_episode_failure(protected_report,field,value):
    candidate=copy.deepcopy(protected_report)
    candidate['conditions'][-1]['trials'][-1][field]=value
    assert training.height_posture_regressions(candidate,protected_report)


@pytest.mark.parametrize('change',['case','height','seed'])
def test_retention_requires_matched_cases_and_independent_seeds(protected_report,change):
    candidate=copy.deepcopy(protected_report)
    if change=='case': candidate['conditions'][0]['name']='different'
    if change=='height': candidate['conditions'][0]['height']=.31
    if change=='seed': candidate['conditions'][0]['trials'][0]['seed']=84
    with pytest.raises(ValueError,match='comparison'):
        training.height_posture_regressions(candidate,protected_report)
