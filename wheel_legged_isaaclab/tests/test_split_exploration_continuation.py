import json

import pytest
import yaml

import continue_joint_pd_split_exploration as controller


def test_native_command_preserves_resume_and_physics_but_keeps_wheel_entropy_at_baseline():
    command=['python','train.py','--checkpoint_path','latest.pt','--num_envs','8192',
             'env.leg_control_mode="implicit_joint_position"']
    before=command.copy()
    revised=controller.split_entropy_command(command,.005,'profile.json')
    assert command==before
    assert revised[:2]==['python',str(controller.ENTRY)]
    assert revised[revised.index('--')+1:-2]==before[2:]
    assert revised[-2:] == ['agent.algorithm.entropy_coef=0.005',
                           'agent.policy.class_name=JointPdExplorationActorCritic']
    assert controller.entropy_multipliers(.005)==[1.,1.,.2,1.,1.,.2]


def test_same_survival_and_height_cannot_hide_a_large_standing_drift_regression():
    history=[dict(survivors=21,height_error_fraction=.315,
                  measured_case_metric_ratios={'stand/speed_mae_m_s':.31}),
             dict(survivors=21,height_error_fraction=.318,
                  measured_case_metric_ratios={'stand/speed_mae_m_s':3.86})]
    assert controller.next_entropy(history,.005)==.0025


def test_small_fluctuations_and_subthreshold_metrics_do_not_change_objective():
    history=[dict(survivors=21,height_error_fraction=.315,
                  measured_case_metric_ratios={'stand/speed_mae_m_s':.31}),
             dict(survivors=21,height_error_fraction=.314,
                  measured_case_metric_ratios={'stand/speed_mae_m_s':.8})]
    assert controller.next_entropy(history,.005)==.005


def report():
    names=['stand','forward_0p3','forward_0p6','reverse_0p3','reverse_0p6',
           'spin_left','spin_right','height_low','height_high']
    cases=[]
    for name in names:
        height=.225 if name=='height_low' else .375 if name=='height_high' else .3
        metrics={key:.01 for key in controller.METRICS}
        metrics['height_mae_m']=.07
        limits={key:.1 for key in controller.METRICS}
        trials=[dict(seed=seed,survived=name not in ('forward_0p6','reverse_0p6'),
                     metrics=metrics.copy(),limits=limits.copy()) for seed in (83,84,85)]
        cases.append(dict(name=name,height=height,trials=trials))
    return dict(total_trials=27,survivors=21,resets=30,passed=False,conditions=cases)


def test_failed_trials_remain_missing_in_complete_review():
    result=controller.review_summary(report())
    assert result['missing_steady_trials']==6
    assert not result['full_gate_passed']
    assert 'forward_0p6/height_mae_m' not in result['measured_case_metric_ratios']


def test_pilot_automatically_continues_latest_checkpoint_with_normal_review_interval(tmp_path,monkeypatch):
    results=tmp_path/'mine_model/results';results.mkdir(parents=True)
    base=results/'coordinator';previous=results/'source'
    def boundary(path,actual):
        (path/'replay').mkdir(parents=True)
        (path/'params').mkdir()
        (path/'status.json').write_text(json.dumps(dict(checkpoint=str(path/'model.pt'),actual_ppo_updates=actual)))
        (path/'experiment_plan.json').write_text('{}')
        (path/'replay/assessment.json').write_text(json.dumps(report()))
        (path/'params/agent.yaml').write_text(yaml.safe_dump(dict(
            policy=dict(class_name='JointPdExplorationActorCritic'),
            algorithm=dict(class_name='PPO',entropy_coef=.005))))
        (path/'native_entropy_profile.json').write_text(json.dumps(dict(
            leg_entropy_coefficient=.005,wheel_entropy_coefficient=.001,
            dimension_multipliers=[1.,1.,.2,1.,1.,.2],
            modified_sampling_or_log_prob=False,modified_inference=False,
            additional_parameters=0,additional_checkpoint_keys=0)))
    boundary(previous,601)
    calls=[]
    def native_block(args,prepare_function):
        calls.append(args)
        boundary(args.output,601+sum(a.updates for a in calls))
        if len(calls)==2:(base/'STOP').touch()
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    monkeypatch.setattr(controller,'ORIGINAL_BLOCK_RUN',native_block)
    monkeypatch.setattr(controller.core.block,'continuation_command',controller.core.block.continuation_command)
    controller.run(previous,base,'owned.service')
    assert [a.updates for a in calls]==[25,100]
    assert calls[1].previous==calls[0].output
    assert json.loads((base/'coordinator.json').read_text())['state']=='explicit_stop_respected'
