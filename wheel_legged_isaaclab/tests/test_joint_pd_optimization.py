"""Autonomous PD trials are evidence-driven and cannot alter hardware or hide failures."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'mine_model/scripts'))
import joint_pd_optimization as optimization
from diagnose_policy_feedback import joint_pd_feedback


def report(yaw=.188, speed=.04, posture=.7, survived=True):
    limits = dict(speed_mae_m_s=.025, yaw_mae_rad_s=.025, height_mae_m=.03, body_p95_deg=2.5,
                  leg_length_mae_m=.03, leg_angle_mae_rad=.02, wheel_speed_mae_rad_s=.15)
    metrics = dict(speed_mae_m_s=speed, yaw_mae_rad_s=yaw, height_mae_m=.003, body_p95_deg=posture,
                   leg_length_mae_m=.002, leg_angle_mae_rad=.001, wheel_speed_mae_rad_s=.3)
    trials = [dict(seed=s, metrics=dict(metrics) if survived else None, limits=limits) for s in (83,84,85)]
    return dict(survivors=3 if survived else 0, total_trials=3, resets=0 if survived else 3,
                conditions=[dict(name='stand', trials=trials)])


def test_goal_quality_does_not_reward_low_wheel_error_at_bad_yaw():
    good, bad = report(yaw=.02), report(yaw=.188)
    for t in good['conditions'][0]['trials']:
        t['metrics']['wheel_speed_mae_rad_s'] = .8
    assert optimization.quality(good) > optimization.quality(bad)


def test_proposals_are_single_reward_variables_not_controller_gains():
    cfg = dict(rewards=dict(yaw_rate_error_sq=-1., orientation=-20., lin_vel_error_sq=-2., base_height_sigma=.0004))
    result = optimization.proposals(cfg, report(posture=9.), set())
    assert result[0][2] == {'rewards.yaw_rate_error_sq': -5.}
    assert result[1][2] == {'rewards.orientation': -40.}
    assert all(len(r[2]) == 1 for r in result)


def test_identical_failed_trials_are_not_repeated():
    cfg = dict(rewards=dict(yaw_rate_error_sq=-1., orientation=-20., lin_vel_error_sq=-2., base_height_sigma=.0004))
    tried = {optimization.proposal_signature({'rewards.yaw_rate_error_sq': -5.})}
    result = optimization.proposals(cfg, report(), tried)
    assert result[0][2]['rewards.yaw_rate_error_sq'] == -2.


def test_bounds_do_not_grow_forever_and_exploration_keeps_weights():
    cfg = dict(rewards=dict(yaw_rate_error_sq=-20., orientation=-80., lin_vel_error_sq=-12., base_height_sigma=.0002))
    result = optimization.proposals(cfg, report(yaw=.8, speed=.6, posture=12.), set())
    assert result[0][1:] == ('exploration', {}, .02, False)


def test_collapse_and_cross_task_regression_cannot_win():
    assert not optimization.safe_candidate(report(survived=False), report())
    assert not optimization.safe_candidate(report(yaw=.01, speed=.4), report())
    assert optimization.safe_candidate(report(yaw=.08, speed=.035), report())


def test_learning_branch_can_continue_without_relabeling_retained_model(tmp_path):
    source = report(yaw=.188,speed=.01)
    measured = report(yaw=.02,speed=.035)
    path=tmp_path/'assessment.json';path.write_text(json.dumps(measured))
    candidate={'name':'learning','checkpoint':'model_2296.pt','assessment':str(path)}
    assert not optimization.safe_candidate(measured,source)
    choice,result=optimization.learning_choice([candidate],source,measured)
    assert choice==candidate and result==measured
    assert source['conditions'][0]['trials'][0]['metrics']['yaw_mae_rad_s']==.188


def test_learning_branch_rejects_collapse_regression_and_nonfinite_metrics(tmp_path):
    source=report(yaw=.188,speed=.01);control=report(yaw=.02,speed=.035)
    for value in (report(survived=False),report(yaw=.01,speed=.4),report(yaw=.5,speed=.01)):
        path=tmp_path/'assessment.json';path.write_text(json.dumps(value))
        choice,_=optimization.learning_choice([{'assessment':str(path)}],source,control)
        assert choice is None
    value=report(yaw=.01,speed=.035)
    value['conditions'][0]['trials'][0]['metrics']['speed_mae_m_s']=float('nan')
    path.write_text(json.dumps(value))
    assert optimization.learning_choice([{'assessment':str(path)}],source,control)[0] is None


def test_continuation_target_is_cumulative_not_the_original_start():
    command,final=optimization.training_command(Path('/fixture/model_2296.pt'),'cumulative',200,{})
    assert command[command.index('--checkpoint_path')+1]=='/fixture/model_2296.pt'
    assert command[command.index('--target_iteration')+1]=='2496'
    assert final==2495


def test_learning_source_needs_matched_exact_input_evidence(tmp_path,monkeypatch):
    import hashlib
    checkpoint=tmp_path/'model_1.pt';checkpoint.write_bytes(b'synthetic checkpoint')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (tmp_path/'assessment.json').write_text(json.dumps({'checkpoint_sha256':digest}))
    diagnosis=dict(checkpoint_sha256=digest,actor_inference_parity_max=0.,physical_contract_changed=False,weights_updated=False)
    path=tmp_path/'joint_pd_command_diagnosis.json';path.write_text(json.dumps(diagnosis))
    monkeypatch.setattr(optimization,'RUNS',tmp_path)
    calls=[];monkeypatch.setattr(optimization,'validate_contract',lambda *a,**k:calls.append((a,k)))
    result=dict(checkpoint=str(checkpoint),assessment=str(tmp_path/'assessment.json'))
    optimization.validate_learning_source(result)
    assert calls[-1][1]['fresh_weights'] is False
    diagnosis['actor_inference_parity_max']=float('nan');path.write_text(json.dumps(diagnosis))
    with pytest.raises(ValueError,match='mismatched'):optimization.validate_learning_source(result)


def test_implicit_position_source_and_all_outputs_are_preserved():
    command, final = optimization.training_command(Path('/fixture/model_1999.pt'), 'trial', 50,
                                                   {'rewards.yaw_rate_error_sq': -5.})
    assert final == 2048
    assert command[command.index('--target_iteration')+1] == '2049'
    assert command[command.index('--task')+1] == 'WheelLeggedJointPD-Flat-v0'
    assert command[command.index('--require_leg_control_mode')+1] == 'implicit_joint_position'
    assert '--wheel_head_only_finetune' not in command
    assert '--reset_optimizer' not in command
    assert '--reset_exploration_std' not in command
    with pytest.raises(ValueError, match='admitted reward'):
        optimization.training_command(Path('/fixture/model_1999.pt'), 'trial', 50, {'wheel_damping': 2.})


def test_actual_runtime_allows_only_recorded_reward_and_log_changes():
    source = dict(log_dir='old', commands={'range': [.2,.8]}, rewards={'orientation': -20.}, wheel_damping=.5)
    current = deepcopy(source)
    current.update(log_dir='new', rewards={'orientation': -40.})
    optimization.validate_changes(source, current, {'rewards.orientation': -40.})
    current['wheel_damping'] = 2.
    with pytest.raises(ValueError, match='Non-reward'):
        optimization.validate_changes(source, current, {'rewards.orientation': -40.})


def test_prepared_height_band_cannot_change_physics_or_acceptance(tmp_path):
    path = tmp_path/'trial.json'
    change = {'rewards.base_height_relative_deadband': .05}
    path.write_text(json.dumps(change))
    assert optimization.prepared_reward_trial(path) == change
    assert optimization.reward_value({'rewards': {}}, 'base_height_relative_deadband') == 0.
    for invalid in ({'wheel_damping': 2.}, {'rewards.base_height_relative_deadband': .1},
                    {'rewards.base_height_relative_deadband': .05, 'rewards.orientation': -80.}):
        path.write_text(json.dumps(invalid))
        with pytest.raises(ValueError):
            optimization.prepared_reward_trial(path)


def test_old_checkpoint_height_equation_is_preserved_unless_explicitly_overridden():
    source = dict(rewards={'base_height_sigma': .0004}, wheel_damping=.5)
    current = dict(rewards={'base_height_sigma': .0004, 'base_height_relative_deadband': 0.}, wheel_damping=.5)
    optimization.validate_changes(source, current, {})
    current['rewards']['base_height_relative_deadband'] = .05
    with pytest.raises(ValueError, match='Unexpected reward'):
        optimization.validate_changes(source, current, {})
    optimization.validate_changes(source, current, {'rewards.base_height_relative_deadband': .05})


def test_guard_accepts_removal_of_known_no_ops_but_not_extra_runtime_settings():
    source = dict(commands={'num_commands': 3, 'ranges_height': [.28, .32]},
                  obs_scales={'torque': .05, 'dof_acc': .0025, 'lin_vel': 2.},
                  rewards={'max_contact_force': 100., 'orientation': -20.}, leg_effort_limit=35.574)
    current = dict(commands={'ranges_height': [.28, .32]}, obs_scales={'lin_vel': 2.},
                   rewards={'orientation': -20.}, leg_effort_limit=35.574)
    optimization.validate_changes(source, current, {})
    current['new_unexpected_controller'] = True
    with pytest.raises(ValueError, match='parameter added'):
        optimization.validate_changes(source, current, {})


def test_user_pause_is_not_misreported_as_a_simulation_fault(tmp_path):
    handler = next(n for n in ast.walk(ast.parse(inspect.getsource(optimization.run)))
                   if isinstance(n, ast.ExceptHandler) and isinstance(n.type, ast.Name)
                   and n.type.id == 'BaseException')
    wrapper = ast.parse('def check(error, output, save):\n    try:\n        raise error\n    except BaseException as error:\n        pass\n')
    wrapper.body[0].body[0].handlers = [handler]
    scope = {'stop_outcome': optimization.stop_outcome}
    exec(compile(ast.fix_missing_locations(wrapper), '<actual_pause_handler>', 'exec'), scope)
    (tmp_path/'STOP').touch()
    events = []
    save = lambda state, **values: events.append((state, values))
    scope['check'](RuntimeError('User stop marker received during replay; no automatic thaw'), tmp_path, save)
    assert events[-1][0] == 'paused_user_request'
    assert events[-1][1]['automatic_resume'] is False
    with pytest.raises(RuntimeError, match='physics_fault'):
        scope['check'](RuntimeError('physics_fault'), tmp_path, save)
    assert events[-1][0] == 'diagnosis_required'


def test_agent_checkpoint_diagnosis_does_not_claim_user_paused(tmp_path):
    (tmp_path/'STOP').write_text(json.dumps({'reason': 'agent_diagnosis_for_contact_regression'}))
    phase, details = optimization.stop_outcome(tmp_path)
    assert phase == 'agent_diagnosis_required'
    assert details['user_pause_requested'] is False and details['goal_remains_active'] is True


def test_exact_counterfactual_preserves_input_and_exposes_yaw_bias():
    state = {}
    for i, n, out in ((0,27,27),(2,27,27),(4,27,27),(6,27,6)):
        state[f'actor.{i}.weight'] = torch.zeros(out,n)
        state[f'actor.{i}.bias'] = torch.zeros(out)
    state['actor.6.bias'][5] = .1
    obs = np.zeros((10,3,27), dtype=np.float32)
    obs[:,:,5] = -1.
    means = np.zeros((10,3,6), dtype=np.float32)
    means[:,:,5] = .1
    before = obs.copy()
    cfg = dict(leg_control_mode='implicit_joint_position', action_scale_vel=15.,
               obs_scales=dict(lin_vel=2., ang_vel=.25, dof_vel=.05),
               rewards=dict(orientation=-20., orientation_roll_multiplier=2., yaw_rate_error_sq=-1.,
                            tracking_ang_vel=1., tracking_sigma_ang=.25, orientation_penalty_clip=30.))
    manifest = dict(policy_dt=.02, seeds=[83,84,85], cases=[dict(name='stand',yaw=0.)],
                    checkpoint='fixture.pt', checkpoint_sha256='fixture', checkpoint_iteration=1999)
    result = joint_pd_feedback(obs, means, manifest, cfg, state)
    assert result['actor_inference_parity_max'] == 0.
    assert result['conditions'][0]['command_counterfactual'][1]['right_minus_left_reference_rad_s'] == 1.5
    np.testing.assert_array_equal(obs,before)
    means[:,:,5] = .5
    with pytest.raises(ValueError, match='does not reproduce'):
        joint_pd_feedback(obs, means, manifest, cfg, state)
