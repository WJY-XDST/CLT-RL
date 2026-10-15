"""Regression evidence: distinguish process loss, diagnosis and explicit pause."""
import pytest

from watch_joint_pd_continuation_service import decision,stagnation_plan
import json
import os

import stop_joint_pd_continuation_service as stop_service
from stop_joint_pd_continuation_service import stop_reason, stop_owned_controller


@pytest.mark.parametrize('phase', ['training', 'training_initialization', 'checkpoint_evaluation',
                                 'block_evaluated', 'best_measured_checkpoint_saved'])
def test_missing_service_in_an_unfinished_run_recovers(phase):
    assert decision({'ActiveState':'inactive'}, {'state':phase}, {}) == 'recover_verified_checkpoint'


def test_live_training_does_not_spawn_duplicate():
    assert decision({'ActiveState':'active'}, {'state':'training'}, {}) == 'service_running'


@pytest.mark.parametrize('phase', ['agent_diagnosis_required', 'diagnosis_required', 'error'])
def test_diagnosed_fault_is_not_silently_restarted(phase):
    assert decision({'ActiveState':'failed'}, {'state':phase}, {}) == 'diagnosis_required'


def test_saved_stop_for_known_host_shutdown_is_recoverable():
    assert decision({}, {'state':'paused_user_request'}, {'reason':'host_shutdown'}) == 'recover_verified_checkpoint'


def test_generic_service_stop_may_be_manual_and_must_not_be_overridden():
    assert decision({}, {'state':'training'}, {'reason':'service_stop_requested'}) == 'explicit_user_pause'


def test_explicit_pause_blocks_automatic_recovery():
    assert decision({}, {'state':'training'}, {}, paused=True) == 'explicit_user_pause'


def test_unknown_manual_stop_is_not_misclassified_as_reboot():
    assert decision({}, {'state':'paused_user_request'}, {'reason':'user_pause'}) == 'explicit_user_pause'


def test_saved_diagnosis_marker_blocks_restart_even_with_stale_status():
    assert decision({}, {'state':'training'}, {'reason':'agent_diagnosis_numerical'}) == 'diagnosis_required'


def test_passed_gate_requires_delivery_verification():
    assert decision({}, {'state':'simulation_accepted_pending_sim2sim_and_randomization',
                         'all_errors_under_10_percent':True}, {}) == 'verify_delivery'


def test_unknown_state_needs_inspection():
    assert decision({}, {'state':'something_new'}, {}) == 'inspect_unknown_state'


def test_only_confirmed_global_shutdown_creates_recoverable_stop():
    assert stop_reason('stopping\n') == 'host_shutdown'
    for state in ('running', 'degraded', 'offline', 'unknown', ''):
        assert stop_reason(state) == 'service_stop_requested'


def test_measured_plateau_has_diagnosed_continuation_instead_of_a_round_limit():
    state=dict(state='agent_diagnosis_required',reason='three_independent_tracking_milestones_stagnant')
    assert decision({},state,{})=='diagnose_stagnation_and_continue'
    assert decision({},state,{},paused=True)=='explicit_user_pause'


def test_only_stalled_height_adapts_its_bounded_training_coefficient():
    report=dict(survivors=27,total_trials=27,resets=0,passed=False)
    plan=stagnation_plan(dict(stagnant_metrics=['height_mae_m']),report,.1)
    assert plan['height_coefficient_after']==pytest.approx(.15)
    assert not plan['weights_reset'] and not plan['optimizer_reset']
    assert not plan['control_or_actuator_values_changed']
    assert stagnation_plan(dict(stagnant_metrics=['height_mae_m']),report,.225)['height_coefficient_after']==.25
    assert stagnation_plan(dict(stagnant_metrics=['wheel_speed_mae_rad_s']),report,.1)['height_coefficient_after']==.1


def test_failed_survival_is_not_reclassified_as_an_ordinary_plateau():
    with pytest.raises(ValueError):
        stagnation_plan(dict(stagnant_metrics=['height_mae_m']),
                        dict(survivors=26,total_trials=27,resets=1,passed=False),.1)


@pytest.mark.parametrize('phase', ['agent_diagnosis_required','training','checkpoint_evaluation'])
def test_post_exit_hook_does_not_create_a_false_manual_pause(tmp_path,monkeypatch,phase):
    monkeypatch.setattr(stop_service,'ROOT',tmp_path)
    output=tmp_path/'mine_model/results/joint_pd_optimization_test'
    output.mkdir(parents=True)
    (output/'.controller.lock').touch()
    state=dict(state=phase,reason='three_independent_tracking_milestones_stagnant')
    (output/'status.json').write_text(json.dumps(state))
    monkeypatch.setattr(stop_service,'request_checkpoint_stop',lambda _:pytest.fail('Post-exit hook must not request a stop'))
    assert stop_owned_controller(output,None)['action']=='main_process_already_exited'
    assert not (output/'STOP').exists()
    assert json.loads((output/'status.json').read_text())==state


def test_live_manual_stop_still_saves_and_preserves_pause(tmp_path,monkeypatch):
    monkeypatch.setattr(stop_service,'ROOT',tmp_path)
    output=tmp_path/'mine_model/results/joint_pd_optimization_test'
    output.mkdir(parents=True)
    (output/'.controller.lock').touch()
    (output/'status.json').write_text(json.dumps(dict(state='training')))
    monkeypatch.setattr(stop_service.subprocess,'run',lambda *a,**k:type('Result',(),{'stdout':'running\n'})())
    calls=[]
    monkeypatch.setattr(stop_service,'request_checkpoint_stop',lambda path:calls.append(path))
    result=stop_owned_controller(output,str(os.getpid()))
    assert result['action']=='cooperative_stop_requested' and result['stop_marker_created']
    marker=json.loads((output/'STOP').read_text())
    assert marker==dict(reason='service_stop_requested',checkpoint_before_exit=True)
    assert calls==[output]
    assert decision({},dict(state='training'),marker)=='explicit_user_pause'
