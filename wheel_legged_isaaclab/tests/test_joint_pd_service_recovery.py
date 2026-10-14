"""Regression evidence: distinguish process loss, diagnosis and explicit pause."""
import pytest

from watch_joint_pd_continuation_service import decision
from stop_joint_pd_continuation_service import stop_reason


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
