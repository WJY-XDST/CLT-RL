import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import finish_joint_pd_native_audit as finish


def records():
    step = dict(actor_gradient_norm_after_global_clip=.9, critic_gradient_norm_after_global_clip=.1,
        all_parameter_gradient_norm_after_global_clip=1., adaptive_learning_rate=1.e-5,
        height_reference_loss_change=-.5,
        before=dict(height_reference_normalized_squared_error=2.),
        after=dict(height_reference_normalized_squared_error=1.5))
    return [dict(update=number, total_environments=8192, native_alignment_recorded=True,
        read_only_hooks=True, added_loss_or_backward=False, probe_RNG_unchanged=True,
        all_environment_alignment_max_errors={key:0. for key in finish.ALIGNMENT_KEYS},
        optimizer_steps=[copy.deepcopy(step) for _ in range(20)]) for number in (1, 2)]


def test_paired_probes_are_not_cross_minibatch_learning_or_acceptance():
    report = finish.summarize_records(records())
    assert report['optimizer_minibatches'] == 40
    assert report['fixed_input_reference_proxy']['improved_minibatch_fraction'] == 1.
    assert report['fixed_input_reference_proxy']['mean_normalized_squared_error_change'] == -.5
    assert report['policy_accepted'] is False
    assert report['independent_tracking_acceptance_not_proven_by_audit']


@pytest.mark.parametrize('kind', ['short', 'wrong_order', 'subset', 'added_loss', 'rng',
    'missing_alignment', 'misaligned', 'nan_alignment', 'missing_steps', 'nan_step', 'wrong_delta'])
def test_rejects_incomplete_or_inconsistent_native_evidence(kind):
    data = records()
    if kind == 'short': data.pop()
    elif kind == 'wrong_order': data.reverse()
    elif kind == 'subset': data[0]['total_environments'] = 32
    elif kind == 'added_loss': data[0]['added_loss_or_backward'] = True
    elif kind == 'rng': data[0]['probe_RNG_unchanged'] = False
    elif kind == 'missing_alignment': data[0]['all_environment_alignment_max_errors'].pop(finish.ALIGNMENT_KEYS[0])
    elif kind == 'misaligned': data[0]['all_environment_alignment_max_errors'][finish.ALIGNMENT_KEYS[0]] = .1
    elif kind == 'nan_alignment': data[0]['all_environment_alignment_max_errors'][finish.ALIGNMENT_KEYS[0]] = float('nan')
    elif kind == 'missing_steps': data[0]['optimizer_steps'].pop()
    elif kind == 'nan_step': data[0]['optimizer_steps'][0]['adaptive_learning_rate'] = float('nan')
    elif kind == 'wrong_delta': data[0]['optimizer_steps'][0]['height_reference_loss_change'] = .5
    with pytest.raises(ValueError):
        finish.summarize_records(data)


@pytest.mark.parametrize('state,pid', [('active', '123'), ('inactive', '123'), ('activating', '0')])
def test_no_finalization_while_native_service_may_be_live(monkeypatch, state, pid):
    monkeypatch.setattr(finish.subprocess, 'check_output',
        lambda *a, **kw: f'ActiveState={state}\nMainPID={pid}\nResult=success\n')
    with pytest.raises(ValueError, match='terminal'):
        finish.terminal_training_service('clt-joint-pd-native-alignment-test.service')


def test_terminal_evidence_is_read_only_and_requires_exact_owned_service(monkeypatch):
    monkeypatch.setattr(finish.subprocess, 'check_output',
        lambda *a, **kw: 'ActiveState=inactive\nMainPID=0\nResult=success\n')
    assert finish.terminal_training_service('clt-joint-pd-native-alignment-test.service')['MainPID'] == '0'
    with pytest.raises(ValueError, match='exact owned'):
        finish.terminal_training_service('unrelated.service')
