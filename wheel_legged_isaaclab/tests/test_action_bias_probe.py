import json
from copy import deepcopy

import pytest
import torch
from types import SimpleNamespace

from wheel_legged_gym_isaaclab.action_bias_probe import build_bias_cases, add_action_bias, command_probe_cases, command_probe_plan, wheel_reference_probe


def specification():
    return dict(cases=['stand'], probes=[
        dict(name='unchanged', bias=[0.] * 6),
        dict(name='wheel_difference', bias=[0., 0., -.02, 0., 0., .02])])


def test_paired_control_keeps_command_and_actor_output_unchanged():
    spec = specification(); before = deepcopy(spec)
    cases, biases, labels = build_bias_cases([('stand', 0., 0., .3), ('forward', .6, 0., .3)], spec)
    assert [c[1:] for c in cases] == [(0., 0., .3), (0., 0., .3)]
    means = torch.arange(12, dtype=torch.float32).reshape(2, 6) / 10
    original = means.clone(); applied = add_action_bias(means, torch.tensor(biases))
    torch.testing.assert_close(applied[0], original[0], rtol=0, atol=0)
    torch.testing.assert_close(means, original, rtol=0, atol=0)
    torch.testing.assert_close(applied[1] - original[1], torch.tensor(biases[1]))
    assert spec == before and labels[0]['probe'] == 'unchanged'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), .1, -.1])
def test_unbounded_or_nonfinite_intervention_is_rejected(bad):
    spec = specification(); spec['probes'][1]['bias'][0] = bad
    with pytest.raises(ValueError):
        build_bias_cases([('stand', 0., 0., .3)], spec)


def test_missing_unmodified_control_cannot_be_used():
    spec = specification(); spec['probes'][0]['bias'][0] = .01
    with pytest.raises(ValueError, match='unchanged'):
        build_bias_cases([('stand', 0., 0., .3)], spec)


def test_diagnostic_manifest_is_rejected_by_full_acceptance(tmp_path):
    from summarize_mine_trial import assess
    (tmp_path/'manifest.json').write_text(json.dumps({'policy_acceptance_test': False}))
    with pytest.raises(ValueError, match='cannot count as policy acceptance'):
        assess(tmp_path)


def test_training_boundary_commands_are_diagnostic_and_cannot_exceed_distribution():
    cfg = SimpleNamespace(ranges_reverse_lin_vel_x=(-.8,-.2), ranges_lin_vel_x=(.2,.8),
                          ranges_ang_vel_yaw=(-.3,.3), ranges_height=(.28,.32))
    case = dict(name='fast_low', speed=.8, yaw=0., height=.28)
    assert command_probe_cases([case], cfg) == [('fast_low', .8, 0., .28)]
    for key, value in [('speed',.81), ('yaw',.31), ('height',.27), ('speed',float('nan'))]:
        with pytest.raises(ValueError, match='outside training range'):
            command_probe_cases([{**case, key:value}], cfg)


def test_command_sequence_preserves_training_period_and_case_identities():
    cfg = SimpleNamespace(ranges_reverse_lin_vel_x=(-.8,-.2), ranges_lin_vel_x=(.2,.8),
                          ranges_ang_vel_yaw=(-.3,.3), ranges_height=(.28,.32), resampling_time=5.)
    forward = dict(name='switching', speed=.8, yaw=0., height=.28)
    reverse = dict(name='switching', speed=-.8, yaw=.3, height=.32)
    spec = dict(phase_seconds=5., phases=[[forward], [reverse]])
    initial, sequence, period = command_probe_plan(spec, cfg)
    assert initial == [('switching', .8, 0., .28)] and sequence[1][0] == (-.8, .3, .32) and period == 5.
    with pytest.raises(ValueError, match='period'):
        command_probe_plan({**spec, 'phase_seconds': .1}, cfg)
    with pytest.raises(ValueError, match='identities'):
        command_probe_plan({**spec, 'phases': [[forward], [{**reverse, 'name': 'different'}]]}, cfg)


def test_wheel_probe_preserves_leg_actions_and_original_checkpoint_requests():
    raw = torch.tensor([[.2, -.3, 1.4, -.2, .3, 1.7]])
    original = raw.clone(); applied = raw.clamp(-1., 1.)
    torch.testing.assert_close(wheel_reference_probe(applied, raw, 1.), applied, rtol=0, atol=0)
    expanded = wheel_reference_probe(applied, raw, 1.5)
    torch.testing.assert_close(expanded[:, [0,1,3,4]], applied[:, [0,1,3,4]], rtol=0, atol=0)
    torch.testing.assert_close(expanded[:, [2,5]], torch.tensor([[1.4,1.5]]))
    torch.testing.assert_close(raw, original, rtol=0, atol=0)
