"""Retain learning feedback for simultaneous failures of the unchanged gate."""
import copy
import hashlib
import json
import math

import pytest
import torch

from test_tracking_goal_feedback import physical_batch
from joint_pd_continuation import continuation_command, optimizer_step, validate_continuation_config
from joint_pd_optimization import prepared_reward_trial, validate_changes
from wheel_legged_gym_isaaclab.tracking_goal_feedback import (
    load_tracking_goal_criteria, tracking_goal_violation_penalty,
    tracking_goal_component_violation_penalty,
)


def test_larger_wheel_error_does_not_erase_height_feedback():
    criteria, _ = load_tracking_goal_criteria()
    values = physical_batch()
    height = torch.tensor([.345], dtype=torch.float64, requires_grad=True)
    wheels = torch.tensor([[15., 12.]], dtype=torch.float64, requires_grad=True)
    values[3], values[9] = height, wheels
    worst = tracking_goal_violation_penalty(*values, criteria)
    components = tracking_goal_component_violation_penalty(*values, criteria)
    old_height, old_wheel = torch.autograd.grad(worst, (height, wheels), retain_graph=True)
    new_height, new_wheel = torch.autograd.grad(components, (height, wheels))
    assert worst.item() == pytest.approx(3.5)
    assert components.item() == pytest.approx(3.625)
    assert old_height.item() == 0.
    assert new_height.item() == pytest.approx(1. / .06)
    assert torch.equal(old_wheel, new_wheel)


@pytest.mark.parametrize('metric', ['speed', 'yaw', 'height', 'length', 'angle', 'wheels', 'body'])
def test_each_unchanged_gate_retains_its_own_feedback(metric):
    criteria, _ = load_tracking_goal_criteria()
    values = physical_batch()
    index, delta = {'speed': (1, .12), 'yaw': (2, .06), 'height': (3, .06),
                    'length': (5, .04), 'angle': (7, .04), 'wheels': (9, 2.)}.get(metric, (None, None))
    if index is not None:
        values[index].flatten()[0] += delta
    else:
        values[4][0] = torch.tensor([math.sin(math.radians(5.)), 0., -math.cos(math.radians(5.))])
    assert tracking_goal_component_violation_penalty(*values, criteria).item() == pytest.approx(.5, abs=1.e-6)


def test_passing_components_cannot_cancel_a_failed_component_and_zero_commands_are_finite():
    criteria, _ = load_tracking_goal_criteria()
    values = physical_batch()
    assert tracking_goal_component_violation_penalty(*values, criteria).item() == 0.
    values[3] += .09
    failed = tracking_goal_component_violation_penalty(*values, criteria)
    values[1] -= .03; values[2] -= .01; values[9] -= .1
    assert torch.equal(failed, tracking_goal_component_violation_penalty(*values, criteria))
    values = physical_batch(); values[0][:, :2] = 0.; values[1].zero_(); values[2].zero_()
    values[9].zero_(); values[10].zero_()
    assert tracking_goal_component_violation_penalty(*values, criteria).item() == 0.


@pytest.mark.parametrize('value', [0., .1, -1.01, float('nan'), float('inf'), True])
def test_unqualified_component_reward_is_rejected(tmp_path, value):
    path = tmp_path / 'change.json'
    path.write_text(json.dumps({'rewards.tracking_goal_component_violation': value}))
    with pytest.raises(ValueError):
        prepared_reward_trial(path)


def test_component_change_preserves_checkpoint_adam_and_all_existing_control(tmp_path):
    path = tmp_path / 'change.json'
    path.write_text(json.dumps({'rewards.tracking_goal_component_violation': -.25}))
    change = prepared_reward_trial(path)
    source = tmp_path / 'model.pt'
    torch.save(dict(iter=3760, model_state_dict={'actor.6.weight': torch.ones(6, 64)},
                    optimizer_state_dict={'state': {0: {'step': torch.tensor(55680.), 'exp_avg': torch.ones(6, 64)}},
                                          'param_groups': [{'lr': .0000759375}]}), source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    commands = dict(height_expansion_from=[.28, .32], height_expansion_start_steps=117024,
                    height_expansion_ramp_steps=48000, ranges_height=[.22534645531, .37534645531])
    previous = dict(commands=commands, rewards=dict(joint_position_action_saturation=-8.,
                    tracking_goal_violation=-.5, tracking_goal_penalty_clip=40.), leg_joint_stiffness=300.)
    argv, final, saved = continuation_command(source, previous, 'component_feedback', 100,
                                             commands, tmp_path / 'STOP', reward_overrides=change,
                                             minimum_wheel_std=.03)
    assert optimizer_step(saved) == 55680 and final == 3859
    assert not any(flag in argv for flag in ('--reset_optimizer', '--reset_exploration_std', '--allow_physics_contract_change'))
    assert 'env.rewards.tracking_goal_component_violation=-0.25' in argv
    current = copy.deepcopy(previous)
    current['rewards'].update(tracking_goal_component_violation=-.25, tracking_goal_component_penalty_clip=200.)
    validate_changes(previous, current, change)
    validate_continuation_config(previous, current, commands, reward_overrides=change)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('check', [validate_changes, validate_continuation_config])
def test_legacy_neutral_defaults_do_not_allow_an_undeclared_reward(check):
    before = dict(commands=dict(height_expansion_from=None, height_expansion_start_steps=0,
                                height_expansion_ramp_steps=48000), rewards=dict(orientation=-60.),
                  leg_joint_stiffness=300.)
    after = copy.deepcopy(before)
    after['rewards'].update(tracking_goal_component_violation=0., tracking_goal_component_penalty_clip=200.)
    if check is validate_changes:
        check(before, after, {})
    else:
        check(before, after, before['commands'])
    after['rewards']['tracking_goal_component_violation'] = -.25
    with pytest.raises(ValueError):
        if check is validate_changes:
            check(before, after, {})
        else:
            check(before, after, before['commands'])
