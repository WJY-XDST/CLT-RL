import ast
from pathlib import Path
from types import SimpleNamespace

import torch

from wheel_legged_gym_isaaclab.training_state import aggregate_episode_logs, install_episode_log_aggregation


def test_end_events_are_counted_once_instead_of_averaging_different_sized_groups():
    key = 'Episode_Termination/forward_fast_unsafe'
    records = [{key: torch.ones(1)}, {key: torch.zeros(99)}]
    result = aggregate_episode_logs(records)
    assert abs(float(result[key]) - .01) < 1.e-8  # Historical mean of step means was .5.
    assert result['Episode_Count/forward_fast_ended'] == 100
    assert result['Episode_Count/forward_fast_unsafe'] == 1


def test_sparse_group_after_first_step_is_retained_and_absent_groups_do_not_make_nan():
    key = 'Episode_Termination/reverse_fast_unsafe'
    result = aggregate_episode_logs([{'Tracking/speed': .2}, {key: torch.zeros(0)},
                                    {'Tracking/speed': .4, key: torch.tensor([1., 0.])}])
    assert result[key] == .5
    torch.testing.assert_close(result['Tracking/speed'], torch.tensor(.3))
    assert aggregate_episode_logs([{key: torch.zeros(0)}]) == {}


def test_rare_numerical_event_survives_four_decimal_log_rounding():
    events = torch.zeros(100000)
    events[99999] = 1.
    result = aggregate_episode_logs([{'Episode_Termination/numerical': events}])
    assert f"{float(result['Episode_Termination/numerical']):.4f}" == '0.0000'
    assert result['Episode_Count/numerical'] == 1
    from guard_mcu_vmc_training import anomaly
    from watch_joint_pd_scratch import numerical_anomaly
    update = {'metrics': {'Episode_Termination/numerical': 0., 'Episode_Count/numerical': 1.}}
    assert anomaly([update], '') == 'numerical_termination'
    assert numerical_anomaly([update], '') == 'numerical_termination'


def test_logger_adapter_preserves_rollout_list_and_original_records():
    captured = []
    runner = SimpleNamespace(log=lambda value: captured.append(value))
    original = [{'a': torch.tensor([1., 2.])}, {'a': torch.tensor([6.])}]
    locs = dict(ep_infos=original, loss_dict={'value': .2})
    install_episode_log_aggregation(runner)
    runner.log(locs)
    assert locs['ep_infos'] is original and len(original) == 2
    assert captured[0]['ep_infos'][0]['a'] == 3.
    assert captured[0]['loss_dict'] is locs['loss_dict']


def production_method(name):
    path = Path(__file__).resolve().parents[1]/'wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py'
    return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == name)


def test_next_physics_step_does_not_reuse_previous_episode_events_or_change_actions():
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[production_method('_pre_physics_step')], type_ignores=[]), '<production>', 'exec'), scope)
    old = {'Episode_Termination/forward_fast_unsafe': torch.ones(1)}
    env = SimpleNamespace(extras={'log': old})
    actions = torch.tensor([[0., .5, -.5, 2., -2., 0.]])
    scope['_pre_physics_step'](env, actions)
    assert env.extras['log'] == {} and old['Episode_Termination/forward_fast_unsafe'] == 1
    torch.testing.assert_close(env._actions, actions.clamp(-1., 1.))


def test_partial_command_resample_updates_only_its_own_global_timestamps():
    scope = {'resample_commands': lambda env, ids: None}
    exec(compile(ast.Module(body=[production_method('_resample_commands_for')], type_ignores=[]), '<production>', 'exec'), scope)
    env = SimpleNamespace(common_step_counter=126234, _command_resample_step=torch.tensor([126000,126000,126000]))
    scope['_resample_commands_for'](env, torch.tensor([1]))
    torch.testing.assert_close(env._command_resample_step, torch.tensor([126000,126234,126000]))
