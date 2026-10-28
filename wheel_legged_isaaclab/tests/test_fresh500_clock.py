"""Fresh full-range commands and bounded, non-resetting continuation."""
from types import SimpleNamespace

import pytest
import torch
import yaml

from wheel_legged_gym_isaaclab.training_state import (
    make_curriculum_clock, load_curriculum_clock, clocked_update_count,
    save_random_initial_checkpoint,
)
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M, training_height_bounds
from joint_pd_continuation import continuation_command
from train_joint_pd_fresh500 import (
    fresh_command, update_experiment_state, next_replay_target,
    TOTAL_UPDATES, BLOCK_UPDATES, FIRST_REPLAY_UPDATES,
)


def saved(step=1000):
    return dict(iter=49, model_state_dict={'actor.6.weight': torch.ones(6, 64)},
        optimizer_state_dict={'state': {0: {'step': torch.tensor(float(step))}}, 'param_groups': [{'lr': .0003}]})


@pytest.mark.parametrize('offset,steps,adam', [(-1, 48, 20), (0, 0, 20), (True, 48, 20), (0, 48, .5)])
def test_clock_rejects_invalid_offsets_or_rates(offset, steps, adam):
    with pytest.raises(ValueError): make_curriculum_clock(offset, steps, adam)


def test_new_clock_counts_complete_updates_not_old_loop_label(tmp_path):
    checkpoint = tmp_path / 'model_49.pt'
    torch.save(saved(), checkpoint)
    clock = make_curriculum_clock(276096, 48, 20)
    (tmp_path / 'params').mkdir()
    (tmp_path / 'params/curriculum_clock.yaml').write_text(yaml.safe_dump(clock))
    assert load_curriculum_clock(checkpoint) == clock
    assert clocked_update_count(saved(), clock) == 50
    command, final, _ = continuation_command(checkpoint, {'rewards': {}}, 'clock', 50, {}, tmp_path / 'STOP')
    assert final == 99 and command[command.index('--target_iteration') + 1] == '100'
    assert '--reset_optimizer' not in command


@pytest.mark.parametrize('step', [0, 999, float('nan')])
def test_clock_refuses_partial_or_nonfinite_adam(step):
    with pytest.raises(ValueError): clocked_update_count(saved(step), make_curriculum_clock(276096, 48, 20))


def test_fresh_command_has_no_old_checkpoint_and_preserves_current_reward(tmp_path):
    checkpoint = tmp_path / 'model_5751.pt'
    torch.save(saved(), checkpoint)
    cfg = {'rewards': {'orientation': -60., 'base_height_relative_deadband': .05},
           'commands': {'ranges_height': list(SAFE_BASE_HEIGHT_RANGE_M)}}
    command = fresh_command(checkpoint, cfg, 'fresh', tmp_path / 'STOP', 276096)
    assert '--checkpoint_path' not in command and '--reset_optimizer' not in command
    assert command[command.index('--target_iteration') + 1] == '200'
    assert command[command.index('--curriculum_step_offset') + 1] == '276096'
    assert '--save_initial_checkpoint' in command
    assert 'env.rewards.orientation=-60.0' in command
    assert TOTAL_UPDATES == 500 and BLOCK_UPDATES == 50 and FIRST_REPLAY_UPDATES == 200


def test_first_fresh_reset_uses_current_full_range_not_early_narrow_curriculum():
    commands = SimpleNamespace(ranges_height=SAFE_BASE_HEIGHT_RANGE_M, height_expansion_from=[.28, .32],
        height_expansion_start_steps=117024, height_expansion_ramp_steps=48000)
    assert training_height_bounds(commands, .3, 276096, 1.) == SAFE_BASE_HEIGHT_RANGE_M
    assert training_height_bounds(commands, .3, 0, 0.) == (.3, .3)


def test_clock_remains_opt_in_for_old_models(tmp_path):
    assert load_curriculum_clock(tmp_path / 'model_5751.pt') is None


def test_inconsistent_or_missing_adam_states_are_not_accepted():
    bad = saved()
    bad['optimizer_state_dict']['state'][1] = {'step': torch.tensor(999.)}
    with pytest.raises(ValueError): clocked_update_count(bad, make_curriculum_clock(1, 48, 20))
    bad['optimizer_state_dict']['state'][1] = {}
    with pytest.raises(ValueError): clocked_update_count(bad, make_curriculum_clock(1, 48, 20))


def test_empty_adam_requires_explicit_initial_snapshot_recovery():
    initial = saved()
    initial['iter'] = 0
    initial['optimizer_state_dict']['state'] = {}
    clock = make_curriculum_clock(276096, 48, 20)
    with pytest.raises(ValueError): clocked_update_count(initial, clock)
    assert clocked_update_count(initial, clock, allow_initial=True) == 0
    initial['iter'] = 1
    with pytest.raises(ValueError): clocked_update_count(initial, clock, allow_initial=True)


def test_initial_snapshot_initializes_rsl_writer_before_save_and_never_overwrites(tmp_path):
    events = []
    runner = SimpleNamespace(_prepare_logging_writer=lambda: events.append('writer'),
                             save=lambda path: events.append(('save', path)))
    path = tmp_path / 'model_initial.pt'
    save_random_initial_checkpoint(runner, path)
    assert events == ['writer', ('save', str(path))]
    path.touch()
    with pytest.raises(FileExistsError): save_random_initial_checkpoint(runner, path)


def test_initial_recovery_loads_the_preserved_random_weights_without_reset_flags(tmp_path):
    initial = saved()
    initial['iter'] = 0
    initial['optimizer_state_dict']['state'] = {}
    path = tmp_path / 'model_initial.pt'
    torch.save(initial, path)
    (tmp_path / 'params').mkdir()
    (tmp_path / 'params/curriculum_clock.yaml').write_text(yaml.safe_dump(make_curriculum_clock(276096, 48, 20)))
    command, final, _ = continuation_command(path, {'rewards': {}}, 'recover_initial', 50, {}, tmp_path / 'STOP')
    assert command[command.index('--checkpoint_path') + 1] == str(path)
    assert command[command.index('--target_iteration') + 1] == '50' and final == 49
    assert '--reset_optimizer' not in command and '--save_initial_checkpoint' not in command


def test_terminal_and_error_reporting_allow_explicit_training_inactive():
    for phase in ('comparison_500_updates_complete', 'agent_diagnosis_required'):
        state = update_experiment_state({}, phase, training_active=False, actual_ppo_updates=500)
        assert state['state'] == phase and state['training_active'] is False
        assert state['live_ppo_updates'] == 500 and state['updated_at']


def test_live_progress_is_not_confused_with_last_verified_fifty_update_milestone():
    state = {'actual_ppo_updates': 50}
    update_experiment_state(state, 'training', latest_update={'iteration': 64})
    assert state['live_ppo_updates'] == 65 and state['actual_ppo_updates'] == 50


@pytest.mark.parametrize('actual,target', [(0, 200), (4, 200), (50, 200), (62, 200),
    (100, 200), (150, 200), (199, 200), (200, 250), (250, 300), (499, 500), (500, 500)])
def test_no_performance_replay_before_200_even_after_recovery(actual, target):
    assert next_replay_target(actual) == target


@pytest.mark.parametrize('actual', [-1, 501, True, 50.5])
def test_replay_schedule_rejects_out_of_scope_update_counts(actual):
    with pytest.raises(ValueError): next_replay_target(actual)
