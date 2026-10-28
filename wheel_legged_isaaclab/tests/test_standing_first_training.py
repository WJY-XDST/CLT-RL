"""CPU-only checks for the fresh, single-chain standing-first experiment."""
import copy
import json

import pytest
import torch
import yaml

import train_joint_pd_standing_first as experiment
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M


def assessment(*, survives=True, error=.001):
    conditions = []
    for name, height in zip(experiment.STAND_CASES, (.3, *SAFE_BASE_HEIGHT_RANGE_M)):
        trials = [dict(seed=s, survived=survives, first_failure_s=.5,
            metrics={key:error for key in (*experiment.READINESS_METRICS, 'height_mae_m')} if survives else None,
            limits={key:.025 if key!='body_p95_deg' else 2.5 for key in experiment.READINESS_METRICS})
            for s in (83, 84, 85)]
        conditions.append(dict(name=name, height=height, survivors=3 if survives else 0, trials=trials))
    return dict(conditions=conditions)


def test_historical_simple_rewards_drop_late_precision_but_retain_physical_balance():
    original = dict(orientation=-60., yaw_rate_error_sq=-10., joint_position_action_saturation=-8.,
        wheel_action_saturation=-2., tracking_goal_violation=-.5,
        tracking_goal_component_violation=-.75, base_height_relative_deadband=.05,
        base_height=2., termination=-20., tracking_lin_vel=2., action_rate=-.02)
    before = copy.deepcopy(original)
    result = experiment.simple_rewards(original)
    assert original == before
    assert result['orientation'] == -20. and result['base_height'] == 2.
    assert result['termination'] == -20. and result['action_rate'] == -.02
    assert result['tracking_goal_component_violation'] == result['wheel_action_saturation'] == 0.


def test_motion_release_is_separate_from_full_height_coverage():
    cfg = dict(ranges_height=(.28, .32), ranges_lin_vel_x=(.2, .8), motion_ramp_steps=48000)
    before = copy.deepcopy(cfg)
    frozen = experiment.commands_for_stage(cfg)
    moving = experiment.commands_for_stage(cfg, motion_start=9600)
    assert cfg == before
    assert frozen['ranges_height'] == moving['ranges_height'] == list(SAFE_BASE_HEIGHT_RANGE_M)
    assert frozen['full_height_range_from_start'] is moving['full_height_range_from_start'] is True
    assert frozen['standing_only_steps'] == 2_000_000_000
    assert moving['standing_only_steps'] == moving['yaw_start_steps'] == moving['reverse_ramp_start_steps'] == 9600
    assert frozen['ranges_lin_vel_x'] == (.2, .8)  # Replay retains real canonical motion targets.


def test_height_revision_adds_only_dense_height_cost_and_preserves_original_input():
    original = dict(base_height_error_sq=0., base_height=2., termination=-20.,
        base_height_penalty_clip=30., tracking_goal_component_violation=-.75)
    before = copy.deepcopy(original)
    basic = experiment.simple_rewards(original)
    revised = experiment.rewards_for_revision(original, {'overrides':{'base_height_error_sq':-50.}})
    assert original == before
    assert {k for k in revised if revised[k] != basic[k]} == {'base_height_error_sq'}
    assert revised['base_height'] == 2. and revised['termination'] == -20.
    assert revised['tracking_goal_component_violation'] == 0.
    assert experiment.rewards_for_revision(original) == basic
    # Native reward uses coefficient * actual squared root-height error * dt.
    penalties = torch.tensor([.01, .05, .10, .15]).square()*revised['base_height_error_sq']
    assert penalties.tolist() == pytest.approx([-.005, -.125, -.5, -1.125])
    assert bool((penalties > -revised['base_height_penalty_clip']).all())
    assert penalties[-1]*.02 == pytest.approx(-.0225)


@pytest.mark.parametrize('coefficient', [True, 0., 50., float('nan'), float('inf'), '-50'])
def test_height_revision_rejects_invalid_or_nonnegative_cost(coefficient):
    with pytest.raises(ValueError, match='finite and negative'):
        experiment.rewards_for_revision({}, {'overrides':{'base_height_error_sq':coefficient}})


def test_height_revision_cannot_restore_another_reward_before_standing():
    with pytest.raises(ValueError, match='only base_height_error_sq'):
        experiment.rewards_for_revision({}, {'overrides':{'base_height_error_sq':-50., 'orientation':-60.}})


def test_readiness_is_not_full_acceptance_or_height_precision_claim():
    report = assessment(error=.001)
    report['conditions'][2]['trials'][0]['metrics']['height_mae_m'] = .1
    result = experiment.standing_readiness(report)
    assert result['ready'] and result['prerequisite_only_not_full_acceptance']
    assert result['height_relative_mae_percent_by_trial'][-3]['percent'] > 10.


def test_a_failed_lifetime_cannot_appear_to_be_a_stable_standing_sample():
    result = experiment.standing_readiness(assessment(survives=False))
    assert result['survivors'] == 0 and not result['ready']
    assert result['mean_first_lifetime_s'] == .5
    assert all(x['percent'] is None for x in result['height_relative_mae_percent_by_trial'])


def test_readiness_reuses_strict_original_limits():
    report = assessment(error=.025)
    assert not experiment.standing_readiness(report)['ready']


def test_missing_low_height_or_missing_seed_cannot_release_motion():
    report = assessment()
    report['conditions'].pop()
    with pytest.raises(ValueError, match='three actual'):
        experiment.standing_readiness(report)
    report = assessment()
    report['conditions'][0]['trials'].pop()
    with pytest.raises(ValueError, match='three independent seeds'):
        experiment.standing_readiness(report)


def test_persistent_failure_is_monitored_but_does_not_cut_short_the_1500_update_warmup():
    history = [dict(stage='standing', actual_ppo_updates=n,
        standing=dict(survivors=0, mean_first_lifetime_s=.5)) for n in (100, 200, 300)]
    assert not experiment.support_stagnation(history[:2])
    assert not experiment.support_stagnation(history)
    for item, updates in zip(history, (1300, 1400, 1500)):
        item['actual_ppo_updates'] = updates
    assert experiment.support_stagnation(history)
    history[-1]['standing']['mean_first_lifetime_s'] = 2.
    assert not experiment.support_stagnation(history)


def test_check_interval_is_exactly_100_actual_updates_and_precision_starts_disabled():
    assert experiment.INTERVAL == 100 and experiment.RESTORE_SCALES[0] == 0.
    assert experiment.BASIC_REWARD_WARMUP_UPDATES == 1500


def source_checkpoint(tmp_path, updates=37):
    path = tmp_path/'model_36.pt'
    torch.save(dict(iter=36, model_state_dict={'marker':torch.ones(1)},
        optimizer_state_dict={'state':{i:{'step':torch.tensor(float(updates*20))} for i in range(17)},
                              'param_groups':[{'lr':.0003}]}), path)
    params = tmp_path/'params'
    params.mkdir()
    (params/'curriculum_clock.yaml').write_text(yaml.safe_dump(dict(
        offset_steps=0, steps_per_update=48, adam_steps_per_update=20)))
    return path


def test_height_revision_pins_real_adam_update_not_iteration_label(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'RUNS', tmp_path)
    assert experiment.load_height_penalty_revision(tmp_path) is None
    source = source_checkpoint(tmp_path)
    revision = dict(overrides={'base_height_error_sq':-50.}, effective_after_actual_ppo_updates=37,
        source_checkpoint=str(source), source_checkpoint_sha256=experiment.sha(source))
    marker = tmp_path/'height_penalty_revision.json'
    marker.write_text(json.dumps(revision))
    assert experiment.load_height_penalty_revision(tmp_path) == revision
    revision['effective_after_actual_ppo_updates'] = 36
    marker.write_text(json.dumps(revision))
    with pytest.raises(ValueError, match='complete Adam'):
        experiment.load_height_penalty_revision(tmp_path)
    revision['effective_after_actual_ppo_updates'] = 37
    revision['source_checkpoint_sha256'] = 'changed'
    marker.write_text(json.dumps(revision))
    with pytest.raises(ValueError, match='intact owned'):
        experiment.load_height_penalty_revision(tmp_path)


def test_height_penalty_continuation_is_one_reward_change_and_never_resets(tmp_path):
    source = source_checkpoint(tmp_path)
    revision = {'overrides':{'base_height_error_sq':-50.}}
    cfg = dict(commands={}, rewards=experiment.rewards_for_revision({'base_height_error_sq':0.}, revision))
    command = experiment.learning_command(source, source, cfg, 'height_penalty', 100, tmp_path/'STOP')
    assert command[command.index('--checkpoint_path')+1] == str(source)
    assert 'env.rewards.base_height_error_sq=-50.0' in command
    assert 'env.rewards.tracking_goal_component_violation=0.0' in command
    assert not any(flag in command for flag in ('--reset_optimizer', '--reset_exploration_std',
        '--curriculum_step_offset', '--save_initial_checkpoint', '--allow_physics_contract_change'))


def test_fresh_command_has_no_checkpoint_or_auxiliary_losses_and_is_headless(tmp_path):
    path = source_checkpoint(tmp_path)
    command = experiment.learning_command(None, path, dict(commands={}, rewards={}),
                                           'fresh', 100, tmp_path/'STOP')
    assert '--checkpoint_path' not in command
    assert '--headless' in command and '--save_initial_checkpoint' in command
    assert command[command.index('--curriculum_step_offset')+1] == '0'
    assert command[command.index('--target_iteration')+1] == '100'
    for flag in ('--joint_mean_guard_coef', '--joint_height_feedback_coef'):
        assert command[command.index(flag)+1] == '0.0'


def test_resume_counts_adam_updates_and_never_resets_saved_state(tmp_path):
    path = source_checkpoint(tmp_path)
    command = experiment.learning_command(path, path, dict(commands={}, rewards={}),
                                           'resume', 100, tmp_path/'STOP')
    assert command[command.index('--target_iteration')+1] == '100'
    assert command[command.index('--checkpoint_path')+1] == str(path)
    assert not any(flag in command for flag in (
        '--curriculum_step_offset', '--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint'))
