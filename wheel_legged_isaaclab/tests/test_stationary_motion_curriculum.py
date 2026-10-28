import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wheel_legged_gym_isaaclab.motion_curriculum import scale_training_motion, training_motion_scale
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
import joint_pd_stationary_stage as stage
from test_joint_pd_priority import report, trial


def completed_report(height=.06):
    result = report()
    for case in result['conditions']:
        for row in case['trials']:
            row['metrics'].update(height_mae_m=height, upright_support=True, command_reached=True,
                wheel_speed_mae_rad_s=.01, leg_length_mae_m=.001, leg_angle_mae_rad=.001)
            row['limits'].update(wheel_speed_mae_rad_s=.15, leg_length_mae_m=.025, leg_angle_mae_rad=.02)
    return result


def sampler():
    path = Path(__file__).resolve().parents[1]/'wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/env_commands.py'
    spec = importlib.util.spec_from_file_location('stationary_env_commands_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    cfg = SimpleNamespace(resampling_time=5., standing_only_steps=0, motion_ramp_steps=48000,
        reverse_ramp_start_steps=0, reverse_ramp_steps=48000, yaw_start_steps=0, yaw_ramp_steps=48000,
        full_height_range_from_start=True, ranges_height=SAFE_BASE_HEIGHT_RANGE_M,
        height_expansion_from=None, height_expansion_start_steps=0, height_expansion_ramp_steps=48000,
        height_boundary_fraction=.4, training_motion_scale=0.,
        ranges_lin_vel_x=(.2,.8), ranges_reverse_lin_vel_x=(-.8,-.2),
        ranges_transition_lin_vel_x=(-.2,.2), ranges_ang_vel_yaw=(-.3,.3),
        speed_boundary_fraction=.3, grouped_low_speed_fraction=1/3., yaw_min_abs_rate=.15,
        yaw_boundary_fraction=.2)
    env = SimpleNamespace(device='cpu', common_step_counter=10**9, step_dt=.02,
        _dedicated_standing_count=2048, _commands=torch.zeros(8192,4),
        _target_lin_vel_x=torch.zeros(8192), cfg=SimpleNamespace(commands=cfg,
            rewards=SimpleNamespace(base_height_target=.3)),
        _command_ranges=torch.tensor([[.2,.8],[-.3,.3],list(SAFE_BASE_HEIGHT_RANGE_M)]))
    return module, env


def test_all_8192_stay_stationary_after_arbitrarily_long_clock_with_full_height_endpoints():
    module, env = sampler()
    for ids in (torch.arange(8192), torch.tensor([0, 2047, 2048, 8191])):
        module.resample_commands(env, ids)
        assert env._commands[:, :2].count_nonzero() == 0
        assert env._target_lin_vel_x.count_nonzero() == 0
    heights = env._commands[:,2]
    assert heights.min().item() == pytest.approx(SAFE_BASE_HEIGHT_RANGE_M[0])
    assert heights.max().item() == pytest.approx(SAFE_BASE_HEIGHT_RANGE_M[1])
    assert (heights == heights.min()).sum() > 1000 and (heights == heights.max()).sum() > 1000


def test_partial_motion_release_changes_only_speed_yaw_and_keeps_height_rng_and_bounds():
    module, env = sampler()
    other = copy.deepcopy(env); other.cfg.commands.training_motion_scale=.125
    rng=torch.random.get_rng_state()
    module.resample_commands(env, torch.arange(8192))
    after=torch.random.get_rng_state()
    torch.random.set_rng_state(rng)
    module.resample_commands(other, torch.arange(8192))
    assert torch.equal(after, torch.random.get_rng_state())
    assert torch.equal(env._commands[:,2], other._commands[:,2])
    assert other._target_lin_vel_x.abs().max() <= .100001
    assert other._target_lin_vel_x.count_nonzero() > 0
    assert other._commands[:,1].abs().max() <= .037501
    assert other._target_lin_vel_x[:2048].count_nonzero() == 0


def test_stationary_training_scale_does_not_zero_fixed_replay_commands():
    module, env = sampler()
    env._command_ranges=torch.tensor([[.3,.3],[.2,.2],[.35,.35]])
    module.resample_commands(env, torch.arange(8192))
    assert torch.all(env._target_lin_vel_x == torch.tensor(.3))
    assert torch.all(env._commands[:,1] == torch.tensor(.2))


@pytest.mark.parametrize('bad', [-.1, 1.01, float('nan'), float('inf'), True])
def test_invalid_motion_scale_rejected(bad):
    with pytest.raises(ValueError):training_motion_scale(SimpleNamespace(training_motion_scale=bad))


def test_missing_scale_preserves_historical_sampler_bitwise_and_without_rng():
    speed,yaw=torch.randn(16),torch.randn(16)
    rng=torch.random.get_rng_state()
    a,b=scale_training_motion(speed,yaw,SimpleNamespace())
    assert a is speed and b is yaw and torch.equal(rng,torch.random.get_rng_state())


def test_stationary_training_selection_cannot_be_mistaken_for_full_acceptance():
    before,after=completed_report(),completed_report(.04)
    row=trial(after,'reverse_0p6');row.update(survived=False,metrics=None,resets=1,first_failure_s=3.)
    after.update(survivors=26,resets=1)
    decision=stage.review(after,before)
    assert decision['promote'] and not decision['policy_accepted']
    assert decision['full_scope_comparison']['regressions']['stability']
    assert not decision['candidate_stationary']['stationary_all_original_metrics_passed']


@pytest.mark.parametrize('metric',['body_p95_deg','speed_mae_m_s','yaw_mae_rad_s'])
def test_static_stability_loss_forbids_promotion_despite_better_height(metric):
    before,after=completed_report(),completed_report(.001)
    trial(after,'height_low')['metrics'][metric]=trial(after,'height_low')['limits'][metric]*2
    assert not stage.review(after,before)['promote']


def test_motion_release_requires_all_original_stationary_metrics_and_prerequisites():
    value=completed_report(.001)
    assert stage.summary(value)['stationary_all_original_metrics_passed']
    trial(value,'height_high')['metrics']['wheel_speed_mae_rad_s']=.16
    assert not stage.summary(value)['stationary_all_original_metrics_passed']
    value=completed_report(.001);trial(value)['metrics']['upright_support']=False
    assert not stage.summary(value)['stationary_all_original_metrics_passed']


def test_command_only_first_block_then_height_cost_on_measured_stagnation():
    cfg=dict(rewards=dict(orientation=-20.,standing_velocity=-50.,yaw_rate_error_sq=-1.,base_height_error_sq=-100.))
    value=completed_report()
    assert stage.next_reward(cfg,value,[],set()) == {}
    decision=stage.review(value,value)
    assert stage.next_reward(cfg,value,[decision],{()}) == {'base_height_error_sq':-250.}


def test_meaningful_height_progress_continues_without_adding_reward_variables():
    before,after=completed_report(),completed_report(.04)
    decision=stage.review(after,before)
    assert stage.next_reward(dict(rewards={}),after,[decision],set()) == {}
