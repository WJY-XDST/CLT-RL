import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import train_joint_pd_posture_repair as repair
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M


@pytest.fixture
def config():
    return dict(leg_control_mode='implicit_joint_position', joint_reset_profile='joint_target_nominal_v1',
        commands=dict(ranges_height=list(SAFE_BASE_HEIGHT_RANGE_M), full_height_range_from_start=True,
            standing_only_steps=0, yaw_start_steps=0, reverse_ramp_start_steps=0, motion_ramp_steps=48000,
            ranges_lin_vel_x=[.2,.8], ranges_reverse_lin_vel_x=[-.8,-.2], ranges_ang_vel_yaw=[-.3,.3]),
        rewards=dict(orientation=-20., base_height_error_sq=-50., tracking_goal_component_violation=-.1,
            base_height=2., tracking_lin_vel=2., tracking_ang_vel=1., lin_vel_error_sq=-2., yaw_rate_error_sq=-1.),
        robot=dict(spawn=dict(usd_path='owner_calibrated.usda')), leg_joint_drive_stiffness=300.,
        leg_joint_drive_damping=3., leg_joint_position_offset=[-.07,-.03,.07,.03])


def test_changes_only_one_existing_reward_without_mutating_source(config):
    before = copy.deepcopy(config)
    result = repair.corrected_config(config)
    assert config == before
    assert result['rewards']['orientation'] == -400.
    result['rewards']['orientation'] = -20.
    assert result == config
    assert repair.UPDATES == 25


def test_replay_uses_full_canonical_scope_and_actual_reward_telemetry():
    command = repair.replay_command(Path('saved.pt'), Path('replay'))
    assert command[command.index('--seconds')+1] == '20'
    assert command[command.index('--seeds')+1:command.index('--seeds')+4] == ['83','84','85']
    assert '--record-reward-terms' in command
    assert '--record_policy_inputs' in command
    assert '--headless' in command
    assert not any(flag in command for flag in ('--command_scale', '--case_names', '--override-json'))


@pytest.mark.parametrize(('group','key','value'), [
    (None,'leg_control_mode','explicit_vmc'),
    (None,'joint_reset_profile','cad_zero_v0'),
    ('commands','ranges_height',[.29,.31]),
    ('commands','full_height_range_from_start',False),
    ('commands','standing_only_steps',100000),
    ('commands','yaw_start_steps',100000),
    ('commands','reverse_ramp_start_steps',100000),
    ('commands','motion_ramp_steps',96000),
    ('rewards','orientation',-400.),
    ('rewards','base_height_error_sq',0.),
    ('rewards','tracking_goal_component_violation',0.),
])
def test_rejects_undiagnosed_or_changed_source(config,group,key,value):
    (config if group is None else config[group])[key] = value
    with pytest.raises(ValueError, match='diagnosed'):
        repair.corrected_config(config)


@pytest.fixture
def report():
    names = ('stand','forward_0p3','forward_0p6','reverse_0p3','reverse_0p6',
             'spin_left','spin_right','height_low','height_high')
    conditions = []
    for name in names:
        metrics = dict.fromkeys(repair.METRICS,.01)
        metrics.update(body_p95_deg=.2,upright_support=True,command_reached=True)
        limits = dict.fromkeys(repair.METRICS,1.)
        limits['body_p95_deg'] = 2.5
        trials = [dict(seed=seed,survived=True,metrics=copy.deepcopy(metrics),limits=copy.deepcopy(limits))
                  for seed in (83,84,85)]
        conditions.append(dict(name=name,height=.3,trials=trials))
    return dict(conditions=conditions,survivors=27,total_trials=27,resets=0,passed=True)


def test_even_perfect_canonical_recovery_cannot_claim_sustained_acceptance(report):
    result = repair.repair_comparison(report,report,report)
    assert result['strict_full_original_gate_passed']
    assert result['best_height_posture_retained']
    assert not result['policy_accepted']
    assert not result['safe_to_continue_unreviewed']
    assert not result['independent_sustained_verification_completed']


def test_speed_gain_cannot_hide_posture_or_height_regressions(report):
    previous = copy.deepcopy(report)
    report['conditions'][1]['trials'][0]['metrics'].update(
        speed_mae_m_s=0., body_p95_deg=3., height_mae_m=.02)
    result = repair.repair_comparison(report,previous,previous)
    assert not result['best_height_posture_retained']
    assert not result['strict_full_original_gate_passed']
    assert {v['metric'] for v in result['best_height_posture_retention_failures']} == {'height_mae_m','body_p95_deg'}
    assert not result['policy_accepted']


def test_improving_failed_source_does_not_erase_original_protected_envelope(report):
    failed = copy.deepcopy(report)
    failed['conditions'][2]['trials'][0]['metrics']['body_p95_deg'] = 3.5
    current = copy.deepcopy(report)
    current['conditions'][2]['trials'][0]['metrics']['body_p95_deg'] = 1.5
    result = repair.repair_comparison(current,failed,report)
    assert result['worst_body_improvement_fraction'] > .5
    assert result['strict_full_original_gate_passed']
    assert not result['best_height_posture_retained']
    assert not result['safe_to_continue_unreviewed']


def test_exact_original_body_limit_is_still_failure(report):
    report['conditions'][2]['trials'][0]['metrics']['body_p95_deg'] = 2.5
    result = repair.repair_comparison(report,report,report)
    assert not result['strict_full_original_gate_passed']
    assert result['best_height_posture_retention_failures']


def test_incomplete_case_scope_cannot_be_used_as_recovery(report):
    narrowed = copy.deepcopy(report)
    narrowed['conditions'].pop()
    with pytest.raises(ValueError, match='all original'):
        repair.repair_comparison(narrowed,report,report)


def test_failed_first_episode_is_reported_not_crashed_or_omitted(report):
    previous = copy.deepcopy(report)
    trial = report['conditions'][4]['trials'][0]
    trial.update(survived=False,metrics=None)
    report.update(survivors=26,resets=1,passed=False)
    result = repair.repair_comparison(report,previous,previous)
    assert result['steady_metrics_missing_trials'] == 1
    assert result['worst_body_p95_deg'] is None
    assert result['worst_body_improvement_fraction'] is None
    assert all(value is None for value in result['full_original_metric_ratios'].values())
    assert result['surviving_trials_worst_body_p95_deg'] == .2
    assert result['surviving_trials_metric_ratios']['body_p95_deg'] == .08
    assert not result['full_scope_steady_metrics_available']
    assert not result['strict_full_original_gate_passed']
    assert not result['best_height_posture_retained']
    assert not result['policy_accepted']


def test_no_surviving_trials_cannot_generate_global_metrics_or_pass(report):
    previous = copy.deepcopy(report)
    for case in report['conditions']:
        for trial in case['trials']:
            trial.update(survived=False,metrics=None)
    report.update(survivors=0,resets=27,passed=False)
    result = repair.repair_comparison(report,previous,previous)
    assert result['steady_metrics_missing_trials'] == 27
    assert result['worst_body_p95_deg'] is None
    assert result['surviving_trials_worst_body_p95_deg'] is None
    assert all(value is None for value in result['surviving_trials_metric_ratios'].values())
    assert not result['strict_full_original_gate_passed']


def test_recovered_current_policy_does_not_invent_missing_source_metrics(report):
    previous = copy.deepcopy(report)
    previous['conditions'][4]['trials'][0].update(survived=False,metrics=None)
    previous.update(survivors=26,resets=1,passed=False)
    result = repair.repair_comparison(report,previous,report)
    assert result['worst_body_p95_deg'] == .2
    assert result['source_worst_body_p95_deg'] is None
    assert result['worst_body_improvement_fraction'] is None
    assert result['strict_full_original_gate_passed']
    assert not result['policy_accepted']
