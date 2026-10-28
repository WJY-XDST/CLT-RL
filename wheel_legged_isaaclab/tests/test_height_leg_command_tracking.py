"""Actual command tracking cannot be inferred from a self-generated reference."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import assess_height_leg_command_tracking as stage


@pytest.fixture
def replay():
    heights = dict(stand=.3, height_low=stage.SAFE_BASE_HEIGHT_RANGE_M[0],
                   height_high=stage.SAFE_BASE_HEIGHT_RANGE_M[1])
    names = ('stand', 'forward_0p3', 'forward_0p6', 'reverse_0p3', 'reverse_0p6',
             'spin_left', 'spin_right', 'height_low', 'height_high')
    manifest = dict(checkpoint='fixture.pt', checkpoint_sha256='fixture',
        cases=[dict(name=name, height=heights.get(name, .3), speed=0., yaw=0.) for name in names],
        seconds_per_case=20., policy_dt=.02, seeds=list(stage.SEEDS), weights_updated=False,
        policy_acceptance_test=True, command_probe=False, leg_control_mode='implicit_joint_position')
    fields = ('time_s', 'case_id', 'seed', 'episode', 'terminated', 'timeout',
        'height_m', 'height_cmd_m', 'speed_m_s', 'speed_cmd_m_s', 'yaw_rate_rad_s', 'yaw_cmd_rad_s',
        'pitch_rad', 'roll_rad', 'left_length_m', 'right_length_m', 'left_length_ref_m',
        'right_length_ref_m', 'left_angle_rad', 'right_angle_rad', 'left_angle_ref_rad',
        'right_angle_ref_rad', 'left_wheel_rad_s', 'right_wheel_rad_s', 'left_wheel_ref_rad_s',
        'right_wheel_ref_rad_s')
    data = np.zeros(27000, dtype=[(field, 'f8') for field in fields])
    for case_id, case in enumerate(manifest['cases']):
        for index, seed in enumerate(stage.SEEDS):
            trial = data[(case_id*3+index)*1000:(case_id*3+index+1)*1000]
            trial['case_id'], trial['seed'] = case_id, seed
            trial['time_s'] = np.arange(1, 1001)*.02
            trial['height_m'] = trial['height_cmd_m'] = case['height']
            for side in stage.SIDES:
                trial[f'{side}_length_m'] = case['height']-stage.UPRIGHT_HEIGHT_OFFSET_M
                trial[f'{side}_length_ref_m'] = trial[f'{side}_length_m']
    return manifest, data, stage.acceptance_criteria()


def test_full_static_stage_does_not_claim_motion_or_full_acceptance(replay):
    result = stage.assess_samples(*replay)
    assert result['static_stage_candidate_passed']
    assert not result['policy_accepted']
    assert not result['motion_release_authorized_by_this_report']
    assert not result['repeated_command_sequence_verified']
    high = [t for t in result['trials'] if t['condition'] == 'height_high'][0]
    assert high['metrics']['legs']['left']['command_target_m'] == pytest.approx(.34)
    assert result['endpoint_responses'][0]['necessary_minimum_span_m'] == pytest.approx(.097)


def test_constant_legs_with_perfect_self_reference_cannot_pass(replay):
    manifest, data, criteria = replay
    for side in stage.SIDES:
        data[f'{side}_length_m'] = data[f'{side}_length_ref_m'] = .26465354469
    result = stage.assess_samples(manifest, data, criteria)
    assert not result['actual_leg_static_command_tracking_passed']
    assert result['height_posture_passed']
    assert all(t['metrics']['legs']['left']['policy_reference_MAE_m'] == 0. for t in result['trials'])
    assert all(not r['passed'] for r in result['endpoint_responses'])


def test_a_bad_right_leg_cannot_be_hidden_by_a_good_left_leg(replay):
    manifest, data, criteria = replay
    data['right_length_m'][data['case_id'] == 8] = .26
    result = stage.assess_samples(manifest, data, criteria)
    assert not result['actual_leg_static_command_tracking_passed']


def test_exact_ten_percent_is_not_strictly_below_ten_percent(replay):
    manifest, data, criteria = replay
    rows = data['case_id'] == 7
    data['left_length_m'][rows] = .19*1.1
    result = stage.assess_samples(manifest, data, criteria)
    assert not result['actual_leg_static_command_tracking_passed']


@pytest.mark.parametrize('field,value', [('terminated', 1.), ('timeout', 1.), ('episode', 1.)])
def test_a_failed_first_episode_cannot_be_replaced_by_a_good_tail(replay, field, value):
    manifest, data, criteria = replay
    data[field][0] = value
    result = stage.assess_samples(manifest, data, criteria)
    assert not result['height_posture_passed']
    assert not result['actual_leg_static_command_tracking_passed']


@pytest.mark.parametrize('field,value', [('height_m', .2), ('pitch_rad', .1), ('roll_rad', .1)])
def test_height_and_posture_fail_independently_of_leg_reference_precision(replay, field, value):
    manifest, data, criteria = replay
    data[field][:1000] = value
    assert not stage.assess_samples(manifest, data, criteria)['height_posture_passed']


def test_zero_command_drift_is_separate_from_motion_training(replay):
    manifest, data, criteria = replay
    data['speed_m_s'][:1000] = .03
    result = stage.assess_samples(manifest, data, criteria)
    assert result['height_posture_passed']
    assert not result['static_stage_candidate_passed']


@pytest.mark.parametrize('change', ['narrow_range', 'duplicate_seed', 'sequence_probe',
                                   'weights_updated', 'different_mode', 'missing_case'])
def test_changed_contract_cannot_count_as_static_evidence(replay, change):
    manifest, data, criteria = replay
    manifest = copy.deepcopy(manifest)
    if change == 'narrow_range': manifest['cases'][-1]['height'] = .32
    if change == 'duplicate_seed': manifest['seeds'] = [83, 83, 85]
    if change == 'sequence_probe': manifest['command_probe'] = True
    if change == 'weights_updated': manifest['weights_updated'] = True
    if change == 'different_mode': manifest['leg_control_mode'] = 'vmc'
    if change == 'missing_case': manifest['cases'].pop()
    with pytest.raises(ValueError): stage.assess_samples(manifest, data, criteria)


@pytest.mark.parametrize('change', ['missing', 'duplicate_time', 'nan', 'command'])
def test_bad_telemetry_cannot_count_as_static_evidence(replay, change):
    manifest, data, criteria = replay
    if change == 'missing': data = data[:-1]
    if change == 'duplicate_time': data['time_s'][0] = data['time_s'][1]
    if change == 'nan': data['left_length_m'][0] = np.nan
    if change == 'command': data['height_cmd_m'][0] = .32
    with pytest.raises(ValueError): stage.assess_samples(manifest, data, criteria)


@pytest.fixture
def saved_replay(replay, tmp_path, monkeypatch):
    manifest, data, criteria = replay
    checkpoint, asset = tmp_path/'fixture.pt', tmp_path/'fixture.usda'
    checkpoint.write_bytes(b'fixture saved weights')
    asset.write_bytes(b'fixture asset')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        asset_sha256=sha(asset), asset_chain_sha256={str(asset): sha(asset)},
        tracking_goal_feedback_criteria=dict(sha256=sha(stage.ACCEPTANCE_PATH), criteria=criteria))
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'completed.json').write_text(json.dumps(dict(completed=True, samples=len(data))))
    monkeypatch.setattr(stage.np, 'genfromtxt', lambda *args, **kwargs: data)
    return tmp_path, manifest, data


def test_readonly_directory_audit_verifies_recorded_hashes(saved_replay):
    directory, manifest, data = saved_replay
    assert stage.assess_directory(directory)['static_stage_candidate_passed']
    Path(manifest['checkpoint']).write_bytes(b'changed weights')
    with pytest.raises(ValueError, match='mismatch'): stage.assess_directory(directory)


def test_old_frozen_policy_can_have_no_reward_feedback_without_weakening_the_gate(saved_replay):
    directory, manifest, data = saved_replay
    manifest['tracking_goal_feedback_criteria'] = None
    (directory/'manifest.json').write_text(json.dumps(manifest))
    report = stage.assess_directory(directory)
    assert not report['replay_reward_feedback_criteria_recorded']
    assert report['original_acceptance_sha256'] == hashlib.sha256(stage.ACCEPTANCE_PATH.read_bytes()).hexdigest()


@pytest.mark.parametrize('change', ['gate_hash', 'gate_limits', 'asset', 'not_completed', 'sample_count'])
def test_directory_audit_rejects_mismatched_provenance(saved_replay, change):
    directory, manifest, data = saved_replay
    if change == 'gate_hash': manifest['tracking_goal_feedback_criteria']['sha256'] = 'mismatched'
    if change == 'gate_limits': manifest['tracking_goal_feedback_criteria']['criteria']['relative_error'] = .2
    if change == 'asset': Path(next(iter(manifest['asset_chain_sha256']))).write_bytes(b'changed asset')
    if change == 'not_completed':
        (directory/'completed.json').write_text(json.dumps(dict(completed=False, samples=len(data))))
    if change == 'sample_count':
        (directory/'completed.json').write_text(json.dumps(dict(completed=True, samples=len(data)-1)))
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError): stage.assess_directory(directory)
