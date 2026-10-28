"""CPU-only height-stage evidence, distinct from policy-reference tracking.

The upright height-to-leg conversion is a measurement reference, never an IK
controller. This supplemental report cannot replace the original complete gate
or establish repeated-command/sustained acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_mine_trial import ACCEPTANCE_PATH, acceptance_criteria, metrics_for_window
from wheel_legged_gym_isaaclab.height_commands import (
    SAFE_BASE_HEIGHT_RANGE_M, UPRIGHT_HEIGHT_OFFSET_M,
)

STATIONARY = ('stand', 'height_low', 'height_high')
SEEDS = (83, 84, 85)
SIDES = ('left', 'right')
ROUND_OFF = 1.e-12


def strict_pass(value, limit):
    return bool(np.isfinite(value) and value < limit-ROUND_OFF)


def trial_metrics(samples, case, duration, criteria):
    """Failed first episodes never contribute a reconstructed steady tail."""
    survived = bool(np.all(samples['episode'] == 0)
                    and not np.any(samples['terminated']) and not np.any(samples['timeout']))
    if not survived:
        return dict(survived=False, height_posture_passed=False,
                    leg_command_passed=False, stationary_drift_passed=False, metrics=None)
    tail = samples[samples['time_s'] >= duration-3.]
    if not len(tail):
        raise ValueError('Missing original three-second steady window')
    if (not np.allclose(samples['height_cmd_m'], case['height'], rtol=0., atol=1.e-7)
            or np.any(abs(samples['speed_cmd_m_s']) > 1.e-7)
            or np.any(abs(samples['yaw_cmd_rad_s']) > 1.e-7)):
        raise ValueError('Stationary height commands differ from the declared case')
    target = float(case['height']-UPRIGHT_HEIGHT_OFFSET_M)
    physical = metrics_for_window(tail, case, criteria)
    legs = {}
    for side in SIDES:
        actual = tail[f'{side}_length_m']
        error = float(np.mean(abs(actual-target)))
        legs[side] = dict(command_target_m=target, measured_mean_m=float(np.mean(actual)),
            command_MAE_m=error, command_relative_MAE=error/target,
            policy_reference_MAE_m=float(np.mean(abs(actual-tail[f'{side}_length_ref_m']))),
            passed=strict_pass(error/target, criteria['relative_error']))
    height_posture = bool(physical['upright_support'] and physical['command_reached']
        and strict_pass(physical['height_mae_m']/case['height'], criteria['relative_error'])
        and strict_pass(physical['body_p95_deg'], criteria['body_p95_deg']))
    drift = bool(strict_pass(physical['speed_mae_m_s'], criteria['speed_absolute_m_s'])
                 and strict_pass(physical['yaw_mae_rad_s'], criteria['yaw_absolute_rad_s']))
    return dict(survived=True, height_posture_passed=height_posture,
        leg_command_passed=all(leg['passed'] for leg in legs.values()),
        stationary_drift_passed=drift,
        metrics=dict(height_relative_MAE=physical['height_mae_m']/case['height'],
            measured_height_mean_m=float(np.mean(tail['height_m'])),
            body_p95_deg=physical['body_p95_deg'], speed_MAE_m_s=physical['speed_mae_m_s'],
            yaw_MAE_rad_s=physical['yaw_mae_rad_s'], legs=legs))


def assess_samples(manifest, samples, criteria):
    """Require the canonical full replay even though only its static stage is assessed."""
    cases = manifest['cases']
    names = [case['name'] for case in cases]
    duration, dt = float(manifest['seconds_per_case']), float(manifest['policy_dt'])
    if (manifest.get('policy_acceptance_test') is not True or manifest.get('weights_updated') is not False
            or manifest.get('command_probe') or manifest.get('sampled_policy_actions')
            or manifest.get('leg_control_mode') != 'implicit_joint_position'
            or manifest['seeds'] != list(SEEDS) or duration != 20. or dt != .02
            or len(cases) != 9 or len(set(names)) != 9 or not set(STATIONARY).issubset(names)):
        raise ValueError('Require the deterministic canonical 20-second, three-seed joint-PD replay')
    by_name = {case['name']: case for case in cases}
    expected_heights = dict(zip(STATIONARY, (.3, *SAFE_BASE_HEIGHT_RANGE_M)))
    if any(not np.isclose(by_name[name]['height'], height, rtol=0., atol=1.e-10)
           or by_name[name]['speed'] != 0. or by_name[name]['yaw'] != 0.
           for name, height in expected_heights.items()):
        raise ValueError('Require the unchanged nominal and both full safe height endpoints')
    count = round(duration/dt)
    if (samples.ndim != 1 or len(samples) != count*len(cases)*len(SEEDS)
            or not all(np.isfinite(samples[key]).all() for key in samples.dtype.names)):
        raise ValueError('Incomplete or nonfinite replay')
    trials = []
    for case_id, case in enumerate(cases):
        for seed in SEEDS:
            data = samples[(samples['case_id'] == case_id) & (samples['seed'] == seed)]
            if len(data) != count or not np.allclose(
                    data['time_s'], np.arange(1, count+1)*dt, rtol=0., atol=1.e-8):
                raise ValueError('Incomplete, duplicate or reordered first-episode timeline')
            if case['name'] in STATIONARY:
                trials.append(dict(condition=case['name'], seed=seed,
                    **trial_metrics(data, case, duration, criteria)))
    lookup = {(trial['condition'], trial['seed']): trial for trial in trials}
    low_target, high_target = (height-UPRIGHT_HEIGHT_OFFSET_M for height in SAFE_BASE_HEIGHT_RANGE_M)
    # Necessary consequence of passing both endpoint MAEs, not a relaxed limit.
    minimum_span = high_target*(1.-criteria['relative_error'])-low_target*(1.+criteria['relative_error'])
    responses = []
    for seed in SEEDS:
        low, high = (lookup[(name, seed)] for name in ('height_low', 'height_high'))
        for side in SIDES:
            span = (high['metrics']['legs'][side]['measured_mean_m']
                    - low['metrics']['legs'][side]['measured_mean_m']) if low['survived'] and high['survived'] else None
            responses.append(dict(seed=seed, side=side, measured_high_minus_low_m=span,
                command_high_minus_low_m=high_target-low_target,
                necessary_minimum_span_m=minimum_span,
                passed=bool(span is not None and span > minimum_span+ROUND_OFF)))
    height_posture = all(trial['height_posture_passed'] for trial in trials)
    leg_command = all(trial['leg_command_passed'] for trial in trials) and all(r['passed'] for r in responses)
    drift = all(trial['stationary_drift_passed'] for trial in trials)
    return dict(checkpoint=manifest['checkpoint'], checkpoint_sha256=manifest['checkpoint_sha256'],
        height_posture_passed=height_posture, actual_leg_static_command_tracking_passed=leg_command,
        stationary_zero_command_drift_passed=drift,
        static_stage_candidate_passed=bool(height_posture and leg_command and drift),
        trials=trials, endpoint_responses=responses,
        upright_conversion_is_measurement_only=True,
        policy_reference_error_is_not_command_error=True,
        repeated_command_sequence_verified=False, independent_sustained_verified=False,
        motion_release_authorized_by_this_report=False,
        original_full_acceptance_replaced=False, policy_accepted=False)


def assess_directory(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    completed = json.loads((directory/'completed.json').read_text())
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    # This field records optional reward feedback, not evaluation criteria.
    # A frozen historical policy can legitimately have no feedback reward.
    feedback = manifest.get('tracking_goal_feedback_criteria')
    criteria = acceptance_criteria()
    gate_digest = digest(ACCEPTANCE_PATH)
    mismatched_feedback = feedback is not None and (
        not isinstance(feedback, dict) or feedback.get('sha256') != gate_digest
        or feedback.get('criteria') != criteria)
    if (not completed.get('completed') or digest(manifest['checkpoint']) != manifest['checkpoint_sha256']
            or mismatched_feedback
            or not manifest.get('asset_chain_sha256') or any(
                digest(path) != saved for path, saved in manifest['asset_chain_sha256'].items())):
        raise ValueError('Checkpoint, original gate, calibrated asset or replay completion mismatch')
    samples = np.genfromtxt(directory/'trace.csv', delimiter=',', names=True)
    if completed['samples'] != len(samples):
        raise ValueError('Replay sample count differs from completion evidence')
    report = assess_samples(manifest, samples, criteria)
    report.update(original_acceptance_sha256=gate_digest,
                  replay_reward_feedback_criteria_recorded=feedback is not None,
                  asset_sha256=manifest['asset_sha256'])
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    report = assess_directory(args.directory)
    output = args.directory/'height_leg_command_assessment.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('trials', 'endpoint_responses')}, indent=2))
