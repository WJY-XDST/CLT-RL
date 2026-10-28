"""Read frozen-policy height-step diagnostics; never replace acceptance."""
import argparse
import json
from pathlib import Path

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import sha


def moments(rows):
    if not len(rows):
        return None
    height_error = np.abs(rows['height_m'] - rows['height_cmd_m'])
    return dict(
        samples=len(rows),
        height_mae_m=float(height_error.mean()),
        height_relative_mae=float((height_error / rows['height_cmd_m']).mean()),
        horizontal_speed_mae_m_s=float(np.hypot(rows['speed_m_s'], rows['lateral_speed_m_s']).mean()),
        yaw_mae_rad_s=float(np.abs(rows['yaw_rate_rad_s']).mean()),
        pitch_peak_deg=float(np.rad2deg(np.abs(rows['pitch_rad'])).max()),
        roll_peak_deg=float(np.rad2deg(np.abs(rows['roll_rad'])).max()),
        body_p95_deg=float(np.quantile(np.rad2deg(np.maximum(np.abs(rows['pitch_rad']), np.abs(rows['roll_rad']))), .95)),
        base_net_force_peak_n=float(rows['base_net_force_n'].max()),
        base_ground_force_peak_n=float(rows['base_ground_force_n'].max()),
        base_collision_clearance_min_m=float(rows['base_collision_min_z_m'].min()),
        joint_tracking_peak_rad=float(rows['active_joint_tracking_max_abs_rad'].max()),
        actuator_torque_telemetry_is_estimate=True)


def summarize(directory, plan_file):
    directory, plan_file = Path(directory).resolve(), Path(plan_file).resolve()
    manifest = json.loads((directory / 'manifest.json').read_text())
    complete = json.loads((directory / 'completed.json').read_text())
    specification = json.loads(plan_file.read_text())
    phases, period = specification['phases'], specification['phase_seconds']
    sequence = [[[case[key] for key in ('speed', 'yaw', 'height')] for case in phase] for phase in phases]
    first_cases = phases[0]
    dt, duration = manifest['policy_dt'], manifest['seconds_per_case']
    expected_samples = round(duration / dt) * len(first_cases) * len(manifest['seeds'])
    if (manifest.get('policy_acceptance_test') is not False
            or not manifest.get('command_probe') or manifest.get('weights_updated') is not False
            or manifest['leg_control_mode'] != 'implicit_joint_position'
            or manifest.get('command_scale') != 1. or manifest.get('curriculum_progress') is not None
            or manifest.get('action_bias_probes') or manifest.get('wheel_probe')
            or manifest.get('wheel_reference_limit_probe') is not None
            or manifest.get('policy_input_remapping') or manifest.get('reset_velocity_uniform_amplitude') != .05
            or manifest['physics_dt'] != .005 or dt != .02
            or manifest['solver_position_iterations'] != 128 or manifest['solver_velocity_iterations'] != 8
            or manifest.get('enable_external_forces_every_iteration') is not True
            or manifest['cases'] != first_cases or manifest['command_sequence'] != sequence
            or manifest['command_phase_seconds'] != period or duration != period * len(phases)
            or manifest['seeds'] != [383, 384, 385]
            or not complete['completed'] or complete['samples'] != expected_samples
            or sha(manifest['checkpoint']) != manifest['checkpoint_sha256']):
        raise ValueError('Require the exact complete frozen direct-PD height-transition probe')
    for name, digest in manifest['asset_chain_sha256'].items():
        if sha(name) != digest:
            raise ValueError('Changed replay asset: ' + name)
    samples = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    if len(samples) != expected_samples or not all(np.isfinite(samples[key]).all() for key in samples.dtype.names):
        raise ValueError('Incomplete or nonfinite diagnostic data')
    rows, events = [], []
    reason_keys = [key for key in samples.dtype.names if key.startswith('fail_')]
    for cid, case in enumerate(first_cases):
        for seed in manifest['seeds']:
            trial = samples[(samples['case_id'] == cid) & (samples['seed'] == seed)]
            timeline = np.arange(1, round(duration / dt) + 1) * dt
            if len(trial) != len(timeline) or not np.allclose(trial['time_s'], timeline, atol=1.e-8, rtol=0.):
                raise ValueError('Missing complete probe timeline')
            phase_ids = np.floor((timeline - dt + 1.e-7) / period).astype(int)
            targets = np.array([sequence[i][cid] for i in phase_ids])
            measured_targets = np.column_stack([trial[key] for key in ('speed_cmd_m_s', 'yaw_cmd_rad_s', 'height_cmd_m')])
            if not np.allclose(targets, measured_targets, atol=1.e-7, rtol=0.):
                raise ValueError('Recorded targets do not match the full-height5s plan')
            survived = bool(not np.any(trial['episode']) and not np.any(trial['terminated']) and not np.any(trial['timeout']))
            for event in trial[trial['terminated'] != 0]:
                events.append(dict(case=case['name'], seed=seed, time_s=float(event['time_s']),
                    height_cmd_m=float(event['height_cmd_m']), reasons=[key for key in reason_keys if event[key]],
                    physical_state={key:float(event[key]) for key in ('height_m', 'pitch_rad', 'roll_rad',
                        'base_net_force_n', 'base_ground_force_n', 'base_collision_min_z_m', 'active_joint_tracking_max_abs_rad')}))
            phase_rows = []
            for i in range(len(phases)):
                part = trial[phase_ids == i]
                age = part['time_s'] - i * period
                phase_rows.append(dict(index=i, target_height_m=sequence[i][cid][2],
                    first_second=moments(part[age <= 1.+dt*.1]),
                    final_three_seconds=moments(part[age > period-3.+dt*.1])))
            rows.append(dict(case=case['name'], seed=seed, survived=survived,
                total=moments(trial), phases=phase_rows))
    result = dict(checkpoint=manifest['checkpoint'], checkpoint_sha256=manifest['checkpoint_sha256'],
        sampled_policy_actions=manifest['sampled_policy_actions'], policy_noise_seed=manifest['policy_noise_seed'],
        total_trials=len(rows), survivors=sum(row['survived'] for row in rows), resets=complete['resets'],
        terminated_samples=len(events), events=events, trials=rows,
        source_sha256={str(directory/name):sha(directory/name) for name in ('manifest.json', 'completed.json', 'trace.csv')},
        command_plan_sha256=sha(plan_file), analysis_sha256=sha(__file__),
        full_height_steps_with_unchanged_control=True, training_modified=False, policy_accepted=False,
        limitation='Small frozen diagnostic separates deterministic means and saved Gaussian actions. '
                   'It cannot establish absence of rare native training failures or replace canonical full27 and independent sustained acceptance.')
    write_json(directory / 'height_transition_diagnosis.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory, args.plan)
    print(json.dumps({key:result[key] for key in ('checkpoint_sha256', 'sampled_policy_actions', 'total_trials', 'survivors', 'resets', 'terminated_samples')}))
