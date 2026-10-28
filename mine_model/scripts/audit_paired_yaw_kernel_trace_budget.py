"""Pair completed fixed replays and account for their signed yaw residuals.

Read only the recorded command, wheel reference/velocity and body yaw. The
three signed residuals add algebraically; they are not a causal plant model
or a direct measurement of slip/torque. Neither PPO nor simulation is run.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
GATE = ROOT / 'mine_model/config/tracking_acceptance.json'
GATE_SHA = '08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d'
NAMES = ('time_s', 'case_id', 'seed', 'yaw_rate_rad_s', 'yaw_cmd_rad_s',
         'height_m', 'height_cmd_m', 'pitch_rad', 'roll_rad',
         'left_wheel_rad_s', 'right_wheel_rad_s',
         'left_wheel_ref_rad_s', 'right_wheel_ref_rad_s',
         'reward_yaw_rate_error_sq', 'reward_track_ang_vel',
         'episode', 'terminated', 'timeout')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def differences(before, after, path=''):
    if isinstance(before, dict) and isinstance(after, dict):
        return [row for key in set(before) | set(after)
                for row in differences(before.get(key), after.get(key), path + '/' + key)]
    return [] if before == after else [(path, before, after)]


def process(folder):
    manifest, complete, cfg, diagnosis = [read(folder / name) for name in (
        'manifest.json', 'completed.json', 'resolved_env.json',
        'motion025_full_height_diagnosis.json')]
    checkpoint = Path(manifest['checkpoint'])
    count = len(manifest['cases']) * len(manifest['seeds'])
    criteria = read(GATE)
    assert diagnosis['criteria'] == criteria and sha(GATE) == GATE_SHA
    assert (complete['completed'] and complete['samples'] == count * 1000
            and manifest['seconds_per_case'] == 20. and manifest['policy_dt'] == .02
            and diagnosis['total_trials'] == count
            and manifest['checkpoint_sha256'] == diagnosis['checkpoint_sha256'] == sha(checkpoint)
            and not manifest['sampled_policy_actions'] and not manifest['weights_updated']
            and not manifest['policy_input_remapping'] and not manifest['action_bias_probes']
            and manifest['wheel_probe'] is False
            and manifest['wheel_reference_limit_probe'] is None
            and cfg['leg_control_mode'] == 'implicit_joint_position'
            and cfg['wheel_control_mode'] == 'implicit_velocity')
    files = [folder / name for name in ('manifest.json', 'completed.json', 'resolved_env.json',
        'motion025_full_height_diagnosis.json', 'policy_inputs.npz', 'trace.csv')]
    files.append(checkpoint)
    pins = {str(path): sha(path) for path in files}
    with (folder / 'trace.csv').open() as stream:
        header = stream.readline().strip().split(',')
    raw = np.loadtxt(folder / 'trace.csv', delimiter=',', skiprows=1,
                     usecols=[header.index(name) for name in NAMES])
    assert raw.shape == (count * 1000, len(NAMES)) and np.isfinite(raw).all()
    raw = raw.reshape(1000, count, len(NAMES))
    data = {name: raw[:, :, index] for index, name in enumerate(NAMES)}
    assert np.max(abs(data['time_s'] - .02 * np.arange(1, 1001)[:, None])) < 1.e-8
    assert np.array_equal(data['case_id'], np.broadcast_to(
        np.repeat(np.arange(len(manifest['cases'])), len(manifest['seeds'])), (1000, count)))
    assert np.array_equal(data['seed'], np.broadcast_to(
        np.tile(manifest['seeds'], len(manifest['cases'])), (1000, count)))
    done = (data['terminated'] != 0) | (data['timeout'] != 0)
    assert int(done.sum()) == complete['resets'] == diagnosis['resets']
    with np.load(folder / 'policy_inputs.npz', allow_pickle=False) as archive:
        means = archive['means'].copy()
        initial = archive['observations'][0].copy()
    assert means.shape == (1000, count, 6) and initial.shape == (count, 27)
    limit, scale = cfg['wheel_action_limit'], cfg['action_scale_vel']
    reference = np.stack([data[name] for name in
                          ('left_wheel_ref_rad_s', 'right_wheel_ref_rad_s')], -1)
    mapping = float(abs(reference - np.clip(means[..., [2, 5]], -limit, limit) * scale).max())
    assert mapping < 2.e-6
    original = {(row['case']['name'], row['seed']): row for row in diagnosis['trials']}
    assert len(original) == count
    radius_over_track = cfg['wheel_radius'] / cfg['wheel_track_width']
    rows = []
    parity = dict(original_yaw_MAE=0., signed_component_closure=0.,
                  original_positive_yaw_reward=0., original_negative_yaw_reward=0.)
    rewards = cfg['rewards']
    for cid, case in enumerate(manifest['cases']):
        for sid, seed in enumerate(manifest['seeds']):
            eid = cid * len(manifest['seeds']) + sid
            prior = original[(case['name'], seed)]
            survived = not done[:, eid].any() and not data['episode'][:, eid].any()
            assert survived == prior['survived']
            row = dict(case=case, seed=seed, survived=bool(survived),
                original_stage_constraints_passed=prior['stage_constraints_passed'],
                original_failed_metrics=prior['failed_metrics'],
                original_failed_prerequisites=prior['failed_prerequisites'],
                metrics=None, signed_components=None)
            if not survived:
                assert prior['metrics'] is None
                rows.append(row)
                continue
            mask = data['time_s'][:, eid] >= 17.
            assert mask.sum() == 151
            tail = {key: value[mask, eid] for key, value in data.items()}
            assert np.max(abs(tail['yaw_cmd_rad_s'] - case['yaw'])) < 1.e-7
            error = tail['yaw_rate_rad_s'] - tail['yaw_cmd_rad_s']
            ref = radius_over_track * (tail['right_wheel_ref_rad_s'] - tail['left_wheel_ref_rad_s'])
            measured = radius_over_track * (tail['right_wheel_rad_s'] - tail['left_wheel_rad_s'])
            components = dict(reference_minus_command=ref - tail['yaw_cmd_rad_s'],
                measured_wheel_proxy_minus_reference=measured - ref,
                body_minus_measured_wheel_proxy=tail['yaw_rate_rad_s'] - measured)
            parity['signed_component_closure'] = max(parity['signed_component_closure'],
                float(abs(sum(components.values()) - error).max()))
            mae = float(abs(error).mean())
            parity['original_yaw_MAE'] = max(parity['original_yaw_MAE'],
                                           abs(mae - prior['metrics']['yaw_mae_rad_s']))
            hratio = abs(tail['height_m'] - tail['height_cmd_m']) / (
                criteria['relative_error'] * abs(tail['height_cmd_m']))
            bratio = np.maximum(abs(tail['pitch_rad']), abs(tail['roll_rad'])) / math.radians(criteria['body_p95_deg'])
            floor = rewards['tracking_task_quality_gate_floor']
            quality = floor + (1. - floor) / ((1. + np.maximum(hratio - .5, 0.)**2)
                                             * (1. + np.maximum(bratio - .5, 0.)**2))
            positive = (rewards['tracking_ang_vel'] * np.exp(-error**2 / rewards['tracking_sigma_ang'])
                        * quality * manifest['policy_dt'])
            negative = np.maximum(rewards['yaw_rate_error_sq'] * error**2,
                                  -rewards['yaw_rate_penalty_clip']) * manifest['policy_dt']
            parity['original_positive_yaw_reward'] = max(parity['original_positive_yaw_reward'],
                float(abs(positive - tail['reward_track_ang_vel']).max()))
            parity['original_negative_yaw_reward'] = max(parity['original_negative_yaw_reward'],
                float(abs(negative - tail['reward_yaw_rate_error_sq']).max()))
            mse = float(np.mean(error**2))
            row.update(metrics=dict(yaw_MAE_rad_s=mae,
                nonzero_yaw_relative_percent=100 * mae / abs(case['yaw']) if case['yaw'] else None,
                yaw_mean_rad_s=float(tail['yaw_rate_rad_s'].mean()),
                signed_bias_rad_s=float(error.mean()),
                centered_body_yaw_std_rad_s=float(tail['yaw_rate_rad_s'].std()),
                bias_squared_fraction_of_MSE=float(error.mean()**2 / mse) if mse else None,
                original_positive_yaw_reward_mean=float(tail['reward_track_ang_vel'].mean()),
                original_negative_yaw_reward_mean=float(tail['reward_yaw_rate_error_sq'].mean()),
                wheel_reference_peak_abs_rad_s=float(abs(reference[mask, eid]).max()),
                wheel_reference_at_bound_frames=int(np.any(
                    abs(reference[mask, eid]) >= limit * scale - 1.e-6, axis=1).sum())),
                signed_components={key: dict(signed_mean_rad_s=float(value.mean()),
                    MAE_rad_s=float(abs(value).mean())) for key, value in components.items()})
            rows.append(row)
    assert len(rows) == count
    assert parity['original_yaw_MAE'] < 1.e-8 and parity['signed_component_closure'] < 1.e-12
    assert parity['original_positive_yaw_reward'] < 1.e-6 and parity['original_negative_yaw_reward'] < 1.e-6
    assert all(sha(path) == digest for path, digest in pins.items())
    return dict(checkpoint_sha256=manifest['checkpoint_sha256'], total_trials=count,
                all_original_trials_retained=True, trials=rows, parity=parity,
                target_mapping_max_error=mapping), manifest, cfg, initial, pins


def audit(source, candidate, destination):
    source, candidate, destination = [Path(path).resolve() for path in (source, candidate, destination)]
    assert not destination.exists() and destination.is_relative_to(ROOT / 'mine_model/results')
    before, sm, scfg, si, sp = process(source)
    after, cm, ccfg, ci, cp = process(candidate)
    assert sm['cases'] == cm['cases'] and sm['seeds'] == cm['seeds'] and np.array_equal(si, ci)
    cfg_diff = differences(scfg, ccfg)
    assert cfg_diff == [('/rewards/tracking_sigma_ang', .01, .0025)], cfg_diff
    pins = {**sp, **cp, str(SCRIPT): sha(SCRIPT), str(GATE): sha(GATE)}
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        source=before, candidate=after, exact_saved_configuration_differences=cfg_diff,
        all_cases_seeds_and_initial27_inputs_bitwise_paired=True,
        original_failed_None_and_zero_command_relative_None_preserved=True,
        original_actual_float32_commands_used_for_MAE_and_reward_reproduction=True,
        formula='body-command=(reference_proxy-command)+(measured_wheel_proxy-reference_proxy)+(body-measured_wheel_proxy)',
        source_pins=pins, simulator_started=False, optimizer_executed=False,
        controller_changed=False, training_modified=False, acceptance_changed=False,
        next_parameter_change_selected=False, policy_accepted=False,
        limitations=[
            'Signed components add; their MAEs do not. A large signed component is descriptive, not causal dominance.',
            'Wheel differential kinematics omit body motion, geometry and tire/contact dynamics. The residual is not a direct slip measurement.',
            'The finite velocity-servo residual can supply load torque. It alone does not prove a damping bug or justify a gain change.',
            'These completed fixed replays do not replace the full735, static, moving/steering, signed height or sustained reviews.',
        ])
    assert all(sha(path) == digest for path, digest in pins.items())
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.candidate, args.output)
    print(json.dumps(dict(verified=True, report=str(args.output), source_parity=result['source']['parity'],
                          candidate_parity=result['candidate']['parity'], policy_accepted=False)))
