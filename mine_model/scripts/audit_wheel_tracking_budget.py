"""Decompose signed speed/yaw residuals using an intact full51 replay.

Wheel rolling proxies separate reference choice, measured wheel following and
body/proxy residuals. They are not a tire model or a new policy controller.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(value):
    return dict(signed_mean=float(value.mean()), MAE=float(np.abs(value).mean()),
                abs_P95=float(np.percentile(np.abs(value), 95)))


def audit(replay, output):
    replay, output = Path(replay).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve previous immutable wheel diagnostics')
    read = lambda name: json.loads((replay / name).read_text())
    manifest, complete = read('manifest.json'), read('completed.json')
    cfg, assessed = read('resolved_env.json'), read('motion025_full_height_diagnosis.json')
    source = Path(manifest['checkpoint'])
    if (not complete['completed'] or complete['samples'] != 51000 or complete['resets']
            or len(manifest['cases']) != 17 or manifest['seeds'] != [483, 484, 485]
            or manifest['seconds_per_case'] != 20. or manifest['policy_dt'] != .02
            or manifest['sampled_policy_actions'] or manifest['weights_updated']
            or manifest['policy_input_remapping'] or manifest['command_scale'] != 1.
            or manifest['action_bias_probes'] or manifest['wheel_probe']
            or manifest['wheel_reference_limit_probe'] is not None
            or sha(source) != manifest['checkpoint_sha256']
            or assessed['checkpoint_sha256'] != manifest['checkpoint_sha256']
            or assessed['criteria']['relative_error'] != .1
            or assessed['total_trials'] != 51 or assessed['survivors'] != 51
            or assessed['resets'] or cfg['leg_control_mode'] != 'implicit_joint_position'
            or cfg['action_scale_vel'] <= 0. or cfg['wheel_radius'] <= 0.
            or cfg['wheel_track_width'] <= 0.):
        raise ValueError('Require the intact deterministic full51 direct-joint source')
    if any(sha(path) != digest for path, digest in assessed['source_files'].items()):
        raise ValueError('Original assessed physical data changed')
    trace = np.genfromtxt(replay / 'trace.csv', delimiter=',', names=True)
    if len(trace) != 51000 or any(not np.isfinite(trace[k]).all() for k in trace.dtype.names):
        raise ValueError('Incomplete or nonfinite physical wheel replay')
    with np.load(replay / 'policy_inputs.npz', allow_pickle=False) as data:
        means = data['means'].copy()
    if means.shape != (1000, 51, 6) or not np.isfinite(means).all():
        raise ValueError('Incomplete raw policy means')
    rows, closure, reference_parity = [], 0., 0.
    radius, track, scale = (cfg[k] for k in ('wheel_radius', 'wheel_track_width', 'action_scale_vel'))
    for cid, case in enumerate(manifest['cases']):
        for sid, seed in enumerate(manifest['seeds']):
            part = trace[(trace['case_id'] == cid) & (trace['seed'] == seed)]
            original = assessed['trials'][cid * 3 + sid]
            if (len(part) != 1000 or original['case'] != case or original['seed'] != seed
                    or not original['survived'] or np.any(part['episode'])
                    or np.any(part['terminated']) or np.any(part['timeout'])
                    or not np.allclose(part['time_s'], np.arange(1, 1001)*.02, rtol=0., atol=1.e-8)):
                raise ValueError('Mismatched full first-episode command/seed timeline')
            raw = means[:, cid * 3 + sid]
            expected = np.clip(raw[:, [2, 5]], -1., 1.) * scale
            reference = np.stack((part['left_wheel_ref_rad_s'], part['right_wheel_ref_rad_s']), -1)
            error = float(np.max(np.abs(expected - reference)))
            reference_parity = max(reference_parity, error)
            if error > 1.e-5:
                raise ValueError('Policy means do not reconstruct the recorded wheel targets')
            mask = part['time_s'] >= 17.
            tail = part[mask]
            actual = np.stack((tail['left_wheel_rad_s'], tail['right_wheel_rad_s']), -1)
            reference, raw = reference[mask], raw[mask]
            channels = {}
            for label, body, command, ref_proxy, actual_proxy, unit in (
                ('speed', tail['speed_m_s'], tail['speed_cmd_m_s'],
                 radius*reference.mean(-1), radius*actual.mean(-1), 'm/s'),
                ('yaw', tail['yaw_rate_rad_s'], tail['yaw_cmd_rad_s'],
                 radius*(reference[:, 1]-reference[:, 0])/track,
                 radius*(actual[:, 1]-actual[:, 0])/track, 'rad/s')):
                components = dict(reference_rolling_proxy_minus_command=ref_proxy-command,
                    actual_wheel_proxy_minus_reference_proxy=actual_proxy-ref_proxy,
                    body_measurement_minus_actual_wheel_proxy=body-actual_proxy)
                total = body-command
                residual = float(np.abs(sum(components.values())-total).max())
                closure = max(closure, residual)
                metric = 'speed_mae_m_s' if label == 'speed' else 'yaw_mae_rad_s'
                if abs(float(np.abs(total).mean())-original['metrics'][metric]) > 1.e-9:
                    raise ValueError('Original independent speed/yaw metric does not reproduce')
                channels[label] = dict(unit=unit, body_minus_command=describe(total),
                    signed_components={key: describe(value) for key, value in components.items()})
            rows.append(dict(case=case, seed=seed, tail_samples=len(tail),
                original_stage_constraints_passed=original['stage_constraints_passed'], channels=channels,
                wheel_reference_mean_rad_s=reference.mean(0).tolist(),
                actual_wheel_mean_rad_s=actual.mean(0).tolist(),
                wheel_following_error_MAE_rad_s=np.abs(actual-reference).mean(0).tolist(),
                wheel_raw_mean_outside_action_box_fraction=(np.abs(raw[:, [2, 5]]) > 1.).mean(0).tolist(),
                leg_raw_mean_outside_action_box_fraction=(np.abs(raw[:, [0, 1, 3, 4]]) > 1.).mean(0).tolist(),
                standing_velocity_reward_mean=float(tail['reward_standing_velocity'].mean()),
                recorded_torque_saturation_fraction_mean=float(tail['torque_saturation_fraction'].mean())))
    if closure > 1.e-12:
        raise ValueError('The signed residual decomposition does not close')
    files = [source, Path(__file__).resolve(), *(replay / name for name in
        ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv',
         'policy_inputs.npz', 'motion025_full_height_diagnosis.json'))]
    result = dict(observed_at=datetime.now().astimezone().isoformat(), checkpoint=str(source),
        checkpoint_sha256=sha(source), complete_trials=51, complete_samples=51000,
        wheel_radius_m=radius, wheel_track_width_m=track,
        wheel_convention='Both wheel rates are recorded forward-positive; right native joint sign is already reversed by the environment.',
        target_from_raw_mean_max_error_rad_s=reference_parity,
        signed_decomposition_closure_max_error=closure, trials=rows,
        source_pins={str(path): sha(path) for path in files}, training_modified=False,
        controller_changed=False, optimizer_executed=False, simulator_started=False,
        acceptance_changed=False, policy_accepted=False,
        limitations=['Only the original final3-second independent steady windows are decomposed.',
            'The signed components add exactly; their MAEs and P95 values do not add.',
            'A body/rolling-proxy residual can include leg motion, wheel geometry, body attitude, contact and slip; it is not a slip estimate.',
            'Wheel tracking error can be the finite servo error needed to produce load torque. It does not alone establish a PD bug or justify a gain change.',
            'Combined applied-torque telemetry is not independent proof that the PhysX internal wheel drive never saturates.',
            'The measured steady decomposition does not explain startup failures or prove a causal training improvement.'])
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.replay, args.output)
    print(json.dumps(dict(report=str(args.output), complete_trials=result['complete_trials'],
        target_reconstruction_error=result['target_from_raw_mean_max_error_rad_s'],
        signed_closure_error=result['signed_decomposition_closure_max_error']), ensure_ascii=False))
