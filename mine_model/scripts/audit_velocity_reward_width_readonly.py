"""Rescore recorded velocity kernels without running a policy or changing training.

This diagnoses reward sensitivity on an existing complete native full-height
low-speed replay. Counterfactual rewards do not predict a trained policy.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.exists() or output.parent != directory.parent.parent:
        raise ValueError('Use a fresh report next to the existing replay bundle')
    paths = {name: directory / name for name in (
        'manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv',
        'motion025_full_height_diagnosis.json')}
    read = lambda name: json.loads(paths[name].read_text())
    manifest, complete, cfg, assessed = (read(name) for name in (
        'manifest.json', 'completed.json', 'resolved_env.json',
        'motion025_full_height_diagnosis.json'))
    if (not complete['completed'] or complete['samples'] != 51000 or complete['resets'] != 0
            or manifest['seeds'] != [483, 484, 485] or manifest['seconds_per_case'] != 20.
            or len(manifest['cases']) != 17 or manifest['weights_updated']
            or manifest['sampled_policy_actions'] or manifest['policy_input_remapping']
            or manifest['command_scale'] != 1. or not manifest['command_probe']
            or manifest['settle_seconds'] != 0. or manifest['command_sequence'] is not None
            or assessed['checkpoint_sha256'] != manifest['checkpoint_sha256']
            or len(assessed['trials']) != 51
            or cfg['rewards']['tracking_task_quality_gate_floor'] != 1.
            or cfg['rewards']['tracking_lin_vel'] != 2.
            or cfg['rewards']['tracking_sigma'] != .25):
        raise ValueError('Require the unchanged full51 native replay and recorded base kernel')
    pinned = {str(path): sha(path) for path in paths.values()}
    for path, digest in assessed['source_files'].items():
        if sha(path) != digest:
            raise ValueError('Recorded replay changed: ' + path)
    data = np.genfromtxt(paths['trace.csv'], delimiter=',', names=True)
    if len(data) != 51000 or not all(np.isfinite(data[key]).all() for key in data.dtype.names):
        raise ValueError('Incomplete or nonfinite recorded transitions')
    dt = cfg['sim']['dt'] * cfg['decimation']
    weight, original = cfg['rewards']['tracking_lin_vel'], cfg['rewards']['tracking_sigma']
    error = data['speed_cmd_m_s'] - data['speed_m_s']
    replayed = weight * np.exp(-error ** 2 / original) * dt
    parity_error = float(np.max(np.abs(replayed - data['reward_track_lin_vel'])))
    if parity_error > 2.e-7:
        raise ValueError('Rescoring does not match the recorded native velocity reward')
    # Do not mix startup or failed/reset episodes into steady-policy evidence.
    tails, rows = [], []
    ramp_error = 0.
    widths = (.25, .1, .05, .025, .01)
    for cid, case in enumerate(manifest['cases']):
        seed_rows = []
        for seed in manifest['seeds']:
            trial = data[(data['case_id'] == cid) & (data['seed'] == seed)]
            if len(trial) != 1000 or not np.allclose(
                    trial['time_s'], np.arange(1, 1001) * dt, rtol=0., atol=1.e-8):
                raise ValueError('Missing full native episode timeline')
            # The recorded command ramps at the configured acceleration from
            # rest. Compare the full applied ramp, not the eventual target.
            expected_speed = np.sign(case['speed']) * np.minimum(abs(case['speed']),
                cfg['commands']['linear_acceleration_limit'] * trial['time_s'])
            ramp_error = max(ramp_error, float(np.max(np.abs(trial['speed_cmd_m_s'] - expected_speed))))
            if (not np.allclose(trial['speed_cmd_m_s'], expected_speed, atol=1.e-7, rtol=0.)
                    or not np.allclose(trial['yaw_cmd_rad_s'], case['yaw'], atol=1.e-7, rtol=0.)
                    or not np.allclose(trial['height_cmd_m'], case['height'], atol=1.e-7, rtol=0.)):
                raise ValueError('Recorded commands differ from the complete command plan')
            survived = not any(np.any(trial[key]) for key in ('episode', 'terminated', 'timeout'))
            seed_rows.append(dict(seed=seed, survived=survived))
            if survived:
                tail = trial[trial['time_s'] >= manifest['seconds_per_case'] - 3.]
                tails.append((cid, seed, tail))
        parts = [tail for case_id, _, tail in tails if case_id == cid]
        if not parts:
            rows.append(dict(case=case, seeds=seed_rows, sensitivity=None))
            continue
        tail = np.concatenate(parts)
        e = tail['speed_cmd_m_s'] - tail['speed_m_s']
        sensitivity = []
        for sigma in widths:
            kernel = np.exp(-e ** 2 / sigma)
            sensitivity.append(dict(tracking_sigma=sigma,
                positive_reward_fraction_of_max=float(kernel.mean()),
                positive_reward_per_s=float((weight * kernel).mean()),
                positive_reward_gain_at_exact_vx_per_s=float((weight * (1. - kernel)).mean()),
                mean_abs_dreward_dactual_vx=float((2. * weight * np.abs(e) / sigma * kernel).mean()),
                mean_signed_dreward_dactual_vx=float((2. * weight * e / sigma * kernel).mean())))
        penalty_e = (e ** 2 + cfg['rewards']['lin_vel_error_lateral_multiplier']
                     * tail['lateral_speed_m_s'] ** 2)
        expected_penalty = np.clip(penalty_e * cfg['rewards']['lin_vel_error_sq'] * dt,
                                   -cfg['rewards']['lin_vel_penalty_clip'] * dt,
                                   cfg['rewards']['clip_single_reward'] * dt)
        penalty_error = float(np.max(np.abs(expected_penalty - tail['reward_lin_vel_error_sq'])))
        if penalty_error > 2.e-7:
            raise ValueError('Existing dense velocity cost was not reproduced')
        rows.append(dict(case=case, seeds=seed_rows, samples=int(len(tail)),
            speed_actual_mean_m_s=float(tail['speed_m_s'].mean()),
            speed_MAE_m_s=float(np.abs(e).mean()),
            existing_squared_speed_cost_per_s=float((tail['reward_lin_vel_error_sq'] / dt).mean()),
            standing_velocity_cost_per_s=float((tail['reward_standing_velocity'] / dt).mean()),
            height_positive_reward_per_s=float((tail['reward_base_height'] / dt).mean()),
            height_squared_cost_per_s=float((tail['reward_base_height_error_sq'] / dt).mean()),
            orientation_cost_per_s=float((tail['reward_orientation'] / dt).mean()),
            existing_dense_velocity_cost_formula_max_error=penalty_error,
            sensitivity=sensitivity))
    if any(sha(path) != digest for path, digest in pinned.items()):
        raise ValueError('Recorded data changed during read-only analysis')
    result = dict(observed_at=datetime.now().astimezone().isoformat(),
        checkpoint_sha256=manifest['checkpoint_sha256'],
        recorded_native_samples=len(data), surviving_tail_trials=len(tails),
        original_native_reward_formula_max_error=parity_error,
        original_applied_command_ramp_max_error_m_s=ramp_error,
        current_tracking_sigma=original, candidate_widths=list(widths),
        per_case=rows, source_pins=pinned, script_sha256=sha(__file__),
        full_height_range=assessed['full_height_range'], read_only=True,
        training_modified=False, simulator_started=False, model_loaded=False,
        optimizer_loaded=False, PPO_updates_performed=0, acceptance_changed=False,
        policy_accepted=False,
        limitations=[
            'Only the positive forward-speed reward is rescored on the same recorded states.',
            'Analytic reward derivatives are not PPO parameter gradients or causal training effects.',
            'The dense forward/lateral squared penalty is already active and is left unchanged.',
            'No width is selected or applied; current native200 and independent27+51 must finish.',
            'Full-scope stability/height/velocity and independent sustained validation remain required.'])
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.output)
    print(json.dumps(dict(report=str(args.output.resolve()),
        recorded_native_samples=result['recorded_native_samples'],
        original_native_reward_formula_max_error=result['original_native_reward_formula_max_error'],
        rows=[row for row in result['per_case'] if row['case']['name'] in
              ('stage025_forward02', 'stage025_reverse02')]), ensure_ascii=False))
