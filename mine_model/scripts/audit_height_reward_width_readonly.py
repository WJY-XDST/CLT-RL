"""Rescore height reward widths on complete existing native replay traces.

This runs no policy, simulator or optimizer. Analytic reward derivatives are
not policy gradients and do not establish the effect of a training change.
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
    if output.exists():
        raise ValueError('Preserve existing evidence')
    paths = {name: directory / name for name in (
        'manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv',
        'motion025_full_height_diagnosis.json')}
    read = lambda name: json.loads(paths[name].read_text())
    manifest, complete, cfg, assessed = (read(name) for name in (
        'manifest.json', 'completed.json', 'resolved_env.json',
        'motion025_full_height_diagnosis.json'))
    reward = cfg['rewards']
    if (not complete['completed'] or manifest['weights_updated']
            or manifest['sampled_policy_actions'] or manifest['policy_input_remapping']
            or manifest['command_scale'] != 1. or not manifest['command_probe']
            or manifest['command_sequence'] is not None
            or manifest['seconds_per_case'] != 20. or manifest['seeds'] != [483, 484, 485]
            or len(manifest['cases']) != 17 or len(assessed['trials']) != 51
            or assessed['checkpoint_sha256'] != manifest['checkpoint_sha256']
            or reward['base_height'] != 2. or reward['base_height_sigma'] != .0025
            or reward['base_height_relative_deadband'] != 0.
            or reward['tracking_task_quality_gate_floor'] != 1.):
        raise ValueError('Require the original full-height51 deterministic reward contract')
    pins = {str(path): sha(path) for path in paths.values()}
    if any(sha(path) != value for path, value in assessed['source_files'].items()):
        raise ValueError('Recorded replay evidence changed')
    data = np.genfromtxt(paths['trace.csv'], delimiter=',', names=True)
    dt = cfg['sim']['dt'] * cfg['decimation']
    steps = round(manifest['seconds_per_case'] / dt)
    expected = steps * len(manifest['cases']) * len(manifest['seeds'])
    if (len(data) != expected or complete['samples'] != expected
            or not all(np.isfinite(data[key]).all() for key in data.dtype.names)):
        raise ValueError('Incomplete or nonfinite native transitions')
    weight, original = reward['base_height'], reward['base_height_sigma']
    error = data['height_m'] - data['height_cmd_m']
    replayed = weight * np.exp(-error ** 2 / original) * dt
    parity = float(np.abs(replayed - data['reward_base_height']).max())
    if parity > 2.e-7:
        raise ValueError('Height kernel does not reproduce recorded native rewards')
    widths = (.0025, .0015, .001, .0005)
    rows, survivors = [], 0
    for cid, case in enumerate(manifest['cases']):
        tails, seeds = [], []
        for seed in manifest['seeds']:
            trial = data[(data['case_id'] == cid) & (data['seed'] == seed)]
            if (len(trial) != steps or not np.allclose(trial['time_s'],
                    np.arange(1, steps + 1) * dt, rtol=0., atol=1.e-8)
                    or not np.allclose(trial['height_cmd_m'], case['height'], rtol=0., atol=1.e-7)):
                raise ValueError('Incomplete trial timeline or changed height commands')
            survived = not any(np.any(trial[key]) for key in ('episode', 'terminated', 'timeout'))
            seeds.append(dict(seed=seed, survived=survived))
            if survived:
                tails.append(trial[-round(3. / dt):])
                survivors += 1
        if not tails:
            rows.append(dict(case=case, seeds=seeds, sensitivity=None))
            continue
        tail = np.concatenate(tails)
        e = tail['height_m'] - tail['height_cmd_m']
        sensitivity = []
        for sigma in widths:
            kernel = np.exp(-e ** 2 / sigma)
            sensitivity.append(dict(base_height_sigma=sigma,
                positive_reward_fraction_of_max=float(kernel.mean()),
                positive_reward_per_s=float((weight * kernel).mean()),
                positive_reward_gain_at_exact_height_per_s=float((weight * (1. - kernel)).mean()),
                mean_abs_dreward_dactual_height=float((2. * weight * np.abs(e) / sigma * kernel).mean())))
        rows.append(dict(case=case, seeds=seeds, samples=len(tail),
            actual_height_mean_m=float(tail['height_m'].mean()),
            height_MAE_m=float(np.abs(e).mean()),
            height_relative_MAE=float((np.abs(e) / tail['height_cmd_m']).mean()),
            original_orientation_cost_per_s=float((tail['reward_orientation'] / dt).mean()),
            original_linear_speed_reward_per_s=float((tail['reward_track_lin_vel'] / dt).mean()),
            sensitivity=sensitivity))
    if survivors != assessed['survivors'] or any(sha(path) != value for path, value in pins.items()):
        raise ValueError('Survivor count or immutable source evidence changed')
    report = dict(observed_at=datetime.now().astimezone().isoformat(),
        checkpoint_sha256=manifest['checkpoint_sha256'], recorded_native_samples=len(data),
        surviving_tail_trials=survivors, original_native_reward_formula_max_error=parity,
        original_base_height_sigma=original, candidate_widths=list(widths), per_case=rows,
        source_sha256=pins, script_sha256=sha(__file__), selected_width=None,
        training_modified=False, simulator_started=False, model_loaded=False,
        optimizer_loaded=False, PPO_updates_performed=0, acceptance_changed=False,
        policy_accepted=False, limitations=[
            'Rescores height rewards on the same recorded states; no new trajectory is predicted.',
            'Analytic reward derivatives are not PPO parameter gradients or causal training effects.',
            'Failed/reset episodes have no steady tail metric, never a zero error.',
            'No width is selected or applied; full200 and independent27+51 remain required.'])
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.directory, args.output)
    print(json.dumps(dict(report=str(args.output.resolve()),
        original_native_reward_formula_max_error=result['original_native_reward_formula_max_error'],
        rows=[row for row in result['per_case'] if row['case']['name'] in
              ('stage025_height_low', 'stage025_low_reverse')]), ensure_ascii=False))
