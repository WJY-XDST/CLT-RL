"""Run unchanged new-model policy on nine flat-ground conditions, three seeds.

No training, resets, gain changes or hidden heading correction. Retain each
independent replay and assess with the existing preparation tolerances.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from core import PROJECT, load_config

ROOT = PROJECT.parent
sys.path.insert(0, str(ROOT / 'mine_model/scripts'))
from summarize_mine_trial import assess, plot

CASES = [('stand', 0., 0., .3), ('forward_0p3', .3, 0., .3), ('forward_0p6', .6, 0., .3),
         ('reverse_0p3', -.3, 0., .3), ('reverse_0p6', -.6, 0., .3),
         ('spin_left', 0., .3, .3), ('spin_right', 0., -.3, .3),
         ('height_low', 0., 0., .28), ('height_high', 0., 0., .32)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--mjcf', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=float, default=12.)
    parser.add_argument('--seeds', type=int, nargs=3, default=[43, 44, 45])
    parser.add_argument('--physics-substeps', type=int, default=50)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.checkpoint.parent / 'params/env.yaml')
    dt = cfg['sim']['dt'] * cfg['decimation']
    nsteps = round(args.seconds / dt)
    import torch
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    manifest = {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_iteration': checkpoint['iter'],
                'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                'seconds_per_case': args.seconds, 'policy_dt': dt, 'seeds': args.seeds,
                'cases': [dict(name=n, speed=v, yaw=y, height=h) for n, v, y, h in CASES],
                'reset_velocity_uniform_amplitude': .05, 'reset_rng': 'Isaac CPU torch Generator',
                'heading': 'constant yaw rate; no heading hold', 'automatic_resets': False,
                'mujoco_physics_substeps': args.physics_substeps,
                'mujoco_integration_dt': cfg['sim']['dt'] / args.physics_substeps,
                'weights_updated': False, 'gains_changed': False,
                'termination': 'immediate posture/height threshold; stricter than delayed Isaac termination'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    del checkpoint
    results, rows = [], []
    for cid, (name, speed, yaw, height) in enumerate(CASES):
        for seed in args.seeds:
            folder = output / name / f'seed{seed}'
            folder.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(Path(__file__).with_name('run.py')), '--checkpoint', str(args.checkpoint.resolve()),
                       '--mjcf', str(args.mjcf.resolve()), '--output', str(folder), '--headless', '--no-heading-hold',
                       '--fixed-command', str(speed), str(yaw), str(height), '--max-steps', str(nsteps),
                       '--phase-duration', str(args.seconds), '--seed', str(seed), '--reset-velocity', '.05',
                       '--reset-rng', 'isaac-torch', '--physics-substeps', str(args.physics_substeps)]
            with (folder / 'replay.log').open('w') as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           env={**os.environ, 'MUJOCO_GL': 'egl', 'OMP_NUM_THREADS': '1'})
            if not (folder / 'summary.json').is_file():
                raise RuntimeError(f'Replay crashed: {folder}/replay.log')
            summary = json.loads((folder / 'summary.json').read_text())
            survived = summary['status'] == 'completed' and summary['steps'] == nsteps and completed.returncode == 0
            result = {'case': name, 'seed': seed, 'survived': survived, 'status': summary['status'],
                      'failure_time_s': summary['failure_time_s'], 'folder': str(folder), 'summary': summary}
            results.append(result)
            with (folder / 'trace.csv').open() as handle:
                individual = list(csv.DictReader(handle))
            for index, record in enumerate(individual):
                r = {k: float(v) for k, v in record.items() if k != 'five_bar_valid'}
                row = {'time_s': r['step'] * dt, 'case_id': cid, 'seed': seed, 'episode': 0,
                       'terminated': int(not survived and index == len(individual) - 1), 'timeout': 0,
                       'speed_m_s': r['vel_x_heading'], 'speed_cmd_m_s': r['cmd_x'],
                       'yaw_rate_rad_s': r['yaw_rate'], 'yaw_cmd_rad_s': r['cmd_yaw'],
                       'height_m': r['base_height'], 'height_cmd_m': r['cmd_height'],
                       'pitch_rad': r['pitch_deg'] * 3.141592653589793 / 180,
                       'roll_rad': r['roll_deg'] * 3.141592653589793 / 180,
                       'closure_gap_max_mm': r['closure_gap_max_mm'],
                       'contact_penetration_max_mm': r['contact_penetration_max_mm'],
                       'stop_penetration_max_mm': r['stop_penetration_max_mm'],
                       'stop_contact_count': r['stop_contact_count'],
                       'five_bar_valid': int(record['five_bar_valid'] == 'True')}
                for side in ('left', 'right'):
                    row.update({side + '_length_m': r[side + '_l0'], side + '_length_ref_m': r[side + '_l0_ref'],
                                side + '_angle_rad': r[side + '_theta0_deg'] * 3.141592653589793 / 180,
                                side + '_angle_ref_rad': r[side + '_angle_ref_rad'],
                                side + '_wheel_rad_s': r[side + '_wheel_rad_s'], side + '_wheel_ref_rad_s': r[side + '_wheel_ref_rad_s']})
                rows.append(row)
            print(f'{len(results)}/27 {name} seed{seed}: {summary["status"]}, simulated {summary["duration_s"]:.2f}s', flush=True)
            (output / 'progress.json').write_text(json.dumps(results, indent=2) + '\n')
    with (output / 'trace.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    all_completed = all(r['survived'] for r in results)
    (output / 'completed.json').write_text(json.dumps({'completed': all_completed, 'trials': results}, indent=2) + '\n')
    if all_completed:
        report = assess(output)
        plot(output)
        print(json.dumps({k: report[k] for k in ('iteration', 'survivors', 'total_trials', 'passed')}, indent=2), flush=True)
    else:
        print('Some independent runs failed; partial traces retained, no complete precision acceptance.', flush=True)
    return 0 if all_completed else 2


if __name__ == '__main__':
    raise SystemExit(main())
