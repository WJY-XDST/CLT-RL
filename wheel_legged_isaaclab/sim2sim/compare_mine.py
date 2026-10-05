"""Compare matched new-model Isaac/MuJoCo replays without hiding failed gates."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--mujoco-folder', default='mujoco_matrix')
    parser.add_argument('--mechanical-folder', default='mechanical_stops')
    args = parser.parse_args()
    root = args.directory.resolve()
    reports = {name: json.loads((root / folder / 'assessment.json').read_text())
               for name, folder in [('Isaac Lab', 'isaac_baseline'), ('MuJoCo', args.mujoco_folder)]}
    if reports['Isaac Lab']['checkpoint_sha256'] != reports['MuJoCo']['checkpoint_sha256']:
        raise ValueError('Checkpoint mismatch')
    traces = {name: np.genfromtxt(root / folder / 'trace.csv', delimiter=',', names=True)
              for name, folder in [('Isaac Lab', 'isaac_baseline'), ('MuJoCo', args.mujoco_folder)]}
    keys = ['speed_mae_m_s', 'yaw_mae_rad_s', 'height_mae_m', 'leg_length_mae_m',
            'leg_angle_mae_rad', 'wheel_speed_mae_rad_s', 'body_p95_deg']
    rows = []
    for name, report in reports.items():
        for condition in report['conditions']:
            for trial in condition['trials']:
                rows.append({'simulator': name, 'case': condition['name'], 'seed': trial['seed'],
                             'passed': trial['passed'], **trial['metrics']})
    with (root / 'simulator_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    comparison = {'checkpoint': reports['MuJoCo']['checkpoint'], 'checkpoint_sha256': reports['MuJoCo']['checkpoint_sha256'],
                  'simulators': {}, 'cases': [], 'same_weights_and_gains': True,
                  'same_initial_velocity_draws': True, 'initial_velocity_amplitude': .05,
                  'steady_window_s': 3, 'boundary': 'Flat ground, 12 s per condition, three seeds; separate contact solvers. Not slope/disturbance or real-robot validation.'}
    for name, report in reports.items():
        trials = [t for c in report['conditions'] for t in c['trials']]
        comparison['simulators'][name] = {'survivors': report['survivors'], 'total_trials': report['total_trials'],
            'passed_trials': sum(t['passed'] for t in trials), 'all_precision_passed': report['passed'],
            'worst_metrics': {key: max(t['metrics'][key] for t in trials) for key in keys}}
    for cid, condition in enumerate(reports['MuJoCo']['conditions']):
        entry = {'name': condition['name'], 'target_speed': condition['speed'], 'target_yaw': condition['yaw'],
                 'target_height': condition['height'], 'worst_mae': {}}
        for name, report in reports.items():
            trials = report['conditions'][cid]['trials']
            entry['worst_mae'][name] = {key: max(t['metrics'][key] for t in trials) for key in keys}
        comparison['cases'].append(entry)
    mu = traces['MuJoCo']
    comparison['closed_chain'] = {'max_recorded_gap_mm': float(max(mu['closure_gap_max_mm'])),
        'max_recorded_stop_contact_penetration_mm': float(max(mu['stop_penetration_max_mm'])),
        'stop_contacts_in_basic_motion': int(max(mu['stop_contact_count'])),
        'max_recorded_all_contact_penetration_mm': float(max(mu['contact_penetration_max_mm'])),
        'all_five_bar_valid': bool(np.all(mu['five_bar_valid'] == 1)),
        'sample_dt_s': .02, 'sampling_boundary': 'Post-control-step samples; not all internal substeps and not original CAD boolean intersection.'}
    if (root / args.mechanical_folder / 'report.json').exists():
        comparison['mechanical_stops'] = json.loads((root / args.mechanical_folder / 'report.json').read_text())
    if (root / args.mechanical_folder / 'cad_nonpenetration.json').exists():
        comparison['mechanical_stops_cad_audit'] = json.loads((root / args.mechanical_folder / 'cad_nonpenetration.json').read_text())
    (root / 'comparison.json').write_text(json.dumps(comparison, indent=2) + '\n')
    names = [c['name'] for c in reports['MuJoCo']['conditions']]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout='constrained')
    for ax, key, label in zip(axes.ravel(), keys[:6], ['Speed', 'Yaw rate', 'Height', 'Leg length', 'Leg angle', 'Wheel speed']):
        for offset, (name, report) in zip([-.18, .18], reports.items()):
            ratios = [max(t['metrics'][key] / t['limits'][key] for t in c['trials']) for c in report['conditions']]
            ax.barh(np.arange(9) + offset, ratios, height=.35, label=name)
        ax.axvline(1, color='black', linestyle='--', lw=1)
        ax.set_yticks(np.arange(9), names); ax.invert_yaxis()
        ax.set_title(label + ': worst seed MAE / tolerance'); ax.grid(axis='x', alpha=.25)
    axes[0, 0].legend()
    fig.suptitle('model_6700: matched flat-ground transfer; dashed line = acceptance limit')
    fig.savefig(root / 'transfer_error_comparison.png', dpi=150); plt.close(fig)
    # All nine conditions: actual error curves for seed43, same axes and units.
    fig, axes = plt.subplots(9, 3, figsize=(15, 20), sharex=True, layout='constrained')
    for cid, case in enumerate(names):
        for name, data in traces.items():
            d = data[(data['case_id'] == cid) & (data['seed'] == 43)]
            errors = [d['speed_m_s'] - d['speed_cmd_m_s'], d['yaw_rate_rad_s'] - d['yaw_cmd_rad_s'],
                      1000 * (d['height_m'] - d['height_cmd_m'])]
            for col, error in enumerate(errors):
                axes[cid, col].plot(d['time_s'], error, linewidth=.7, alpha=.85, label=name)
                axes[cid, col].axhline(0, color='grey', lw=.4); axes[cid, col].grid(alpha=.2)
        axes[cid, 0].set_ylabel(case)
    for col, label in enumerate(['Speed error (m/s)', 'Yaw-rate error (rad/s)', 'Height error (mm)']):
        axes[0, col].set_title(label); axes[-1, col].set_xlabel('Time (s)')
    axes[0, 0].legend()
    fig.suptitle('Same actor and commands, seed43: Isaac Lab vs MuJoCo')
    fig.savefig(root / 'paired_error_curves.png', dpi=150); plt.close(fig)
    return comparison


if __name__ == '__main__':
    main()
