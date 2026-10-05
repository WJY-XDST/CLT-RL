"""Compare identical-checkpoint replays before adopting simplified leg colliders."""
import argparse
import json
from pathlib import Path
import numpy as np
from summarize_mine_trial import assess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifests = [json.loads((p/'manifest.json').read_text()) for p in (args.baseline, args.candidate)]
    for key in ('checkpoint_sha256', 'cases', 'seeds', 'seconds_per_case', 'physics_dt', 'policy_dt'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('Unpaired comparison: ' + key)
    reports = [assess(p) for p in (args.baseline, args.candidate)]
    traces = [np.genfromtxt(p/'trace.csv', delimiter=',', names=True) for p in (args.baseline, args.candidate)]
    metrics = {}
    increases = []
    for key in reports[0]['conditions'][0]['trials'][0]['limits']:
        values = [[t['metrics'][key] for c in r['conditions'] for t in c['trials'] if t['metrics']] for r in reports]
        if len(values[0]) != reports[0]['total_trials'] or len(values[1]) != len(values[0]):
            increases.append('survival')
            continue
        metrics[key] = {'baseline_mean': float(np.mean(values[0])), 'candidate_mean': float(np.mean(values[1])),
                        'baseline_max': max(values[0]), 'candidate_max': max(values[1]),
                        'maximum_paired_increase': float(np.max(np.array(values[1])-values[0]))}
        # Only roundoff is ignored. Even one worse condition prevents adoption.
        if metrics[key]['maximum_paired_increase'] > 1e-9:
            increases.append(key)
    same_telemetry = len(traces[0]) == len(traces[1]) and traces[0].dtype.names == traces[1].dtype.names
    difference = max(float(np.max(abs(traces[0][k]-traces[1][k]))) for k in traces[0].dtype.names) if same_telemetry else None
    result = {'baseline': str(args.baseline.resolve()), 'candidate': str(args.candidate.resolve()),
              'source_checkpoint_sha256': manifests[0]['checkpoint_sha256'], 'metrics': metrics,
              'increased_metrics': increases, 'maximum_telemetry_difference': difference,
              'adopt_simplified_legs': not increases and reports[1]['survivors']==reports[1]['total_trials'],
              'scope': 'Paired flat-ground policy replay only; fall/obstacle contact shape equivalence is not established.'}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'collision_comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 3, figsize=(12, 9), sharex=True, layout='constrained')
    for column, case in enumerate((0, 1, 5)):
        for index, trace in enumerate(traces):
            d = trace[(trace['case_id']==case)&(trace['seed']==manifests[0]['seeds'][0])]
            errors = [d['speed_m_s']-d['speed_cmd_m_s'], d['yaw_rate_rad_s']-d['yaw_cmd_rad_s'],
                      1000*(d['height_m']-d['height_cmd_m']),
                      1000*(d['left_length_m']-d['left_length_ref_m'])]
            for row, y in enumerate(errors):
                axes[row, column].plot(d['time_s'], y, '-' if index==0 else '--', lw=1,
                                      label=('Original leg CAD', 'Simplified ordinary links')[index])
                axes[row, column].grid(alpha=.2)
        axes[0, column].set_title(('Stand', 'Forward 0.3 m/s', 'Spin 0.3 rad/s')[column])
        axes[-1, column].set_xlabel('Time (s)')
    for axis, label in zip(axes[:, 0], ('Speed error (m/s)', 'Yaw error (rad/s)', 'Height error (mm)', 'Left leg length error (mm)')):
        axis.set_ylabel(label)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.output/'collision_ab_errors.png', dpi=160)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
