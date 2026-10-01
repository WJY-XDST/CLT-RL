"""Compare a 15-phase heading-feedback replay against the same rate-only replay."""

import argparse
import csv
import json
import math
from pathlib import Path

from assess_yaw import yaw_sequence
from assess_tracking import percentile


def assess(path, feedback):
    with Path(path).open() as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    sequence = yaw_sequence()
    if len(rows) != len(sequence) * 250:
        raise ValueError("Incomplete heading test")
    reference = 0.
    errors, phases = [], []
    for i, row in enumerate(rows):
        speed, rate, height = sequence[i // 250]
        if not all(math.isfinite(v) for v in row.values()) or row['step'] != i:
            raise ValueError("Invalid or missing heading sample")
        if abs(row['sim_time_s'] - .02 * i) > 1e-5 or row['env_id'] != 0:
            raise ValueError("Incorrect time base or environment")
        if abs(row['cmd_x_target'] - speed) > 1e-5 or abs(row['height_cmd'] - height) > 1e-5:
            raise ValueError("Wrong evaluation commands")
        reference += math.degrees(rate * .02)
        if feedback:
            if row.get('heading_feedback') != 1 or abs(row['yaw_rate_requested'] - rate) > 1e-5:
                raise ValueError("Missing heading feedback or wrong requested rate")
            if abs(row['yaw_turn_reference_deg'] - reference) > 1e-4 or abs(row['yaw_rate_applied']) > .500001:
                raise ValueError("Reference was changed by feedback, or corrected command exceeds range")
        errors.append(row['yaw_turn_measured_deg'] - reference)
    for i, (speed, rate, height) in enumerate(sequence):
        start, end = i * 250 + 150, (i + 1) * 250
        phases.append({'phase': i, 'speed': speed, 'yaw_rate': rate,
                       'angle_p95_deg': percentile([abs(v) for v in errors[start:end]]),
                       'angle_final_deg': errors[end - 1]})
    resets = sum(r['terminated'] + r['time_out'] for r in rows)
    p95 = max(p['angle_p95_deg'] for p in phases)
    return {'trace': str(path), 'resets': resets, 'max_angle_error_deg': max(map(abs, errors)),
            'final_angle_error_deg': errors[-1], 'worst_steady_angle_p95_deg': p95,
            'provisional_angle_limit_deg': 1., 'passed': resets == 0 and p95 < 1.,
            'phases': phases}, rows, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('baseline', type=Path)
    args = parser.parse_args()
    candidate, rows, errors = assess(args.trace, True)
    baseline, old_rows, old_errors = assess(args.baseline, False)
    result = {'candidate': candidate, 'rate_only': baseline,
              'scope': 'same checkpoint and seed; heading controller comparison, not policy-training improvement or navigation acceptance'}
    output = args.trace.parent
    (output / 'heading_assessment.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    times = [r['sim_time_s'] for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    axes[0].plot(times, [r['yaw_turn_reference_deg'] for r in rows], '--', label='Requested turn')
    axes[0].plot(times, [r['yaw_turn_measured_deg'] for r in rows], label='Heading feedback')
    axes[0].plot(times, [r['yaw_turn_measured_deg'] for r in old_rows], label='Rate only', alpha=.6)
    axes[0].set_ylabel('Turn (deg)')
    axes[1].plot(times, errors, label='Heading error with feedback')
    axes[1].plot(times, old_errors, label='Rate-only heading error', alpha=.6)
    axes[1].axhspan(-1, 1, alpha=.15, color='green', label='Provisional +/-1 deg')
    axes[1].set_ylabel('Error (deg)')
    for key, label in [('yaw_rate_requested','Requested rate'), ('yaw_rate_applied','Corrected command'), ('yaw_rate_body','Measured rate')]:
        axes[2].plot(times, [r[key] for r in rows], label=label, linewidth=1)
    axes[2].set_ylabel('Yaw rate (rad/s)'); axes[2].set_xlabel('Simulation time (s)')
    for ax in axes:
        ax.grid(alpha=.25); ax.legend()
    fig.savefig(output / 'heading_comparison.png', dpi=140); plt.close(fig)
    print(json.dumps({k:{m:v for m,v in result[k].items() if m!='phases'} for k in ('candidate','rate_only')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
