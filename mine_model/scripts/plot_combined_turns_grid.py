"""Plot every paired signed-speed/yaw/height command without hiding failures."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


def plot(source, prefix):
    source, prefix = Path(source).resolve(), Path(prefix).resolve()
    destinations = [prefix.with_suffix(s) for s in ('.png', '.svg', '.json')]
    if any(p.exists() for p in destinations):
        raise ValueError('Preserve the previous immutable plot')
    report = json.loads(source.read_text())
    rows = report['paired_trials']
    if len(rows) != 324 or not report['same_reset_seeds_initial_observations_and_physical_configuration_verified']:
        raise ValueError('Require the complete paired324-trial grid and native configuration/initial-state proof')
    speeds = sorted({t['case']['speed'] for t in rows})
    yaws = sorted({t['case']['yaw'] for t in rows})
    heights = sorted({t['case']['height'] for t in rows})
    if (len(speeds), len(yaws), len(heights)) != (6, 6, 3):
        raise ValueError('Retain all signed speeds, yaws and actual height endpoints')
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.8))
    colors = ['#f3a5a0', '#f5d88f', '#a5d8ad']
    cells = []
    for rid, owner in enumerate(('baseline', 'candidate')):
        for hid, height in enumerate(heights):
            ax = axes[rid, hid]
            status = np.zeros((6, 6), dtype=int)
            annotations = {}
            for yi, yaw in enumerate(yaws):
                for vi, speed in enumerate(speeds):
                    selected = [t for t in rows if t['case']['speed'] == speed
                        and t['case']['yaw'] == yaw and t['case']['height'] == height]
                    if len(selected) != 3 or len({t['seed'] for t in selected}) != 3:
                        raise ValueError('Every command must retain exactly three paired reset seeds')
                    passed = sum(t[owner + '_passed'] for t in selected)
                    survived = sum(t[owner + '_survived'] for t in selected)
                    if passed > survived:
                        raise ValueError('A failed-survival trial cannot pass tracking')
                    status[yi, vi] = 0 if survived < 3 else 2 if passed == 3 else 1
                    annotations[yi, vi] = f'P{passed} S{survived}'
                    cells.append(dict(owner=owner, speed=speed, yaw=yaw, height=height,
                        total_seeds=3, passed=passed, survived=survived))
            ax.imshow(status, cmap=ListedColormap(colors), vmin=-.5, vmax=2.5,
                origin='lower', interpolation='nearest', aspect='auto')
            for (yi, vi), label in annotations.items():
                ax.text(vi, yi, label, ha='center', va='center', fontsize=9, color='#202020')
            ax.set_xticks(range(6), [f'{v:+g}' for v in speeds])
            ax.set_yticks(range(6), [f'{w:+g}' for w in yaws])
            ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
            ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
            ax.grid(which='minor', color='white', linewidth=1.2)
            ax.tick_params(which='minor', bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            clock = 5201 if owner == 'baseline' else report['actual_updates']
            ax.set_title(f'{owner.capitalize()} {clock} | height {height:.3f} m', fontsize=12, pad=9)
            ax.set_xlabel('Forward speed command (m/s)')
            ax.set_ylabel('Yaw rate command (rad/s)')
    fig.suptitle('Travel while turning: every command, including unstable trials', fontsize=16, y=.98)
    fig.legend([Patch(facecolor=c) for c in colors],
        ['At least one survival failure', 'All survived; tracking failed', 'All passed'],
        loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.5, .045), fontsize=10)
    fig.text(.5, .023, 'Each cell: P = passed seeds, S = survived seeds, out of 3. Original 10% rules and absolute floors retained.',
        ha='center', fontsize=10)
    fig.subplots_adjust(left=.065, right=.985, top=.915, bottom=.15, wspace=.26, hspace=.48)
    fig.savefig(destinations[0], dpi=170, facecolor='white')
    fig.savefig(destinations[1], facecolor='white')
    plt.close(fig)
    value = dict(observed_at=datetime.now().astimezone().isoformat(), source=str(source),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_grid_trials_per_model=324, complete_commands_per_model=108,
        all_failed_trials_retained=True, cells=cells, policy_accepted=False)
    with destinations[2].open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return destinations


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--summary', type=Path, required=True)
    cli.add_argument('--prefix', type=Path, required=True)
    args = cli.parse_args()
    print(json.dumps([str(p) for p in plot(args.summary, args.prefix)]))
