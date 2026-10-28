"""Read all frozen trials and separate signed reference, drive and body residuals.

Rolling wheel proxies are diagnostic algebra, not measurements of tire slip or
contact force. This program cannot start a simulator, update a policy or select
training commands, gains, rewards or acceptance tolerances.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve()
FIELDS = ('time_s', 'case_id', 'seed', 'yaw_rate_rad_s', 'left_wheel_rad_s',
          'right_wheel_rad_s', 'left_wheel_ref_rad_s', 'right_wheel_ref_rad_s')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def residuals(command, body, wheel_proxy, reference_proxy):
    values = dict(reference_bias=reference_proxy - command,
                  drive_gap=wheel_proxy - reference_proxy,
                  wheel_body_proxy_gap=body - wheel_proxy)
    error = body - command
    closure = float(np.max(np.abs(sum(values.values()) - error)))
    assert closure < 1.e-12
    return values, error, closure


def summarize(rows, side):
    present = [row[side] for row in rows if row[side] is not None]
    nonzero = [row for row in rows if row['case']['yaw'] and row[side] is not None]
    return dict(trials=len(rows), available_budgets=len(present),
                unavailable_budgets=len(rows) - len(present),
                mean_absolute_signed_components={key: float(np.mean([abs(v[key]) for v in present]))
                    if present else None for key in ('reference_bias', 'drive_gap', 'wheel_body_proxy_gap')},
                largest_signed_mean_component_counts=dict(Counter(max(
                    ('reference_bias', 'drive_gap', 'wheel_body_proxy_gap'), key=lambda key: abs(v[key]))
                    for v in present)),
                nonzero_yaw_trials=len(nonzero),
                nonzero_yaw_strict_failures=sum(row[side]['relative_yaw_percent'] >= 10. for row in nonzero),
                mean_nonzero_yaw_relative_percent=float(np.mean([row[side]['relative_yaw_percent']
                    for row in nonzero])) if nonzero else None)


def audit(comparison_path, output):
    comparison_path, output = Path(comparison_path).resolve(), Path(output).resolve()
    comparison = read(comparison_path)
    assert comparison['complete'] and comparison['new_PPO_updates'] == 0 and not output.exists()
    assert comparison['all_requested_scopes_and_failures_retained']
    assert comparison['all_initial27_observations_bitwise_equal']
    assert comparison['original_numeric_comparison_unchanged']
    pins = dict(comparison['source_pins'])
    pins.update({str(comparison_path): sha(comparison_path), str(SCRIPT): sha(SCRIPT)})
    indexed = {(row['scope'], row['case']['name'], row['seed']): row
               for row in comparison['all_pairs']}
    assert len(indexed) == comparison['scoped_trials']
    budgets, max_closure, max_metric_error = {}, 0., 0.
    for scope in comparison['scopes']:
        for side, directory in (('source', scope['source']), ('candidate', scope['candidate'])):
            directory = Path(directory)
            manifest, cfg = [read(directory / name) for name in ('manifest.json', 'resolved_env.json')]
            assert not manifest['weights_updated'] and not manifest['sampled_policy_actions']
            assert manifest['checkpoint_sha256'] == comparison['source_checkpoint_sha256']
            radius, track = cfg['wheel_radius'], cfg['wheel_track_width']
            assert radius > 0. and track > 0.
            path = directory / 'trace.csv'
            trace = np.genfromtxt(path, delimiter=',', names=True, usecols=FIELDS)
            assert all(np.isfinite(trace[key]).all() for key in FIELDS)
            for index, case in enumerate(manifest['cases']):
                for seed in manifest['seeds']:
                    key = scope['scope'], case['name'], seed
                    pair = indexed[key]
                    assert pair['case'] == case
                    row = budgets.setdefault(key, dict(scope=key[0], case=case, seed=seed))
                    row[side + '_survived'] = pair[side + '_survived']
                    row[side + '_strict_failed_axes'] = pair[side + '_strict_failed_axes']
                    row[side] = None
                    if not pair[side + '_survived']:
                        continue
                    tail = trace[(trace['case_id'] == index) & (trace['seed'] == seed)
                                 & (trace['time_s'] >= manifest['seconds_per_case'] - 3.)]
                    assert len(tail) > 0
                    reference = radius / track * (tail['right_wheel_ref_rad_s'] - tail['left_wheel_ref_rad_s'])
                    wheel = radius / track * (tail['right_wheel_rad_s'] - tail['left_wheel_rad_s'])
                    values, error, closure = residuals(case['yaw'], tail['yaw_rate_rad_s'], wheel, reference)
                    mae = float(np.mean(np.abs(error)))
                    metric_error = abs(mae - pair[side + '_metrics']['yaw_mae_rad_s'])
                    assert metric_error < 1.e-7
                    max_closure, max_metric_error = max(max_closure, closure), max(max_metric_error, metric_error)
                    row[side] = dict(**{name: float(np.mean(value)) for name, value in values.items()},
                        samples=len(tail), yaw_signed_bias=float(np.mean(error)), yaw_MAE=mae,
                        yaw_temporal_std=float(np.std(error)),
                        relative_yaw_percent=100. * mae / abs(case['yaw']) if case['yaw'] else None)
            pins.update({str(directory / name): sha(directory / name)
                         for name in ('trace.csv', 'manifest.json', 'resolved_env.json')})
    rows = list(budgets.values())
    assert len(rows) == len(indexed) and all('source' in row and 'candidate' in row for row in rows)
    assert all(sha(path) == digest for path, digest in pins.items())
    populations = dict(all_trials=rows)
    for side in ('source', 'candidate'):
        populations[side + '_nonzero_yaw_failed'] = [row for row in rows if row['case']['yaw']
            and (row[side] is None or row[side]['relative_yaw_percent'] >= 10.)]
        populations[side + '_nonzero_yaw_passed'] = [row for row in rows if row['case']['yaw']
            and row[side] is not None and row[side]['relative_yaw_percent'] < 10.]
    result = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  scoped_trials=len(rows), groups={name: {side: summarize(value, side)
                      for side in ('source', 'candidate')} for name, value in populations.items()},
                  all_trials=rows, signed_component_closure_max_error=max_closure,
                  original_yaw_MAE_reproduction_max_error=max_metric_error, source_pins=pins,
                  all_failed_and_passing_trials_retained=True, failed_budgets_not_zeroed=True,
                  training_modified=False, optimizer_executed=False, simulator_started=False,
                  acceptance_changed=False, policy_accepted=False,
                  limitations=['The signed proxy decomposition does not isolate tire slip, contact force or a PD defect.',
                               'A large mean component can cancel another component and is not alone a cause.',
                               'Comparisons are frozen tests; they do not establish trained or final sustained acceptance.'])
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps(dict(verified=True, scoped_trials=len(rows), groups=result['groups'],
                          closure_max_error=max_closure, original_MAE_max_error=max_metric_error)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.comparison, args.output)
