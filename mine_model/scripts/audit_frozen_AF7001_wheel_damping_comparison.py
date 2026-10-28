"""Read complete frozen damping trials without selecting gains or weights.

The original scorer is rerun on every cached/native scope. Only explicitly
completed scopes may be summarized; partial coverage never means acceptance.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

import numpy as np

import run_frozen_AF7001_wheel_damping_diagnostic as driver
import review_source5201_motion075_combined_turns as numeric


SCRIPT = Path(__file__).resolve()
CHANNELS = (('speed_mae_m_s', 'speed'), ('yaw_mae_rad_s', 'yaw'), ('height_mae_m', 'height'))
FIXED = ('speed_mae_m_s', 'yaw_mae_rad_s', 'height_mae_m', 'leg_angle_mae_rad', 'body_p95_deg')


def relative(row):
    if row['metrics'] is None:
        return None
    return {metric: 100. * row['metrics'][metric] / abs(row['case'][command])
            if row['case'][command] else None for metric, command in CHANNELS}


def strict(row):
    errors = relative(row)
    return bool(row['stage_constraints_passed'] and errors is not None
                and all(value is None or value < 10. for value in errors.values()))


def summary(rows):
    defined = [row for row in rows if row['metrics'] is not None]
    yaw = [row for row in defined if row['case']['yaw']]
    return dict(trials=len(rows), survivors=sum(row['survived'] for row in rows),
        undefined_failed_metrics=len(rows) - len(defined),
        original_passed=sum(row['stage_constraints_passed'] for row in rows),
        original_AND_literal10_passed=sum(strict(row) for row in rows),
        nonzero_yaw_trials=len(yaw),
        yaw_strict10_passed=sum(relative(row)['yaw_mae_rad_s'] < 10. for row in yaw),
        worst_nonzero_yaw_percent=max((relative(row)['yaw_mae_rad_s'] for row in yaw), default=None),
        mean_nonzero_yaw_MAE_rad_s=float(np.mean([row['metrics']['yaw_mae_rad_s'] for row in yaw])) if yaw else None,
        worst_nonzero_speed_percent=max((relative(row)['speed_mae_m_s'] for row in defined
                                        if row['case']['speed']), default=None),
        worst_height_percent=max((relative(row)['height_mae_m'] for row in defined), default=None),
        worst_body_P95_deg=max((row['metrics']['body_p95_deg'] for row in defined), default=None),
        failures=[row for row in rows if not row['stage_constraints_passed']])


def audit(destination, require_complete=True):
    destination = Path(destination).resolve()
    assert not destination.exists() and destination.parent == driver.OUTPUT
    proof = driver.read_and_check_preparation()
    baseline = driver.read(driver.OUTPUT / 'baseline05_native_qualification.json')
    assert baseline['qualified'] and baseline['all_original_flags_identical']
    results, pins = {}, {str(SCRIPT): driver.sha(SCRIPT), str(driver.PREPARATION): driver.sha(driver.PREPARATION)}
    for scope in proof['scopes']:
        directory = driver.OUTPUT / ('damping075_' + scope['name'])
        if not (directory / 'metrics_diagnostic_only.json').exists():
            assert not require_complete, 'Finish every frozen native scope'
            continue
        fixture = Path(scope['fixture'])
        before_cfg, after_cfg = [driver.read(path / 'resolved_env.json') for path in (fixture, directory)]
        expected = copy.deepcopy(before_cfg)
        expected['log_dir'] = after_cfg['log_dir']
        expected['wheel_damping'] = .75
        expected['robot']['actuators']['wheels']['damping'] = .75
        assert after_cfg == expected, 'Only the declared wheel drive D may differ'
        manifests = [driver.read(path / 'manifest.json') for path in (fixture, directory)]
        assert manifests[0]['cases'] == manifests[1]['cases']
        assert manifests[0]['seeds'] == manifests[1]['seeds'] == scope['seeds']
        assert all(m['checkpoint_sha256'] == driver.SOURCE_SHA and m['seconds_per_case'] == 20.
                   and not m['sampled_policy_actions'] and not m['weights_updated'] for m in manifests)
        with np.load(fixture / 'policy_inputs.npz', allow_pickle=False) as old, np.load(
                directory / 'policy_inputs.npz', allow_pickle=False) as new:
            for key in ('observations', 'means'):
                assert np.array_equal(old[key][0], new[key][0])
            assert new['observations'].shape == (1000, scope['count'], 27)
        reports = []
        for path, prior in ((fixture, Path(scope['original_report'])),
                            (directory, directory / 'metrics_diagnostic_only.json')):
            reader, captured, _ = numeric.reader(manifests[0]['cases'], scope['seeds'])
            rebuilt = reader(path, Path(proof['checkpoint']))
            saved = driver.read(prior)
            assert len(captured) == 1 and rebuilt['trials'] == saved['trials']
            assert rebuilt['total_trials'] == len(rebuilt['trials']) == scope['count']
            reports.append(rebuilt)
            for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz'):
                pins[str(path / name)] = driver.sha(path / name)
            pins[str(prior)] = driver.sha(prior)
        before, after = [report['trials'] for report in reports]
        assert [(r['case'], r['seed']) for r in before] == [(r['case'], r['seed']) for r in after]
        assert all(left['limits'][key] == right['limits'][key]
                   for left, right in zip(before, after) for key in FIXED)
        results[scope['name']] = dict(source05=summary(before), frozen075=summary(after),
            literal_rescues=[dict(case=b['case'], seed=b['seed'], source_relative=relative(a),
                                 probe_relative=relative(b)) for a, b in zip(before, after)
                             if strict(b) and not strict(a)],
            literal_regressions=[dict(case=b['case'], seed=b['seed'], source_relative=relative(a),
                                     probe_relative=relative(b)) for a, b in zip(before, after)
                                 if strict(a) and not strict(b)],
            initial27_and_initial6_means_bitwise_paired=True,
            original_numeric_report_trials_reproduced_exactly=True,
            all_original_cases_seeds_fixed_gates_and_failure_rows_retained=True)
    complete = len(results) == len(proof['scopes']) == 4
    if require_complete:
        assert complete and driver.read(driver.OUTPUT / 'completed.json')['completed']
    driver.read_and_check_preparation()
    value = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        completed_all_four_scopes=complete, completed_probe_trials=sum(v['frozen075']['trials'] for v in results.values()),
        checkpoint_sha256=driver.SOURCE_SHA, frozen_controller_factor={'wheel_damping': [.5, .75]},
        results=results, original_thresholds_absolute_floors_windows_failed_None_unchanged=True,
        canonical735_static810_120s_signed_and_small_yaw_reviews_not_replaced=True,
        gain_not_selected=True, source_not_selected=True, training_modified=False,
        optimizer_executed=False, acceptance_changed=False, policy_accepted=False,
        source_pins={**proof['source_pins'], **pins})
    with destination.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(completed_all_four_scopes=complete,
        results={k: {x: {n: v[x][n] for n in ('trials', 'survivors', 'original_passed',
            'original_AND_literal10_passed', 'yaw_strict10_passed', 'worst_nonzero_yaw_percent',
            'worst_nonzero_speed_percent', 'worst_height_percent', 'worst_body_P95_deg')}
            for x in ('source05', 'frozen075')} for k, v in results.items()})), flush=True)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-completed-prefix-only', action='store_true')
    args = parser.parse_args()
    audit(args.output, require_complete=not args.allow_completed_prefix_only)
