"""Reproduce and summarize every original error in a completed plain27 block.

Original numeric readers, inclusive151-frame windows, limits and flags are
retained. Joint PD tracking is supplementary telemetry without a new gate.
No model, optimizer, command, controller or simulator is changed or executed.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from joint_pd_continuation import sha
import review_plain27_exploration_block200 as owner
import diagnose_consolidation_motion_height_steps as numeric

SCRIPT = Path(__file__).resolve()
METRICS = ('speed_mae_m_s', 'yaw_mae_rad_s', 'height_mae_m', 'body_p95_deg',
    'leg_length_mae_m', 'leg_angle_mae_rad', 'wheel_speed_mae_rad_s')


def audit(launch, static_qualification, destination):
    launch, destination = Path(launch).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Preserve the earlier immutable error report')
    bound = owner.build(launch, static_qualification)
    training = Path(bound.launch['output'])
    verified_path = training / 'final_checkpoint_verification_readonly_20261019.json'
    verified = json.loads(verified_path.read_text())
    source = Path(verified['checkpoint'])
    if (not verified['verified'] or verified['actual_ppo_updates'] != bound.target_clock
            or sha(source) != verified['checkpoint_sha256']
            or any(sha(p) != h for p, h in verified['source_files'].items())
            or destination.parent != training):
        raise ValueError('Require this exact verified completed200 checkpoint')
    manifest_path = training / 'retained_four_motion_envelopes_review_20261019.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['checkpoint_sha256'] != sha(source) or set(manifest['suites']) != set(map(str, owner.SCALES)):
        raise ValueError('Require every original full-height scope for this checkpoint')
    files = [SCRIPT, owner.SCRIPT, Path(numeric.__file__), launch, verified_path, manifest_path,
        source, numeric.acceptance.CURRENT, numeric.acceptance.POLICY,
        Path(numeric.acceptance.assessor.__file__)]
    pins = {str(p): sha(p) for p in files}
    rows, scopes, parity = [], {}, 0.
    criteria = numeric.acceptance.profile()['criteria']

    def trace_at(directory, cases, seeds):
        nonlocal pins
        path = directory / 'trace.csv'
        trace = np.genfromtxt(path, delimiter=',', names=True)
        if len(trace) != 1000 * cases * len(seeds) or any(not np.isfinite(trace[k]).all() for k in trace.dtype.names):
            raise ValueError('Keep every complete finite raw timeline, including failures')
        pins[str(path)] = sha(path)
        return trace

    def append(scope, case, trial, trace, cid):
        part = trace[(trace['case_id'] == cid) & (trace['seed'] == trial['seed'])]
        if len(part) != 1000 or not np.allclose(part['time_s'], np.arange(1, 1001) * .02, atol=1.e-8, rtol=0.):
            raise ValueError('Require the complete original20s case/seed timeline')
        tail = part[part['time_s'] >= 17.]
        if len(tail) != 151:
            raise ValueError('Preserve the inclusive original final3s window')
        joint = tail['active_joint_tracking_max_abs_rad']
        passed = trial.get('stage_constraints_passed',
            trial['task_goal_passed'] and trial['reference_tracking_passed'])
        rows.append(dict(scope=scope, case=case, seed=trial['seed'], metrics=trial['metrics'],
            limits=trial['limits'], passed=passed,
            flags={k: trial[k] for k in ('task_goal_passed', 'reference_tracking_passed',
                'failed_metrics', 'failed_prerequisites')},
            joint_PD_per_step_max4_mean_rad=float(np.mean(joint)),
            joint_PD_per_step_max4_P95_rad=float(np.percentile(joint, 95))))

    canonical = training / 'replay'
    assessment = json.loads((canonical / 'assessment.json').read_text())
    raw_manifest = json.loads((canonical / 'manifest.json').read_text())
    if (assessment['checkpoint_sha256'] != sha(source) or raw_manifest['checkpoint_sha256'] != sha(source)
            or assessment['total_trials'] != 27 or raw_manifest['seeds'] != [83, 84, 85]):
        raise ValueError('Retain every original canonical case and seed')
    for name in ('assessment.json', 'manifest.json', 'completed.json'):
        pins[str(canonical / name)] = sha(canonical / name)
    trace = trace_at(canonical, 9, [83, 84, 85])
    for cid, condition in enumerate(assessment['conditions']):
        case = {k: condition[k] for k in ('name', 'speed', 'yaw', 'height')}
        for trial in condition['trials']:
            tail = trace[(trace['case_id'] == cid) & (trace['seed'] == trial['seed']) & (trace['time_s'] >= 17.)]
            metrics = numeric.acceptance.assessor.metrics_for_window(tail, case, criteria)
            limits = numeric.limits(tail, case, criteria)
            for key in METRICS:
                parity = max(parity, abs(metrics[key] - trial['metrics'][key]), abs(limits[key] - trial['limits'][key]))
            flags = numeric.acceptance.assessor.tracking_components(trial['survived'], metrics,
                {k: v - 1.e-12 for k, v in limits.items()})
            if any(flags[k] != trial[k] for k in flags):
                raise ValueError('Original canonical task/reference flags differ')
            append('canonical27', case, trial, trace, cid)
    scopes['canonical27'] = dict(passed=sum(t['passed'] for t in rows), trials=27,
        survivors=assessment['survivors'], resets=assessment['resets'])
    for scale in owner.SCALES:
        entry = manifest['suites'][str(scale)]
        directory = Path(entry['output']) / 'candidate'
        report_path = directory / 'motion025_full_height_diagnosis.json'
        report = json.loads(report_path.read_text())
        rebuilt = bound.api.reader(scale, directory, source)
        keys = ('checkpoint', 'checkpoint_sha256', 'total_trials', 'survivors', 'resets',
            'stage_all_constraints_passed', 'criteria', 'full_height_range', 'trials', 'source_files')
        if any(rebuilt[k] != report[k] for k in keys):
            raise ValueError('Original full51 numeric metrics, limits or flags differ')
        pins.update(report['source_files'])
        pins[str(report_path)] = sha(report_path)
        trace = trace_at(directory, 17, [483, 484, 485])
        scope = f'motion_scale{scale}'
        for index, trial in enumerate(report['trials']):
            append(scope, trial['case'], trial, trace, index // 3)
        scopes[scope] = {k: entry[k] for k in ('passed', 'total_trials', 'survivors', 'resets')}
        scopes[scope]['trials'] = scopes[scope].pop('total_trials')
    if len(rows) != 231 or parity > 1.e-12 or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Original231 numerical parity or frozen evidence changed')
    worst = {}
    for key in (*METRICS, 'joint_PD_per_step_max4_mean_rad', 'joint_PD_per_step_max4_P95_rad'):
        value_at = lambda row: row['metrics'][key] if key in METRICS else row[key]
        trial = max(rows, key=value_at)
        worst[key] = dict(value=value_at(trial), case=trial['case'], seed=trial['seed'], scope=trial['scope'])
        if key in METRICS:
            worst[key]['limit'] = trial['limits'][key]
        else:
            worst[key]['original_acceptance_limit_defined'] = False
    relative = {}
    for key, command in (('speed_mae_m_s', 'speed'), ('yaw_mae_rad_s', 'yaw'), ('height_mae_m', 'height')):
        trial = max((r for r in rows if r['case'][command]),
            key=lambda row: row['metrics'][key] / abs(row['case'][command]))
        relative[key] = dict(percent=100 * trial['metrics'][key] / abs(trial['case'][command]),
            absolute_error=trial['metrics'][key], case=trial['case'], seed=trial['seed'],
            original_limit=trial['limits'][key])
    result = dict(observed_at=datetime.now().astimezone().isoformat(), actual_ppo_updates=bound.target_clock,
        checkpoint=str(source), checkpoint_sha256=sha(source), scope_counts=scopes, all_231_worst=worst,
        maximum_nonzero_relative_errors=relative,
        all7_metrics_and_limits_raw_reproduction_max_error=parity,
        original_windows='Inclusive final3s of20s;151frames; canonical27 and four17x3 full-height suites retained',
        relative_error_target=.1, original_absolute_floors_preserved=True,
        all_relative_errors_under10percent=all(v['percent'] < 10. for v in relative.values()),
        deterministic_trials=231, deterministic_resets=sum(s['resets'] for s in scopes.values()),
        trials=rows, source_pins=pins, independent_current_checkpoint_reviews_not_implied=True,
        weights_updated=False, optimizer_executed=False, simulation_started=False,
        training_modified=False, acceptance_changed=False, policy_accepted=False)
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training-launch', type=Path, required=True)
    cli.add_argument('--static-qualification', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.training_launch, args.static_qualification, args.output)
    print(json.dumps(dict(actual_updates=result['actual_ppo_updates'], scopes=result['scope_counts'],
        raw_reproduction_max_error=result['all7_metrics_and_limits_raw_reproduction_max_error'],
        report=str(args.output), optimizer_executed=False)))
