"""Read every original tracking gate without dropping failed-survival trials.

Canonical27, all four full-height51 suites and optional wider frozen probes
retain their original readers, limits, flags and inclusive151-frame tails.
Metrics left undefined by an original reader remain None after a failure.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from joint_pd_continuation import sha
import audit_completed_plain27_all_errors as reference
import review_plain27_exploration_block200 as exploration_owner
import review_plain27_single_reward_block200 as reward_owner
import diagnose_consolidation_motion_height_steps as numeric

SCRIPT = Path(__file__).resolve()
METRICS = reference.METRICS


def audit(launch, static_qualification, destination, wider=(), reference_report=None):
    launch, destination = Path(launch).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Preserve the earlier immutable error report')
    launch_data = json.loads(launch.read_text())
    delta = launch_data['next_configuration_changes']
    owner = reward_owner if delta['rewards'] else exploration_owner
    bound = owner.build(launch, static_qualification)
    training = Path(bound.launch['output'])
    verified_path = training / 'final_checkpoint_verification_readonly_20261019.json'
    verified = json.loads(verified_path.read_text())
    source = Path(verified['checkpoint'])
    if (not verified['verified'] or verified['actual_ppo_updates'] != bound.target_clock
            or sha(source) != verified['checkpoint_sha256'] or destination.parent != training
            or any(sha(p) != h for p, h in verified['source_files'].items())):
        raise ValueError('Require the exact completed retained-state200 block')
    manifest_path = training / 'retained_four_motion_envelopes_review_20261019.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['checkpoint_sha256'] != sha(source) or set(manifest['suites']) != set(map(str, exploration_owner.SCALES)):
        raise ValueError('Require all four original full-height scopes')
    pins = {str(p): sha(p) for p in (SCRIPT, reference.SCRIPT, owner.SCRIPT,
        Path(numeric.__file__), launch, Path(static_qualification), verified_path, manifest_path,
        source, numeric.acceptance.CURRENT, numeric.acceptance.POLICY,
        Path(numeric.acceptance.assessor.__file__))}
    criteria = numeric.acceptance.profile()['criteria']
    scopes, rows, parity, checked, differences_found = {}, [], 0., 0, []

    def add_directory(scope, directory, cases, seeds, trials):
        nonlocal parity, checked
        native, complete = [json.loads((directory / name).read_text()) for name in ('manifest.json', 'completed.json')]
        if (native['checkpoint_sha256'] != sha(source) or native['cases'] != cases
                or native['seeds'] != seeds or not complete['completed']
                or complete['samples'] != 1000 * len(cases) * len(seeds)):
            raise ValueError('Require the original source, cases, seeds and complete trajectories')
        trace = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
        if (len(trace) != complete['samples'] or len(trials) != len(cases) * len(seeds)
                or any(not np.isfinite(trace[k]).all() for k in trace.dtype.names)):
            raise ValueError('Require every complete finite timeline, including all failed trials')
        resets = int(np.count_nonzero((trace['terminated'] != 0) | (trace['timeout'] != 0)))
        if resets != complete['resets']:
            raise ValueError('Raw OR reset count differs')
        added = []
        for cid, case in enumerate(cases):
            for sid, seed in enumerate(seeds):
                trial = trials[cid * len(seeds) + sid]
                if trial['seed'] != seed:
                    raise ValueError('Original case/seed ordering differs')
                part = trace[(trace['case_id'] == cid) & (trace['seed'] == seed)]
                if len(part) != 1000 or not np.allclose(part['time_s'], np.arange(1, 1001) * .02, atol=1.e-8, rtol=0.):
                    raise ValueError('Preserve each original20s timeline')
                tail = part[part['time_s'] >= 17.]
                if len(tail) != 151:
                    raise ValueError('Preserve the inclusive final3s,151-frame original window')
                trial_resets = int(np.count_nonzero((part['terminated'] != 0) | (part['timeout'] != 0)))
                if trial['survived'] != (trial_resets == 0):
                    raise ValueError('Raw survival differs from original report')
                # The original full51 reader leaves failed metrics undefined,
                # and computes reference-based limits from the first episode
                # in that case. A post-reset final tail is not interchangeable.
                limit_reference = tail if trial['survived'] else part[part['episode'] == 0]
                limits = numeric.limits(limit_reference, case, criteria)
                for key in METRICS:
                    parity = max(parity, abs(limits[key] - trial['limits'][key]))
                    if abs(limits[key] - trial['limits'][key]) > 1.e-12:
                        differences_found.append(dict(scope=scope, case=case['name'], seed=seed,
                            key=key, reproduced=limits[key], original=trial['limits'][key],
                            original_metrics_defined=trial['metrics'] is not None))
                metrics = trial['metrics']
                if metrics is None:
                    if trial['survived'] or not trial_resets or (trial['task_goal_passed'] or trial['reference_tracking_passed']):
                        raise ValueError('Undefined original metrics require a retained failed-survival trial')
                else:
                    recomputed = numeric.acceptance.assessor.metrics_for_window(tail, case, criteria)
                    for key in METRICS:
                        parity = max(parity, abs(recomputed[key] - metrics[key]))
                    flags = numeric.acceptance.assessor.tracking_components(trial['survived'], recomputed,
                        {k: v - 1.e-12 for k, v in limits.items()})
                    if any(flags[k] != trial[k] for k in flags):
                        raise ValueError('Original tracking flags differ')
                    checked += 1
                passed = trial.get('stage_constraints_passed',
                    trial['task_goal_passed'] and trial['reference_tracking_passed'])
                row = dict(scope=scope, case=case, seed=seed, survived=trial['survived'],
                    raw_OR_resets=trial_resets, metrics=metrics, limits=trial['limits'], passed=passed,
                    flags={k: trial[k] for k in ('task_goal_passed', 'reference_tracking_passed',
                        'failed_metrics', 'failed_prerequisites')},
                    joint_PD_per_step_max4_mean_rad=float(np.mean(tail['active_joint_tracking_max_abs_rad'])),
                    joint_PD_per_step_max4_P95_rad=float(np.percentile(tail['active_joint_tracking_max_abs_rad'], 95)),
                    raw_min_height_over_entire_trial_m=float(part['height_m'].min()),
                    raw_peak_abs_pitch_over_entire_trial_deg=float(np.rad2deg(abs(part['pitch_rad'])).max()))
                rows.append(row)
                added.append(row)
        scopes[scope] = dict(passed=sum(t['passed'] for t in added), trials=len(added),
            survivors=sum(t['survived'] for t in added), resets=resets,
            original_metrics_undefined_trials=sum(t['metrics'] is None for t in added))
        for name in ('trace.csv', 'manifest.json', 'completed.json'):
            pins[str(directory / name)] = sha(directory / name)

    canonical = training / 'replay'
    assessment = json.loads((canonical / 'assessment.json').read_text())
    if assessment['checkpoint_sha256'] != sha(source) or assessment['total_trials'] != 27:
        raise ValueError('Require the canonical27 for this exact checkpoint')
    pins[str(canonical / 'assessment.json')] = sha(canonical / 'assessment.json')
    cases = [{k: c[k] for k in ('name', 'speed', 'yaw', 'height')} for c in assessment['conditions']]
    add_directory('canonical27', canonical, cases, [83, 84, 85],
        [t for c in assessment['conditions'] for t in c['trials']])
    for scale in exploration_owner.SCALES:
        directory = Path(manifest['suites'][str(scale)]['output']) / 'candidate'
        path = directory / 'motion025_full_height_diagnosis.json'
        report = json.loads(path.read_text())
        rebuilt = bound.api.reader(scale, directory, source)
        keys = ('checkpoint', 'checkpoint_sha256', 'total_trials', 'survivors', 'resets',
            'stage_all_constraints_passed', 'criteria', 'full_height_range', 'trials', 'source_files')
        if any(rebuilt[k] != report[k] for k in keys):
            raise ValueError('Original full51 raw reproduction differs')
        pins.update(report['source_files'])
        pins[str(path)] = sha(path)
        add_directory(f'motion_scale{scale}', directory,
            [report['trials'][i * 3]['case'] for i in range(17)], [483, 484, 485], report['trials'])
    if len(rows) != 231:
        raise ValueError('Every mandatory original canonical27/four51 trial must be retained')
    for report_path in map(lambda p: Path(p).resolve(), wider):
        report = json.loads(report_path.read_text())
        if (report['checkpoint_sha256'] != sha(source) or report['total_trials'] != 51
                or report['criteria'] != criteria or report['training_modified'] or report['optimizer_updated']
                or not report['original_metric_formulas_limits_and_windows_unchanged']
                or not report['frozen_wider_command_diagnosis_only']):
            raise ValueError('Optional wider suites must retain the same frozen source and original criteria')
        pins.update(report['source_files'])
        pins[str(report_path)] = sha(report_path)
        add_directory(f"wider_frozen_scale{report['tested_command_scale']}", report_path.parent,
            [report['trials'][i * 3]['case'] for i in range(17)], [483, 484, 485], report['trials'])
    changed_pins = [p for p, h in pins.items() if sha(p) != h]
    if parity > 1.e-12 or changed_pins:
        raise ValueError('Original numeric parity, flags or frozen dependencies changed: '
            + json.dumps(dict(max_error=parity, examples=differences_found[:5], changed_pins=changed_pins)))
    reference_trials = 0
    if reference_report is not None:
        reference_report = Path(reference_report).resolve()
        prior = json.loads(reference_report.read_text())
        if prior['checkpoint_sha256'] != sha(source) or len(prior['trials']) != 231:
            raise ValueError('Require the original same-source231 reference report')
        for old, new in zip(prior['trials'], rows[:231]):
            if any(old[k] != new[k] for k in old):
                raise ValueError('Same-source original231 detailed report changed')
            reference_trials += 1
        pins[str(reference_report)] = sha(reference_report)
    valid = [row for row in rows if row['metrics'] is not None]
    worst = {}
    for key in METRICS:
        trial = max(valid, key=lambda row: row['metrics'][key])
        worst[key] = dict(value=trial['metrics'][key], limit=trial['limits'][key],
            case=trial['case'], seed=trial['seed'], scope=trial['scope'])
    result = dict(observed_at=datetime.now().astimezone().isoformat(),
        actual_ppo_updates=bound.target_clock, checkpoint=str(source), checkpoint_sha256=sha(source),
        scope_counts=scopes, total_trials=len(rows), total_survivors=sum(t['survived'] for t in rows),
        total_passed=sum(t['passed'] for t in rows), original_metrics_undefined_trials=len(rows) - len(valid),
        original_numeric_trials_reproduced=checked, all7_metrics_and_limits_raw_reproduction_max_error=parity,
        original231_detailed_reference_parity_trials=reference_trials,
        worst_defined_original_metrics=worst, worst_values_scope='Only trials with metrics defined by the original reader; all failures remain in trials and scope counts.',
        undefined_original_metrics_are_not_zeroed_or_recomputed_as_passing=True,
        original_inclusive151_frame_windows_and_strict_gate_flags_preserved=True,
        relative_error_target=.1, original_absolute_floors_preserved=True,
        optional_wider_probes_do_not_authorize_faster_training=True, trials=rows, source_pins=pins,
        weights_updated=False, optimizer_executed=False, simulator_started=False,
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
    cli.add_argument('--wider-reports', type=Path, nargs='*', default=[])
    cli.add_argument('--reference-report', type=Path)
    args = cli.parse_args()
    result = audit(args.training_launch, args.static_qualification, args.output,
        args.wider_reports, args.reference_report)
    print(json.dumps({k: result[k] for k in ('total_trials', 'total_survivors', 'total_passed',
        'original_metrics_undefined_trials', 'all7_metrics_and_limits_raw_reproduction_max_error',
        'original231_detailed_reference_parity_trials')}))
