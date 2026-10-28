"""Reproduce original tracking reports from complete raw case/seed timelines.

Canonical, full-height and combined-turn batches share the original metrics,
limits and strict flags. Original failed-trial metrics remain undefined.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from joint_pd_continuation import ROOT, sha
import diagnose_consolidation_motion_height_steps as numeric
from audit_completed_plain27_all_errors import METRICS

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'


def audit(verification, specification, destination):
    verification, specification, destination = [Path(p).resolve() for p in
        (verification, specification, destination)]
    final = json.loads(verification.read_text())
    inputs = json.loads(specification.read_text())
    checkpoint = Path(final['checkpoint'])
    digest = sha(checkpoint)
    if (destination.exists() or not destination.is_relative_to(RESULTS)
            or not final['verified'] or final['checkpoint_sha256'] != digest
            or final['all17_Adam_step'] != 20 * final['actual_ppo_updates']
            or final['exact_actual_update_delta'] != 200
            or any(final['all200_all4096_alignment_max_errors'].values())
            or not final['all200_original_RNG_unchanged']
            or any(sha(p) != h for p, h in final['source_files'].items())
            or not inputs or not isinstance(inputs, list)):
        raise ValueError('Require complete retained-state200 evidence and fresh owned read-only output')
    criteria = numeric.acceptance.profile()['criteria']
    pins = {str(p): sha(p) for p in (SCRIPT, verification, specification, checkpoint,
        Path(numeric.__file__), Path(numeric.acceptance.assessor.__file__),
        numeric.acceptance.CURRENT, numeric.acceptance.POLICY)}
    records, scopes, maximum_error = [], {}, 0.
    for entry in inputs:
        scope, kind = entry['scope'], entry['kind']
        directory, path = Path(entry['directory']).resolve(), Path(entry['report']).resolve()
        if scope in scopes or kind not in ('canonical', 'fixed_command_probe'):
            raise ValueError('Preserve unique explicit report scopes and original reader semantics')
        report = json.loads(path.read_text())
        manifest, complete = [json.loads((directory / name).read_text()) for name in
            ('manifest.json', 'completed.json')]
        if kind == 'canonical':
            cases = [{k: c[k] for k in ('name', 'speed', 'yaw', 'height')}
                for c in report['conditions']]
            trials = [t for c in report['conditions'] for t in c['trials']]
        else:
            trials = report['trials']
            cases = [trials[i * len(manifest['seeds'])]['case']
                for i in range(len(manifest['cases']))]
        if (report['checkpoint_sha256'] != digest or manifest['checkpoint_sha256'] != digest
                or manifest['cases'] != cases or manifest['seconds_per_case'] != 20.
                or manifest['policy_dt'] != .02 or not complete['completed']
                or manifest['weights_updated'] or manifest['policy_input_remapping']
                or manifest['sampled_policy_actions'] or manifest['action_bias_probes']
                or manifest['wheel_probe'] or manifest['wheel_reference_limit_probe'] is not None
                or manifest['command_scale'] != 1.
                or complete['samples'] != 1000 * len(cases) * len(manifest['seeds'])
                or report['total_trials'] != len(trials)):
            raise ValueError('Require original frozen direct-policy cases, clocks and complete trials')
        if report.get('criteria', report.get('acceptance_criteria')) != criteria:
            raise ValueError('Original10% criteria and absolute floors must remain unchanged')
        trace = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
        if len(trace) != complete['samples'] or any(not np.isfinite(trace[k]).all() for k in trace.dtype.names):
            raise ValueError('Retain every finite raw sample, including all failed episodes')
        resets = int(np.count_nonzero((trace['terminated'] != 0) | (trace['timeout'] != 0)))
        if resets != complete['resets'] or report['resets'] != resets:
            raise ValueError('Original report and raw OR reset counts differ')
        added = []
        for cid, case in enumerate(cases):
            for sid, seed in enumerate(manifest['seeds']):
                trial = trials[cid * len(manifest['seeds']) + sid]
                if trial['seed'] != seed or trial.get('case', case) != case:
                    raise ValueError('Original trial identity differs')
                part = trace[(trace['case_id'] == cid) & (trace['seed'] == seed)]
                if len(part) != 1000 or not np.allclose(part['time_s'], np.arange(1, 1001) * .02,
                        atol=1.e-8, rtol=0.):
                    raise ValueError('Preserve each complete original20s timeline')
                count = int(np.count_nonzero((part['terminated'] != 0) | (part['timeout'] != 0)))
                survived = not count and not np.any(part['episode'])
                if survived != trial['survived']:
                    raise ValueError('Original survival flags differ')
                tail = part[part['time_s'] >= 17.]
                if len(tail) != 151:
                    raise ValueError('Retain the original inclusive151-frame final window')
                first = part[part['episode'] == 0]
                if kind == 'canonical':
                    first_tail = first[first['time_s'] >= 17.]
                    limit_reference = first_tail if len(first_tail) else first
                else:
                    limit_reference = tail if survived else first
                limits = numeric.limits(limit_reference, case, criteria)
                for metric in METRICS:
                    maximum_error = max(maximum_error, abs(limits[metric] - trial['limits'][metric]))
                metrics = numeric.acceptance.assessor.metrics_for_window(tail, case, criteria) if survived else None
                if (metrics is None) != (trial['metrics'] is None):
                    raise ValueError('Retain original failed-trial None metrics')
                if metrics is not None:
                    for metric in METRICS:
                        maximum_error = max(maximum_error, abs(metrics[metric] - trial['metrics'][metric]))
                flags = numeric.acceptance.assessor.tracking_components(survived, metrics,
                    {k: v - 1.e-12 for k, v in limits.items()})
                if any(flags[k] != trial[k] for k in flags):
                    raise ValueError('Original strict tracking flags differ')
                row = dict(scope=scope, case=case, seed=seed, survived=survived,
                    raw_OR_resets=count, metrics=trial['metrics'], limits=trial['limits'],
                    flags=flags, passed=trial.get('stage_constraints_passed', trial.get('passed',
                        flags['task_goal_passed'] and flags['reference_tracking_passed'])),
                    original_static_leg_command_checks=trial.get('actual_static_leg_commands'),
                    nonzero_command_relative_error_percent=(None if metrics is None else
                        {metric: (100 * metrics[metric] / abs(case[command]) if case[command] else None)
                            for metric, command in (('speed_mae_m_s', 'speed'),
                                ('yaw_mae_rad_s', 'yaw'), ('height_mae_m', 'height'))}))
                records.append(row)
                added.append(row)
        scopes[scope] = dict(total_trials=len(added), survivors=sum(t['survived'] for t in added),
            passed=sum(t['passed'] for t in added), resets=resets,
            original_metrics_undefined_trials=sum(t['metrics'] is None for t in added))
        pins.update(report.get('source_files', {}))
        for p in (path, *(directory / name for name in ('manifest.json', 'completed.json', 'trace.csv'))):
            pins[str(p)] = sha(p)
    if maximum_error > 1.e-12 or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Resolve raw metric/limit parity or changed immutable inputs')
    defined = [t for t in records if t['metrics'] is not None]
    worst = {}
    for metric in METRICS:
        if not defined:
            worst[metric] = None
            continue
        t = max(defined, key=lambda t: t['metrics'][metric])
        worst[metric] = dict(value=t['metrics'][metric], limit=t['limits'][metric],
            scope=t['scope'], case=t['case'], seed=t['seed'])
    result = dict(observed_at=datetime.now().astimezone().isoformat(), verified=True,
        actual_ppo_updates=final['actual_ppo_updates'], checkpoint=str(checkpoint), checkpoint_sha256=digest,
        total_trials=len(records), total_survivors=sum(t['survived'] for t in records),
        total_passed=sum(t['passed'] for t in records), scope_counts=scopes,
        original_metrics_undefined_trials=len(records) - len(defined),
        all7_metrics_and_limits_raw_reproduction_max_error=maximum_error,
        original_case_seed_and_strict_flags_preserved=True,
        failed_trial_None_metrics_not_zeroed=True, original_reference_window_rules_preserved=True,
        original_absolute_floors_and_nonzero_relative_percent_reported_separately=True,
        worst_defined_original_metrics=worst, trials=records, source_pins=pins,
        optimizer_executed=False, simulator_started=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False, goal_remains_active=True)
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--verification', type=Path, required=True)
    cli.add_argument('--specification', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.verification, args.specification, args.output)
    print(json.dumps({k: result[k] for k in ('verified', 'actual_ppo_updates', 'total_trials',
        'total_survivors', 'total_passed', 'original_metrics_undefined_trials',
        'all7_metrics_and_limits_raw_reproduction_max_error')}))
