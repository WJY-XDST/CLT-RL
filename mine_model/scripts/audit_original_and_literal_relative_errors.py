"""Keep original tracking gates and separately verify literal nonzero10%.

An absolute floor can pass a nonzero target whose relative error exceeds10%.
Zero targets keep their original absolute checks; failed-trial None metrics
remain undefined and cannot count as a pass. This changes no acceptance file.
"""
import argparse
from collections import Counter
from datetime import datetime
import json
import math
from pathlib import Path

from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
GATE = ROOT / 'mine_model/config/tracking_acceptance.json'
GATE_SHA = '08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d'
CHANNELS = (('speed_mae_m_s', 'speed'), ('yaw_mae_rad_s', 'yaw'), ('height_mae_m', 'height'))


def audit(source, destination):
    source, destination = [Path(p).resolve() for p in (source, destination)]
    report = json.loads(source.read_text())
    pins = {str(p): sha(p) for p in (SCRIPT, source, GATE)}
    if (destination.exists() or not destination.is_relative_to(ROOT / 'mine_model/results')
            or not report['verified'] or report['total_trials'] != len(report['trials'])
            or report['all7_metrics_and_limits_raw_reproduction_max_error'] != 0.
            or pins[str(GATE)] != GATE_SHA or report['source_pins'].get(str(GATE)) != GATE_SHA):
        raise ValueError('Require immutable original10% raw-verification evidence and a fresh owned output')
    violations, undefined, counts = [], [], Counter()
    literal_passed = both_passed = original_passed = 0
    for trial in report['trials']:
        identity = {k: trial[k] for k in ('scope', 'case', 'seed')}
        original_passed += bool(trial['passed'])
        if trial['metrics'] is None:
            if trial['nonzero_command_relative_error_percent'] is not None or trial['passed']:
                raise ValueError('Original failed None metrics cannot become zero or a pass')
            undefined.append(dict(identity, metrics=None, nonzero_relative_error_percent=None,
                original_passed=False, literal_relative10_passed=False))
            continue
        values = {}
        for metric, command in CHANNELS:
            error, target = trial['metrics'][metric], trial['case'][command]
            if not math.isfinite(error) or error < 0. or not math.isfinite(target):
                raise ValueError('Require finite nonnegative original MAEs and finite commands')
            value = 100. * error / abs(target) if target else None
            recorded = trial['nonzero_command_relative_error_percent'][metric]
            if ((value is None) != (recorded is None)
                    or value is not None and (not math.isfinite(recorded) or abs(value - recorded) > 1.e-12)):
                raise ValueError('Reproduce the original nonzero-target denominator exactly')
            values[metric] = value
        failed = [k for k, v in values.items() if v is not None and not v < 10.]
        counts.update(failed)
        literal_passed += not failed
        both_passed += not failed and bool(trial['passed'])
        if failed:
            violations.append(dict(identity, original_passed=trial['passed'],
                failed_nonzero_relative_metrics=failed, nonzero_relative_error_percent=values,
                original_failed_metrics=trial['flags']['failed_metrics']))
    if (original_passed != report['total_passed']
            or len(undefined) != report['original_metrics_undefined_trials']
            or any(sha(p) != h for p, h in pins.items())):
        raise ValueError('Original counts, failed-trial semantics or immutable inputs changed')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), verified=True,
        actual_ppo_updates=report['actual_ppo_updates'], checkpoint_sha256=report['checkpoint_sha256'],
        total_trials=report['total_trials'], original_seven_metric_floor_aware_passed=original_passed,
        nonzero_command_relative10_only_passed=literal_passed,
        original_gates_and_nonzero_relative10_both_passed=both_passed,
        nonzero_relative_failed_metric_counts=dict(counts), violations=violations,
        original_failed_undefined_trials=len(undefined), undefined_trial_records=undefined,
        zero_command_relative_errors_remain_None=True,
        original_absolute_floors_not_changed_or_substituted=True,
        supplementary_literal_report_does_not_replace_original_acceptance=True,
        source_pins=pins, training_modified=False, optimizer_executed=False,
        simulator_started=False, acceptance_changed=False, policy_accepted=False,
        full_goal_acceptance_unproven=True)
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, args.destination)
    print(json.dumps({k: result[k] for k in ('verified', 'total_trials',
        'original_seven_metric_floor_aware_passed', 'original_gates_and_nonzero_relative10_both_passed',
        'original_failed_undefined_trials')}))
