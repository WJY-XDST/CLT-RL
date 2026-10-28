"""Run paired frozen-policy stress tests; retain incomplete and failed trials.

No PPO, model promotion, hardware output or relaxed scoring is provided here.
One-factor and combined distributions are external input files. Evaluation
cases never select a different perturbation distribution or controller.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import review_frozen_joint_pd_command_matrix as matrix


def summary(rows):
    result = dict(trials=len(rows), survivors=sum(t['survived'] for t in rows),
                  original_passed=sum(t['original_passed'] for t in rows),
                  strict10_passed=sum(t['original_AND_literal10_passed'] for t in rows),
                  failed_metrics_retained_as_null=sum(t['metrics'] is None for t in rows))
    for axis in ('speed', 'yaw', 'height'):
        selected = [t for t in rows if t['nonzero_relative_percent'][axis] is not None]
        values = [t['nonzero_relative_percent'][axis] for t in selected]
        result[axis] = dict(valid_nonzero_trials=len(values),
            mean_relative_MAE_percent=sum(values) / len(values) if values else None,
            worst_relative_MAE_percent=max(values) if values else None,
            failed_relative10=sum(v >= 10. for v in values))
    return result


def run(plan_path):
    plan = matrix.read(plan_path)
    matrix.verify_pins(plan['source_pins'])
    output = plan_path.parent
    baseline, reports, raw = None, {}, {}
    for entry in plan['profiles']:
        destination = Path(entry['output'])
        complete = destination / 'complete_frozen_command_matrix.json'
        if complete.exists():
            report = matrix.read(complete)
            matrix.verify_pins(report['source_pins'])
        else:
            report = matrix.run(destination)
        assert report['checkpoint_sha256'] == plan['checkpoint_sha256']
        rows = {(t['case']['name'], t['seed']): t for t in report['all_trials']}
        assert len(rows) == plan['trial_count_per_profile']
        if baseline is None:
            assert entry['name'] == 'nominal'
            baseline = rows
        assert rows.keys() == baseline.keys()
        assert all(rows[k]['case'] == baseline[k]['case'] for k in rows)
        paired = dict(recovered=sum(not baseline[k]['original_AND_literal10_passed'] and
                                     rows[k]['original_AND_literal10_passed'] for k in rows),
                      newly_failed=sum(baseline[k]['original_AND_literal10_passed'] and
                                       not rows[k]['original_AND_literal10_passed'] for k in rows),
                      newly_lost_survival=sum(baseline[k]['survived'] and not rows[k]['survived'] for k in rows))
        worst = sorted((t for t in rows.values() if t['nonzero_relative_percent']['yaw'] is not None),
                       key=lambda t: t['nonzero_relative_percent']['yaw'], reverse=True)[:12]
        reports[entry['name']] = dict(summary=summary(list(rows.values())), versus_nominal=paired,
                                     worst_yaw_trials=worst, output=str(destination))
        raw[entry['name']] = report['all_trials']
        matrix.write(output / 'progress.json', dict(observed_at=datetime.now(timezone.utc).isoformat(),
            completed_profiles=list(reports), reports=reports, policy_accepted=False,
            training_active=False, hardware_enabled=False))
    result = dict(complete=True, observed_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=plan['checkpoint'], checkpoint_sha256=plan['checkpoint_sha256'],
        actual_ppo_updates=plan['actual_ppo_updates'], profiles=reports, all_trials=raw,
        all_failures_retained=True, original_numeric_gates_unchanged=True,
        engineering_stress_ranges_not_measured_hardware_distributions=True,
        sensor_noise_applies_to_actor_only=True, timing_resolution_ms=plan['timing_resolution_ms'],
        geometry_optimization_included=False, optimizer_executed=False,
        hardware_enabled=False, policy_accepted=False,
        final_acceptance_and_hardware_validation_still_required=True,
        source_pins={**plan['source_pins'], str(plan_path): matrix.sha(plan_path)})
    matrix.write(output / 'complete_robustness_comparison.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.plan.resolve())
        print(json.dumps({name: value['summary'] for name, value in result['profiles'].items()}, ensure_ascii=False))
    except BaseException as error:
        matrix.write(args.plan.resolve().parent / 'inspection_required.json',
            dict(error=repr(error), existing_results_preserved=True,
                 no_automatic_blind_retry=True, training_active=False, hardware_enabled=False,
                 policy_accepted=False))
        raise
