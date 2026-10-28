"""Reuse the saved-trace yaw/reward reconstruction for the frozen drive factor.

Only report filenames are adapted. Retain every original trial, failedNone,
float32 command,151-frame last-three-second window and reward reconstruction.
"""
import ast
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path

import numpy as np

import audit_paired_yaw_kernel_trace_budget as original
import run_frozen_AF7001_wheel_damping_diagnostic as driver


SCRIPT = Path(__file__).resolve()


def process(directory, report_name):
    tree = ast.parse(inspect.getsource(original.process))
    before, edits = ast.dump(tree), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'motion025_full_height_diagnosis.json':
            edits.append((node, node.value))
            node.value = report_name
    assert len(edits) == 2
    modified = copy.deepcopy(tree)
    for node, value in edits:
        node.value = value
    assert ast.dump(tree) == before
    scope = dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(modified), str(original.SCRIPT), 'exec'), scope)
    return scope['process'](directory)


def audit(destination):
    destination = Path(destination).resolve()
    assert not destination.exists() and destination.parent == driver.OUTPUT
    proof = driver.read_and_check_preparation()
    finished = driver.read(driver.OUTPUT / 'completed.json')
    assert finished['completed'] and finished['probe_trials'] == 318
    paired = driver.read(driver.OUTPUT / 'complete318_paired_readonly_20261023.json')
    assert paired['verified'] and paired['completed_all_four_scopes']
    results, pins = {}, {str(SCRIPT): driver.sha(SCRIPT), str(original.SCRIPT): driver.sha(original.SCRIPT)}
    for entry in proof['scopes']:
        source, candidate = Path(entry['fixture']), driver.OUTPUT / ('damping075_' + entry['name'])
        a, am, ac, ai, ap = process(source, Path(entry['original_report']).name)
        z, zm, zc, zi, zp = process(candidate, 'metrics_diagnostic_only.json')
        assert a['checkpoint_sha256'] == z['checkpoint_sha256'] == driver.SOURCE_SHA
        assert am['cases'] == zm['cases'] and am['seeds'] == zm['seeds'] and np.array_equal(ai, zi)
        delta = sorted(original.differences(ac, zc))
        assert delta == [('/robot/actuators/wheels/damping', .5, .75), ('/wheel_damping', .5, .75)], delta
        arows, zrows = a['trials'], z['trials']
        assert len(arows) == len(zrows) == entry['count']
        changes = []
        for before, after in zip(arows, zrows):
            assert before['case'] == after['case'] and before['seed'] == after['seed']
            assert (before['metrics'] is None) == (after['metrics'] is None)
            changes.append(dict(case=after['case'], seed=after['seed'],
                original_source_passed=before['original_stage_constraints_passed'],
                original_probe_passed=after['original_stage_constraints_passed'],
                source_failed_metrics=before['original_failed_metrics'],
                probe_failed_metrics=after['original_failed_metrics'],
                signed_bias_change_rad_s=(after['metrics']['signed_bias_rad_s'] - before['metrics']['signed_bias_rad_s'])
                    if before['metrics'] is not None else None,
                signed_component_changes_rad_s={key: after['signed_components'][key]['signed_mean_rad_s']
                    - before['signed_components'][key]['signed_mean_rad_s'] for key in before['signed_components']}
                    if before['metrics'] is not None else None))
        results[entry['name']] = dict(source05=a, frozen075=z, changes=changes,
            exact_saved_configuration_differences=delta,
            all_initial27_inputs_bitwise_paired=True)
        pins.update(ap)
        pins.update(zp)
    driver.read_and_check_preparation()
    assert all(driver.sha(path) == digest for path, digest in pins.items())
    value = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        checkpoint_sha256=driver.SOURCE_SHA, trials_per_side=318, scopes=results,
        original_process_AST_restored_identical_except_report_filename=True,
        full_original_yaw_MAE_and_both_rewards_reproduced=True,
        formula='body-command=(reference_proxy-command)+(measured_wheel_proxy-reference_proxy)+(body-measured_wheel_proxy)',
        network_weights_and_saved_std_unchanged=True, controller_factor={'wheel_damping': [.5, .75]},
        all_original_failures_and_zero_command_relative_None_preserved=True,
        numerical_performance_gates_and_windows_unchanged=True,
        source_pins={**proof['source_pins'], **pins}, simulator_started=False,
        optimizer_executed=False, training_modified=False, acceptance_changed=False, policy_accepted=False,
        limitation='One frozen gain intervention supports an effect of drive D on closed-loop tracking. '
            'The signed component allocation remains descriptive; wheel kinematics are not a direct measurement '
            'of tire slip or actual implicit drive torque. It does not replace any full candidate review.')
    with destination.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(verified=True, trials_per_side=318,
        max_yaw_MAE_reproduction_error=max(v[side]['parity']['original_yaw_MAE']
            for v in results.values() for side in ('source05', 'frozen075')),
        max_signed_component_closure=max(v[side]['parity']['signed_component_closure']
            for v in results.values() for side in ('source05', 'frozen075')))), flush=True)
    return value


if __name__ == '__main__':
    audit(driver.OUTPUT / 'complete318_signed_yaw_budget_readonly_20261023.json')
