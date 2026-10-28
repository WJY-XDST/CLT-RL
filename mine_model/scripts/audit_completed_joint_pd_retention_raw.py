"""Check complete raw timelines behind one checkpoint's own retention review.

This verifies evidence integrity and reports the original gates separately.
It never changes a gate, promotes a curriculum or accepts a final model.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
RESULTS = SCRIPT.parents[1] / 'results'
SPECS = (
    ('heldout', 30., [183, 184, 185], 9, False),
    ('sustained', 60., [283, 284, 285], 9, False),
    ('height_steps_deterministic', 50., [383, 384, 385], 3, False),
    ('height_steps_sampled', 50., [383, 384, 385], 3, True),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(training, retention, output):
    training, retention, output = [Path(p).resolve() for p in (training, retention, output)]
    if (training.parent != RESULTS or retention.parent != RESULTS
            or output.parent != RESULTS or output.exists()):
        raise ValueError('Preserve a fresh owned raw-verification report')
    final_path = training / 'final_checkpoint_verification_readonly_20261019.json'
    final = json.loads(final_path.read_text())
    report_path = retention / 'static_retention.json'
    report = json.loads(report_path.read_text())
    source = Path(final['checkpoint'])
    actual, digest = final['actual_ppo_updates'], sha(source)
    if (not final['verified'] or final['all17_Adam_step'] != 20 * actual
            or final['exact_actual_update_delta'] != 200
            or final['checkpoint_sha256'] != digest or report['checkpoint_sha256'] != digest
            or not report['same_original_thresholds_and_windows']
            or report['policy_accepted'] or report['training_modified']
            or any(sha(p) != h for p, h in final['source_files'].items())):
        raise ValueError('Require the exact unchanged completed source and its own original review')
    files = [SCRIPT, final_path, report_path, source]
    modes = {}
    for label, seconds, seeds, cases, sampled in SPECS:
        directory = retention / label
        manifest, complete = [json.loads((directory / name).read_text())
            for name in ('manifest.json', 'completed.json')]
        frames = round(seconds / .02)
        expected_rows = frames * cases * len(seeds)
        if (manifest['checkpoint_sha256'] != digest or manifest['seconds_per_case'] != seconds
                or manifest['seeds'] != seeds or len(manifest['cases']) != cases
                or manifest['sampled_policy_actions'] is not sampled
                or manifest['reset_velocity_uniform_amplitude'] != .05
                or manifest['physics_dt'] != .005 or manifest['policy_dt'] != .02
                or manifest['solver_position_iterations'] != 128
                or manifest['solver_velocity_iterations'] != 8
                or manifest['enable_external_forces_every_iteration'] is not True
                or manifest['leg_control_mode'] != 'implicit_joint_position'
                or manifest['weights_updated'] or manifest['policy_input_remapping']
                or manifest['action_bias_probes'] or manifest['wheel_probe']
                or manifest['wheel_reference_limit_probe'] is not None
                or manifest['command_scale'] != 1. or manifest['curriculum_progress'] is not None
                or not complete['completed'] or complete['samples'] != expected_rows):
            raise ValueError('Require all original seeds, commands, reset distribution and native physics')
        for path, expected in manifest['asset_chain_sha256'].items():
            if sha(path) != expected:
                raise ValueError('A physical asset changed during the frozen review')
        data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
        if len(data) != expected_rows or any(not np.isfinite(data[key]).all() for key in data.dtype.names):
            raise ValueError('Preserve every complete finite episode and failure')
        resets = int(np.count_nonzero((data['terminated'] > 0) | (data['timeout'] > 0)))
        if resets != complete['resets']:
            raise ValueError('Raw terminated-or-timeout counts differ from the native completion')
        survivors = 0
        for cid in range(cases):
            for seed in seeds:
                rows = data[(data['case_id'] == cid) & (data['seed'] == seed)]
                if len(rows) != frames or not np.allclose(rows['time_s'], np.arange(1, frames + 1) * .02,
                        rtol=0., atol=1.e-8):
                    raise ValueError('Missing raw case/seed samples or timeline')
                survivors += not any(np.any(rows[key]) for key in ('episode', 'terminated', 'timeout'))
        modes[label] = dict(seconds=seconds, seeds=seeds, total_trials=cases * len(seeds),
            survivors=int(survivors), raw_rows=len(data), raw_OR_resets=resets,
            sampled_actions=sampled, reset_velocity_amplitude=.05,
            complete_case_seed_timelines_verified=True)
        files.extend(directory / name for name in ('manifest.json', 'completed.json', 'trace.csv', 'resolved_env.json'))
    independent = {}
    for label, stable, windows in (('heldout', 20., 18), ('sustained', 40., 38)):
        directory = retention / 'fixed_height_review' / label
        paths = [directory / name for name in ('assessment_final_gate.json', 'assessment_stability.json')]
        fixed, continuous = [json.loads(path.read_text()) for path in paths]
        if (fixed['checkpoint_sha256'] != digest or continuous['checkpoint_sha256'] != digest
                or fixed['total_trials'] != 27 or continuous['total_trials'] != 27
                or fixed['resets'] != modes[label]['raw_OR_resets']
                or continuous['stability']['stable_seconds'] != stable
                or continuous['stability']['window_seconds'] != 3.
                or continuous['stability']['stride_seconds'] != 1.
                or continuous['stability']['windows_per_trial'] != windows
                or continuous['stability']['strict_less_than_limits'] is not True
                or fixed['acceptance_criteria']['relative_error'] != .1
                or report['stages'][label]['full27_passed'] != fixed['passed']):
            raise ValueError('Retain every original full27 and overlapping steady-window gate')
        independent[label] = dict(final_full27_passed=fixed['passed'],
            overlapping_full27_passed=continuous['passed'] and continuous['stability']['passed'],
            stability_spec=continuous['stability'])
        files.extend(paths)
    step_reports = {}
    for label in ('height_steps_deterministic', 'height_steps_sampled'):
        path = Path(report['height_step_reports'][label])
        step = json.loads(path.read_text())
        if (step['checkpoint_sha256'] != digest or step['total_trials'] != 9
                or step['resets'] != modes[label]['raw_OR_resets']
                or step['survivors'] != modes[label]['survivors']):
            raise ValueError('The original height-step reports differ from complete raw episodes')
        step_reports[label] = step
        files.append(path)
    if sha(source) != digest:
        raise ValueError('The frozen checkpoint changed')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), raw_integrity_verified=True,
        actual_updates=actual, checkpoint=str(source), checkpoint_sha256=digest,
        all17_Adam_step=20 * actual, source_training_output=str(training),
        source_static_retention_report=str(report_path), static_retention_verified=report['static_retention_verified'],
        canonical27_passed=report['stages']['baseline']['full27_passed'],
        heldout30_full27_passed=independent['heldout']['final_full27_passed'],
        sustained60_full27_passed=independent['sustained']['final_full27_passed'],
        heldout_and_sustained_overlapping_windows=independent,
        deterministic50s_full_height_steps_passed=step_reports['height_steps_deterministic']['height_steps_passed'],
        saved_Gaussian50s_height_step_survival_passed=report['sampled_exploration_survived'],
        saved_Gaussian_all_numeric_metrics_passed=step_reports['height_steps_sampled']['height_steps_passed'],
        saved_Gaussian_all_numeric_metrics_not_claimed=True, modes=modes,
        full_project_command_envelope_acceptance_unproven=True, full_goal_acceptance_passed=False,
        relative_error_target=.1, acceptance_changed=False, faster_curriculum_authorized=False,
        policy_accepted=False, source_pins={str(path): sha(path) for path in files},
        training_modified=False, optimizer_executed=False, hardware_enabled=False, shutdown=False)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training-output', type=Path, required=True)
    cli.add_argument('--static-output', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.training_output, args.static_output, args.output)
    print(json.dumps({k: result[k] for k in ('actual_updates', 'raw_integrity_verified',
        'static_retention_verified', 'heldout30_full27_passed', 'sustained60_full27_passed', 'policy_accepted')}))
