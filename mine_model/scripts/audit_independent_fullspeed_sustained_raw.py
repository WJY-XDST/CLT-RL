"""Reproduce a completed signed120s review from its complete native evidence.

This is a read-only companion to the original review worker. It retains the
original54 trials, all98 overlapping windows, strict gates and failed None
metrics. It cannot run a simulator, an optimizer or select another PPO block.
"""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from joint_pd_continuation import ROOT, sha
import verify_independent_fullspeed_sustained as original

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
GATE = ROOT / 'mine_model/config/tracking_acceptance.json'
GATE_SHA = '08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d'
CHANNELS = (('speed_mae_m_s', 'speed'), ('yaw_mae_rad_s', 'yaw'), ('height_mae_m', 'height'))


def read(path):
    return json.loads(Path(path).read_text())


def intact(pins):
    changed = [p for p, digest in pins.items() if sha(p) != digest]
    if changed:
        raise ValueError('Immutable native evidence changed: ' + repr(changed[:8]))


def identity(trial):
    return json.dumps([trial['case'], trial['seed']], sort_keys=True)


def compare_original_trials(rebuilt, reported):
    fields = ('case', 'seed', 'survived', 'first_failure_s', 'raw_OR_resets', 'metrics',
        'limits', 'task_goal_passed', 'reference_tracking_passed', 'failed_metrics',
        'failed_prerequisites', 'passed', 'original_steady_windows_per_trial')
    if len(rebuilt) != len(reported):
        raise ValueError('Keep every original sustained trial, including failures')
    for actual, expected in zip(rebuilt, reported):
        if any(actual[k] != expected[k] for k in fields):
            raise ValueError('Original sustained metrics, windows, limits or strict flags differ')
        if not actual['survived'] and (actual['metrics'] is not None or actual['passed']):
            raise ValueError('A failed first episode must retain None metrics and cannot pass')


def audit(training, review, destination, baseline=None):
    training, review, destination = [Path(p).resolve() for p in (training, review, destination)]
    if (training.parent != RESULTS or review.parent != RESULTS or destination.exists()
            or not destination.is_relative_to(RESULTS)):
        raise ValueError('Use owned completed evidence and a fresh output')
    final_path = training / 'final_checkpoint_verification_readonly_20261019.json'
    complete_path = review / 'complete_original_review.json'
    original_path = review / 'sustained120/independent_fullspeed_sustained.json'
    plan_path = training / 'experiment_plan.json'
    plan = read(plan_path)
    final, complete, reported = [read(p) for p in (final_path, complete_path, original_path)]
    source, directory = Path(final['checkpoint']), review / 'sustained120/candidate'
    digest, actual = sha(source), final['actual_ppo_updates']
    if (not final['verified'] or final['exact_actual_update_delta'] != 200
            or final['all17_Adam_step'] != 20 * actual
            or any(final['all200_all4096_alignment_max_errors'].values())
            or not final['all200_original_RNG_unchanged']
            or any(r['checkpoint_sha256'] != digest for r in (final, complete, reported))
            or any(r['actual_updates'] != actual for r in (complete, reported))
            or not complete['all_required_review_data_complete'] or sha(GATE) != GATE_SHA):
        raise ValueError('Require exact retained-state200, completed original scope and unchanged10% gates')
    intact(final['source_files'])
    manifest, finished, cfg = [read(directory / p) for p in
        ('manifest.json', 'completed.json', 'resolved_env.json')]
    reference_path = training / 'replay/resolved_env.json'
    reference = deepcopy(read(reference_path))
    reference['log_dir'], reference['episode_length_s'] = cfg['log_dir'], 122.
    reference['scene']['num_envs'] = reference['terrain']['num_envs'] = 54
    # The command probe supplies the explicit full-height commands. Its
    # resolved random sampler is intentionally fixed at the nominal height.
    # Derive the qualified case endpoints from the saved training contract.
    cases = original.cases_for(plan['training_config'])
    if (cfg != reference or manifest['checkpoint_sha256'] != digest
            or manifest['cases'] != cases or manifest['seeds'] != [1283, 1284, 1285]
            or manifest['seconds_per_case'] != 120. or not manifest['command_probe']
            or manifest['command_sequence'] is not None or manifest['command_phase_seconds'] is not None
            or manifest['weights_updated'] or manifest['sampled_policy_actions']
            or manifest['policy_input_remapping'] or manifest['policy_acceptance_test']
            or manifest['action_bias_probes'] or manifest['wheel_probe']
            or manifest['command_scale'] != 1. or manifest['curriculum_progress'] is not None
            or any(manifest.get(k) is not None for k in ('wheel_reference_limit_probe',
                'joint_saturation_reward_probe', 'tracking_goal_component_reward_probe',
                'tracking_goal_reward_probe', 'wheel_action_saturation_reward_probe'))
            or manifest['reset_velocity_uniform_amplitude'] != .05
            or manifest['physics_dt'] != .005 or manifest['policy_dt'] != .02
            or manifest['solver_position_iterations'] != 128 or manifest['solver_velocity_iterations'] != 8
            or manifest['enable_external_forces_every_iteration'] is not True
            or manifest['leg_control_mode'] != 'implicit_joint_position'
            or manifest['saved_wheel_action_limit'] != cfg['wheel_action_limit']
            or manifest['wheel_reference_max_rad_s'] != cfg['action_scale_vel'] * cfg['wheel_action_limit']
            or not finished['completed'] or finished['samples'] != 324000):
        raise ValueError('Preserve original cases, resets, observations and normal saved controller/physics')
    intact(manifest['asset_chain_sha256'])
    files = [SCRIPT, Path(original.__file__), Path(original.numeric.__file__),
        Path(original.numeric.acceptance.assessor.__file__), original.numeric.acceptance.POLICY,
        GATE, final_path, complete_path, original_path, plan_path, reference_path, source]
    files.extend(directory / name for name in
        ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz'))
    pins = {str(p): sha(p) for p in files}
    pins.update(manifest['asset_chain_sha256'])
    raw = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    if len(raw) != 324000 or any(not np.isfinite(raw[k]).all() for k in raw.dtype.names):
        raise ValueError('Retain every finite raw frame, including every failed first episode')
    data = raw.reshape(6000, 54)
    with np.load(directory / 'policy_inputs.npz') as z:
        observations, means = z['observations'], z['means']
        if (observations.shape != (6000, 54, 27) or means.shape != (6000, 54, 6)
                or not np.isfinite(observations).all() or not np.isfinite(means).all()
                or z['seeds'].tolist() != manifest['seeds'] or float(z['policy_dt']) != .02):
            raise ValueError('Keep the complete ordinary27-input/6-output policy timeline')
    references = np.stack([data[k] for k in ('left_wheel_ref_rad_s', 'right_wheel_ref_rad_s')], -1)
    cap = cfg['wheel_action_limit']
    expected = cfg['action_scale_vel'] * np.clip(means[..., [2, 5]], -cap, cap)
    mapping_error = float(abs(references - expected).max())
    resets = int(np.count_nonzero((data['terminated'] != 0) | (data['timeout'] != 0)))
    if resets != finished['resets'] or mapping_error > 1.e-7:
        raise ValueError('Raw OR resets or saved wheel action execution mapping differ')
    trials = original.score(data, cases, manifest['seeds'], 120., 100.)
    compare_original_trials(trials, reported['trials'])
    counts = dict(total_trials=54, survivors=sum(t['survived'] for t in trials),
        passed=sum(t['passed'] for t in trials), raw_OR_resets=resets)
    if any(reported[k] != v for k, v in counts.items()):
        raise ValueError('Original whole-scope counts differ from raw episodes')
    paired = None
    if baseline is not None:
        baseline = Path(baseline).resolve()
        previous = read(baseline)
        if not previous['raw_integrity_verified'] or previous['cases'] != cases or previous['seeds'] != manifest['seeds']:
            raise ValueError('Pair the identical original signed sustained cases and seeds')
        intact(previous['source_pins'])
        with np.load(Path(previous['native_directory']) / 'policy_inputs.npz') as z:
            same_initial = np.array_equal(observations[0], z['observations'][0])
        if not same_initial:
            raise ValueError('Paired initial physical policy inputs differ')
        left = {identity(t): t for t in previous['trials']}
        if len(left) != len(trials) or set(left) != {identity(t) for t in trials}:
            raise ValueError('Every paired trial must retain its full case/seed identity')
        pairs = [(left[identity(t)], t) for t in trials]
        paired = dict(baseline=str(baseline), initial_native_observations_bitwise_equal=True,
            baseline_actual_updates=previous['actual_updates'], candidate_actual_updates=actual,
            survival_rescues=sum(not a['survived'] and b['survived'] for a, b in pairs),
            survival_regressions=sum(a['survived'] and not b['survived'] for a, b in pairs),
            numeric_rescues=sum(not a['passed'] and b['passed'] for a, b in pairs),
            numeric_regressions=sum(a['passed'] and not b['passed'] for a, b in pairs))
        pins[str(baseline)] = sha(baseline)
    defined = [t for t in trials if t['metrics'] is not None]
    failed = Counter(k for t in trials for k in t['failed_metrics'])
    nonzero_failures = Counter()
    literal_both = 0
    for trial in trials:
        relative = None if trial['metrics'] is None else {
            metric: 100 * trial['metrics'][metric] / abs(trial['case'][command]) if trial['case'][command] else None
            for metric, command in CHANNELS}
        trial['nonzero_command_relative_error_percent'] = relative
        violations = [] if relative is None else [k for k, v in relative.items() if v is not None and not v < 10.]
        nonzero_failures.update(violations)
        literal_both += relative is not None and not violations and bool(trial['passed'])
    intact(pins)
    result = dict(observed_at=datetime.now().astimezone().isoformat(), raw_integrity_verified=True,
        checkpoint=str(source), checkpoint_sha256=digest, actual_updates=actual,
        all17_Adam_step=20 * actual, native_directory=str(directory), cases=cases, seeds=manifest['seeds'],
        **counts, trials=trials, original_undefined_failed_trials=54-len(defined),
        original_all7_metrics_limits_flags_and_failed_None_reproduction_max_error=0.,
        original98_overlapping_150frame_windows_per_trial_preserved=True,
        failed_metric_counts=dict(failed), nonzero_relative_failed_metric_counts=dict(nonzero_failures),
        original_gates_and_nonzero_relative10_both_passed=literal_both,
        worst_defined_metrics={metric:max((t['metrics'][metric] for t in defined), default=None)
            for metric in ('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','body_p95_deg')},
        worst_nonzero_command_relative_percent={metric:max((t['nonzero_command_relative_error_percent'][metric]
            for t in defined if t['nonzero_command_relative_error_percent'][metric] is not None), default=None)
            for metric, _ in CHANNELS}, paired_comparison=paired,
        actual_normal_saved_wheel_mapping_max_error_rad_s=mapping_error,
        source_pins=pins, training_modified=False, optimizer_executed=False, simulator_started=False,
        acceptance_changed=False, original_static_and_all657_and_transitions_not_replaced=True,
        policy_accepted=False, full_goal_acceptance_unproven=True)
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training', type=Path, required=True)
    cli.add_argument('--review', type=Path, required=True)
    cli.add_argument('--destination', type=Path, required=True)
    cli.add_argument('--baseline', type=Path)
    args = cli.parse_args()
    result = audit(args.training, args.review, args.destination, args.baseline)
    print(json.dumps({k:result[k] for k in ('actual_updates','raw_integrity_verified',
        'total_trials','survivors','passed','raw_OR_resets','original_gates_and_nonzero_relative10_both_passed')}))
