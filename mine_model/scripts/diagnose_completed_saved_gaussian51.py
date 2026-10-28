"""Compare a completed checkpoint's own deterministic51 and saved-Gaussian51.

Only the original sampled-policy flag changes. All saved weights, std, physical
configuration, cases, reset streams and original numeric readers are retained.
"""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import diagnose_source2601_wheel_sampling as sampled_reader
import review_plain27_single_reward_block200 as owner
from audit_plain27_after_queued_reviews import wait_unit

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'


def qualify(training_launch, full_review_launch, after_launch, destination, output):
    training_launch, full_review_launch, after_launch, destination, output = [Path(p).resolve()
        for p in (training_launch, full_review_launch, after_launch, destination, output)]
    launch, review, after = [json.loads(p.read_text()) for p in
        (training_launch, full_review_launch, after_launch)]
    if (destination.parent != RESULTS or destination.exists() or output.parent != RESULTS or output.exists()
            or not after['queued'] or after['script_sha256'] != sha(after['script'])
            or after['target_actual_updates'] != launch['target_actual_updates']
            or review['training_launch'] != str(training_launch) or not review['queued']):
        raise ValueError('Require one fresh exact-target frozen diagnosis after the named original reviews')
    bound = owner.build(training_launch, review['static_qualification'])
    training = Path(launch['output'])
    final_path = training / 'final_checkpoint_verification_readonly_20261019.json'
    final = json.loads(final_path.read_text())
    checkpoint = Path(final['checkpoint'])
    state, cfg = bound.native.require_source(checkpoint)
    if (not final['verified'] or final['actual_ppo_updates'] != bound.target_clock
            or final['all17_Adam_step'] != 20 * bound.target_clock
            or final['checkpoint_sha256'] != sha(checkpoint) or len(state['optimizer_state_dict']['state']) != 17
            or any(float(v['step']) != 20 * bound.target_clock for v in state['optimizer_state_dict']['state'].values())):
        raise ValueError('Require the actual completed checkpoint with every original Adam clock')
    baseline = Path(review['outputs'][0]) / 'candidate'
    report_path = baseline / 'motion025_full_height_diagnosis.json'
    report = json.loads(report_path.read_text())
    if bound.api.reader(.25, baseline, checkpoint) != report or report['checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('Original deterministic51 baseline no longer reproduces every raw trial')
    sampled_reader.reader()  # Restores AST and changes only two metadata guards.
    files = [SCRIPT, owner.SCRIPT, Path(sampled_reader.__file__), sampled_reader.original.SCRIPT,
        training_launch, full_review_launch, after_launch, Path(after['request']),
        final_path, checkpoint, report_path, baseline / 'manifest.json', baseline / 'trace.csv',
        baseline / 'policy_inputs.npz', sampled_reader.original.acceptance.CURRENT,
        sampled_reader.original.acceptance.POLICY]
    pins = {**report['source_files'], **final['source_files'], **{str(p): sha(p) for p in files}}
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Pinned weights, native controls or original numeric readers changed')
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        training_launch=str(training_launch), training_output=str(training), output=str(output),
        runtime=str(bound.native.SCRIPT), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        actual_updates=bound.target_clock, all17_Adam_step=20 * bound.target_clock,
        baseline=str(baseline), after_unit=after['unit'], static_output=review['static_output'],
        cases=sampled_reader.original.cases(), seeds=sampled_reader.original.SEEDS, seconds=20.,
        saved_std=state['model_state_dict']['std'].tolist(), source_pins=pins,
        original_deterministic51_reader_raw_reproduction_verified=True,
        original_sampled51_metadata_adapter_reused_without_numeric_changes=True,
        only_saved_policy_sampling_flag_changes=True, own_static_required_before_native=True,
        optimizer_executed=False, simulator_started=False, policy_accepted=False)
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def run(qualification):
    qualification = Path(qualification).resolve()
    spec = json.loads(qualification.read_text())
    if not spec['qualified'] or any(sha(p) != h for p, h in spec['source_pins'].items()):
        raise ValueError('Preserve the source-bound saved-Gaussian diagnosis qualification')
    output = Path(spec['output'])
    if output.exists():
        raise ValueError('Preserve the previous native run; never repeat it from a reader timeout')
    output.mkdir()
    wait_unit(spec['after_unit'], output / 'status.json')
    static = json.loads((Path(spec['static_output']) / 'static_retention.json').read_text())
    if not static['static_retention_verified'] or static['checkpoint_sha256'] != spec['checkpoint_sha256']:
        raise ValueError('Resolve this same checkpoint own static review before further native diagnosis')
    runtime_spec = importlib.util.spec_from_file_location('_saved_gaussian51_exact_runtime', spec['runtime'])
    runtime = importlib.util.module_from_spec(runtime_spec)
    runtime_spec.loader.exec_module(runtime)
    checkpoint = Path(spec['checkpoint'])
    runtime.require_source(checkpoint)
    command_plan = output / 'command_plan.json'
    write_json(command_plan, spec['cases'])
    target = output / 'saved_Gaussian'
    argv = runtime.replay_command(checkpoint, target)
    index = argv.index('--seeds') + 1
    argv[index:index + 3] = list(map(str, spec['seeds']))
    argv[argv.index('--seconds') + 1] = str(spec['seconds'])
    argv.extend(['--command-probe-json', str(command_plan), '--sample-policy-actions'])
    if '--record_policy_inputs' not in argv:
        argv.append('--record_policy_inputs')
    if any(x in argv for x in ('--overrides_json', '--heading-hold', '--export_dir', runtime.FLAG)):
        raise ValueError('No model, controller, reward, command scale or saved-std override')
    write_json(output / 'plan.json', dict(qualification=str(qualification), qualification_sha256=sha(qualification),
        command=argv, source_pins=spec['source_pins'], only_saved_Gaussian_sampling_enabled=True,
        policy_accepted=False))
    write_json(output / 'status.json', dict(state='saved_Gaussian', diagnostic_active=True,
        training_active=False, checkpoint_sha256=spec['checkpoint_sha256'], policy_accepted=False))
    with (output / 'saved_Gaussian.log').open('x') as log:
        subprocess.run(argv, cwd=ROOT, env=sampled_reader.environment(), stdout=log,
            stderr=subprocess.STDOUT, check=True)
    function, writes = sampled_reader.reader()
    result = function(target, checkpoint)
    if len(writes) != 1 or writes[0][1] != result:
        raise ValueError('Original sampled reader output contract differs')
    write_json(target / 'sampled_tracking_diagnosis.json', result)
    baseline = Path(spec['baseline'])
    a, b = [json.loads((p / 'manifest.json').read_text()) for p in (baseline, target)]
    configs = [json.loads((p / 'resolved_env.json').read_text()) for p in (baseline, target)]
    configs[1]['log_dir'] = configs[0]['log_dir']
    if (configs[0] != configs[1] or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or a['checkpoint_sha256'] != b['checkpoint_sha256'] or a['sampled_policy_actions']
            or not b['sampled_policy_actions'] or b['policy_noise_seed'] != 43):
        raise ValueError('Saved-model, reset streams, commands or physical configuration changed')
    with np.load(baseline / 'policy_inputs.npz') as x, np.load(target / 'policy_inputs.npz') as y:
        same_initial = np.array_equal(x['observations'][0], y['observations'][0]) and np.array_equal(x['means'][0], y['means'][0])
    if not same_initial:
        raise ValueError('The paired native runs did not start from identical observations and means')
    traces = [np.genfromtxt(p / 'trace.csv', delimiter=',', names=True) for p in (baseline, target)]
    records, pins = [], dict(spec['source_pins'])
    radius, width = configs[0]['wheel_radius'], configs[0]['wheel_track_width']
    for mode, directory, trace in zip(('deterministic', 'saved_Gaussian'), (baseline, target), traces):
        complete = json.loads((directory / 'completed.json').read_text())
        resets = int(np.count_nonzero((trace['terminated'] != 0) | (trace['timeout'] != 0)))
        if len(trace) != 51000 or resets != complete['resets'] or any(not np.isfinite(trace[k]).all() for k in trace.dtype.names):
            raise ValueError('Preserve every original finite raw timeline and OR reset event')
        for cid, case in enumerate(spec['cases']):
            for seed in spec['seeds']:
                part = trace[(trace['case_id'] == cid) & (trace['seed'] == seed)]
                tail = part[part['time_s'] >= 17.]
                survived = not np.any((part['terminated'] != 0) | (part['timeout'] != 0))
                if len(part) != 1000 or len(tail) != 151:
                    raise ValueError('Retain complete20s and original151-frame tails')
                metrics = None
                if survived:
                    yaw = tail['yaw_rate_rad_s'] - tail['yaw_cmd_rad_s']
                    reference = radius / width * (tail['right_wheel_ref_rad_s'] - tail['left_wheel_ref_rad_s'])
                    actual = radius / width * (tail['right_wheel_rad_s'] - tail['left_wheel_rad_s'])
                    metrics = dict(yaw_signed_error=float(yaw.mean()), yaw_MAE=float(abs(yaw).mean()),
                        yaw_error_temporal_std=float(yaw.std()),
                        reference_proxy_minus_command=float((reference - tail['yaw_cmd_rad_s']).mean()),
                        actual_wheel_minus_reference_proxy=float((actual - reference).mean()),
                        body_minus_actual_wheel_proxy=float((tail['yaw_rate_rad_s'] - actual).mean()),
                        height_MAE_m=float(abs(tail['height_m'] - tail['height_cmd_m']).mean()))
                records.append(dict(mode=mode, case=case, seed=seed, survived=bool(survived), metrics=metrics))
        for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz'):
            pins[str(directory / name)] = sha(directory / name)
    if any(sha(p) != h for p, h in pins.items()) or sha(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Immutable frozen source or diagnostic inputs changed')
    summary = dict(observed_at=datetime.now().astimezone().isoformat(), checkpoint_sha256=spec['checkpoint_sha256'],
        actual_updates=spec['actual_updates'], bitwise_equal_initial_observations_and_means=True,
        same_saved_std_case_seed_and_physical_configuration=True, original_numeric_readers_unchanged=True,
        saved_Gaussian_original_numeric_passed=sum(t['stage_constraints_passed'] for t in result['trials']),
        saved_Gaussian_survivors=result['survivors'], saved_Gaussian_resets=result['resets'],
        trials=records, source_pins=pins, training_modified=False, optimizer_executed=False,
        acceptance_changed=False, faster_curriculum_authorized=False, policy_accepted=False,
        limitation='One saved Gaussian stream per reset seed. The intervention measures the total noise effect on own trajectories; clipping versus other nonlinear dynamics is not causally isolated. Neither learning benefit nor final acceptance follows.')
    write_json(output / 'own_deterministic_saved_Gaussian51_comparison.json', summary)
    write_json(output / 'status.json', dict(state='agent_review_required', diagnostic_active=False,
        training_active=False, checkpoint_sha256=spec['checkpoint_sha256'], policy_accepted=False))
    return summary


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--qualify-only', action='store_true')
    cli.add_argument('--training-launch', type=Path)
    cli.add_argument('--full-review-launch', type=Path)
    cli.add_argument('--after-launch', type=Path)
    cli.add_argument('--output', type=Path)
    args = cli.parse_args()
    result = (qualify(args.training_launch, args.full_review_launch, args.after_launch,
        args.qualification, args.output) if args.qualify_only else run(args.qualification))
    print(json.dumps({k: result[k] for k in ('qualified', 'actual_updates', 'saved_Gaussian_survivors', 'policy_accepted') if k in result}))
