"""Qualify and run a matched wheel-noise pair after the source static review."""
import argparse
from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np

import diagnose_source2601_wheel_sampling as original
import probe_frozen_joint_pd_sampling as probe
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from auto_train_mine import write_json

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'


def qualify(training, full51, static, static_unit, output, wheel_std):
    training, full51, static = [Path(value).resolve() for value in (training, full51, static)]
    plan_path, verification = training / 'experiment_plan.json', training / 'final_checkpoint_verification_readonly_20261019.json'
    plan, verified = [json.loads(path.read_text()) for path in (plan_path, verification)]
    report_path = full51 / 'candidate/motion025_full_height_diagnosis.json'
    report = json.loads(report_path.read_text())
    checkpoint = Path(verified['checkpoint'])
    paths = [Path(path) for path, digest in plan['source_files'].items()
        if digest == plan['profile']['native_runtime_sha256'] and Path(path).parent == ROOT / 'mine_model/experiments']
    if (len(paths) != 1 or not verified['verified'] or report['checkpoint_sha256'] != sha(checkpoint)
            or verified['checkpoint_sha256'] != sha(checkpoint) or report['total_trials'] != 51
            or len(report['trials']) != 51 or plan['bounded_training_updates'] != 200):
        raise ValueError('Require the intact completed200/full27/full51 source')
    loader = importlib.util.spec_from_file_location('_wheel_sampling_qualification_runtime', paths[0])
    runtime = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(runtime)
    state, cfg = runtime.require_source(checkpoint)
    actual = saved_updates(checkpoint)
    if (not tensor_finite(state) or len(state['optimizer_state_dict']['state']) != 17
            or any(float(value['step']) != 20 * actual for value in state['optimizer_state_dict']['state'].values())
            or cfg['leg_control_mode'] != 'implicit_joint_position' or cfg['observation_space'] != 27
            or not plan['ordinary_full_policy_PPO'] or plan['auxiliary_losses_enabled']):
        raise ValueError('Require finite plain27 direct-joint PPO with all17 Adam states')
    saved = state['model_state_dict']['std'].tolist()
    changed = list(saved)
    for index in (2, 5):
        changed[index] = float(np.float32(wheel_std))
    if wheel_std <= 0. or any(changed[index] == saved[index] for index in (2, 5)):
        raise ValueError('Declare one positive changed local wheel sampling amplitude')
    tree, path, evidence = probe.evaluator_tree(runtime)
    compile(tree, str(path), 'exec')
    original.reader()  # Original metrics/windows/limits, sampling flag allowed.
    pins = {**plan['source_files'], **verified['source_files'], **report['source_files']}
    files = [SCRIPT, probe.SCRIPT, probe.template.SCRIPT, original.SCRIPT, original.original.SCRIPT,
        runtime.SCRIPT, checkpoint, checkpoint.parent / 'params/agent.yaml', checkpoint.parent / 'params/env.yaml',
        verification, plan_path, report_path, original.original.acceptance.CURRENT, original.original.acceptance.POLICY]
    pins.update({str(file): sha(file) for file in files})
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('Pinned source or original readers changed')
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        training_output=str(training), full51_output=str(full51), source_static_output=str(static),
        source_static_unit=static_unit, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        actual_updates=actual, all17_Adam_step=20 * actual, runtime=str(runtime.SCRIPT),
        runtime_sha256=sha(runtime.SCRIPT), saved_std=saved, sampling_vectors=dict(saved=saved, probe=changed),
        changed_action_indices=[2, 5], original_hooks=evidence, original_numeric_readers_preserved=True,
        cases=original.original.cases(), seeds=original.original.SEEDS, seconds=20., source_pins=pins,
        weights_updated=False, optimizer_updated=False, saved_std_updated=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False, native_simulations_started=0)
    with Path(output).open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def wait_static(spec):
    path = Path(spec['source_static_output']) / 'static_retention.json'
    while True:
        try:
            result = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            result = None
        if result is not None:
            if result['checkpoint_sha256'] != spec['checkpoint_sha256']:
                raise ValueError('Own static review belongs to another checkpoint')
            return result
        raw = subprocess.run(['systemctl', '--user', 'show', spec['source_static_unit'],
            '-p', 'ActiveState', '-p', 'SubState', '-p', 'MainPID', '-p', 'Result'],
            capture_output=True, text=True, check=True).stdout
        state = dict(line.split('=', 1) for line in raw.strip().splitlines())
        if (state['Result'] != 'success' or state['ActiveState'] not in ('active', 'activating')
                or state['SubState'] == 'exited' and state['MainPID'] == '0'):
            raise ValueError('Source review ended without a complete own artifact; do not repeat PPO')
        time.sleep(5.)


def command(runtime, spec, specification, directory, commands, label):
    argv = runtime.replay_command(Path(spec['checkpoint']), directory)
    argv[argv.index(str(runtime.SCRIPT))] = str(probe.SCRIPT)
    index = argv.index('--seeds') + 1
    argv[index:index + 3] = list(map(str, spec['seeds']))
    argv[argv.index('--seconds') + 1] = str(spec['seconds'])
    argv.extend(['--command-probe-json', str(commands), '--sample-policy-actions', '--record_policy_inputs',
        '--specification', str(specification), '--sampling-label', label])
    return argv


def verify(directory, spec, label):
    metadata = json.loads((directory / 'manifest.json').read_text())['sampling_std_probe']
    integrity = json.loads((directory / 'sampling_probe_integrity.json').read_text())
    with np.load(directory / 'sampling_noise.npz', allow_pickle=False) as arrays:
        valid = (np.array_equal(arrays['sampling_std'], np.asarray(spec['sampling_vectors'][label], np.float32))
            and np.array_equal(arrays['saved_std'], np.asarray(spec['saved_std'], np.float32))
            and arrays['epsilon'].shape == (1000, 3, 6) and np.isfinite(arrays['epsilon']).all())
    if (not valid or metadata['checkpoint_sha256'] != spec['checkpoint_sha256']
            or metadata['probe_entry_sha256'] != sha(probe.SCRIPT)
            or metadata != {key: integrity[key] for key in metadata}
            or not integrity['all_model_tensors_bitwise_unchanged']
            or integrity['gaussian_draw_count'] != 1000 or integrity['additional_RNG_draws']):
        raise ValueError('Changed saved model, sampling vector or incomplete original Gaussian capture')
    function, writes = original.reader()
    result = function(directory, Path(spec['checkpoint']))
    if len(writes) != 1 or writes[0][1] != result:
        raise ValueError('Unexpected original-reader side effect')
    result.update(sampling_diagnostic=True, saved_actor_std=spec['saved_std'],
        sampling_std=spec['sampling_vectors'][label], original_numeric_metrics_windows_limits_unchanged=True,
        diagnostic_cannot_establish_learning_improvement_or_policy_acceptance=True,
        limitation='Matched frozen20s noise diagnosis only. Every failure and original numeric limit/window remains; no trained convergence or deterministic acceptance is implied.')
    write_json(directory / 'sampled_tracking_diagnosis.json', result)
    return result


def run(specification, output, wait):
    spec, runtime = probe.bound(specification)
    output = Path(output).resolve()
    if output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_wheel_sampling_pair_'):
        raise ValueError('Use a fresh owned paired diagnosis output')
    output.mkdir()
    if wait:
        write_json(output / 'status.json', dict(state='waiting_for_own_static_review', training_active=False,
            diagnostic_active=False, policy_accepted=False))
        wait_static(spec)
    commands = output / 'command_plan.json'
    write_json(commands, spec['cases'])
    argv = {label: command(runtime, spec, specification, output / label, commands, label)
            for label in ('saved', 'probe')}
    write_json(output / 'plan.json', dict(specification=str(specification), specification_sha256=sha(specification),
        commands=argv, source_pins=spec['source_pins'], checkpoint_sha256=spec['checkpoint_sha256'],
        saved_std=spec['saved_std'], sampling_vectors=spec['sampling_vectors'], policy_accepted=False))
    reports = {}
    for label in ('saved', 'probe'):
        write_json(output / 'status.json', dict(state=label, diagnostic_active=True, training_active=False,
            checkpoint_sha256=spec['checkpoint_sha256'], policy_accepted=False))
        with (output / (label + '.log')).open('x') as log:
            subprocess.run(argv[label], cwd=ROOT, env=original.environment(),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        reports[label] = verify(output / label, spec, label)
    with np.load(output / 'saved/sampling_noise.npz') as a, np.load(output / 'probe/sampling_noise.npz') as b:
        same_noise = np.array_equal(a['epsilon'], b['epsilon'])
    with np.load(output / 'saved/policy_inputs.npz') as a, np.load(output / 'probe/policy_inputs.npz') as b:
        same_initial = np.array_equal(a['observations'][0], b['observations'][0]) and np.array_equal(a['means'][0], b['means'][0])
    configs = [json.loads((path / 'resolved_env.json').read_text()) for path in
        (Path(spec['full51_output']) / 'candidate', output / 'saved', output / 'probe')]
    for cfg in configs:
        cfg['log_dir'] = configs[0]['log_dir']
    if (not same_noise or not same_initial or not all(cfg == configs[0] for cfg in configs)
            or any(sha(path) != digest for path, digest in spec['source_pins'].items())):
        raise ValueError('Source, physical configuration or original noise pairing differs')
    result = dict(checkpoint_sha256=spec['checkpoint_sha256'], results={label: dict(
        survivors=report['survivors'], resets=report['resets'],
        diagnostic_constraint_passes=sum(row['stage_constraints_passed'] for row in report['trials']))
        for label, report in reports.items()}, exact_same_Gaussian_draws=True,
        bitwise_equal_initial_observations_and_means=True, environment_equal_except_log_directory=True,
        source_pins=spec['source_pins'], weights_updated=False, optimizer_updated=False, saved_std_updated=False,
        acceptance_changed=False, policy_accepted=False,
        limitation='One common noise seed43 and three reset seeds; frozen diagnosis, not trained improvement or policy acceptance.')
    write_json(output / 'paired_wheel_sampling_diagnosis.json', result)
    write_json(output / 'status.json', dict(state='agent_review_required', diagnostic_active=False,
        training_active=False, checkpoint_sha256=spec['checkpoint_sha256'], policy_accepted=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualify-only', action='store_true')
    parser.add_argument('--training-output', type=Path)
    parser.add_argument('--full51-output', type=Path)
    parser.add_argument('--static-output', type=Path)
    parser.add_argument('--static-unit')
    parser.add_argument('--probe-wheel-std', type=float)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--wait-for-static', action='store_true')
    args = parser.parse_args()
    if args.qualify_only:
        if any(value is None for value in (args.training_output, args.full51_output,
                args.static_output, args.static_unit, args.probe_wheel_std)):
            parser.error('Require source/full51/static/unit and one local wheel amplitude')
        result = qualify(args.training_output, args.full51_output, args.static_output,
            args.static_unit, args.qualification, args.probe_wheel_std)
    else:
        if args.output is None:
            parser.error('Require --output')
        result = run(args.qualification.resolve(), args.output, args.wait_for_static)
    print(json.dumps({key: result[key] for key in ('qualified', 'results', 'policy_accepted') if key in result}))
