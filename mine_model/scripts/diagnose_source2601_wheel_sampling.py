"""Matched frozen51-case wheel exploration diagnosis, never PPO or acceptance."""
import argparse
import ast
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
import subprocess
import textwrap

import numpy as np

import joint_pd_source2601_sampling_std_probe as probe
import diagnose_motion025_full_height as original
import validate_yaw025_static_retention10 as retention
from joint_pd_continuation import ROOT, sha
from train_source200_stationary_height200 import environment
from auto_train_mine import write_json

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261019_002748_664445_source2401_yaw025_unchanged200'
FULL51 = RESULTS / 'joint_pd_motion025_yaw025_unchanged_full_height_20261019_source2601'
STATIC = RESULTS / 'joint_pd_static_retention_yaw025_unchanged_20261019_source2601'


def reader():
    """Original numeric metrics/windows; require sampling rather than forbid it."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.verify)))
    before, edits = ast.dump(tree), []
    for old, new in (
            ("cfg['commands']['ranges_height'] != list(SAFE_BASE_HEIGHT_RANGE_M)",
             "cfg['commands']['ranges_height'] != [.3, .3]"),
            ("manifest['sampled_policy_actions']", "not manifest['sampled_policy_actions']")):
        expected = ast.dump(ast.parse(old, mode='eval').body)
        found = []
        for parent in ast.walk(tree):
            for field, value in ast.iter_fields(parent):
                if isinstance(value, ast.AST) and ast.dump(value) == expected:
                    found.append((parent, field, None, value))
                elif isinstance(value, list):
                    found.extend((parent, field, index, item) for index, item in enumerate(value)
                        if isinstance(item, ast.AST) and ast.dump(item) == expected)
        if len(found) != 1:
            raise ValueError('Requalify exact diagnostic reader guard: ' + old)
        parent, field, index, prior = found[0]
        edits.append((parent, field, index, prior))
        replacement = ast.parse(new, mode='eval').body
        if index is None:
            setattr(parent, field, replacement)
        else:
            getattr(parent, field)[index] = replacement
    changed = copy.deepcopy(tree)
    for parent, field, index, prior in reversed(edits):
        if index is None:
            setattr(parent, field, prior)
        else:
            getattr(parent, field)[index] = prior
    if ast.dump(tree) != before:
        raise ValueError('Original metric or threshold reader changed')
    writes = []
    scope = dict(original.__dict__, write_json=lambda path, value: writes.append((Path(path), value)))
    exec(compile(ast.fix_missing_locations(changed), str(original.SCRIPT), 'exec'), scope)
    return scope['verify'], writes


def source():
    module = retention.adapted()
    checkpoint = module.reviewed_source(TRAINING)
    module.require_full51(FULL51, checkpoint)
    static = json.loads((STATIC / 'static_retention.json').read_text())
    if (sha(checkpoint) != probe.SOURCE_SHA or not static['static_retention_verified']
            or static['checkpoint_sha256'] != probe.SOURCE_SHA
            or not static['deterministic_full_height_steps_passed']
            or not static['sampled_exploration_survived']):
        raise ValueError('Require this checkpoint own completed independent static retention')
    return checkpoint


def command(checkpoint, output, command_plan, std):
    argv = probe.runtime.replay_command(checkpoint, output)
    argv[argv.index(str(probe.runtime.SCRIPT))] = str(probe.SCRIPT)
    i = argv.index('--seeds') + 1
    argv[i:i + 3] = list(map(str, original.SEEDS))
    argv[argv.index('--seconds') + 1] = str(original.SECONDS)
    argv.extend(['--command-probe-json', str(command_plan), '--sample-policy-actions',
                 '--record_policy_inputs', '--wheel-sampling-std', str(std)])
    return argv


def verify(directory, checkpoint, std):
    load = lambda name: json.loads((directory / name).read_text())
    manifest, integrity = load('manifest.json'), load('sampling_probe_integrity.json')
    proof = manifest['sampling_std_probe']
    expected = np.asarray([.05, .05, std, .05, .05, std], dtype=np.float32)
    saved = np.asarray([.05, .05, .03, .05, .05, .03], dtype=np.float32)
    draws = np.load(directory / 'sampling_noise.npz')
    inputs = np.load(directory / 'policy_inputs.npz')
    if (proof['checkpoint_sha256'] != probe.SOURCE_SHA
            or proof['probe_entry_sha256'] != sha(probe.SCRIPT)
            or proof != {key: integrity[key] for key in proof}
            or not integrity['all_model_tensors_bitwise_unchanged']
            or integrity['additional_RNG_draws'] or integrity['gaussian_draw_count'] != 1000
            or not np.array_equal(draws['sampling_std'], expected)
            or not np.array_equal(draws['saved_std'], saved)
            or draws['epsilon'].shape != (1000, 3, 6)
            or not np.isfinite(draws['epsilon']).all()
            or inputs['observations'].shape != (1000, 51, 27)
            or inputs['means'].shape != (1000, 51, 6)):
        raise ValueError('Changed source, sampling vector or incomplete diagnostic capture')
    function, writes = reader()
    result = function(directory, checkpoint)
    if len(writes) != 1 or writes[0][1] != result:
        raise ValueError('Unexpected original-reader side effect')
    result.update(sampling_diagnostic=True, sampling_std=expected.tolist(),
        saved_actor_std=saved.tolist(), original_numeric_metrics_windows_limits_unchanged=True,
        diagnostic_cannot_establish_learning_improvement_or_policy_acceptance=True,
        limitation='Frozen sampled-policy20s full-height diagnosis only. Different noise scales '
                   'are not trained candidates or deterministic acceptance. All failed trials are retained.')
    data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    for trial in result['trials']:
        cid = next(i for i, c in enumerate(original.cases()) if c == trial['case'])
        part = data[(data['case_id'] == cid) & (data['seed'] == trial['seed'])]
        tail = part[part['time_s'] >= 17.]
        trial['extra_diagnostic'] = (dict(wheel_reference_step_rms_rad_s=float(np.sqrt(np.mean(
            np.diff(np.stack([tail['left_wheel_ref_rad_s'], tail['right_wheel_ref_rad_s']], axis=-1), axis=0)**2))))
            if trial['survived'] else None)
    write_json(directory / 'sampled_tracking_diagnosis.json', result)
    return result


def run(output, qualify_only=False):
    checkpoint = source()
    _, _, ast_proof = probe.evaluator_tree()
    # Compile and restore the original reader before any native simulation.
    reader()
    if output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_wheel_sampling_pair_'):
        raise ValueError('Use a new owned output; never repeat completed native evidence')
    output.mkdir()
    command_plan = output / 'command_plan.json'
    write_json(command_plan, original.cases())
    files = [SCRIPT, probe.SCRIPT, probe.runtime.SCRIPT, Path(original.__file__),
        Path(retention.__file__), original.acceptance.CURRENT, original.acceptance.POLICY,
        ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py', checkpoint,
        checkpoint.parent / 'params/env.yaml', checkpoint.parent / 'params/agent.yaml', command_plan,
        TRAINING / 'final_checkpoint_all200_saved_clocks_readonly_20261019.json',
        FULL51 / 'candidate/motion025_full_height_diagnosis.json', STATIC / 'static_retention.json']
    pins = {str(path): sha(path) for path in files}
    plan = dict(created_at=datetime.now().astimezone().isoformat(), checkpoint=str(checkpoint),
        checkpoint_sha256=probe.SOURCE_SHA, source_pins=pins, AST_proof=ast_proof,
        cases=original.cases(), seeds=original.SEEDS, seconds=20.,
        paired_wheel_sampling_std=[.03, .02], leg_sampling_std=.05,
        actor_and_all17_Adam_unchanged=True, PPO_updates_performed=0,
        qualification_only=qualify_only, acceptance_changed=False, policy_accepted=False)
    commands = {str(std): command(checkpoint, output / label, command_plan, std)
        for label, std in (('saved_wheel003', .03), ('probe_wheel002', .02))}
    plan['commands'] = commands
    write_json(output / 'plan.json', plan)
    if qualify_only:
        return dict(qualified=True, native_started=False, **plan)
    results = {}
    for label, std in (('saved_wheel003', .03), ('probe_wheel002', .02)):
        write_json(output / 'status.json', dict(state=label, diagnostic_active=True,
            training_active=False, checkpoint_sha256=probe.SOURCE_SHA, policy_accepted=False))
        with (output / (label + '.log')).open('x') as log:
            subprocess.run(commands[str(std)], cwd=ROOT, env=environment(),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        results[label] = verify(output / label, checkpoint, std)
        if any(sha(path) != digest for path, digest in pins.items()):
            raise ValueError('Frozen diagnosis inputs changed')
    old_draws = np.load(output / 'saved_wheel003/sampling_noise.npz')
    new_draws = np.load(output / 'probe_wheel002/sampling_noise.npz')
    before_inputs = np.load(output / 'saved_wheel003/policy_inputs.npz')
    after_inputs = np.load(output / 'probe_wheel002/policy_inputs.npz')
    if (not np.array_equal(old_draws['epsilon'], new_draws['epsilon'])
            or not np.array_equal(before_inputs['observations'][0], after_inputs['observations'][0])
            or not np.array_equal(before_inputs['means'][0], after_inputs['means'][0])):
        raise ValueError('Paired Gaussian draws or initial state/source inference differ')
    before_cfg = json.loads((output / 'saved_wheel003/resolved_env.json').read_text())
    after_cfg = json.loads((output / 'probe_wheel002/resolved_env.json').read_text())
    before_cfg['log_dir'] = after_cfg['log_dir']
    if before_cfg != after_cfg:
        raise ValueError('Paired environment differs beyond output directory')
    summary, changes = {}, []
    for label, result in results.items():
        summary[label] = dict(survivors=result['survivors'], resets=result['resets'],
            diagnostic_constraint_passes=sum(row['stage_constraints_passed'] for row in result['trials']))
    for old, new in zip(results['saved_wheel003']['trials'], results['probe_wheel002']['trials']):
        if (old['case'], old['seed']) != (new['case'], new['seed']):
            raise ValueError('Matched cases differ')
        changes.append(dict(case=old['case']['name'], seed=old['seed'],
            old_survived=old['survived'], new_survived=new['survived'],
            metric_changes=({k: new['metrics'][k] - old['metrics'][k] for k in old['limits']}
                if old['survived'] and new['survived'] else None)))
    report = dict(checkpoint_sha256=probe.SOURCE_SHA, results=summary, paired_metric_changes=changes,
        exact_same_Gaussian_draws=True, bitwise_equal_initial_observations_and_means=True,
        environment_equal_except_log_directory=True, source_pins=pins,
        actor_and_optimizer_updated=False, acceptance_changed=False, policy_accepted=False,
        limitation='Matched frozen noise diagnosis; not evidence of training convergence or final acceptance.')
    write_json(output / 'paired_wheel_sampling_diagnosis.json', report)
    write_json(output / 'status.json', dict(state='agent_review_required', diagnostic_active=False,
        training_active=False, checkpoint_sha256=probe.SOURCE_SHA, policy_accepted=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualify-only', action='store_true')
    args = parser.parse_args()
    result = run(args.output.resolve(), args.qualify_only)
    print(json.dumps(dict(qualified=result.get('qualified'), results=result.get('results'),
        policy_accepted=False)), flush=True)
