"""Original25/37.5pct full-height51 and own-static reviews of the exact next200 block."""
import argparse
import ast
import copy
from datetime import datetime
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
import joint_pd_source3601_motion0375_runtime as runtime
import diagnose_motion025_full_height as original
import resume_motion025_full_height_diagnosis as corrected
import validate_quality50_static_retention10 as static_template
from review_unchanged_yaw025_full_height import wait_completed
from verify_completed_plain27_joint_pd_trial import verify as verify_checkpoint

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
STATIC_QUALIFICATION = RESULTS / 'joint_pd_source3601_motion0375_static_qualification_20261019.json'


def retention():
    spec = importlib.util.spec_from_file_location('_motion0375_original_static_readers', static_template.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.runtime, module.SCRIPT, module.QUALIFICATION = runtime, SCRIPT, STATIC_QUALIFICATION
    for name, expected_count in (('reviewed_source', 1), ('qualify', 1), ('run', 2)):
        tree = ast.parse(inspect.getsource(getattr(module, name)))
        before, edits = ast.dump(tree), []
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and type(node.value) is float and node.value == .25:
                edits.append(node)
                node.value = runtime.MOTION_SCALE
        if len(edits) != expected_count:
            raise ValueError('Requalify the exact declared static training-stage metadata: ' + name)
        modified = copy.deepcopy(tree)
        for node in edits:
            node.value = .25
        if ast.dump(tree) != before:
            raise ValueError('Original static source guard, metrics, thresholds or windows changed')
        exec(compile(ast.fix_missing_locations(modified), static_template.__file__, 'exec'), module.__dict__)
    original_dependencies = module.dependencies

    def dependencies():
        return dict(original_dependencies(), **{str(Path(static_template.__file__)): sha(static_template.__file__)})

    module.dependencies = dependencies
    return module


def launch_at(path):
    path = Path(path).resolve()
    launch = json.loads(path.read_text())
    if (launch['source_actual_updates'] != 3601 or launch['target_actual_updates'] != 3801
            or launch['source_all17_Adam_step'] != 72020 or launch['target_all17_Adam_step'] != 76020
            or launch['source_checkpoint_sha256'] != runtime.ANCHOR_SHA
            or launch['next_configuration_changes'] != runtime.DELTA
            or launch['controller_sha256'] != sha(launch['controller'])
            or launch['decision_sha256'] != sha(runtime.DECISION)
            or Path(launch['decision']) != runtime.DECISION):
        raise ValueError('Require this exact one-field source3601->3801 native launch')
    runtime.repair_ready()
    return launch


def cases(scale):
    return [dict(case, name=(case['name'] if scale == .25 else
        case['name'].replace('stage025_', 'envelope_probe_')),
        speed=case['speed'] * (scale / .25), yaw=case['yaw'] * (scale / .25)) for case in original.cases()]


def reader(scale, directory, source):
    function = corrected.corrected_reader()
    writes = []
    function.__globals__['cases'] = lambda: cases(scale)
    function.__globals__['write_json'] = lambda path, result: writes.append((Path(path), result))
    result = function(directory, source)
    if writes != [(Path(directory) / 'motion025_full_height_diagnosis.json', result)]:
        raise ValueError('Original51 reader output contract differs')
    return result


def baseline(scale):
    decision = runtime.repair_ready()
    report_path = (Path(decision['full51_output']) / 'candidate/motion025_full_height_diagnosis.json'
        if scale == .25 else Path(decision['wider_frozen_report']))
    report = json.loads(report_path.read_text())
    source = Path(decision['source_checkpoint'])
    rebuilt = reader(scale, report_path.parent, source)
    keys = ('checkpoint', 'checkpoint_sha256', 'total_trials', 'survivors', 'resets',
        'stage_all_constraints_passed', 'criteria', 'full_height_range', 'trials', 'source_files')
    if any(rebuilt[k] != report[k] for k in keys):
        raise ValueError('Original baseline51 raw metrics differ at scale ' + str(scale))
    pins = {**report['source_files'], str(report_path): sha(report_path)}
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Retained full51 baseline changed')
    return report, pins


def qualify(launch_path, checkpoint, qualification):
    launch, qualification = launch_at(launch_path), Path(qualification).resolve()
    if qualification.parent != RESULTS or qualification.exists():
        raise ValueError('Preserve one owned full-height qualification')
    _, cfg = runtime.require_source(checkpoint)
    if cfg['commands']['ranges_height'] != list(original.SAFE_BASE_HEIGHT_RANGE_M):
        raise ValueError('Retain every original safe height endpoint')
    pins, summaries = {}, {}
    for scale in (.25, runtime.MOTION_SCALE):
        report, source_pins = baseline(scale)
        pins.update(source_pins)
        checked = original.command_probe_cases(cases(scale), SimpleNamespace(**cfg['commands']))
        expected = [tuple(c[k] for k in ('name', 'speed', 'yaw', 'height')) for c in cases(scale)]
        if checked != expected:
            raise ValueError('Command cases differ from unchanged project bounds')
        summaries[str(scale)] = dict(trials=report['total_trials'], survivors=report['survivors'],
            resets=report['resets'], passed=sum(t['stage_constraints_passed'] for t in report['trials']))
    pins.update({str(p): sha(p) for p in (SCRIPT, runtime.SCRIPT, Path(original.SCRIPT),
        Path(corrected.SCRIPT), Path(static_template.__file__), Path(launch_path).resolve(), runtime.DECISION)})
    static = retention().qualify(checkpoint, Path(launch['output']), STATIC_QUALIFICATION)
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        launch_path=str(Path(launch_path).resolve()), launch_sha256=sha(launch_path),
        training_output=launch['output'], unit=launch['unit'], target_actual_updates=3801,
        target_all17_Adam_step=76020, baseline_51000_samples_recomputed_at_both_scales=True,
        baseline_native_repeated=False, original_metric_formulas_thresholds_and_windows_unchanged=True,
        original_static_active_source_rejection_passed=static['active_training_rejected'],
        static_qualification=str(STATIC_QUALIFICATION), static_qualification_sha256=sha(STATIC_QUALIFICATION),
        baseline_summaries=summaries, source_pins=pins, PPO_updates_performed=0,
        native_simulations_started=0, policy_accepted=False)
    with qualification.open('x') as stream:
        json.dump(result, stream, indent=2)
    return result


def review51(source, output, scale):
    output = Path(output).resolve()
    expected_prefix = 'joint_pd_motion025_' if scale == .25 else 'joint_pd_motion0375_'
    if output.parent != RESULTS or output.exists() or not output.name.startswith(expected_prefix):
        raise ValueError('Use a fresh declared owned full-height51 output')
    before, pins = baseline(scale)
    runtime.require_source(source)
    # Compare actual native configurations after replay; only training sampler metadata and log_dir differ.
    decision = runtime.repair_ready()
    baseline_dir = (Path(decision['full51_output']) / 'candidate' if scale == .25
        else Path(decision['wider_frozen_report']).parent)
    output.mkdir()
    command_file, candidate = output / 'command_plan.json', output / 'candidate'
    write_json(command_file, cases(scale))
    argv = runtime.replay_command(source, candidate)
    argv[argv.index('--seconds') + 1] = str(original.SECONDS)
    i = argv.index('--seeds') + 1
    argv[i:i + 3] = list(map(str, original.SEEDS))
    argv.extend(['--command-probe-json', str(command_file)])
    if '--record_policy_inputs' not in argv:
        argv.append('--record_policy_inputs')
    if any(x in argv for x in ('--overrides_json', '--command-scale', '--sample-policy-actions',
            '--heading-hold', '--export_dir', runtime.FLAG)):
        raise ValueError('Frozen51 commands are explicit and physical/configuration overrides are forbidden')
    write_json(output / 'plan.json', dict(checkpoint=str(source), checkpoint_sha256=sha(source),
        tested_command_scale=scale, training_motion_scale=runtime.MOTION_SCALE, command=argv,
        baseline_native_repeated=False, source_pins=pins, thresholds_changed=False, policy_accepted=False))
    write_json(output / 'status.json', dict(state='candidate', diagnostic_active=True,
        training_active=False, policy_accepted=False))
    with (output / 'candidate.log').open('x') as log:
        subprocess.run(argv, cwd=ROOT, env=original.environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
    after = reader(scale, candidate, source)
    native_before = json.loads((baseline_dir / 'resolved_env.json').read_text())
    native_after = json.loads((candidate / 'resolved_env.json').read_text())
    native_before['commands']['training_motion_scale'] = runtime.MOTION_SCALE
    native_before['log_dir'] = native_after['log_dir']
    if native_before != native_after or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Frozen native physics/control/reward/source differs beyond the one declared sampler field')
    write_json(candidate / 'motion025_full_height_diagnosis.json', after)
    write_json(output / 'paired_source3601_source3801_full_height.json', dict(
        source_checkpoint_sha256=runtime.ANCHOR_SHA, checkpoint_sha256=sha(source),
        tested_command_scale=scale, training_motion_scale=runtime.MOTION_SCALE,
        source_passed=sum(t['stage_constraints_passed'] for t in before['trials']),
        candidate_passed=sum(t['stage_constraints_passed'] for t in after['trials']),
        source_survivors=before['survivors'], candidate_survivors=after['survivors'],
        candidate_resets=after['resets'], original51_metric_reader_and_all_thresholds_unchanged=True,
        canonical_acceptance_replaced=False, policy_accepted=False))
    write_json(output / 'status.json', dict(state='agent_review_required', diagnostic_active=False,
        training_active=False, total_trials=51, survivors=after['survivors'], resets=after['resets'],
        passed_trials=sum(t['stage_constraints_passed'] for t in after['trials']),
        tested_command_scale=scale, policy_accepted=False))
    return after


def run(launch_path, qualification, output025, output0375, static_output):
    launch = launch_at(launch_path)
    proof = json.loads(Path(qualification).read_text())
    if (not proof['qualified'] or proof['launch_sha256'] != sha(launch_path)
            or proof['static_qualification_sha256'] != sha(STATIC_QUALIFICATION)
            or any(sha(p) != h for p, h in proof['source_pins'].items())):
        raise ValueError('Require intact original-reader and source qualification')
    training = Path(launch['output'])
    wait_completed(SimpleNamespace(EXPECTED_TRAINING=training, EXPECTED_UNIT=launch['unit']),
        Path(output025).with_name(Path(output025).name + '_waiter.json'))
    verified = verify_checkpoint(training)
    source = retention().reviewed_source(training)
    if (saved_updates(source) != 3801 or not verified['verified']
            or verified['all17_Adam_step'] != 76020):
        raise ValueError('Require exact200/full27/current3801/all17 continuous states')
    review51(source, output025, .25)
    review51(source, output0375, runtime.MOTION_SCALE)
    return retention().run(training, output025, static_output)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training-launch', type=Path, required=True)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--qualify-checkpoint', type=Path)
    cli.add_argument('--output025', type=Path)
    cli.add_argument('--output0375', type=Path)
    cli.add_argument('--static-output', type=Path)
    args = cli.parse_args()
    result = (qualify(args.training_launch, args.qualify_checkpoint, args.qualification)
        if args.qualify_checkpoint else run(args.training_launch, args.qualification,
            args.output025, args.output0375, args.static_output))
    print(json.dumps({k: result[k] for k in ('qualified', 'static_retention_verified', 'policy_accepted') if k in result}))
