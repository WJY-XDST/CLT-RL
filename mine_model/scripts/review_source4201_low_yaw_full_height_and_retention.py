"""Original four51 suites and own complete static retention after the yaw-coverage block."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
import source4201_low_yaw_plan as selection
import plain27_sampler_metadata_adapter as adapter
import review_source3601_motion0375_full_height_and_retention as template

SCRIPT = Path(__file__).resolve()
RESULTS = selection.RESULTS
STATIC_QUALIFICATION = RESULTS / 'joint_pd_source4201_low_yaw_static_qualification_20261019.json'
SCALES = (.25, .375, .5, .625)
runtime = selection.api_runtime()
_api = adapter.private_module(template.__file__, '_source4201_low_yaw_original_readers')
_api.SCRIPT, _api.runtime, _api.STATIC_QUALIFICATION = SCRIPT, runtime, STATIC_QUALIFICATION
ADAPTER_EVIDENCE = adapter.retarget(_api, ('launch_at', 'qualify'),
    integers={'*': {3601: 4201, 3801: 4401, 72020: 84020, 76020: 88020}},
    strings={'source3601': 'source4201', 'source3801': 'source4401', 'motion0375_': 'low_yaw_'})


def baseline_path(scale):
    if scale not in SCALES:
        raise ValueError('Keep all four original full-height command suites')
    return Path(selection.ready()['retained_full_height_baselines'][str(scale)])


def baseline(scale):
    path = baseline_path(scale)
    report = json.loads(path.read_text())
    rebuilt = _api.reader(scale, path.parent, Path(selection.ready()['source_checkpoint']))
    keys = ('checkpoint', 'checkpoint_sha256', 'total_trials', 'survivors', 'resets',
        'stage_all_constraints_passed', 'criteria', 'full_height_range', 'trials', 'source_files')
    if any(rebuilt[k] != report[k] for k in keys):
        raise ValueError('Original raw51 metric reproduction differs')
    pins = {**report['source_files'], str(path): sha(path)}
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Original source baseline changed')
    return report, pins


_api.baseline = baseline
_original_retention = _api.retention


def retention():
    module = _original_retention()
    original = module.dependencies

    def dependencies():
        return dict(original(), **{str(p): sha(p) for p in
            (SCRIPT, selection.SCRIPT, adapter.SCRIPT, Path(template.__file__))})

    module.dependencies = dependencies
    return module


_api.retention = retention


def qualify(launch_path, checkpoint, qualification):
    _, cfg = runtime.require_source(checkpoint)
    if cfg['commands']['yaw_min_abs_rate'] != 0.:
        raise ValueError('Qualify the unchanged-state metadata fixture with the declared yaw minimum')
    additional, pins = {}, {}
    reference = json.loads((baseline_path(.625).parent / 'resolved_env.json').read_text())
    for scale in (.375, .5):
        report, source_pins = baseline(scale)
        pins.update(source_pins)
        checked = template.original.command_probe_cases(_api.cases(scale), SimpleNamespace(**cfg['commands']))
        expected = [tuple(c[k] for k in ('name', 'speed', 'yaw', 'height')) for c in _api.cases(scale)]
        if checked != expected:
            raise ValueError('Original command cases changed')
        native = json.loads((baseline_path(scale).parent / 'resolved_env.json').read_text())
        native['log_dir'] = reference['log_dir']
        if native != reference:
            raise ValueError('Retained source physical configuration comparator differs')
        additional[str(scale)] = dict(trials=51, survivors=report['survivors'], resets=report['resets'],
            passed=sum(t['stage_constraints_passed'] for t in report['trials']))
    result = _api.qualify(launch_path, checkpoint, qualification)
    result['source_pins'].update(pins)
    result['source_pins'].update({str(p): sha(p) for p in
        (selection.SCRIPT, adapter.SCRIPT, Path(template.__file__))})
    result['baseline_summaries'].update(additional)
    result.update(reviewed_command_scales=list(SCALES),
        baseline_51000_samples_recomputed_at_all_four_scales=True,
        only_native_configuration_difference={'commands.yaw_min_abs_rate': [.15, 0.]},
        retained_native_configuration_comparisons_qualified=True,
        metadata_adapter_evidence=ADAPTER_EVIDENCE)
    write_json(Path(qualification), result)
    return result


def review51(source, output, scale):
    output = Path(output).resolve()
    prefix = 'joint_pd_motion025_' if scale == .25 else 'joint_pd_low_yaw_'
    if output.parent != RESULTS or output.exists() or not output.name.startswith(prefix):
        raise ValueError('Require fresh owned full-height output')
    before, pins = baseline(scale)
    _, cfg = runtime.require_source(source)
    if cfg['commands']['yaw_min_abs_rate'] != 0.:
        raise ValueError('Candidate must retain the qualified zero minimum')
    output.mkdir()
    command_file, candidate = output / 'command_plan.json', output / 'candidate'
    write_json(command_file, _api.cases(scale))
    argv = runtime.replay_command(source, candidate)
    argv[argv.index('--seconds') + 1] = str(template.original.SECONDS)
    i = argv.index('--seeds') + 1
    argv[i:i + 3] = list(map(str, template.original.SEEDS))
    argv.extend(['--command-probe-json', str(command_file)])
    if '--record_policy_inputs' not in argv:
        argv.append('--record_policy_inputs')
    if any(x in argv for x in ('--overrides_json', '--command-scale', '--sample-policy-actions',
            '--heading-hold', '--export_dir', runtime.FLAG)):
        raise ValueError('No native physical, action, input or acceptance overrides')
    write_json(output / 'plan.json', dict(checkpoint=str(source), checkpoint_sha256=sha(source),
        tested_command_scale=scale, training_motion_scale=.625, command=argv,
        source_pins=pins, original_numeric_readers_unchanged=True, policy_accepted=False))
    write_json(output / 'status.json', dict(state='candidate', diagnostic_active=True,
        training_active=False, policy_accepted=False))
    with (output / 'candidate.log').open('x') as log:
        subprocess.run(argv, cwd=ROOT, env=template.original.environment(), stdout=log,
            stderr=subprocess.STDOUT, check=True)
    after = _api.reader(scale, candidate, source)
    native_before = json.loads((baseline_path(scale).parent / 'resolved_env.json').read_text())
    native_after = json.loads((candidate / 'resolved_env.json').read_text())
    native_before['commands']['yaw_min_abs_rate'] = 0.
    native_before['log_dir'] = native_after['log_dir']
    if native_before != native_after or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Physical config differs beyond the one declared command field')
    write_json(candidate / 'motion025_full_height_diagnosis.json', after)
    write_json(output / 'paired_source4201_candidate4401_full_height.json', dict(
        source_checkpoint_sha256=selection.ANCHOR_SHA, checkpoint_sha256=sha(source),
        tested_command_scale=scale, training_motion_scale=.625,
        source_passed=sum(t['stage_constraints_passed'] for t in before['trials']),
        candidate_passed=sum(t['stage_constraints_passed'] for t in after['trials']),
        source_survivors=before['survivors'], candidate_survivors=after['survivors'],
        candidate_resets=after['resets'], original51_metrics_and_thresholds_unchanged=True,
        canonical_acceptance_replaced=False, policy_accepted=False))
    write_json(output / 'status.json', dict(state='agent_review_required', diagnostic_active=False,
        training_active=False, total_trials=51, survivors=after['survivors'], resets=after['resets'],
        passed_trials=sum(t['stage_constraints_passed'] for t in after['trials']),
        tested_command_scale=scale, policy_accepted=False))
    return after


def run(launch_path, qualification, outputs, static_output):
    launch = _api.launch_at(launch_path)
    proof = json.loads(Path(qualification).read_text())
    if (not proof['qualified'] or proof['launch_sha256'] != sha(launch_path)
            or proof['static_qualification_sha256'] != sha(STATIC_QUALIFICATION)
            or proof['reviewed_command_scales'] != list(SCALES)
            or len(outputs) != 4 or any(sha(p) != h for p, h in proof['source_pins'].items())):
        raise ValueError('Require all four original suites and own-static reader qualification')
    training = Path(launch['output'])
    template.wait_completed(SimpleNamespace(EXPECTED_TRAINING=training, EXPECTED_UNIT=launch['unit']),
        Path(outputs[0]).with_name(Path(outputs[0]).name + '_waiter.json'))
    verified = template.verify_checkpoint(training)
    source = retention().reviewed_source(training)
    if saved_updates(source) != 4401 or not verified['verified'] or verified['all17_Adam_step'] != 88020:
        raise ValueError('Require exactly200 new updates and continuous all17 clocks')
    reports = [review51(source, output, scale) for output, scale in zip(outputs, SCALES)]
    write_json(training / 'retained_four_motion_envelopes_review_20261019.json', dict(
        observed_at=datetime.now().astimezone().isoformat(), checkpoint=str(source),
        checkpoint_sha256=sha(source), source_checkpoint_sha256=selection.ANCHOR_SHA,
        suites={str(scale): dict(output=str(Path(output).resolve()), total_trials=51,
            survivors=report['survivors'], resets=report['resets'],
            passed=sum(t['stage_constraints_passed'] for t in report['trials']))
            for output, scale, report in zip(outputs, SCALES, reports)},
        original_metrics_and_thresholds_unchanged=True, policy_accepted=False))
    return retention().run(training, outputs[-1], static_output)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training-launch', type=Path, required=True)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--qualify-checkpoint', type=Path)
    cli.add_argument('--outputs', type=Path, nargs=4)
    cli.add_argument('--static-output', type=Path)
    args = cli.parse_args()
    result = (qualify(args.training_launch, args.qualify_checkpoint, args.qualification)
        if args.qualify_checkpoint else run(args.training_launch, args.qualification,
            args.outputs, args.static_output))
    print(json.dumps({k: result[k] for k in ('qualified', 'static_retention_verified', 'policy_accepted') if k in result}))
