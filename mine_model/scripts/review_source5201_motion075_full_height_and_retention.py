"""Original full-height51 readers at all five stages and own static review."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
import joint_pd_source5201_motion075_runtime as native
import plain27_sampler_metadata_adapter as adapter
import review_source3601_motion0375_full_height_and_retention as template

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
STATIC_QUALIFICATION = RESULTS / 'joint_pd_source5201_motion075_static_qualification_20261019.json'
SCALES = (.25, .375, .5, .625, .75)


def ready():
    decision = native.repair_ready()
    prior = Path(decision['training_output'])
    summary = json.loads((prior / 'retained_four_motion_envelopes_review_20261019.json').read_text())
    return dict(decision, full51_output=summary['suites']['0.25']['output'])


runtime = SimpleNamespace(**dict(native.__dict__, repair_ready=ready))
_api = adapter.private_module(template.__file__, '_source5201_original_fullheight_and_static')
_api.SCRIPT, _api.runtime, _api.STATIC_QUALIFICATION = SCRIPT, runtime, STATIC_QUALIFICATION
ADAPTER_EVIDENCE = adapter.retarget(_api, ('launch_at', 'qualify', 'review51'),
    integers={'*': {3601: 5201, 3801: 5401, 72020: 104020, 76020: 108020}},
    strings={'source3601': 'source5201', 'source3801': 'source5401', 'motion0375_': 'motion075_'})


def baseline_path(scale):
    decision = ready()
    if scale == .75:
        return Path(decision['wider_frozen_report'])
    if scale not in SCALES:
        raise ValueError('Retain every declared original command suite')
    summary = json.loads((Path(decision['training_output']) /
        'retained_four_motion_envelopes_review_20261019.json').read_text())
    return Path(summary['suites'][str(scale)]['output']) / 'candidate/motion025_full_height_diagnosis.json'


def baseline(scale):
    path = baseline_path(scale)
    report = json.loads(path.read_text())
    rebuilt = _api.reader(scale, path.parent, Path(ready()['source_checkpoint']))
    keys = ('checkpoint', 'checkpoint_sha256', 'total_trials', 'survivors', 'resets',
        'stage_all_constraints_passed', 'criteria', 'full_height_range', 'trials', 'source_files')
    if any(rebuilt[k] != report[k] for k in keys):
        raise ValueError('Original baseline51 raw metrics or flags differ')
    pins = dict(report['source_files'], **{str(path): sha(path)})
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('A retained baseline dependency changed')
    return report, pins


_api.baseline = baseline
_original_retention = _api.retention


def retention():
    module = _original_retention()
    original = module.dependencies

    def dependencies():
        return dict(original(), **{str(p): sha(p) for p in (SCRIPT, adapter.SCRIPT, Path(template.__file__))})

    module.dependencies = dependencies
    return module


_api.retention = retention


def qualify(launch_path, checkpoint, qualification):
    _, cfg = native.require_source(checkpoint)
    scout_cfg = json.loads((baseline_path(.75).parent / 'resolved_env.json').read_text())
    summaries, pins = {}, {}
    for scale in SCALES:
        report, deps = baseline(scale)
        pins.update(deps)
        expected = [tuple(c[k] for k in ('name', 'speed', 'yaw', 'height')) for c in _api.cases(scale)]
        checked = template.original.command_probe_cases(_api.cases(scale), SimpleNamespace(**cfg['commands']))
        actual = json.loads((baseline_path(scale).parent / 'resolved_env.json').read_text())
        actual['log_dir'] = scout_cfg['log_dir']
        if checked != expected or actual != scout_cfg:
            raise ValueError('Original cases or retained native physical configurations differ')
        summaries[str(scale)] = dict(trials=51, survivors=report['survivors'], resets=report['resets'],
            passed=sum(t['stage_constraints_passed'] for t in report['trials']))
    result = _api.qualify(launch_path, checkpoint, qualification)
    result['source_pins'].update(pins)
    result['source_pins'].update({str(p): sha(p) for p in (SCRIPT, adapter.SCRIPT, Path(template.__file__))})
    result.update(reviewed_command_scales=list(SCALES), baseline_summaries=summaries,
        all255_source_trials_raw_recomputed=True, original_numeric_readers_unchanged=True,
        metadata_adapter_evidence=ADAPTER_EVIDENCE)
    write_json(Path(qualification), result)
    return result


def run(launch_path, qualification, outputs, static_output):
    launch = _api.launch_at(launch_path)
    proof = json.loads(Path(qualification).read_text())
    if (not proof['qualified'] or proof['launch_sha256'] != sha(launch_path)
            or proof['static_qualification_sha256'] != sha(STATIC_QUALIFICATION)
            or proof['reviewed_command_scales'] != list(SCALES) or len(outputs) != 5
            or any(sha(p) != h for p, h in proof['source_pins'].items())):
        raise ValueError('Require the exact original five51/static qualification')
    for output, scale in zip(outputs, SCALES):
        output = Path(output).resolve()
        prefix = 'joint_pd_motion025_' if scale == .25 else 'joint_pd_motion075_'
        if output.parent != RESULTS or output.exists() or not output.name.startswith(prefix):
            raise ValueError('Preserve fresh owned outputs for every retained scope')
    training = Path(launch['output'])
    template.wait_completed(SimpleNamespace(EXPECTED_TRAINING=training, EXPECTED_UNIT=launch['unit']),
        Path(outputs[0]).with_name(Path(outputs[0]).name + '_waiter.json'))
    verified = template.verify_checkpoint(training)
    source = retention().reviewed_source(training)
    if (not verified['verified'] or saved_updates(source) != 5401 or verified['all17_Adam_step'] != 108020):
        raise ValueError('Require exact200, canonical27 and every original Adam clock')
    reports = [_api.review51(source, output, scale) for output, scale in zip(outputs, SCALES)]
    summary = dict(observed_at=datetime.now().astimezone().isoformat(), checkpoint=str(source),
        checkpoint_sha256=sha(source), source_checkpoint_sha256=native.ANCHOR_SHA,
        suites={str(scale): dict(output=str(Path(out).resolve()), total_trials=51,
            survivors=report['survivors'], resets=report['resets'],
            passed=sum(t['stage_constraints_passed'] for t in report['trials']))
            for out, scale, report in zip(outputs, SCALES, reports)},
        original_metrics_and_thresholds_unchanged=True, policy_accepted=False)
    # The retained four-scope artifact keeps its exact historical schema.
    write_json(training / 'retained_four_motion_envelopes_review_20261019.json',
        dict(summary, suites={k: v for k, v in summary['suites'].items() if k != '0.75'}))
    write_json(training / 'retained_five_motion_envelopes_review_20261019.json', summary)
    return retention().run(training, outputs[-1], static_output)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training-launch', type=Path, required=True)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--qualify-checkpoint', type=Path)
    cli.add_argument('--outputs', type=Path, nargs=5)
    cli.add_argument('--static-output', type=Path)
    args = cli.parse_args()
    result = (qualify(args.training_launch, args.qualify_checkpoint, args.qualification)
        if args.qualify_checkpoint else run(args.training_launch, args.qualification, args.outputs, args.static_output))
    print(json.dumps({k: result[k] for k in ('qualified', 'static_retention_verified', 'policy_accepted') if k in result}))
