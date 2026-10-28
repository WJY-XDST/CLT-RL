"""Preserve all four51 suites and original static checks after an unchanged200 block."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
import types

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
import train_source4401_consolidation200 as selection
import plain27_sampler_metadata_adapter as adapter
import review_source3601_motion0375_full_height_and_retention as template
import review_source4201_low_yaw_full_height_and_retention as shared

SCRIPT = Path(__file__).resolve()
RESULTS = selection.RESULTS
STATIC_QUALIFICATION = RESULTS / 'joint_pd_source4401_consolidation_static_qualification_20261019.json'
SCALES = (.25, .375, .5, .625)
runtime = selection.api_runtime()
_api = adapter.private_module(template.__file__, '_source4401_consolidation_original_readers')
_api.SCRIPT, _api.runtime, _api.STATIC_QUALIFICATION = SCRIPT, runtime, STATIC_QUALIFICATION
EVIDENCE = adapter.retarget(_api, ('launch_at', 'qualify', 'review51'),
    integers={'*': {3601: 4401, 3801: 4601, 72020: 88020, 76020: 92020}},
    strings={'source3601': 'source4401', 'source3801': 'source4601', 'motion0375_': 'low_yaw_'})
_original_retention = _api.retention

# These are the already-qualified four-source selection/recomputation helpers.
# Bind only the declared source/output/runtime metadata; keep their code intact.
for _name in ('baseline_path', 'baseline', 'retention'):
    _function = getattr(shared, _name)
    globals()[_name] = types.FunctionType(_function.__code__, globals(), _name,
        _function.__defaults__, _function.__closure__)
_api.baseline = baseline
_retention = retention


def retention():
    module = _retention()
    original = module.dependencies

    def dependencies():
        return dict(original(), **{str(Path(shared.__file__)): sha(shared.__file__)})

    module.dependencies = dependencies
    return module


_api.retention = retention
_qualify = types.FunctionType(shared.qualify.__code__, globals(), 'qualify',
    shared.qualify.__defaults__, shared.qualify.__closure__)


def qualify(launch_path, checkpoint, qualification):
    result = _qualify(launch_path, checkpoint, qualification)
    result.update(only_native_configuration_difference={}, all_training_configuration_fields_unchanged=True,
        metadata_adapter_evidence=EVIDENCE)
    result['source_pins'][str(Path(shared.__file__))] = sha(shared.__file__)
    write_json(Path(qualification), result)
    return result


def run(launch_path, qualification, outputs, static_output):
    launch = _api.launch_at(launch_path)
    proof = json.loads(Path(qualification).read_text())
    if (not proof['qualified'] or proof['launch_sha256'] != sha(launch_path)
            or proof['static_qualification_sha256'] != sha(STATIC_QUALIFICATION)
            or proof['reviewed_command_scales'] != list(SCALES) or len(outputs) != 4
            or not proof['all_training_configuration_fields_unchanged']
            or any(sha(p) != h for p, h in proof['source_pins'].items())):
        raise ValueError('Require intact all-four original-reader and unchanged-state qualification')
    training = Path(launch['output'])
    template.wait_completed(SimpleNamespace(EXPECTED_TRAINING=training, EXPECTED_UNIT=launch['unit']),
        Path(outputs[0]).with_name(Path(outputs[0]).name + '_waiter.json'))
    verified = template.verify_checkpoint(training)
    source = retention().reviewed_source(training)
    if saved_updates(source) != 4601 or not verified['verified'] or verified['all17_Adam_step'] != 92020:
        raise ValueError('Require exact200 and every continuous original Adam state')
    reports = [_api.review51(source, output, scale) for output, scale in zip(outputs, SCALES)]
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
