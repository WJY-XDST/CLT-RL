"""Original four51 and own static reviews for one retained exploration-only block.

Source clocks and artifact paths come from the immutable actual launch. Native
commands, numeric readers, static specifications and thresholds are reused.
"""
import argparse
import builtins
from datetime import datetime
import dis
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import types

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
import plain27_sampler_metadata_adapter as adapter
import review_source3601_motion0375_full_height_and_retention as template
import review_source4201_low_yaw_full_height_and_retention as shared

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
SCALES = (.25, .375, .5, .625)


def build(launch_path, static_qualification):
    launch_path, static_qualification = Path(launch_path).resolve(), Path(static_qualification).resolve()
    launch = json.loads(launch_path.read_text())
    decision_path, runtime_path = Path(launch['decision']), Path(launch['runtime'])
    start, target = launch['source_actual_updates'], launch['target_actual_updates']
    delta = launch['next_configuration_changes']
    if (target != start + 200 or launch['source_all17_Adam_step'] != 20 * start
            or launch['target_all17_Adam_step'] != 20 * target
            or sha(launch['controller']) != launch['controller_sha256']
            or sha(runtime_path) != launch['runtime_sha256']
            or sha(decision_path) != launch['decision_sha256']
            or any(delta[k] for k in ('commands', 'rewards', 'controller', 'physics'))):
        raise ValueError('Require the actual exact200 launch with unchanged environment and control')
    native = adapter.private_module(runtime_path, '_reviewed_original_native_exploration_runtime')

    def ready():
        decision = json.loads(decision_path.read_text())
        static = json.loads(Path(decision['source_static_retention_report']).read_text())
        if (decision['source_checkpoint_sha256'] != launch['source_checkpoint_sha256']
                or decision['source_actual_updates'] != start or decision['source_all17_Adam_step'] != 20 * start
                or decision['next_configuration_changes'] != delta
                or not static['static_retention_verified']
                or static['checkpoint_sha256'] != launch['source_checkpoint_sha256']
                or decision['relative_error_target'] != .1 or decision['acceptance_changed']
                or any(sha(p) != h for p, h in decision['source_pins'].items())):
            raise ValueError('Preserve the reviewed actual source and original10% gate')
        if launch['source_checkpoint_sha256'] == native.ANCHOR_SHA:
            native.repair_ready()
        return decision

    decision = ready()
    baseline_manifest = Path(decision['training_output']) / 'retained_four_motion_envelopes_review_20261019.json'
    baselines = json.loads(baseline_manifest.read_text())
    if (set(baselines['suites']) != set(map(str, SCALES))
            or baselines['checkpoint_sha256'] != launch['source_checkpoint_sha256']):
        raise ValueError('Preserve all four source scopes for this actual checkpoint')
    runtime = SimpleNamespace(**dict(native.__dict__, ANCHOR_SHA=launch['source_checkpoint_sha256'],
        DECISION=decision_path, DELTA=delta, repair_ready=ready, MOTION_SCALE=.625))
    selection = SimpleNamespace(SCRIPT=Path(launch['controller']), RESULTS=RESULTS,
        ANCHOR_SHA=launch['source_checkpoint_sha256'], ready=ready)
    api = adapter.private_module(template.__file__, '_original_four51_static_exploration_readers')
    api.SCRIPT, api.runtime, api.STATIC_QUALIFICATION = SCRIPT, runtime, static_qualification
    evidence = adapter.retarget(api, ('launch_at', 'qualify'), integers={'*': {
        3601: start, 3801: target, 72020: 20 * start, 76020: 20 * target}},
        strings={'source3601': 'source' + str(start), 'source3801': 'source' + str(target)})
    worker = ModuleType('_complete_original_exploration_review_namespace')
    # Retain the actual full defining namespace, including subprocess. Exported
    # functions cannot be safely rebound into a hand-picked incomplete globals.
    worker.__dict__.update(shared.__dict__)
    worker.__dict__.update(SCRIPT=SCRIPT, RESULTS=RESULTS, SCALES=SCALES,
        STATIC_QUALIFICATION=static_qualification, runtime=runtime, selection=selection,
        _api=api, _original_retention=api.retention, ADAPTER_EVIDENCE=evidence)

    def baseline_path(scale):
        if scale not in SCALES:
            raise ValueError('Retain all four original full-height scopes')
        return Path(baselines['suites'][str(scale)]['output']) / 'candidate/motion025_full_height_diagnosis.json'

    worker.baseline_path = baseline_path
    for name in ('baseline', 'retention', 'qualify', 'review51'):
        fn = getattr(shared, name)
        worker.__dict__[name] = types.FunctionType(fn.__code__, worker.__dict__, name,
            fn.__defaults__, fn.__closure__)
    api.baseline, api.retention = worker.baseline, worker.retention
    label = f'paired_source{start}_candidate{target}_full_height.json'
    replay_evidence = adapter.retarget(worker, ('review51',), exact_strings={
        'paired_source4201_candidate4401_full_height.json': label})
    for function in (worker.baseline, worker.retention, worker.qualify, worker.review51):
        for instruction in dis.get_instructions(function):
            if (instruction.opname == 'LOAD_GLOBAL' and instruction.argval not in function.__globals__
                    and not hasattr(builtins, instruction.argval)):
                raise ValueError('Unresolved original-reader global: ' + instruction.argval)
    return SimpleNamespace(launch=launch, api=api, worker=worker, native=native,
        static_qualification=static_qualification, baseline_manifest=baseline_manifest,
        source_clock=start, target_clock=target, replay_metadata_evidence=replay_evidence)


def qualify(launch_path, static_qualification, checkpoint, qualification):
    bound = build(launch_path, static_qualification)
    _, cfg = bound.native.require_source(checkpoint)
    if cfg['commands']['training_motion_scale'] != .625:
        raise ValueError('This helper retains all four current motion envelopes')
    result = bound.worker.qualify(launch_path, checkpoint, qualification)
    result.update(only_native_configuration_difference={}, native_environment_configuration_unchanged=True,
        exploration_only_difference=bound.launch['next_configuration_changes']['exploration'],
        complete_original_replay_function_namespace_verified=True,
        actual_source_clock=bound.source_clock, actual_target_clock=bound.target_clock)
    for p in (SCRIPT, Path(shared.__file__), Path(template.__file__), adapter.SCRIPT,
            bound.baseline_manifest, Path(bound.launch['controller']), Path(bound.launch['runtime'])):
        result['source_pins'][str(p)] = sha(p)
    write_json(Path(qualification), result)
    return result


def verified_checkpoint(training, bound):
    path = training / 'final_checkpoint_verification_readonly_20261019.json'
    if not path.exists():
        return template.verify_checkpoint(training)
    result = json.loads(path.read_text())
    plan = json.loads((training / 'experiment_plan.json').read_text())
    if (not result['verified'] or result['source_actual_updates'] != bound.source_clock
            or result['actual_ppo_updates'] != bound.target_clock or result['exact_actual_update_delta'] != 200
            or result['source_checkpoint_sha256'] != bound.launch['source_checkpoint_sha256']
            or sha(result['checkpoint']) != result['checkpoint_sha256']
            or result['all17_source_Adam_step'] != 20 * bound.source_clock
            or result['all17_Adam_step'] != 20 * bound.target_clock
            or not result['initial_source_model_Adam_std_and_actual_LR_bitwise_preserved']
            or not result['all_model_and_Adam_tensors_finite']
            or not all(result['all_six_actor_output_heads_changed'])
            or not result['all200_original_RNG_unchanged']
            or any(result['all200_all4096_alignment_max_errors'].values())
            or any(sha(p) != h for p, h in result['source_files'].items())
            or any(sha(p) != h for p, h in plan['source_files'].items())):
        raise ValueError('Resolve changed existing final verification; never repeat PPO')
    return result


def run(launch_path, static_qualification, qualification, outputs, static_output):
    bound = build(launch_path, static_qualification)
    proof = json.loads(Path(qualification).read_text())
    if (not proof['qualified'] or proof['launch_sha256'] != sha(launch_path)
            or proof['static_qualification_sha256'] != sha(static_qualification)
            or proof['reviewed_command_scales'] != list(SCALES) or len(outputs) != 4
            or not proof['native_environment_configuration_unchanged']
            or any(sha(p) != h for p, h in proof['source_pins'].items())):
        raise ValueError('Require the intact exact-source four51/static qualification')
    training = Path(bound.launch['output'])
    template.wait_completed(SimpleNamespace(EXPECTED_TRAINING=training, EXPECTED_UNIT=bound.launch['unit']),
        Path(outputs[0]).with_name(Path(outputs[0]).name + '_waiter.json'))
    verified = verified_checkpoint(training, bound)
    source = bound.worker.retention().reviewed_source(training)
    if (saved_updates(source) != bound.target_clock or not verified['verified']
            or verified['all17_Adam_step'] != 20 * bound.target_clock):
        raise ValueError('Require the completed target and every original continuous Adam state')
    reports = [bound.worker.review51(source, output, scale) for output, scale in zip(outputs, SCALES)]
    write_json(training / 'retained_four_motion_envelopes_review_20261019.json', dict(
        observed_at=datetime.now().astimezone().isoformat(), checkpoint=str(source), checkpoint_sha256=sha(source),
        source_checkpoint_sha256=bound.launch['source_checkpoint_sha256'],
        suites={str(scale): dict(output=str(Path(output).resolve()), total_trials=51,
            survivors=report['survivors'], resets=report['resets'],
            passed=sum(t['stage_constraints_passed'] for t in report['trials']))
            for output, scale, report in zip(outputs, SCALES, reports)},
        original_metrics_and_thresholds_unchanged=True,
        replay_metadata_evidence=bound.replay_metadata_evidence, policy_accepted=False))
    return bound.worker.retention().run(training, outputs[-1], static_output)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--training-launch', type=Path, required=True)
    cli.add_argument('--static-qualification', type=Path, required=True)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--qualify-checkpoint', type=Path)
    cli.add_argument('--outputs', type=Path, nargs=4)
    cli.add_argument('--static-output', type=Path)
    args = cli.parse_args()
    result = (qualify(args.training_launch, args.static_qualification, args.qualify_checkpoint, args.qualification)
        if args.qualify_checkpoint else run(args.training_launch, args.static_qualification,
            args.qualification, args.outputs, args.static_output))
    print(json.dumps({k: result[k] for k in
        ('qualified', 'static_retention_verified', 'policy_accepted') if k in result}))
