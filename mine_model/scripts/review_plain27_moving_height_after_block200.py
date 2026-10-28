"""Keep the original moving-height review after an actual retained200 block.

Only paths, native runtime ownership and completed optimizer clocks are bound
from the launch. The original plan, commands, limits and numeric assessor stay
unchanged. This module never performs PPO or changes the saved distribution.
"""
import argparse
import builtins
from datetime import datetime
import dis
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from qualify_source801_drift50_transfer import differences
import plain27_sampler_metadata_adapter as adapter
import review_plain27_exploration_block200 as owner
import diagnose_consolidation_motion_height_steps as original

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'


def build(review_launch, qualification, output):
    review_launch, qualification, output = [Path(p).resolve() for p in
        (review_launch, qualification, output)]
    queued = json.loads(review_launch.read_text())
    if (not queued['queued'] or queued['script'] != str(owner.SCRIPT)
            or queued['script_sha256'] != sha(owner.SCRIPT)
            or queued['qualification_sha256'] != sha(queued['qualification'])
            or queued['static_qualification_sha256'] != sha(queued['static_qualification'])
            or output.parent != RESULTS or not output.name.startswith('joint_pd_motion_height_steps_')):
        raise ValueError('Require the intact exact-target original-review launch')
    bound = owner.build(queued['training_launch'], queued['static_qualification'])
    proof = json.loads(Path(queued['qualification']).read_text())
    if (not proof['qualified'] or proof['launch_sha256'] != sha(queued['training_launch'])
            or queued['training_output'] != bound.launch['output']
            or any(sha(p) != h for p, h in proof['source_pins'].items())):
        raise ValueError('Require the CPU-qualified full original reviewers')
    worker = adapter.private_module(original.__file__, '_original_plain27_moving_height_review')
    worker.SCRIPT, worker.QUALIFICATION, worker.OUTPUT = SCRIPT, qualification, output
    worker.controller = SimpleNamespace(runtime=bound.native)
    worker.review = SimpleNamespace(retention=bound.worker.retention)
    evidence = adapter.retarget(worker, ('run',), integers={'*': {
        4601: bound.target_clock, 92020: 20 * bound.target_clock}})
    for function in (worker.argv, worker.assess, worker.run):
        for instruction in dis.get_instructions(function):
            if (instruction.opname == 'LOAD_GLOBAL' and instruction.argval not in function.__globals__
                    and not hasattr(builtins, instruction.argval)):
                raise ValueError('Unresolved original-reader global: ' + instruction.argval)
    return SimpleNamespace(worker=worker, bound=bound, queued=queued,
        review_launch=review_launch, qualification=qualification, output=output, evidence=evidence)


def qualify(review_launch, qualification, output, checkpoint):
    context = build(review_launch, qualification, output)
    if context.qualification.exists() or context.output.exists():
        raise ValueError('Preserve previous qualifications and native results')
    bound, worker, queued = context.bound, context.worker, context.queued
    checkpoint = Path(checkpoint).resolve()
    _, cfg = bound.native.require_source(checkpoint)
    source = Path(bound.launch['source_checkpoint'])
    _, prior_cfg = bound.native.require_source(source, transition=True)
    if set(differences(prior_cfg, cfg)) - {'/log_dir'}:
        raise ValueError('The untrained metadata fixture must preserve actual environment configuration')
    checked, sequence, period = worker.command_probe_plan(worker.plan(), SimpleNamespace(**cfg['commands']))
    if (len(checked) != 9 or len(sequence) != 10 or period != 5.
            or cfg['commands']['training_motion_scale'] != .625
            or cfg['min_root_height'] != .22 or cfg['low_height_terminal_time_s'] != .15
            or max(abs(c['speed']) for c in worker.plan()['phases'][0]) > .625 * .8):
        raise ValueError('Preserve the complete previous command plan and fixed physical cutoffs')
    prior = Path(bound.launch['previous']) / 'replay'
    assessment = json.loads((prior / 'assessment.json').read_text())
    manifest = json.loads((prior / 'manifest.json').read_text())
    trace = np.genfromtxt(prior / 'trace.csv', delimiter=',', names=True)
    criteria = worker.acceptance.profile()['criteria']
    maximum_error, count = 0., 0
    for cid, case in enumerate(assessment['conditions']):
        for trial in case['trials']:
            tail = trace[(trace['case_id'] == cid) & (trace['seed'] == trial['seed']) & (trace['time_s'] >= 17.)]
            physical = worker.acceptance.assessor.metrics_for_window(tail, case, criteria)
            limits = worker.limits(tail, case, criteria)
            for key, value in limits.items():
                maximum_error = max(maximum_error, abs(value - trial['limits'][key]),
                    abs(physical[key] - trial['metrics'][key]))
            flags = worker.acceptance.assessor.tracking_components(trial['survived'], physical,
                {key: value - 1.e-12 for key, value in limits.items()})
            if any(flags[k] != trial[k] for k in flags):
                raise ValueError('Preserve all original task/reference component flags')
            count += 1
    if count != 27 or maximum_error > 1.e-12 or manifest['checkpoint_sha256'] != sha(source):
        raise ValueError('Require raw metrics, limits and flags for the actual original source27')
    files = [SCRIPT, owner.SCRIPT, Path(original.__file__), context.review_launch,
        Path(queued['training_launch']), Path(queued['qualification']), Path(queued['static_qualification']),
        Path(bound.launch['controller']), Path(bound.launch['runtime']), source, checkpoint,
        prior / 'assessment.json', prior / 'manifest.json', prior / 'trace.csv',
        worker.acceptance.CURRENT, worker.acceptance.POLICY, Path(worker.acceptance.assessor.__file__),
        Path(worker.static.__file__), Path(worker.command_probe_plan.__code__.co_filename)]
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        output=str(context.output), training_output=bound.launch['output'], training_unit=bound.launch['unit'],
        after_review_unit=queued['unit'], review_launch=str(context.review_launch),
        review_launch_sha256=sha(context.review_launch), training_launch=queued['training_launch'],
        source_actual_updates=bound.source_clock, target_actual_updates=bound.target_clock,
        target_all17_Adam_step=20 * bound.target_clock, cases=9, seeds=worker.SEEDS,
        phases=10, phase_seconds=5., samples_per_native_run=67500, command_plan=worker.plan(),
        original27_numeric_limit_flag_parity_max_error=maximum_error, original27_trials_checked=count,
        original_previous_plan_limits_and_assessor_reused=True, complete_original_function_namespace_verified=True,
        fixed_min_root_height_unchanged=.22, low_height_terminal_time_unchanged=.15,
        source_pins={str(p): sha(p) for p in files}, run_metadata_evidence=context.evidence,
        command_construction={label: worker.argv(checkpoint, context.output / label, sampled)
            for label, sampled in [('deterministic', False), ('saved_Gaussian', True)]},
        PPO_updates_performed=0, simulator_started=False, acceptance_changed=False,
        original_canonical_and_static_retention_not_replaced=True, policy_accepted=False)
    with context.qualification.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def run(review_launch, qualification, output):
    context = build(review_launch, qualification, output)
    proof = json.loads(context.qualification.read_text())
    if (not proof['qualified'] or proof['review_launch_sha256'] != sha(context.review_launch)
            or proof['target_actual_updates'] != context.bound.target_clock
            or proof['target_all17_Adam_step'] != 20 * context.bound.target_clock
            or proof['output'] != str(context.output)
            or any(sha(p) != h for p, h in proof['source_pins'].items())):
        raise ValueError('Preserve the exact qualified source, clocks and original numerical readers')
    result = context.worker.run()
    source = Path(result['checkpoint'])
    digest = sha(source)
    modes = {}
    pins = {str(p): sha(p) for p in (source, SCRIPT, context.qualification)}
    for label in ('deterministic', 'saved_Gaussian'):
        directory = context.output / label
        report, manifest, complete = [json.loads((directory / n).read_text()) for n in
            ('motion_height_steps_diagnosis.json', 'manifest.json', 'completed.json')]
        data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
        raw_resets = int(np.count_nonzero((data['terminated'] > 0) | (data['timeout'] > 0)))
        if (manifest['checkpoint_sha256'] != digest or report['checkpoint_sha256'] != digest
                or raw_resets != complete['resets'] or raw_resets != report['resets']):
            raise ValueError('Resolve raw OR termination/timeout counts and frozen-source identity')
        modes[label] = dict(raw_OR_reset_count=raw_resets, raw_rows=len(data),
            complete_resets=complete['resets'], report_resets=report['resets'], checkpoint_sha256=digest)
        pins.update({str(directory / n): sha(directory / n) for n in
            ('trace.csv', 'manifest.json', 'completed.json', 'motion_height_steps_diagnosis.json')})
    if sha(source) != digest or any(sha(p) != h for p, h in proof['source_pins'].items()):
        raise ValueError('Read-only review dependencies or frozen source changed')
    write_json(context.output / 'motion_height_steps_raw_integrity_readonly_20261019.json', dict(
        verified=True, checkpoint_sha256=digest, modes=modes, source_pins=pins,
        original_readiness_unchanged=result['supplemental_steps_ready'],
        run_metadata_evidence=context.evidence, training_modified=False,
        acceptance_changed=False, optimizer_executed=False, policy_accepted=False))
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--review-launch', type=Path, required=True)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--qualify-checkpoint', type=Path)
    args = cli.parse_args()
    result = (qualify(args.review_launch, args.qualification, args.output, args.qualify_checkpoint)
        if args.qualify_checkpoint else run(args.review_launch, args.qualification, args.output))
    print(json.dumps({k: result[k] for k in
        ('qualified', 'actual_updates', 'supplemental_steps_ready', 'policy_accepted') if k in result}))
