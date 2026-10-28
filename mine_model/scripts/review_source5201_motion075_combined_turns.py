"""Paired straight-travel/turning review using the original physical metrics.

Keep canonical, pivot-turn, full-height and sustained reviews. Add the complete
signed-speed/yaw/height grid in three36-case batches, within the native50-case
limit. No model, control, reward, exploration or acceptance setting changes.
"""
import argparse
import ast
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
from audit_plain27_after_queued_reviews import wait_unit
import diagnose_motion025_full_height as original
import review_source5201_motion075_full_height_and_retention as full

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
SEEDS = [683, 684, 685]
SPEEDS = (.2, .6, .8)
YAWS = (.075, .15, .3)


def batches(cfg):
    lo, hi = cfg['commands']['ranges_height']
    result = {}
    for height_name, height in (('low', lo), ('nominal', .3), ('high', hi)):
        cases = []
        for speed in SPEEDS:
            for travel, sign in (('forward', 1.), ('reverse', -1.)):
                for yaw in YAWS:
                    for turn, turn_sign in (('left', 1.), ('right', -1.)):
                        token = lambda x: str(x).replace('.', 'p')
                        cases.append(dict(name=f'{height_name}_{travel}_{token(speed)}_{turn}_{token(yaw)}',
                            speed=sign * speed, yaw=turn_sign * yaw, height=height))
        checked = original.command_probe_cases(cases, SimpleNamespace(**cfg['commands']))
        if len(cases) != 36 or checked != [tuple(c[k] for k in ('name', 'speed', 'yaw', 'height')) for c in cases]:
            raise ValueError('Retain the full combined-command grid inside unchanged project bounds')
        result[height_name] = cases
    return result


def reader(cases, seeds):
    tree = ast.parse(inspect.getsource(original.verify))
    before, edits, counts = ast.dump(tree), [], {}

    def change(node, field, value, label):
        edits.append((node, field, copy.deepcopy(getattr(node, field))))
        setattr(node, field, value)
        counts[label] = counts.get(label, 0) + 1

    height_guard = ast.dump(ast.parse("cfg['commands']['ranges_height'] != list(SAFE_BASE_HEIGHT_RANGE_M)", mode='eval').body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and ast.dump(node) == height_guard:
            change(node, 'comparators', ast.parse("cfg['commands']['ranges_height'] != [.3,.3]", mode='eval').body.comparators,
                'resolved_sampler_metadata')
        elif isinstance(node, ast.Constant) and type(node.value) is int:
            if node.value == 51000:
                change(node, 'value', len(cases) * len(seeds) * 1000, 'declared_native_row_count')
            elif node.value == 51:
                change(node, 'value', len(cases) * len(seeds), 'declared_trial_count')
    if counts != dict(resolved_sampler_metadata=1, declared_native_row_count=2, declared_trial_count=1):
        raise ValueError('Requalify exactly the original sampler and case-count metadata guards')
    changed = copy.deepcopy(tree)
    for node, field, value in reversed(edits):
        setattr(node, field, value)
    if ast.dump(tree) != before:
        raise ValueError('Original metric formulas, physical checks, thresholds or windows changed')
    writes = []
    scope = dict(original.__dict__, cases=lambda: cases, SEEDS=seeds,
        write_json=lambda p, v: writes.append((Path(p), v)))
    exec(compile(ast.fix_missing_locations(changed), str(original.SCRIPT), 'exec'), scope)
    return scope['verify'], writes, dict(changes=counts, restored_AST_identical=True)


def qualify(training_launch, review_launch, moving_launch, destination):
    training_launch, review_launch, moving_launch, destination = [Path(p).resolve() for p in
        (training_launch, review_launch, moving_launch, destination)]
    launch = full._api.launch_at(training_launch)
    review, moving = [json.loads(p.read_text()) for p in (review_launch, moving_launch)]
    proof = json.loads(Path(review['qualification']).read_text())
    if (destination.parent != RESULTS or destination.exists() or not review['queued']
            or review['script'] != str(full.SCRIPT) or review['script_sha256'] != sha(full.SCRIPT)
            or review['training_launch'] != str(training_launch)
            or not proof['qualified'] or proof['reviewed_command_scales'] != list(full.SCALES)
            or proof['launch_sha256'] != sha(training_launch)
            or review['qualification_sha256'] != sha(review['qualification'])
            or not moving['queued'] or moving['training_output'] != launch['output']
            or moving['review_launch'] != str(review_launch)
            or moving['script_sha256'] != sha(moving['script'])
            or moving['qualification_sha256'] != sha(moving['qualification'])):
        raise ValueError('Retain the actual exact200/five51/static/moving review ownership')
    source = Path(launch['source_checkpoint'])
    baseline_runtime = full.native.previous
    saved, cfg = baseline_runtime.require_source(source)
    if (sha(source) != launch['source_checkpoint_sha256'] or saved_updates(source) != 5201
            or any(float(v['step']) != 104020 for v in saved['optimizer_state_dict']['state'].values())):
        raise ValueError('Preserve the reviewed baseline and every original Adam state')
    grid = batches(cfg)
    baseline = full.baseline_path(.25)
    original_report = json.loads(baseline.read_text())
    verify, writes, evidence = reader(original.cases(), original.SEEDS)
    rebuilt = verify(baseline.parent, source)
    if rebuilt != original_report or writes != [(baseline, rebuilt)]:
        raise ValueError('Original baseline51 raw formulas, gates and flags no longer reproduce')
    output = RESULTS / 'joint_pd_combined_turns_motion075_20261019_source5201_candidate5401'
    if output.exists():
        raise ValueError('Keep fresh paired native output and preserve earlier reviews')
    paths = (SCRIPT, original.SCRIPT, full.SCRIPT, training_launch, review_launch, moving_launch,
        Path(review['qualification']), Path(moving['qualification']), source, baseline,
        baseline_runtime.SCRIPT, original.acceptance.CURRENT, original.acceptance.POLICY)
    pins = {**proof['source_pins'], **original_report['source_files'],
        **{str(p): sha(p) for p in paths}}
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Preserve immutable source data, native controls and original10% criteria')
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        training_launch=str(training_launch), training_output=launch['output'],
        review_launch=str(review_launch), moving_launch=str(moving_launch),
        after_unit=moving['unit'], output=str(output), static_output=review['static_output'],
        source_checkpoint=str(source), source_checkpoint_sha256=sha(source),
        baseline_runtime=str(baseline_runtime.SCRIPT), target_actual_updates=5401,
        target_all17_Adam_step=108020, speeds=list(SPEEDS), yaws=list(YAWS),
        heights=cfg['commands']['ranges_height'], batches=grid, seeds=SEEDS, seconds=20.,
        combined_cases_per_model=108, total_trials_per_model=324,
        expected_raw_rows_per_model=324000,
        original51_raw_metric_limit_and_flag_parity_verified=True,
        original_reader_metadata_adapter_evidence=evidence,
        original_numeric_formulas_strict_gates_and151_frame_windows_preserved=True,
        original_canonical_pivot_and_sustained_tests_not_replaced=True,
        source_pins=pins, PPO_updates_performed=0, simulator_started=False,
        training_modified=False, acceptance_changed=False, policy_accepted=False)
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def run(qualification):
    qualification = Path(qualification).resolve()
    spec = json.loads(qualification.read_text())
    if not spec['qualified'] or any(sha(p) != h for p, h in spec['source_pins'].items()):
        raise ValueError('Preserve the exact qualified combined-turn matrix')
    output = Path(spec['output'])
    if output.exists():
        raise ValueError('Inspect previous native results; never repeat PPO from a reader failure')
    output.mkdir()
    wait_unit(spec['after_unit'], output / 'status.json')
    training = Path(spec['training_output'])
    final = json.loads((training / 'final_checkpoint_verification_readonly_20261019.json').read_text())
    candidate = full.retention().reviewed_source(training)
    static = json.loads((Path(spec['static_output']) / 'static_retention.json').read_text())
    if (not final['verified'] or final['actual_ppo_updates'] != 5401 or final['all17_Adam_step'] != 108020
            or final['checkpoint_sha256'] != sha(candidate)
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != sha(candidate)):
        raise ValueError('Require this candidate own completed200/canonical/static evidence')
    checkpoints = dict(baseline=Path(spec['source_checkpoint']), candidate=candidate)
    runtimes = dict(baseline=full.native.previous, candidate=full.native)
    hashes = {name: sha(path) for name, path in checkpoints.items()}
    reports = {name: [] for name in checkpoints}
    for name, checkpoint in checkpoints.items():
        runtime = runtimes[name]
        runtime.require_source(checkpoint)
        for label, cases in spec['batches'].items():
            command_file = output / (label + '_commands.json')
            if not command_file.exists():
                write_json(command_file, cases)
            directory = output / name / label
            argv = runtime.replay_command(checkpoint, directory)
            argv[argv.index('--seconds') + 1] = '20'
            index = argv.index('--seeds') + 1
            argv[index:index + 3] = list(map(str, spec['seeds']))
            argv.extend(['--command-probe-json', str(command_file)])
            if '--record_policy_inputs' not in argv:
                argv.append('--record_policy_inputs')
            if any(x in argv for x in ('--overrides_json', '--command-scale', '--sample-policy-actions',
                    '--heading-hold', '--export_dir', runtime.FLAG)):
                raise ValueError('No input, controller, reward or reference-clamp override')
            write_json(output / f'{name}_{label}_launch.json', dict(command=argv,
                checkpoint_sha256=hashes[name], qualification_sha256=sha(qualification)))
            write_json(output / 'status.json', dict(state=f'{name}_{label}', diagnostic_active=True,
                training_active=False, policy_accepted=False))
            with (output / f'{name}_{label}.log').open('x') as stream:
                subprocess.run(argv, cwd=ROOT, env=original.environment(), stdout=stream,
                    stderr=subprocess.STDOUT, check=True)
            verify, writes, evidence = reader(cases, spec['seeds'])
            report = verify(directory, checkpoint)
            if len(writes) != 1 or report['total_trials'] != 108:
                raise ValueError('The original numeric-reader output contract differs')
            data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
            raw_resets = int(np.count_nonzero((data['terminated'] != 0) | (data['timeout'] != 0)))
            if len(data) != 108000 or raw_resets != report['resets']:
                raise ValueError('Keep every complete trial and raw OR reset')
            report.update(original_numeric_reader_metadata_evidence=evidence,
                straight_travel_and_turning_simultaneously=True,
                raw_OR_reset_count=raw_resets, original_other_reviews_not_replaced=True)
            write_json(directory / 'combined_turn_tracking_diagnosis.json', report)
            reports[name].extend(report['trials'])
            if sha(checkpoint) != hashes[name] or any(sha(p) != h for p, h in spec['source_pins'].items()):
                raise ValueError('Frozen model or pinned native/metric source changed')
    paired = []
    for label in spec['batches']:
        a, b = [output / name / label for name in ('baseline', 'candidate')]
        configs = [json.loads((p / 'resolved_env.json').read_text()) for p in (a, b)]
        configs[0]['commands']['training_motion_scale'] = configs[1]['commands']['training_motion_scale']
        configs[0]['log_dir'] = configs[1]['log_dir']
        with np.load(a / 'policy_inputs.npz') as x, np.load(b / 'policy_inputs.npz') as y:
            if not np.array_equal(x['observations'][0], y['observations'][0]) or configs[0] != configs[1]:
                raise ValueError('Paired native initial states or physical configurations differ')
    for old, new in zip(reports['baseline'], reports['candidate']):
        if old['case'] != new['case'] or old['seed'] != new['seed']:
            raise ValueError('Preserve every paired combined-command trial')
        paired.append(dict(case=new['case'], seed=new['seed'], baseline_survived=old['survived'],
            candidate_survived=new['survived'], baseline_passed=old['stage_constraints_passed'],
            candidate_passed=new['stage_constraints_passed'], baseline_metrics=old['metrics'],
            candidate_metrics=new['metrics'], limits=new['limits']))
    result = dict(observed_at=datetime.now().astimezone().isoformat(),
        baseline_checkpoint_sha256=hashes['baseline'], checkpoint_sha256=hashes['candidate'],
        actual_updates=5401, results={name: dict(total_trials=len(rows),
            survivors=sum(t['survived'] for t in rows),
            passed=sum(t['stage_constraints_passed'] for t in rows),
            original_failed_None_metrics=sum(t['metrics'] is None for t in rows))
            for name, rows in reports.items()}, paired_trials=paired,
        same_reset_seeds_initial_observations_and_physical_configuration_verified=True,
        scope='Signed speeds .2/.6/.8 x signed yaws .075/.15/.3 x low/nominal/high height x3 reset seeds.',
        original_other_reviews_not_replaced=True, source_pins=spec['source_pins'],
        training_modified=False, optimizer_executed=False, acceptance_changed=False,
        policy_accepted=False, goal_remains_active=True)
    write_json(output / 'paired_combined_turns_summary.json', result)
    write_json(output / 'status.json', dict(state='agent_review_required', diagnostic_active=False,
        training_active=False, results=result['results'], policy_accepted=False))
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--qualify-only', action='store_true')
    cli.add_argument('--training-launch', type=Path)
    cli.add_argument('--review-launch', type=Path)
    cli.add_argument('--moving-launch', type=Path)
    args = cli.parse_args()
    value = (qualify(args.training_launch, args.review_launch, args.moving_launch, args.qualification)
        if args.qualify_only else run(args.qualification))
    print(json.dumps({k: value[k] for k in ('qualified', 'target_actual_updates', 'actual_updates', 'results', 'policy_accepted') if k in value}))
