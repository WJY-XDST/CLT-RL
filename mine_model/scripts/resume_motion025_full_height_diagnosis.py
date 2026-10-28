"""Resume frozen diagnostics after a resolved-config reader schema correction.

The stock evaluator disables random height sampling with (.3,.3), then writes
the17 independent per-environment case targets. Verify actual full-range trace
commands, not the intentionally disabled sampler. Keep the original driver,
native baseline, criteria, checkpoints and training implementation immutable.
"""
import argparse
import ast
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
import subprocess
import textwrap

import diagnose_motion025_full_height as original
from auto_train_mine import write_json
from joint_pd_continuation import sha

SCRIPT = Path(__file__).resolve()


def corrected_reader():
    tree = ast.parse(textwrap.dedent(inspect.getsource(original.verify)))
    before = ast.dump(tree)
    old = ast.parse("cfg['commands']['ranges_height'] != list(SAFE_BASE_HEIGHT_RANGE_M)", mode='eval').body
    new = ast.parse("cfg['commands']['ranges_height'] != [.3, .3]", mode='eval').body
    matches = [node for node in ast.walk(tree) if ast.dump(node) == ast.dump(old)]
    if len(matches) != 1:
        raise ValueError('Requalify the exact original resolved-sampler guard')
    node = matches[0]
    prior = copy.deepcopy(node.comparators)
    node.comparators = copy.deepcopy(new.comparators)
    modified = copy.deepcopy(tree)
    node.comparators = prior
    if ast.dump(tree) != before:
        raise ValueError('Reader changed beyond the one resolved-sampler expectation')
    scope = dict(original.__dict__)
    exec(compile(ast.fix_missing_locations(modified), str(original.SCRIPT), 'exec'), scope)
    return scope['verify']


def run(output, *, reader_only=False):
    output = Path(output).resolve()
    if output.parent != original.RESULTS or not output.name.startswith('joint_pd_motion025_pair_'):
        raise ValueError('Resume only the existing owned frozen diagnosis')
    plan = json.loads((output / 'plan.json').read_text())
    pins = plan['source_pins']

    def unchanged():
        if any(sha(path) != digest for path, digest in pins.items()):
            raise ValueError('Original frozen driver, commands, models or criteria changed')

    unchanged()
    candidate = original.source_for_diagnosis(Path(plan['training_output']))
    baseline = Path(plan['baseline_checkpoint'])
    if candidate != Path(plan['checkpoint']):
        raise ValueError('Candidate differs from the completed original review')
    for source, runtime in ((baseline, original.stationary), (candidate, original.motion)):
        _, cfg = runtime.require_source(source)
        if cfg['commands']['ranges_height'] != list(original.SAFE_BASE_HEIGHT_RANGE_M):
            raise ValueError('Saved checkpoint lost the complete safe height contract')
    reader = corrected_reader()
    proof_path = output / 'reader_schema_guard_correction.json'
    baseline_report = output / 'baseline1401/motion025_full_height_diagnosis.json'
    if proof_path.exists():
        proof = json.loads(proof_path.read_text())
        if proof['resume_script_sha256'] != sha(SCRIPT) or proof['baseline_report_sha256'] != sha(baseline_report):
            raise ValueError('Qualified corrected reader or derived baseline changed')
        old = json.loads(baseline_report.read_text())
        if any(sha(path) != digest for path, digest in old['source_files'].items()):
            raise ValueError('Completed native baseline raw data changed')
    else:
        old = reader(output / 'baseline1401', baseline)
        write_json(proof_path, dict(observed_at=datetime.now().astimezone().isoformat(),
            original_driver_sha256=sha(original.SCRIPT), resume_script_sha256=sha(SCRIPT),
            corrected_guard="resolved cfg.commands.ranges_height == [.3,.3], because stock evaluator line212 disables its random sampler",
            unchanged_full_height_checks='Saved checkpoint full range; all17 declared cases; every recorded height command checked against each full-range target',
            baseline_report_sha256=sha(baseline_report), baseline_native_samples_reused=51000,
            baseline_survivors=old['survivors'], baseline_resets=old['resets'],
            original_native_baseline_repeated=False, thresholds_changed=False,
            training_modified=False, policy_accepted=False))
    unchanged()
    if reader_only:
        return dict(corrected_reader_qualified=True, baseline_samples_reused=51000,
                    baseline_survivors=old['survivors'], candidate_native_started=False,
                    thresholds_changed=False, policy_accepted=False)
    destination = output / 'candidate'
    if not (destination / 'completed.json').exists():
        if destination.exists():
            raise ValueError('Inspect incomplete candidate native evidence before rerunning')
        argv = original.command(candidate, destination, output / 'command_plan.json')
        write_json(output / 'candidate_launch.json', dict(command=argv, checkpoint_sha256=sha(candidate)))
        write_json(output / 'status.json', dict(state='candidate', diagnostic_active=True,
            training_active=False, reused_native_baseline=True, policy_accepted=False))
        with (output / 'candidate.log').open('w') as log:
            subprocess.run(argv, cwd=original.ROOT, env=original.environment(),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    new = reader(destination, candidate)
    unchanged()
    before_cfg = json.loads((output / 'baseline1401/resolved_env.json').read_text())
    after_cfg = json.loads((destination / 'resolved_env.json').read_text())
    comparable = copy.deepcopy(before_cfg)
    comparable['log_dir'] = after_cfg['log_dir']
    comparable['commands']['training_motion_scale'] = after_cfg['commands']['training_motion_scale']
    if (before_cfg['commands']['training_motion_scale'] != 0.
            or after_cfg['commands']['training_motion_scale'] != .25 or comparable != after_cfg):
        raise ValueError('Paired native physics/control configurations changed')
    paired = []
    for left, right in zip(old['trials'], new['trials']):
        if (left['case'], left['seed']) != (right['case'], right['seed']):
            raise ValueError('Paired cases or seeds differ')
        delta = ({key: right['metrics'][key] - left['metrics'][key] for key in right['limits']}
                 if left['survived'] and right['survived'] else None)
        paired.append(dict(case=right['case']['name'], seed=right['seed'], metric_changes=delta,
            baseline_survived=left['survived'], candidate_survived=right['survived']))
    result = dict(baseline_checkpoint_sha256=sha(baseline), checkpoint_sha256=sha(candidate),
        results={name: {key: value[key] for key in
            ('total_trials', 'survivors', 'resets', 'stage_all_constraints_passed')}
                 for name, value in (('baseline1401', old), ('candidate', new))},
        paired_metric_changes=paired, source_pins=pins,
        paired_native_configurations_equal_except_motion_metadata_and_log_directory=True,
        baseline_native_samples_reused=51000, baseline_native_repeated=False,
        reader_correction_proof=str(proof_path), resume_script_sha256=sha(SCRIPT),
        training_modified=False, thresholds_changed=False,
        canonical_acceptance_replaced=False, automatic_source_selection=False, policy_accepted=False)
    target = output / 'paired_motion025_full_height_diagnosis.json'
    if target.exists():
        raise ValueError('Preserve the completed paired report')
    write_json(target, result)
    write_json(output / 'status.json', dict(state='diagnosis_complete', diagnostic_active=False,
        training_active=False, policy_accepted=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-baseline-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.output, reader_only=args.verify_baseline_only), ensure_ascii=False))
