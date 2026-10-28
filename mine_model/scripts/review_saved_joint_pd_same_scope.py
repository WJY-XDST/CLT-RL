"""Freeze a saved policy and replay every case/seed in a retained scope.

This has no training entry point. It preserves scope boundaries, case names,
seed triples, original numerical readers and every failed trial. It never
selects training cases or promotes the policy automatically.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import joint_pd_calibrated_common_noise_runtime as runtime
import review_frozen_joint_pd_command_matrix as matrix
import summarize_frozen_joint_pd_matrix as summaries
from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases


SCRIPT = Path(__file__).resolve()


def contracts(batches):
    return [(job['name'], job['cases'], job['seeds']) for job in batches]


def qualify(*, recipe, checkpoint, verification, original_scope, output):
    runtime.configure(recipe)
    saved, cfg = runtime.require_source(checkpoint)
    verified = matrix.read(verification)
    actual = runtime.saved_updates(checkpoint)
    assert verified['verified'] and verified['all50_native_updates_verified']
    assert verified['actual_ppo_updates'] == actual
    assert Path(verified['checkpoint']).resolve() == checkpoint.resolve()
    assert verified['checkpoint_sha256'] == runtime.sha(checkpoint)
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert {int(v['step']) for v in saved['optimizer_state_dict']['state'].values()} == {actual * 20}
    source = matrix.read(original_scope)
    assert source['qualified'] and source['numerical_reader_extraction_AST_identical']
    assert source['original735_case_seed_contracts_identical']
    assert source['trial_count'] == 735 and source['unique_numeric_commands'] == 219
    assert source['seconds_per_case'] == 20.
    matrix.verify_pins(source['source_pins'])
    reader, previous = matrix.verify_reader()
    assert not output.exists(), 'Do not overwrite or repeat a qualified replay'
    output.mkdir()
    jobs, identities, numeric_commands = [], [], set()
    pins = {str(path.resolve()): runtime.sha(path) for path in (
        SCRIPT, runtime.SCRIPT, recipe, checkpoint, verification, original_scope,
        reader, previous, *sorted((checkpoint.parent / 'params').glob('*.yaml')))}
    for old in source['batches']:
        cases = old['cases']
        expected = [tuple(c[k] for k in ('name', 'speed', 'yaw', 'height')) for c in cases]
        assert command_probe_cases(cases, SimpleNamespace(**cfg['commands'])) == expected
        seeds = old['seeds']
        assert len(seeds) == 3 and len(set(seeds)) == 3
        destination = output / old['name']
        case_file = output / (old['name'] + '_commands.json')
        matrix.write(case_file, cases)
        argv = runtime.replay_command(checkpoint, destination)
        argv[argv.index('--seconds') + 1] = '20'
        at = argv.index('--seeds') + 1
        argv[at:at + 3] = [str(seed) for seed in seeds]
        argv += ['--command-probe-json', str(case_file)]
        jobs.append(dict(name=old['name'], cases=cases, seeds=seeds,
                         command=argv, replay=str(destination), case_file=str(case_file)))
        pins[str(case_file)] = runtime.sha(case_file)
        identities.extend((case['name'], seed) for case in cases for seed in seeds)
        numeric_commands.update(tuple(case[k] for k in ('speed', 'yaw', 'height')) for case in cases)
    assert contracts(jobs) == contracts(source['batches'])
    assert len(identities) == len(set(identities)) == 735
    assert len(numeric_commands) == 219
    report = dict(qualified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint), checkpoint_sha256=runtime.sha(checkpoint),
        actual_updates=actual, recipe=str(recipe), trial_count=735,
        unique_numeric_commands=219, seconds_per_case=20., batches=jobs,
        source_pins=pins, original_scope_qualification=str(original_scope),
        original735_case_seed_contracts_identical=True,
        numerical_reader_extraction_AST_identical=True,
        evaluation_cases_enter_training=False, simulator_started=False,
        optimizer_executed=False, training_modified=False,
        sampled_policy_actions=False, policy_accepted=False,
        full735_pass_does_not_replace_independent_sustained_test=True)
    matrix.verify_pins(pins)
    matrix.write(output / 'qualification.json', report)
    return report


def run(output, before_report):
    result = matrix.run(output)
    before, rows = summaries.load(before_report)
    assert len(rows) == 735
    after = output / 'complete_frozen_command_matrix.json'
    comparison = summaries.run(after, output / 'paired_original735_comparison.json', before_report)
    assert comparison['pairs'] == 735 and comparison['all_commands_and_seeds_matched']
    assert result['summary']['trials'] == 735
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--original-scope', type=Path)
    parser.add_argument('--before-report', type=Path)
    args = parser.parse_args()
    if args.mode == 'qualify':
        assert args.recipe and args.checkpoint and args.verification and args.original_scope
        result = qualify(recipe=args.recipe.resolve(), checkpoint=args.checkpoint.resolve(),
            verification=args.verification.resolve(), original_scope=args.original_scope.resolve(),
            output=args.output.resolve())
        print(json.dumps(dict(qualified=result['qualified'], trial_count=result['trial_count'],
                              actual_updates=result['actual_updates'], simulator_started=False)))
    else:
        assert args.before_report
        result = run(args.output.resolve(), args.before_report.resolve())
        print(json.dumps(dict(verified=result['verified'], pairs=result['pairs'],
            strict10_passed=result['after']['summary']['strict10_passed'], policy_accepted=False)))
