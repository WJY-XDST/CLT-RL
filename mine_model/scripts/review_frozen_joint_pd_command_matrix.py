"""Replay a qualified fixed-command matrix without changing training or gates.

Cases and seed triples are evaluation inputs. The saved actor, ordinary native
replay and extracted original numerical reader retain their existing behavior.
There is no PPO entry point and no automatic policy promotion.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
sys.path[:0] = [str(ROOT / part) for part in (
    'mine_model/scripts', 'mine_model/experiments', 'wheel_legged_isaaclab')]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n')
    temporary.replace(path)


def verify_pins(pins):
    for path, digest in pins.items():
        assert sha(path) == digest, path


def verify_reader():
    """Qualify the extraction without importing the retired trial launcher."""
    current = SCRIPT.with_name('assess_frozen_joint_pd_tracking.py')
    previous = SCRIPT.with_name('review_historical_source3001_conditional_yaw200.py')
    for name in ('read', 'assess'):
        nodes = [next(node for node in ast.parse(path.read_text()).body
                      if isinstance(node, ast.FunctionDef) and node.name == name)
                 for path in (current, previous)]
        assert ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False)
    return current, previous


def qualify(args):
    from types import SimpleNamespace
    import joint_pd_uniform_control_runtime_v3 as runtime
    from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases

    runtime.configure(args.recipe)
    checkpoint = args.checkpoint.resolve()
    saved, cfg = runtime.require_source(checkpoint)
    verification = read(args.verification)
    actual = runtime.saved_updates(checkpoint)
    assert verification['verified'] and verification['actual_ppo_updates'] == actual
    assert Path(verification['checkpoint']).resolve() == checkpoint
    assert verification['checkpoint_sha256'] == sha(checkpoint)
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert all(float(state['step']) == actual * 20
               for state in saved['optimizer_state_dict']['state'].values())
    assert cfg['leg_control_mode'] == 'implicit_joint_position'
    matrix = read(args.matrix)
    assert matrix['schema'] == 'frozen_joint_pd_command_matrix_v1'
    cases, seed_groups = matrix['cases'], matrix['seed_groups']
    assert cases and len({case['name'] for case in cases}) == len(cases)
    assert seed_groups and all(len(group) == 3 for group in seed_groups)
    seeds = [seed for group in seed_groups for seed in group]
    assert all(type(seed) is int and seed >= 0 for seed in seeds)
    assert len(set(seeds)) == len(seeds)
    seconds = matrix['seconds_per_case']
    assert type(seconds) in (int, float) and seconds >= 20.
    reader, previous = verify_reader()
    batches = [cases[index:index + 35] for index in range(0, len(cases), 35)]
    for batch in batches:
        expected = [tuple(case[key] for key in ('name', 'speed', 'yaw', 'height'))
                    for case in batch]
        assert command_probe_cases(batch, SimpleNamespace(**cfg['commands'])) == expected
    assert not args.output.exists(), 'Never overwrite an existing qualified matrix'
    args.output.mkdir(parents=True)
    jobs, pins = [], dict(runtime.recipe()['source_pins'])
    paths = [SCRIPT, runtime.SCRIPT, reader, previous, args.recipe, checkpoint,
             args.verification, args.matrix, *sorted((checkpoint.parent / 'params').glob('*.yaml'))]
    for seed_index, group in enumerate(seed_groups):
        for batch_index, batch in enumerate(batches):
            label = f'seeds{seed_index}_batch{batch_index}'
            commands = args.output / (label + '_commands.json')
            write(commands, batch)
            destination = args.output / label
            argv = runtime.replay_command(checkpoint, destination)
            argv[argv.index('--seconds') + 1] = str(seconds)
            at = argv.index('--seeds') + 1
            argv[at:at + 3] = [str(seed) for seed in group]
            argv += ['--command-probe-json', str(commands.resolve())]
            jobs.append(dict(name=label, command=argv, replay=str(destination.resolve()),
                             cases=batch, seeds=group, case_file=str(commands.resolve())))
            paths.append(commands)
    pins.update({str(path.resolve()): sha(path) for path in paths})
    verify_pins(pins)
    report = dict(qualified=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                  actual_updates=actual, matrix=str(args.matrix.resolve()),
                  recipe=str(args.recipe.resolve()), batches=jobs,
                  trial_count=len(cases) * len(seeds), seconds_per_case=seconds,
                  source_pins=pins, numerical_reader_extraction_AST_identical=True,
                  evaluation_cases_enter_training=False, optimizer_executed=False,
                  simulator_started=False, policy_accepted=False)
    write(args.output / 'qualification.json', report)
    return report


def run(output):
    from assess_frozen_joint_pd_tracking import assess

    request = output / 'qualification.json'
    plan = read(request)
    assert plan['qualified'] and plan['numerical_reader_extraction_AST_identical']
    verify_pins(plan['source_pins'])
    rows, summaries, pins = [], [], dict(plan['source_pins'])
    for job in plan['batches']:
        destination = Path(job['replay'])
        assert not destination.exists(), 'Inspect an existing replay instead of repeating it'
        with (output / (job['name'] + '.log')).open('x') as stream:
            subprocess.run(job['command'], cwd=ROOT, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        manifest, completed = [read(destination / name) for name in ('manifest.json', 'completed.json')]
        assert completed['completed'] and manifest['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert manifest['cases'] == job['cases'] and manifest['seeds'] == job['seeds']
        assert manifest['seconds_per_case'] == plan['seconds_per_case']
        assert manifest['reset_velocity_uniform_amplitude'] == .05
        assert not any(manifest.get(key) for key in (
            'weights_updated', 'sampled_policy_actions', 'policy_input_remapping',
            'action_bias_probes', 'wheel_probe', 'wheel_reference_limit_probe'))
        result = assess(destination)
        assert result['verified'] and result['summary']['trials'] == len(job['cases']) * 3
        rows.extend(result['trials'])
        summaries.append(dict(name=job['name'], summary=result['summary']))
        for name in ('manifest.json', 'completed.json', 'trace.csv', 'resolved_env.json',
                     'policy_inputs.npz', 'original_and_literal10_conditional_assessment.json'):
            path = destination / name
            pins[str(path)] = sha(path)
        write(output / 'progress.json', dict(completed_batches=summaries, policy_accepted=False))
    assert len(rows) == plan['trial_count']
    assert len({(row['case']['name'], row['seed']) for row in rows}) == len(rows)
    summary = dict(trials=len(rows), survivors=sum(row['survived'] for row in rows),
                   original_passed=sum(row['original_passed'] for row in rows),
                   original_AND_literal10_passed=sum(row['original_AND_literal10_passed'] for row in rows),
                   missing_failed_trial_metrics=sum(row['metrics'] is None for row in rows))
    pins[str(request)] = sha(request)
    verify_pins(pins)
    result = dict(complete=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  checkpoint_sha256=plan['checkpoint_sha256'], actual_updates=plan['actual_updates'],
                  summary=summary, batches=summaries, all_trials=rows, source_pins=pins,
                  all_failures_retained=True, original_numerical_reader_unchanged=True,
                  training_modified=False, optimizer_executed=False,
                  acceptance_changed=False, policy_accepted=False)
    write(output / 'complete_frozen_command_matrix.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--matrix', type=Path)
    args = parser.parse_args()
    if args.mode == 'qualify':
        assert args.recipe and args.checkpoint and args.verification and args.matrix
        result = qualify(args)
    else:
        result = run(args.output.resolve())
    print(json.dumps({key: result[key] for key in ('qualified', 'complete', 'trial_count', 'summary')
                      if key in result}, ensure_ascii=False))
