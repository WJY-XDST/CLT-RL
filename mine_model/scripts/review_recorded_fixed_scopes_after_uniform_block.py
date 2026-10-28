"""Retain complete recorded fixed-command scopes after a uniform curriculum step.

Source telemetry is reused. Every source case/seed is replayed with the new
frozen actor and scored by the unchanged reader. No condition enters PPO or
selects an action, reward, controller gain or sampler branch.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import joint_pd_uniform_motion_envelope_runtime_v4 as runtime
import compare_fixed_command_global_motion_replays as paired
import review_historical_source3001_conditional_yaw200 as numeric
from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]


def qualify(args):
    runtime.configure(args.recipe)
    source, _, cfg = runtime.baseline()
    pointer = paired.read(args.pointer)
    assert pointer['source_actual_updates'] == runtime.saved_updates(source)
    assert pointer['target_actual_updates'] == runtime.saved_updates(source) + 200
    assert not args.output.exists()
    pins = {str(path.resolve()): paired.sha(path) for path in
        (SCRIPT, args.recipe, args.pointer, Path(paired.__file__), Path(numeric.__file__), source)}
    jobs = []
    for i, folder in enumerate(args.source):
        assessment, manifest, physical, _, inputs = paired.original.load(folder)
        assert manifest['checkpoint_sha256'] == paired.sha(source)
        assert manifest['command_probe'] and len(manifest['cases']) <= 50 and len(manifest['seeds']) == 3
        assert manifest['seconds_per_case'] == 20. and manifest['policy_dt'] == .02
        assert manifest['reset_velocity_uniform_amplitude'] == .05
        expected_commands = json.loads(json.dumps(cfg['commands']))
        expected_commands.update(ranges_lin_vel_x=[0., 0.],
            ranges_height=[cfg['rewards']['base_height_target']] * 2,
            heading_command=False, grouped_training=False, resampling_time=1000.)
        assert physical['commands'] == expected_commands
        expected = [tuple(row[key] for key in ('name', 'speed', 'yaw', 'height')) for row in manifest['cases']]
        assert command_probe_cases(manifest['cases'], SimpleNamespace(**cfg['commands'])) == expected
        jobs.append(dict(name=f'prior_scope{i}', source=str(folder.resolve()), cases=manifest['cases'],
                         seeds=manifest['seeds'], seconds=20., trials=assessment['summary']['trials']))
        pins.update(inputs)
    args.output.mkdir()
    for job in jobs:
        path = args.output / (job['name'] + '_commands.json')
        path.write_text(json.dumps(job['cases'], indent=2) + '\n')
        job['command_file'] = str(path.resolve())
        pins[str(path.resolve())] = paired.sha(path)
    spec = dict(qualified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        source_checkpoint_sha256=paired.sha(source), source_actual_updates=runtime.saved_updates(source),
        training=pointer['planned_production_output'], recipe=str(args.recipe.resolve()),
        after_owned_unit=args.after_owned_unit, jobs=jobs, trials=sum(job['trials'] for job in jobs),
        source_native_telemetry_reused=True, all_source_cases_and_seeds_retained=True,
        source_pins=pins, training_modified=False, optimizer_executed=False, policy_accepted=False)
    (args.output / 'qualification.json').write_text(json.dumps(spec, indent=2) + '\n')
    return spec


def run(output):
    spec = paired.read(output / 'qualification.json')
    training = Path(spec['training'])
    marker = training / 'independent_review/complete_stage_review_with_global_envelope189.json'
    while not marker.exists():
        text = subprocess.run(['systemctl', '--user', 'show', spec['after_owned_unit'], '-p', 'MainPID', '-p', 'SubState'],
                              capture_output=True, text=True, check=True).stdout
        state = dict(line.split('=', 1) for line in text.splitlines() if '=' in line)
        assert state['MainPID'] != '0', ('Inspect incomplete owned block without repeating PPO', state)
        time.sleep(15.)
    assert paired.read(marker)['complete']
    runtime.configure(Path(spec['recipe']))
    assert all(paired.sha(path) == digest for path, digest in spec['source_pins'].items())
    proof = paired.read(training / 'final_checkpoint_verification_readonly.json')
    assert proof['verified'] and proof['source_checkpoint_sha256'] == spec['source_checkpoint_sha256']
    assert proof['actual_ppo_updates'] == spec['source_actual_updates'] + 200
    checkpoint = Path(proof['checkpoint'])
    runtime.require_source(checkpoint)
    comparisons, pins = [], dict(spec['source_pins'])
    for job in spec['jobs']:
        destination = output / job['name']
        assert not destination.exists(), 'Never repeat or overwrite a partial recorded scope'
        argv = runtime.replay_command(checkpoint, destination)
        argv[argv.index('--seconds') + 1] = str(job['seconds'])
        index = argv.index('--seeds') + 1
        argv[index:index + 3] = list(map(str, job['seeds']))
        argv += ['--command-probe-json', job['command_file']]
        with (output / (job['name'] + '.log')).open('x') as log:
            subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
        numeric.assess(destination)
        report, inputs = paired.compare_pair(Path(job['source']), destination,
            runtime.recipe()['command_changes'], job['name'])
        comparisons.append(report)
        pins.update(inputs)
        (output / 'progress.json').write_text(json.dumps(dict(completed_scopes=[v['scope'] for v in comparisons],
            summaries={v['scope']:v['paired_summary'] for v in comparisons}, policy_accepted=False), indent=2) + '\n')
    rows = [row for comparison in comparisons for row in comparison['all_pairs']]
    assert len(rows) == spec['trials']
    assert len({(scope['scope'], row['case']['name'], row['seed'])
                for scope in comparisons for row in scope['all_pairs']}) == spec['trials']
    for path in (marker, training / 'final_checkpoint_verification_readonly.json', checkpoint):
        pins[str(path)] = paired.sha(path)
    assert all(paired.sha(path) == digest for path, digest in pins.items())
    physical_keys = {(*(row['case'][k] for k in ('height', 'speed', 'yaw')), row['seed']) for row in rows}
    result = dict(complete=True, observed_at=datetime.now(timezone.utc).isoformat(), comparisons=comparisons,
        total_scoped_trials=len(rows), distinct_physical_case_seed_count=len(physical_keys),
        candidate_checkpoint_sha256=paired.sha(checkpoint), all_source_scopes_retained=True,
        original_numeric_scoring_unchanged=True, initial27_observations_bitwise_equal=True,
        source_pins=pins, training_modified=False, optimizer_executed=False, policy_accepted=False,
        limitation='Scoped summaries remain separate because previous scopes can share physical command coordinates.')
    (output / 'complete_recorded_fixed_scopes.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--qualify-only', action='store_true')
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--pointer', type=Path)
    parser.add_argument('--source', type=Path, nargs='+')
    parser.add_argument('--after-owned-unit')
    args = parser.parse_args()
    if args.qualify_only:
        assert args.recipe and args.pointer and args.source and args.after_owned_unit
        qualify(args)
    else:
        run(args.output.resolve())
