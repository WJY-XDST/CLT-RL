"""Independently compare a verified saved prefix on existing recorded scopes.

The cases, seeds, initial reset convention and numerical comparison are reused.
Both the original controller baseline and its frozen changed-controller baseline
are retained, so the effect of subsequent PPO can be distinguished. No training
or policy promotion is performed by this tool.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from types import FunctionType

import numpy as np

import joint_pd_uniform_control_runtime_v3 as runtime
from auto_train_mine import write_json
from assess_frozen_joint_pd_tracking import assess
from compare_uniform_joint_pd_replay_retention import compare, load, summarize_pairs
from joint_pd_continuation import ROOT, sha


def read(path):
    return json.loads(Path(path).read_text())


def compare_declared_controller(source, candidate, scope):
    """Admit only the recorded damping difference; reuse all numeric math."""
    _, _, before, _, _ = load(source)
    _, _, after, _, _ = load(candidate)
    first, second = before['wheel_damping'], after['wheel_damping']
    assert before['robot']['actuators']['wheels']['damping'] == first
    assert after['robot']['actuators']['wheels']['damping'] == second

    def admitted_load(path):
        values = load(path)
        if Path(path).resolve() == Path(source).resolve():
            values = list(values)
            values[2] = copy.deepcopy(values[2])
            values[2]['wheel_damping'] = second
            values[2]['robot']['actuators']['wheels']['damping'] = second
        return values

    qualified = FunctionType(compare.__code__, dict(compare.__globals__, load=admitted_load),
                            compare.__name__, compare.__defaults__, compare.__closure__)
    result, pins = qualified(source, candidate, scope)
    result['declared_physical_configuration_change'] = {'wheel_damping': [first, second],
        'robot.actuators.wheels.damping': [first, second]}
    result['numeric_compare_function_unchanged'] = True
    return result, pins


def qualify(args):
    runtime.configure(args.recipe)
    proof = read(args.proof)
    checkpoint = Path(proof['checkpoint'])
    assert proof['verified'] and sha(checkpoint) == proof['checkpoint_sha256']
    runtime.require_source(checkpoint)
    assert proof['actual_ppo_updates'] == runtime.saved_updates(checkpoint)
    baseline = read(args.baselines)
    assert baseline['complete']
    chosen = [s for s in baseline['scopes'] if any(s['scope'].startswith(p + '/') for p in args.groups)]
    assert chosen and set(args.groups) == {s['scope'].split('/')[0] for s in chosen}
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    jobs, pins = [], {str(p.resolve()): sha(p) for p in (Path(__file__), args.recipe, args.proof, args.baselines, checkpoint)}
    for index, scope in enumerate(chosen):
        original, frozen = Path(scope['source']), Path(scope['candidate'])
        manifest = read(frozen / 'manifest.json')
        assert manifest['checkpoint_sha256'] == runtime.recipe()['source_checkpoint_sha256']
        assert manifest['reset_velocity_uniform_amplitude'] == .05
        commands = args.output / f'scope_{index}_commands.json'
        write_json(commands, manifest['cases'])
        destination = args.output / f'scope_{index}'
        argv = runtime.replay_command(checkpoint, destination)
        argv[argv.index('--seconds') + 1] = str(manifest['seconds_per_case'])
        at = argv.index('--seeds') + 1
        argv[at:at + 3] = [str(seed) for seed in manifest['seeds']]
        argv += ['--command-probe-json', str(commands)]
        jobs.append(dict(scope=scope['scope'], command=argv, output=str(destination),
                         original_baseline=str(original), frozen_baseline=str(frozen),
                         cases=manifest['cases'], seeds=manifest['seeds'],
                         seconds=manifest['seconds_per_case']))
        for p in (commands, original / 'manifest.json', frozen / 'manifest.json',
                  original / 'completed.json', frozen / 'completed.json'):
            pins[str(p)] = sha(p)
    result = dict(qualified=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  recipe=str(args.recipe.resolve()), checkpoint=str(checkpoint),
                  checkpoint_sha256=sha(checkpoint), actual_updates=proof['actual_ppo_updates'],
                  jobs=jobs, scoped_trials=sum(len(j['cases']) * len(j['seeds']) for j in jobs),
                  all_existing_requested_scopes_retained=True, source_pins=pins,
                  cases_enter_training=False, policy_accepted=False)
    write_json(args.output / 'qualification.json', result)
    return result


def run(output):
    spec = read(output / 'qualification.json')
    assert spec['qualified']
    runtime.configure(Path(spec['recipe']))
    pins = dict(spec['source_pins'])
    assert all(sha(p) == v for p, v in pins.items())
    comparisons = []
    for index, job in enumerate(spec['jobs']):
        destination = Path(job['output'])
        assert not destination.exists(), 'Inspect existing replay; never repeat a finished scope'
        write_json(output / 'status.json', dict(state='independent_replay', active_scope=job['scope'],
                   completed_scopes=len(comparisons), training_active=False, policy_accepted=False))
        with (output / f'scope_{index}.log').open('x') as log:
            subprocess.run(job['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest, completed = read(destination / 'manifest.json'), read(destination / 'completed.json')
        assert completed['completed'] and manifest['checkpoint_sha256'] == spec['checkpoint_sha256']
        assert manifest['cases'] == job['cases'] and manifest['seeds'] == job['seeds']
        assert manifest['seconds_per_case'] == job['seconds']
        assert not any(manifest.get(k) for k in ('weights_updated', 'sampled_policy_actions',
                      'policy_input_remapping', 'action_bias_probes', 'wheel_probe', 'wheel_reference_limit_probe'))
        assess(destination)
        a, pa = compare_declared_controller(Path(job['original_baseline']), destination, job['scope'])
        b, pb = compare(Path(job['frozen_baseline']), destination, job['scope'])
        with np.load(Path(job['frozen_baseline']) / 'policy_inputs.npz') as before, np.load(destination / 'policy_inputs.npz') as after:
            assert before['observations'].shape == after['observations'].shape
            assert np.array_equal(before['observations'][0], after['observations'][0])
        comparisons.append(dict(scope=job['scope'], end_to_end=a, PPO_only=b,
                                initial27_observations_bitwise_equal=True))
        pins.update(pa); pins.update(pb)
        write_json(output / 'progress.json', dict(completed_scopes=[c['scope'] for c in comparisons],
            end_to_end=summarize_pairs([r for c in comparisons for r in c['end_to_end']['all_pairs']]),
            PPO_only=summarize_pairs([r for c in comparisons for r in c['PPO_only']['all_pairs']]),
            policy_accepted=False))
    assert sum(len(c['end_to_end']['all_pairs']) for c in comparisons) == spec['scoped_trials']
    assert all(sha(p) == v for p, v in pins.items())
    result = dict(complete=True, observed_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=spec['checkpoint'], checkpoint_sha256=spec['checkpoint_sha256'],
        actual_updates=spec['actual_updates'], scoped_trials=spec['scoped_trials'], comparisons=comparisons,
        end_to_end=summarize_pairs([r for c in comparisons for r in c['end_to_end']['all_pairs']]),
        PPO_only=summarize_pairs([r for c in comparisons for r in c['PPO_only']['all_pairs']]),
        source_pins=pins, all_failures_retained=True, numeric_reader_unchanged=True,
        optimizer_executed=False, training_modified=False, policy_accepted=False,
        final_full_speed_and_sustained_acceptance_still_required=True)
    write_json(output / 'complete_verified_prefix_comparison.json', result)
    write_json(output / 'status.json', dict(state='complete_agent_decision_required', training_active=False, policy_accepted=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--proof', type=Path)
    parser.add_argument('--baselines', type=Path)
    parser.add_argument('--groups', nargs='+')
    args = parser.parse_args()
    result = qualify(args) if args.mode == 'qualify' else run(args.output.resolve())
    print(json.dumps({k: result[k] for k in ('qualified', 'complete', 'scoped_trials') if k in result}))
