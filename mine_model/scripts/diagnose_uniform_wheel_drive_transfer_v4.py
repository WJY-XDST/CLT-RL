"""Frozen, bounded wheel-drive intervention on complete recorded replay scopes.

Both wheels and every command use one shared damping value. No source case is
removed, no PPO step runs, and this diagnostic cannot promote a configuration.
The qualified native observer and numerical comparison are reused unchanged.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
sys.path[:0] = [str(ROOT / 'mine_model/scripts'), str(ROOT / 'mine_model/experiments'),
               str(ROOT / 'wheel_legged_isaaclab')]

import diagnose_uniform_wheel_drive_transfer_v3 as delegate
from auto_train_mine import write_json
from joint_pd_continuation import sha

def qualify(args):
    import joint_pd_uniform_control_runtime_v3 as runtime
    from compare_uniform_joint_pd_replay_retention import load

    assert not args.output.exists()
    assert math.isfinite(args.damping) and args.damping > 0.
    runtime.configure(args.recipe)
    batches, sources, pins, unique_trials = [], set(), {}, set()
    for index, baseline in enumerate(args.baseline):
        assessment, manifest, config, trials, inputs = load(baseline)
        source = Path(manifest['checkpoint']).resolve()
        saved, source_config = runtime.require_source(source)
        assert sha(source) == manifest['checkpoint_sha256']
        assert len(saved['optimizer_state_dict']['state']) == 17
        assert all(float(state['step']) == runtime.saved_updates(source) * 20
                   for state in saved['optimizer_state_dict']['state'].values())
        assert manifest['command_probe'] and len(manifest['seeds']) == 3
        assert 0 < len(manifest['cases']) <= 50
        assert manifest['policy_dt'] == .02 and manifest['seconds_per_case'] >= 20.
        before = config['wheel_damping']
        assert before == config['robot']['actuators']['wheels']['damping'] == source_config['wheel_damping']
        # Extend the previously studied range only by a declared factor of two.
        # This is an intervention bound, never an acceptance-tolerance change.
        assert .5 <= args.damping / before <= 2. and args.damping != before
        assert config['rewards'] == source_config['rewards']
        assert assessment['summary']['trials'] == len(trials)
        sources.add((str(source), sha(source), runtime.saved_updates(source), before))
        for case in manifest['cases']:
            for seed in manifest['seeds']:
                key = case['name'], seed
                assert key not in unique_trials, 'Retain unambiguous scoped trial identities'
                unique_trials.add(key)
        batches.append(dict(name=f'scope_{index}', baseline=str(baseline.resolve()),
                            cases=manifest['cases'], seeds=manifest['seeds'],
                            seconds=manifest['seconds_per_case'], trials=len(trials)))
        pins.update(inputs)
        recorded = baseline / 'policy_inputs.npz'
        assert recorded.is_file()
        pins[str(recorded.resolve())] = sha(recorded)
    assert len(sources) == 1 and batches
    source_name, source_sha, updates, before = sources.pop()
    source = Path(source_name)
    _, evaluator, ast_proof = delegate.original.evaluator_tree(runtime)
    paths = [SCRIPT, Path(delegate.__file__), Path(delegate.original.__file__), runtime.SCRIPT,
             args.recipe, source, evaluator,
             SCRIPT.with_name('compare_uniform_joint_pd_replay_retention.py'),
             SCRIPT.with_name('review_historical_source3001_conditional_yaw200.py')]
    paths += sorted((source.parent / 'params').glob('*.yaml'))
    pins.update(runtime.recipe()['source_pins'])
    pins.update({str(path.resolve()): sha(path) for path in paths})
    spec = dict(qualified=True, observed_at=datetime.now(timezone.utc).isoformat(),
                recipe=str(args.recipe.resolve()), source_checkpoint=str(source),
                source_checkpoint_sha256=source_sha, actual_updates=updates,
                baseline_wheel_damping=before, probe_wheel_damping=args.damping,
                intervention_factor=args.damping / before,
                bounded_gain_factor_range=[.5, 2.], batches=batches,
                trials=sum(batch['trials'] for batch in batches),
                evaluator_AST_qualification=ast_proof, source_pins=pins,
                all_baseline_cases_and_seeds_retained=True,
                condition_specific_control=False, optimizer_executed=False,
                new_PPO_updates=0, training_configuration_selected=False,
                acceptance_changed=False, policy_accepted=False,
                hardware_enabled=False, shutdown=False)
    delegate.verify_pins(spec)
    args.output.mkdir()
    write_json(args.output / 'qualification.json', spec)
    print({key: spec[key] for key in ['qualified', 'trials', 'actual_updates',
                                    'baseline_wheel_damping', 'probe_wheel_damping']})


def run(args):
    import numpy as np
    old_script, old_writer = delegate.SCRIPT, delegate.write_json
    old_compare = delegate.original.compare_declared_drive

    def compare(source, candidate, *values):
        result, pins = old_compare(source, candidate, *values)
        initial, means = [], []
        for folder in (source, candidate):
            path = Path(folder) / 'policy_inputs.npz'
            with np.load(path, allow_pickle=False) as arrays:
                initial.append(arrays['observations'][0].copy())
                means.append(arrays['means'][0].copy())
            pins[str(path.resolve())] = sha(path)
        assert np.array_equal(*initial), 'Inspect initial-state differences before interpreting drive transfer'
        first_mean_error = float(np.max(np.abs(means[0] - means[1])))
        assert first_mean_error < 1.e-6
        result.update(initial27_observations_bitwise_equal=True,
                      first_unchanged_actor_mean_max_absolute_difference=first_mean_error)
        return result, pins

    def write(path, value):
        path = Path(path)
        if path.name == 'complete_frozen_uniform_wheel_drive525.json':
            path = path.with_name('complete_frozen_uniform_wheel_drive.json')
            value = dict(value)
            value['paired_summary'] = value.pop('paired525_summary')
            value['all_pairs'] = value.pop('all525_pairs')
            value['scoped_trial_count'] = len(value['all_pairs'])
            value['generic_scope_count'] = len(value['comparisons'])
            value['original_native_observer_and_numeric_comparison_unchanged'] = True
        return old_writer(path, value)

    delegate.SCRIPT, delegate.write_json = SCRIPT, write
    delegate.original.compare_declared_drive = compare
    try:
        return delegate.run(args)
    finally:
        delegate.SCRIPT, delegate.write_json = old_script, old_writer
        delegate.original.compare_declared_drive = old_compare


def replay(request, arguments):
    old_script = delegate.SCRIPT
    delegate.SCRIPT = SCRIPT
    try:
        # The native observer writes its proof before the evaluator creates its
        # result directory. Prepare only that directory; preserve all native flags.
        output = Path(arguments[arguments.index('--output') + 1])
        output.mkdir(parents=True, exist_ok=True)
        return delegate.replay(request, arguments)
    finally:
        delegate.SCRIPT = old_script


def parse_replay_arguments(arguments):
    # Only consume workflow metadata. Every evaluator flag, especially --output,
    # must reach the original evaluator parser unchanged.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--request', type=Path, required=True)
    args, native_arguments = parser.parse_known_args(arguments)
    return args.request, native_arguments


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'replay':
        request, native_arguments = parse_replay_arguments(sys.argv[2:])
        replay(request, native_arguments)
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['qualify', 'run'])
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--baseline', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--damping', type=float)
    args, remaining = parser.parse_known_args()
    if args.mode == 'qualify':
        assert not remaining and args.recipe and args.baseline and args.output and args.damping
        qualify(args)
    else:
        assert not remaining and args.output
        run(args)
