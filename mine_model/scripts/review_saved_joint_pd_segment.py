"""Supply verified legacy-reader metadata for saved ordinary-PPO segments.

The original training plan, checkpoints and numerical readers stay immutable.
Use a separate diagnostic view for missing descriptive fields; bind the event
reader's old200 counter to the actual segment. No robot or PPO math changes.
"""
import argparse
import ast
import copy
import json
from pathlib import Path
import sys

import yaml

import train_saved_joint_pd_review50 as original
import continue_saved_joint_pd_bundle as bundle
import plain27_sampler_metadata_adapter as adapter
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
ORIGINAL_HEALTH = original.health50
ORIGINAL_PROCESSES = bundle.processes


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def diagnostic_view(output):
    plan = original.read(output / 'experiment_plan.json')
    final = original.read(output / 'final_checkpoint_verification_readonly.json')
    assert final['verified'] and final['all50_native_updates_verified']
    assert final['actual_ppo_updates'] == plan['target_actual_ppo_updates']
    assert sha(final['checkpoint']) == final['checkpoint_sha256']
    native = original.read(output / 'native_learning_signal/plan.json')
    guard = original.read(output / 'native_learning_signal/leg_exploration_floor_runtime.json')
    assert guard['profile'] == plan['profile'] and guard['before_std'] == guard['after_std']
    assert guard['all17_Adam_bitwise_preserved'] and guard['actor_and_critic_bitwise_preserved']
    assert native['requested_updates'] == plan['bounded_training_updates'] == 50
    agent_path = Path(final['checkpoint']).parent / 'params/agent.yaml'
    agent = yaml.safe_load(agent_path.read_text())
    additions = dict(exploration_guard={**{k: plan['profile'][k] for k in
        ('minimum_leg_std', 'minimum_wheel_std', 'global_minimum_std')}, 'saved_initial_std_reset': False},
        optimizer_minibatches_per_actual_update=agent['algorithm']['num_learning_epochs'] * agent['algorithm']['num_mini_batches'],
        training_motion_scale=plan['training_config']['commands']['training_motion_scale'],
        full_height_range=plan['training_config']['commands']['ranges_height'],
        external_observation_size=plan['training_config']['observation_space'],
        policy_class=agent['policy']['class_name'])
    assert additions['optimizer_minibatches_per_actual_update'] == 20
    assert additions['external_observation_size'] == 27 and additions['policy_class'] == 'ActorCritic'
    view = output.with_name(output.name + '_diagnostic_view')
    expected = copy.deepcopy(plan)
    for key, value in additions.items():
        assert key not in plan or plan[key] == value
        expected[key] = value
    if view.exists():
        assert original.read(view / 'experiment_plan.json') == expected
    else:
        view.mkdir()
        write(view / 'experiment_plan.json', expected)
    for name in ('native_learning_signal', 'training.log', 'resume_verification.json',
                 'final_checkpoint_verification_readonly.json', 'review_interval_runtime.json'):
        target = view / name
        if target.exists():
            assert target.is_symlink() and target.resolve() == (output / name).resolve()
        else:
            target.symlink_to(output / name, target_is_directory=(output / name).is_dir())
    write(view / 'verified_diagnostic_metadata.json', dict(qualified=True,
        original_training_plan=str(output / 'experiment_plan.json'),
        original_training_plan_sha256=sha(output / 'experiment_plan.json'),
        diagnostic_plan_sha256=sha(view / 'experiment_plan.json'),
        verified_descriptive_fields=additions, actual_agent_yaml_sha256=sha(agent_path),
        native_guard_receipt_sha256=sha(output / 'native_learning_signal/leg_exploration_floor_runtime.json'),
        training_plan_and_checkpoint_unmodified=True, training_modified=False,
        helper_sha256=sha(SCRIPT), policy_accepted=False))
    return view


def events(output, through):
    import audit_native_terminal_event_ages as reader
    plan = original.read(output / 'experiment_plan.json')
    assert plan['native_num_envs'] == 8192 and 0 < through <= plan['bounded_training_updates'] <= 200
    api = adapter.private_module(reader.__file__, '_saved_segment_original_event_reader')
    evidence = adapter.retarget(api, ('audit',), integers={'*': {4096: 8192}},
        strings={'4096': '8192'}, keyword_names={
            'all4096_ended_episodes': 'all8192_ended_episodes',
            'all4096_unsafe_events': 'all8192_unsafe_events',
            'all4096_numerical_events': 'all8192_numerical_events',
            'all_prefix_native4096_alignment_errors_zero': 'all_prefix_native8192_alignment_errors_zero'},
        comparisons={'audit': [("plan['bounded_training_updates']", 200, plan['bounded_training_updates'])]})
    path, _ = api.audit(output, through)
    write(output / f'event_reader_boundary_metadata_prefix_{through:04d}.json', dict(
        qualified=True, adapter_evidence=evidence, original_numerical_reader_unchanged=True,
        report=str(path), report_sha256=sha(path), helper_sha256=sha(SCRIPT),
        training_modified=False, policy_accepted=False))


def compare(output, through):
    import compare_plain27_recorded_joint_targets as reader
    from audit_saved_action_limits import transform
    from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit

    admitted = []

    def expected_difference(configs):
        before, after = configs
        assert 'targeted_turn_command_fraction' not in after['commands']
        left, right = original.runtime.normalize_saved_bundle(before), copy.deepcopy(after)
        for cfg in (left, right):
            cfg.setdefault('wheel_action_limit', checkpoint_wheel_action_limit(cfg))
            cfg.pop('log_dir', None)
        assert left == right, 'Only the retired sampler descriptor and serialized defaults may differ'
        # The original recursive reader reports a dict-key removal as
        # /commands/keys, not /commands/<removed_field>. Full equality above
        # first proves that no other command value or physical contract changed.
        actual = set(reader.differences(before, after))
        admitted.append(sorted(actual))
        return actual

    tree = transform(reader.compare, [("{'/log_dir'}", 'expected_difference(cfgs)'),
        ('np.clip(r[..., (2, 5)], -1., 1.)',
         'np.clip(r[..., (2, 5)], -saved_limit(cfg), saved_limit(cfg))')])
    scope = dict(reader.__dict__, SCRIPT=SCRIPT, expected_difference=expected_difference,
                 saved_limit=checkpoint_wheel_action_limit)
    exec(compile(ast.fix_missing_locations(tree), str(reader.SCRIPT), 'exec'), scope)
    path, _ = scope['compare'](output, through)
    write(output / f'saved_contract_comparison_prefix_{through:04d}.json', dict(qualified=True,
        canonicalized_full_configuration_equal=True, original_difference_paths=admitted,
        motor_mapping_FK_and_original_recorded_input_math_unchanged=True,
        saved_wheel_action_limit_used=True, report=str(path), report_sha256=sha(path),
        helper_sha256=sha(SCRIPT), training_modified=False, policy_accepted=False))


def processes(delegate):
    base = ORIGINAL_PROCESSES(delegate)

    class ReadOnlyReaders(type(base)):
        def routed(self, command):
            command = super().routed(command)
            if isinstance(command, (list, tuple)) and len(command) > 2:
                if (Path(command[1]).name == 'audit_historical_source3001_count_bound.py'
                        and '--reader' in command and command[command.index('--reader') + 1] == 'events'):
                    return [command[0], str(SCRIPT), 'events', command[2], '--through', command[command.index('--through') + 1]]
                if Path(command[1]).name == 'continue_saved_joint_pd_bundle.py' and command[2] == 'compare':
                    return [command[0], str(SCRIPT), 'compare', *command[3:]]
            return command

    return ReadOnlyReaders(delegate)


def health(output):
    view = diagnostic_view(output)
    bundle.processes = processes
    try:
        ORIGINAL_HEALTH(view)
    finally:
        bundle.processes = ORIGINAL_PROCESSES
    write(output / 'original11_health50_complete.json', dict(qualified=True,
        original11_report=str(view / 'original11_health50_complete.json'),
        diagnostic_metadata=str(view / 'verified_diagnostic_metadata.json'),
        original_numeric_readers_retained=True, actual_updates=50,
        original_training_plan_and_all50_updates_unchanged=True, policy_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('health', 'train', 'events', 'compare'))
    parser.add_argument('directory', type=Path, nargs='?')
    parser.add_argument('--through', type=int)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    if args.mode in ('events', 'compare'):
        globals()[args.mode](args.directory.resolve(), args.through)
    else:
        request = original.read(args.request)
        original.runtime.configure(request['recipe'])
        assert all(sha(path) == digest for path, digest in request['source_pins'].items())
        if args.mode == 'health':
            health(Path(request['output']))
        else:
            original.health50 = health
            original.run(args.request)
