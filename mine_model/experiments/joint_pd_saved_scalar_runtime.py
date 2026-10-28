"""Declared global scalar experiments on a saved ordinary-PPO controller.

Change only explicitly listed numeric reward/command fields. Every environment
uses the same values; no observation, operating condition or replay case picks
an alternate rule. Initial NN, learned std, Adam and adaptive LR stay intact.
"""
import argparse
import copy
import math
from pathlib import Path
import sys

import joint_pd_saved_bundle_review50_runtime as implementation

SCRIPT = Path(__file__).resolve()
saved_namespace = implementation.recipe.__globals__
base_recipe = saved_namespace['old_recipe']
saved_profile = implementation.profile_for_count
saved_expected = implementation.expected_configuration
saved_command = implementation.command


def recipe():
    value = base_recipe()
    assert value['saved_bundle_ordinary_sampler_transfer']
    assert value['removed_training_extension'] == 'targeted_turn_command_fraction'
    assert not value['control_changes'] and not value['case_specific_training_logic']
    assert value['source_exploration'] == value['exploration']
    changes = {}
    for group in ('rewards', 'commands'):
        prefix = 'reward' if group == 'rewards' else 'command'
        overrides = value.get(prefix + '_overrides', {})
        declared = value.get(prefix + '_changes', {})
        assert set(overrides) == set(declared)
        for name, after in overrides.items():
            before, saved_after = declared[name]
            assert all(type(v) in (float, int) and math.isfinite(v)
                       for v in (before, after, saved_after))
            assert saved_after == after and before != after
        if declared:
            changes[group] = declared
    assert changes and changes == value['effective_change_relative_to_source']
    return value


def expected_configuration(cfg, count):
    result = saved_expected(cfg, count)
    for group in ('rewards', 'commands'):
        prefix = 'reward' if group == 'rewards' else 'command'
        for key, (before, after) in recipe().get(prefix + '_changes', {}).items():
            assert cfg[group][key] == before
            result[group][key] = after
    return result


def profile_for_count(count):
    result = saved_profile(count)
    result.update(profile='plain27_saved_uniform_scalar_v1',
        uniform_scalar_runtime_sha256=implementation.sha(SCRIPT),
        uniform_reward_overrides=recipe()['reward_overrides'],
        uniform_command_overrides=recipe().get('command_overrides', {}),
        declared_scalar_changes=recipe()['effective_change_relative_to_source'],
        same_scalar_values_in_every_environment=True,
        saved_initial_std_reset=False)
    return result


def command(source, cfg, label, updates, commands, stop_file):
    argv, final, saved = saved_command(source, cfg, label, updates, commands, stop_file)
    for key, value in recipe().get('command_overrides', {}).items():
        prefix = 'env.commands.' + key + '='
        indices = [i for i, item in enumerate(argv) if item.startswith(prefix)]
        assert len(indices) == 1
        argv[indices[0]] = prefix + str(value)
    return argv, final, saved


saved_namespace['recipe'] = recipe
implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.profile_for_count = profile_for_count
implementation.expected_configuration = expected_configuration
implementation.command = command
sys.modules['joint_pd_saved_scalar_runtime'] = implementation


def compare(output, through):
    import review_saved_joint_pd_segment as segment
    old_normalize = implementation.normalize_saved_bundle

    def declared_expected(cfg):
        result = old_normalize(cfg)
        for group in ('rewards', 'commands'):
            prefix = 'reward' if group == 'rewards' else 'command'
            for key, (before, after) in recipe().get(prefix + '_changes', {}).items():
                assert result[group][key] == before
                result[group][key] = after
        return result

    implementation.normalize_saved_bundle = declared_expected
    try:
        segment.compare(output, through)
    finally:
        implementation.normalize_saved_bundle = old_normalize
    segment.write(output / f'declared_scalar_reader_contract_prefix_{through:04d}.json', dict(
        verified=True, only_admitted_changes=recipe()['effective_change_relative_to_source'],
        original_mapping_FK_recorded_action_math_and_limits_unchanged=True,
        original_numeric_reader_retained=True, training_modified=False, policy_accepted=False))


def train(request):
    import review_saved_joint_pd_segment as segment
    api = segment.original
    old_write, old_health, old_processes = api.write_json, api.health50, segment.processes

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            changes = recipe()['effective_change_relative_to_source']
            for group in ('rewards', 'commands'):
                value['next_configuration_changes'][group] = changes.get(group, {})
            value['only_reward_change'] = changes.get('rewards', {})
            value['only_command_change'] = changes.get('commands', {})
            value['only_runtime_sampling_change'] = {}
            value.update(declared_uniform_scalar_changes=changes,
                no_condition_specific_training_control_or_sampling=True)
        return old_write(path, value)

    def processes(delegate):
        base = old_processes(delegate)

        class ScalarReaders(type(base)):
            def routed(self, argv):
                argv = super().routed(argv)
                if (isinstance(argv, (list, tuple)) and len(argv) > 2
                        and Path(argv[1]).name == segment.SCRIPT.name and argv[2] == 'compare'):
                    return [argv[0], str(SCRIPT), 'compare', '--recipe',
                            str(implementation.RECIPE_PATH), *argv[3:]]
                return argv
        return ScalarReaders(delegate)

    api.write_json, api.health50, segment.processes = write, segment.health, processes
    try:
        return api.run(request)
    finally:
        api.write_json, api.health50, segment.processes = old_write, old_health, old_processes


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        boundary = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:boundary])
        implementation.configure(args.recipe)
        implementation.native(args.observation_output, sys.argv[boundary + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:])
        implementation.configure(args.recipe)
        implementation.replay(remaining)
    elif mode == 'qualify':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        from continue_saved_joint_pd_bundle import qualify
        qualify(args.recipe.resolve(), args.output.resolve())
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        train(args.request.resolve())
    elif mode == 'compare':
        parser.add_argument('directory', type=Path)
        parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        compare(args.directory.resolve(), args.through)
    else:
        raise ValueError(mode)
else:
    sys.modules[__name__] = implementation
