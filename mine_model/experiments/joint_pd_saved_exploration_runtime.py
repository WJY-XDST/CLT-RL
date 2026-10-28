"""Ordinary saved-policy PPO with a declared, common exploration-floor change.

Only the future optimizer projection floor changes. Initial network, all17
Adam states, six learned std values and adaptive LR must load bitwise intact.
No command, height, speed, sign or case selects a different rule.
"""
import argparse
import copy
from pathlib import Path
import sys

import joint_pd_saved_bundle_review50_runtime as implementation


SCRIPT = Path(__file__).resolve()
saved_recipe_namespace = implementation.recipe.__globals__
base_recipe = saved_recipe_namespace['old_recipe']
saved_profile = implementation.profile_for_count


def recipe():
    value = base_recipe()
    assert value['saved_bundle_ordinary_sampler_transfer']
    assert value['removed_training_extension'] == 'targeted_turn_command_fraction'
    assert not value['reward_changes'] and not value['control_changes']
    assert not value['case_specific_training_logic']
    keys = {'global_minimum_std', 'minimum_leg_std', 'minimum_wheel_std'}
    before, after = value['source_exploration'], value['exploration']
    assert set(before) == set(after) == keys
    assert all(type(v) in (int, float) and 0. < v < 1. for v in before.values())
    changes = {key: [before[key], after[key]] for key in keys if before[key] != after[key]}
    assert changes and value['exploration_changes'] == changes
    assert value['effective_change_relative_to_source'] == {'exploration': changes}
    assert value['initial_std_reset'] is False
    return value


def profile_for_count(count):
    result = saved_profile(count)
    changes = recipe()['exploration_changes']
    # These identify changed floor entries, not a reset or new network heads.
    heads = []
    if 'minimum_leg_std' in changes:
        heads.extend((0, 1, 3, 4))
    if 'minimum_wheel_std' in changes:
        heads.extend((2, 5))
    result.update(profile='plain27_saved_uniform_exploration_projection_v1',
        uniform_exploration_runtime_sha256=implementation.sha(SCRIPT),
        uniform_exploration_changes=changes, changed_noise_heads=sorted(heads),
        saved_initial_std_reset=False, all_six_std_entries_trainable=True)
    return result


# Saved-bundle functions have their own module globals. Update the recipe
# lookup there as well as the retained implementation; no source file changes.
saved_recipe_namespace['recipe'] = recipe
implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.profile_for_count = profile_for_count
sys.modules['joint_pd_saved_exploration_runtime'] = implementation


def train(request):
    import review_saved_joint_pd_segment as segment
    api = segment.original
    old_write, old_health = api.write_json, api.health50

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            changes = recipe()['exploration_changes']
            value['next_configuration_changes']['exploration'] = changes
            value['only_exploration_change'] = changes
            value['only_runtime_sampling_change'] = {}
            value.update(uniform_exploration_projection_change_only=True,
                saved_initial_std_reset=False, source_exploration=recipe()['source_exploration'],
                actual_future_projection_floors=recipe()['exploration'],
                no_condition_specific_training_control_or_sampling=True)
        return old_write(path, value)

    api.write_json, api.health50 = write, segment.health
    try:
        return api.run(request)
    finally:
        api.write_json, api.health50 = old_write, old_health


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
    else:
        raise ValueError(mode)
else:
    sys.modules[__name__] = implementation
