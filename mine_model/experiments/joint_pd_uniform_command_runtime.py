"""Ordinary PPO with one declared global change to task share or motion ceiling.

The existing class sampler, rewards, direct joint targets and PPO are reused.
An experiment changes either the standing fraction, or the common motion scale
and its algebraically coupled minimum yaw magnitude. No operating case selects
a different controller, reward, sample list or policy.
"""
import argparse
import copy
import math
from pathlib import Path
import sys

import joint_pd_uniform_control_runtime_v3 as implementation


SCRIPT = Path(__file__).resolve()
BASE_SCRIPT = Path(implementation.__file__).resolve()
old_recipe = implementation.recipe
old_baseline = implementation.baseline
old_expected = implementation.expected_configuration
old_profile = implementation.profile_for_count
old_command = implementation.command
old_private = implementation.private_for_count


def validate_commands(value, cfg=None):
    overrides, changes = value['command_overrides'], value['command_changes']
    keys = set(overrides)
    assert keys == set(changes)
    assert keys in ({'standing_env_fraction'}, {'training_motion_scale', 'yaw_min_abs_rate'})
    assert not value['case_specific_training_logic']
    assert all(type(x) in (int, float) and math.isfinite(x) for x in overrides.values())
    for key, pair in changes.items():
        assert len(pair) == 2 and pair[0] != pair[1] and pair[1] == overrides[key]
        if cfg is not None:
            assert cfg['commands'][key] == pair[0]
    if keys == {'standing_env_fraction'}:
        assert all(0. < x < 1. for x in changes['standing_env_fraction'])
        if cfg is not None:
            assert cfg['commands']['grouped_training'] and cfg['commands']['fixed_standing_groups']
    else:
        first, second = changes['training_motion_scale']
        low, high = changes['yaw_min_abs_rate']
        assert 0. < first < second <= 1. and high >= 0.
        assert math.isclose(first * low, second * high, abs_tol=1.e-12, rel_tol=0.)
        if cfg is not None:
            assert high <= min(abs(v) for v in cfg['commands']['ranges_ang_vel_yaw'])
    return overrides


def recipe():
    value = old_recipe()
    validate_commands(value)
    assert not value['reward_changes'] and not value['control_changes']
    assert not value['effective_change_relative_to_source']
    return value


def baseline():
    source, saved, cfg = old_baseline()
    validate_commands(recipe(), cfg)
    return source, saved, cfg


def expected_configuration(cfg, count):
    result = old_expected(cfg, count)
    result['commands'].update(validate_commands(recipe(), cfg))
    return result


def standing_count(count):
    commands = expected_configuration(baseline()[2], count)['commands']
    assert commands['grouped_training'] and commands['fixed_standing_groups']
    return int(count * commands['standing_env_fraction'])


def profile_for_count(count):
    result = old_profile(count)
    result.update(global_command_overrides=copy.deepcopy(recipe()['command_overrides']),
        global_command_changes=copy.deepcopy(recipe()['command_changes']),
        fixed_standing_environment_count=standing_count(count),
        immutable_v3_runtime_sha256=implementation.sha(BASE_SCRIPT),
        original_continuous_sampler_and_roles_unchanged=True,
        no_case_specific_sampling_or_control=True)
    return result


def command(source, cfg, label, updates, commands, stop_file):
    argv, final, saved = old_command(source, cfg, label, updates, commands, stop_file)
    for key, value in validate_commands(recipe(), cfg).items():
        prefix = 'env.commands.' + key + '='
        indices = [i for i, item in enumerate(argv) if item.startswith(prefix)]
        assert len(indices) == 1, (key, indices)
        argv[indices[0]] = prefix + str(value)
    return argv, final, saved


def private_for_count(count, selected_profile=None, *, is_replay=False):
    module = old_private(count, selected_profile, is_replay=is_replay)
    previous = module.observe_environment

    def observe(env, output):
        result = previous(env, output)
        if not is_replay:
            raw = env.unwrapped
            expected = standing_count(count)
            assert raw._dedicated_standing_count == expected
            from wheel_legged_gym_isaaclab.yaw_training import TASK_NAMES, grouped_task_ids
            import torch
            ids = torch.arange(count)
            groups = grouped_task_ids(ids, dedicated_standing_count=expected)
            counts = torch.bincount(groups, minlength=len(TASK_NAMES)).tolist()
            assert sum(counts) == count and counts[0] == expected
            implementation.write_json(Path(output) / 'global_command_population_contract.json', dict(
                verified=True, actual_num_envs=raw.num_envs,
                actual_fraction=raw.cfg.commands.standing_env_fraction,
                actual_dedicated_standing_count=expected,
                motion_environment_count=count - expected,
                startup_role_counts=dict(zip(TASK_NAMES, counts)),
                command_sampler_instance_replacements=0,
                command_value_dependent_controller_or_reward=False,
                observer_changes_native_state=False, policy_accepted=False))
        return result

    module.observe_environment = observe
    return module


implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.baseline = baseline
implementation.expected_configuration = expected_configuration
implementation.profile_for_count = profile_for_count
implementation.command = command
implementation.private_for_count = private_for_count
implementation.validate_global_command_metadata = validate_commands
implementation.standing_count = standing_count


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
    else:
        raise ValueError(mode)
else:
    sys.modules[__name__] = implementation
