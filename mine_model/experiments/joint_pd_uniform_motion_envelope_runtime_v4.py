"""Extend the immutable v3 runtime with one global motion-envelope scalar.

Ordinary PPO, rewards, action mapping, controller and the class sampler are
reused unchanged. The second configuration value preserves the prior absolute
small-turn support as the common linear/yaw ceiling increases.
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


def validate_commands(value, cfg=None):
    overrides, changes = value["command_overrides"], value["command_changes"]
    allowed = {"training_motion_scale", "yaw_min_abs_rate"}
    assert set(overrides) == set(changes) == allowed
    assert not value["case_specific_training_logic"]
    assert all(type(x) in (int, float) and math.isfinite(x) for x in overrides.values())
    before_scale, after_scale = changes["training_motion_scale"]
    before_minimum, after_minimum = changes["yaw_min_abs_rate"]
    assert 0. < before_scale < after_scale <= 1.
    assert after_scale == overrides["training_motion_scale"]
    assert after_minimum == overrides["yaw_min_abs_rate"] and after_minimum >= 0.
    assert math.isclose(before_minimum*before_scale, after_minimum*after_scale, abs_tol=1.e-12, rel_tol=0.)
    if cfg is not None:
        commands = cfg["commands"]
        assert before_scale == commands["training_motion_scale"]
        assert before_minimum == commands["yaw_min_abs_rate"]
        assert after_minimum <= min(abs(x) for x in commands["ranges_ang_vel_yaw"])
    return overrides


def recipe():
    value = old_recipe()
    validate_commands(value)
    assert not value["reward_changes"] and not value["control_changes"]
    return value


def baseline():
    source, saved, cfg = old_baseline()
    validate_commands(recipe(), cfg)
    return source, saved, cfg


def expected_configuration(cfg, count):
    result = old_expected(cfg, count)
    result["commands"].update(validate_commands(recipe(), cfg))
    return result


def profile_for_count(count):
    result = old_profile(count)
    result.update(global_command_overrides=copy.deepcopy(recipe()["command_overrides"]),
                  global_command_changes=copy.deepcopy(recipe()["command_changes"]),
                  immutable_v3_runtime_sha256=implementation.sha(BASE_SCRIPT),
                  original_continuous_sampler_and_roles_unchanged=True,
                  prior_absolute_small_turn_support_preserved=True)
    return result


def command(source, cfg, label, updates, commands, stop_file):
    argv, final, saved = old_command(source, cfg, label, updates, commands, stop_file)
    for key, value in validate_commands(recipe(), cfg).items():
        prefix = "env.commands." + key + "="
        indices = [i for i, item in enumerate(argv) if item.startswith(prefix)]
        assert len(indices) == 1, (key, indices)
        argv[indices[0]] = prefix + str(value)
    return argv, final, saved


implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.baseline = baseline
implementation.expected_configuration = expected_configuration
implementation.profile_for_count = profile_for_count
implementation.command = command
implementation.validate_global_command_metadata = validate_commands
implementation.global_motion_envelope_extension = True


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    if mode == "native":
        parser.add_argument("--observation-output", type=Path, required=True)
        boundary = sys.argv.index("--")
        args = parser.parse_args(sys.argv[1:boundary])
        implementation.configure(args.recipe)
        implementation.native(args.observation_output, sys.argv[boundary+1:])
    elif mode == "replay":
        args, remaining = parser.parse_known_args(sys.argv[1:])
        implementation.configure(args.recipe)
        implementation.replay(remaining)
    else:
        raise ValueError(mode)
else:
    # Every reused module sees the same configuration, count and profile
    # globals; assignments such as NUM_ENVS=4096 cannot be lost in a facade.
    sys.modules[__name__] = implementation
