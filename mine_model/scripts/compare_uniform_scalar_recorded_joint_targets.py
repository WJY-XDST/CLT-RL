"""Original mapping/FK reader with count and legacy-schema metadata declared."""
import argparse
import ast
import copy
from pathlib import Path

from audit_saved_action_limits import transform
import compare_plain27_recorded_joint_targets as original
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit


SCRIPT = Path(__file__).resolve()


def expected_difference(configs):
    before, after = configs
    assert before["commands"] == after["commands"]
    assert all(cfg["scene"]["num_envs"] == cfg["terrain"]["num_envs"] in (4096, 8192) for cfg in configs)
    assert checkpoint_wheel_action_limit(before) == checkpoint_wheel_action_limit(after) == 1.
    result = {"/log_dir"}
    if "wheel_action_limit" not in before:
        result.add("/wheel_action_limit")
    result.update("/" + key + "/num_envs" for key in ("scene", "terrain") if before[key]["num_envs"] != after[key]["num_envs"])
    return result


def differences(before, after):
    normalized = copy.deepcopy(before)
    missing = "wheel_action_limit" not in normalized
    assert checkpoint_wheel_action_limit(before) == checkpoint_wheel_action_limit(after) == 1.
    normalized.setdefault("wheel_action_limit", 1.)
    result = original.differences(normalized, after)
    return result + (["/wheel_action_limit"] if missing else [])


def compare(output, through):
    tree = transform(original.compare, [("{'/log_dir'}", "expected_difference(cfgs)"),
        ("np.clip(r[..., (2, 5)], -1., 1.)", "np.clip(r[..., (2, 5)], -saved_limit(cfg), saved_limit(cfg))")])
    scope = dict(original.__dict__, SCRIPT=SCRIPT, expected_difference=expected_difference,
        differences=differences, saved_limit=checkpoint_wheel_action_limit)
    exec(compile(ast.fix_missing_locations(tree), str(original.SCRIPT), "exec"), scope)
    return scope["compare"](Path(output), through)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--through", type=int, required=True)
    args = cli.parse_args()
    print(compare(args.output, args.through)[0])
