"""Original action/FK audit with exactly two declared global command differences."""
import argparse
import copy
import json
from pathlib import Path

import compare_uniform_control_recorded_joint_targets_v3 as original


SCRIPT = Path(__file__).resolve()


def compare(output, through):
    plan = json.loads((Path(output)/"experiment_plan.json").read_text())
    overrides = plan["profile"]["global_command_overrides"]
    assert set(overrides) == {"training_motion_scale", "yaw_min_abs_rate"}
    previous, previous_script = original.expected_difference, original.SCRIPT

    def expected(configs):
        before, after = configs
        normalized = copy.deepcopy(before)
        normalized["commands"].update(overrides)
        assert normalized["commands"] == after["commands"]
        paths = previous((normalized, after))
        paths.update("/commands/"+key for key, value in overrides.items()
                     if before["commands"][key] != value)
        return paths

    original.expected_difference, original.SCRIPT = expected, SCRIPT
    try:
        return original.compare(output, through)
    finally:
        original.expected_difference, original.SCRIPT = previous, previous_script


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--through", type=int, required=True)
    args = parser.parse_args()
    print(compare(args.output, args.through)[0])
