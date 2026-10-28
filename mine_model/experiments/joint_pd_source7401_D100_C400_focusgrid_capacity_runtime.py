"""Five isolated updates of the weighted grid, retaining source std floors."""
import argparse
import json
from pathlib import Path
import sys

import joint_pd_AF7001_wheelD075_capacity_runtime as original
import joint_pd_source7401_D100_C400_focusgrid_runtime as production
import plain27_sampler_metadata_adapter as adapter
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
api = adapter.private_module(original.SCRIPT, "_source7401_leg0075_original_capacity")
CLOCK_METADATA = adapter.retarget(api, ("native",), integers={"*": {7006: 7406}})
api.SCRIPT, api.production = SCRIPT, production


def request_ready():
    production.repair_ready()
    value = json.loads(api.MANIFEST.read_text())
    source = Path(value["source_checkpoint"])
    _, cfg = api.require_source(source)
    assert value["counts"] == [8192, 4096] and api.NUM_ENVS in value["counts"]
    assert value["probe_updates"] == 5 and value["source_actual_updates"] == 7401
    assert value["source_all17_Adam_step"] == 148020
    assert value["source_checkpoint_sha256"] == sha(source) == production.ANCHOR_SHA
    assert cfg["scene"]["num_envs"] == value["source_num_envs"] == 8192
    assert value["proposed_reward_overrides"] == {"rewards.yaw_rate_error_sq": -400.}
    assert value["proposed_controller_overrides"] == {
        "wheel_damping": 1., "robot.actuators.wheels.damping": 1.}
    assert value["only_controller_change"] == value["only_reward_change"] == {}
    assert value["only_exploration_change"] == production.DELTA["exploration"]
    assert value["only_command_change"] == production.DELTA["commands"]
    assert value["proposed_exploration_floors"] == [.01, .01, .005, .01, .01, .005]
    assert all(sha(path) == digest for path, digest in value["source_pins"].items())
    return value


def install_leg_exploration(output):
    return production.install_leg_exploration(output, profile_api=api.profile)


api.request_ready, api.install_leg_exploration = request_ready, install_leg_exploration


def __getattr__(name):
    return getattr(api, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-request", type=Path, required=True)
    parser.add_argument("--observation-output", type=Path, required=True)
    parser.add_argument("--capacity-num-envs", type=int, required=True)
    boundary = sys.argv.index("--")
    args = parser.parse_args(sys.argv[1:boundary])
    api.MANIFEST, api.NUM_ENVS = args.capacity_request.resolve(), args.capacity_num_envs
    api.native(args.observation_output.resolve(), sys.argv[boundary + 1:])
