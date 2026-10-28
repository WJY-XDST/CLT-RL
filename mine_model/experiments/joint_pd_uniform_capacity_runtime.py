"""Five isolated real PPO updates using the exact uniform production recipe."""
import argparse
import copy
import json
from pathlib import Path
import sys

import joint_pd_uniform_exploration_runtime as production
from auto_train_mine import write_json
from joint_pd_continuation import sha
from train_joint_pd_standing_seed import saved_updates


SCRIPT = Path(__file__).resolve()
FLAG = "--capacity_population_change"
MANIFEST = None
NUM_ENVS = None


def request_ready():
    request = json.loads(MANIFEST.read_text())
    production.configure(request["uniform_recipe"])
    saved, cfg = production.require_source(Path(request["source_checkpoint"]))
    assert request["source_checkpoint_sha256"] == sha(request["source_checkpoint"])
    assert request["source_actual_updates"] == saved_updates(request["source_checkpoint"])
    assert request["source_all17_Adam_step"] == request["source_actual_updates"] * 20
    assert request["source_num_envs"] == cfg["scene"]["num_envs"]
    assert request["counts"] == [8192, 4096] and request["probe_updates"] == 5
    assert NUM_ENVS in request["counts"] and not request["capacity_weights_may_be_promoted"]
    assert all(sha(path) == digest for path, digest in request["source_pins"].items())
    return request


def profile():
    result = copy.deepcopy(production.profile_for_count(NUM_ENVS))
    result.update(capacity_only_probe=True, capacity_num_envs=NUM_ENVS, capacity_probe_updates=5,
        capacity_runtime_sha256=sha(SCRIPT), capacity_request_sha256=sha(MANIFEST),
        capacity_weights_may_not_be_promoted=True)
    return result


def native(output, arguments):
    request = request_ready()
    assert arguments[arguments.index("--checkpoint_path") + 1] == request["source_checkpoint"]
    assert int(arguments[arguments.index("--target_iteration") + 1]) == request["source_actual_updates"] + 5
    arguments = list(arguments)
    if FLAG in arguments:
        arguments.remove(FLAG)
    write_json(Path(output).parent / "capacity_runtime_contract.json", dict(
        capacity_only=True, profile=profile(), source_actual_updates=request["source_actual_updates"],
        source_all17_Adam_step=request["source_all17_Adam_step"], num_envs=NUM_ENVS,
        probe_updates=5, rollout_steps=48, Adam_minibatches_per_update=20,
        original_uniform_reward_and_continuous_sampler=True, capacity_weights_may_be_promoted=False))
    return production.native(output, arguments, selected_count=NUM_ENVS, selected_profile=profile, updates=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-request", type=Path, required=True)
    parser.add_argument("--observation-output", type=Path, required=True)
    parser.add_argument("--capacity-num-envs", type=int, required=True)
    boundary = sys.argv.index("--")
    args = parser.parse_args(sys.argv[1:boundary])
    MANIFEST, NUM_ENVS = args.capacity_request.resolve(), args.capacity_num_envs
    native(args.observation_output.resolve(), sys.argv[boundary + 1:])
