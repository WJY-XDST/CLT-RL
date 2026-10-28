"""Five real isolated PPO updates of the exact source7201/D100 recipe."""
import argparse
import json
from pathlib import Path
import sys

import joint_pd_AF7001_wheelD075_capacity_runtime as original
import joint_pd_wheel005_source7201_D100_yaw400_runtime as production
import plain27_sampler_metadata_adapter as adapter
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
api = adapter.private_module(original.SCRIPT, '_source7201_D100_yaw400_original_capacity')
api.SCRIPT, api.production = SCRIPT, production
CLOCK_METADATA = adapter.retarget(api, ('native',), integers={'*': {7006: 7206}})


def request_ready():
    production.repair_ready()
    request = json.loads(api.MANIFEST.read_text())
    source = Path(request['source_checkpoint'])
    _, cfg = api.require_source(source)
    assert request['counts'] == [8192, 4096] and api.NUM_ENVS in request['counts']
    assert request['probe_updates'] == 5 and request['source_actual_updates'] == 7201
    assert request['source_all17_Adam_step'] == 144020
    assert request['source_checkpoint_sha256'] == sha(source) == production.ANCHOR_SHA
    assert cfg['scene']['num_envs'] == request['source_num_envs'] == 8192
    assert request['proposed_reward_overrides'] == {'rewards.yaw_rate_error_sq': -400.}
    assert request['proposed_controller_overrides'] == {
        'wheel_damping': 1., 'robot.actuators.wheels.damping': 1.}
    assert request['proposed_exploration_floors'] == [.01, .01, .005, .01, .01, .005]
    args = request['prototype_command']
    assert float(args[args.index('--min_exploration_std') + 1]) == .005
    assert float(args[args.index('--min_wheel_exploration_std') + 1]) == .005
    assert all(sha(path) == digest for path, digest in request['source_pins'].items())
    return request


def install_leg_exploration(output):
    return production.install_leg_exploration(output, profile_api=api.profile)


api.request_ready = request_ready
api.install_leg_exploration = install_leg_exploration


def __getattr__(name):
    return getattr(api, name)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--capacity-request', type=Path, required=True)
    cli.add_argument('--observation-output', type=Path, required=True)
    cli.add_argument('--capacity-num-envs', type=int, required=True)
    boundary = sys.argv.index('--')
    args = cli.parse_args(sys.argv[1:boundary])
    api.MANIFEST = args.capacity_request.resolve()
    api.NUM_ENVS = args.capacity_num_envs
    api.native(args.observation_output.resolve(), sys.argv[boundary + 1:])
