"""Five isolated ordinary PPO updates for the declared wheel-drive recipe."""
import copy
import json
from pathlib import Path
import sys

from auto_train_mine import write_json
from joint_pd_continuation import sha
import joint_pd_AF7001_wheelD075_runtime as production


SCRIPT = Path(__file__).resolve()
FLAG = '--capacity_population_change'
NUM_ENVS = None
MANIFEST = None


def require_source(path, transition=False, *, initial_transfer=False):
    assert not initial_transfer and sha(path) == production.ANCHOR_SHA
    return production.require_source(path)


def request_ready():
    production.repair_ready()
    request = json.loads(MANIFEST.read_text())
    source = Path(request['source_checkpoint'])
    _, cfg = require_source(source)
    assert request['counts'] == [8192, 4096] and NUM_ENVS in request['counts']
    assert request['probe_updates'] == 5 and request['source_actual_updates'] == 7001
    assert request['source_checkpoint_sha256'] == sha(source)
    assert request['source_all17_Adam_step'] == 140020 and cfg['scene']['num_envs'] == request['source_num_envs']
    assert request['proposed_reward_overrides'] == {}
    assert request['proposed_controller_overrides'] == {
        'wheel_damping': .75, 'robot.actuators.wheels.damping': .75}
    assert all(sha(path) == digest for path, digest in request['source_pins'].items())
    return request


def profile():
    value = copy.deepcopy(production.profile_for_count(NUM_ENVS))
    value.update(capacity_only_probe=True, capacity_num_envs=NUM_ENVS, capacity_probe_updates=5,
        capacity_runtime_sha256=sha(SCRIPT), capacity_request_sha256=sha(MANIFEST),
        capacity_weights_may_not_be_promoted=True)
    return value


def private_runtime(source=None):
    if source is not None:
        require_source(source)
    module = production.private_for_count(NUM_ENVS, profile)
    module.profile, module.require_source = profile, require_source
    return module


def install_leg_exploration(output):
    return production.bind(production.family.install_leg_exploration, profile=profile)(output)


def native(output, arguments):
    import joint_pd_review_interval200 as interval
    import joint_pd_source2201_yaw025_runtime as template
    request = request_ready()
    source = Path(arguments[arguments.index('--checkpoint_path') + 1]).resolve()
    assert str(source) == request['source_checkpoint']
    assert int(arguments[arguments.index('--num_envs') + 1]) == NUM_ENVS
    assert int(arguments[arguments.index('--target_iteration') + 1]) == 7006
    assert (FLAG in arguments) == (NUM_ENVS != request['source_num_envs'])
    assert not any(flag in arguments for flag in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint'))
    interval.UPDATES = 5
    production.family._sampler_api.install_sampling_field_restore()
    write_json(Path(output).parent / 'capacity_runtime_contract.json', dict(capacity_only=True,
        num_envs=NUM_ENVS, probe_updates=5, rollout_steps=48, Adam_minibatches_per_update=20,
        only_proposed_controller_change=production.DELTA['controller'], profile=profile(),
        source_model_all17_Adam_std_and_actual_LR_not_reset=True,
        capacity_weights_may_be_promoted=False, policy_accepted=False))
    return production.bind(template.native, SCRIPT=SCRIPT, FLAG=FLAG,
        repair_ready=request_ready, require_source=require_source, private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)(output, arguments)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-request', type=Path, required=True)
    parser.add_argument('--observation-output', type=Path, required=True)
    parser.add_argument('--capacity-num-envs', type=int, required=True)
    boundary = sys.argv.index('--')
    args = parser.parse_args(sys.argv[1:boundary])
    MANIFEST = args.capacity_request.resolve()
    NUM_ENVS = args.capacity_num_envs
    native(args.observation_output.resolve(), sys.argv[boundary + 1:])
