"""Isolated capacity PPO using the unchanged measured direct-joint recipe.

The production files are never edited. Only two observer count assertions
are adapted to the requested number and its 25% standing population. The
ordinary rollout, reward, controller, loss and Adam implementation remain
the existing implementations. Saved weights carry a capacity-only profile.
"""
import copy
import json
from pathlib import Path
import sys

import plain27_sampler_metadata_adapter as adapter
import joint_pd_preserved6801_yawcost200_runtime as family
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
NUM_ENVS = None
UPDATES = None
MANIFEST = None


def profile():
    value = copy.deepcopy(family.profile())
    value.update(capacity_only_probe=True, capacity_num_envs=NUM_ENVS,
        capacity_probe_updates=UPDATES, capacity_runtime_sha256=sha(SCRIPT),
        capacity_request_sha256=sha(MANIFEST),
        capacity_weights_may_not_be_promoted=True)
    return value


def sampler_factory():
    if NUM_ENVS not in (4096, 8192):
        raise ValueError('Only the declared current and preferred counts are admitted')
    # Start from the uncompiled source so the restoration proof is meaningful.
    module = adapter.private_module(family.previous.SCRIPT,
        '_capacity_only_retained_targeted_sampler')
    for name in ('SCRIPT', 'ANCHOR_SHA', 'FLAG', 'DELTA', 'DECISION', 'TRAINING',
                 'recipe', 'require_source', 'repair_ready'):
        setattr(module, name, getattr(family, name))
    module.profile = profile
    module._provider = family._plant
    evidence = adapter.retarget(module, ('private_runtime',),
        integers={'*': {6601: 6801, 1024: NUM_ENVS // 4, 4096: NUM_ENVS}},
        strings={'targeted_turn_runtime_evidence.json':
                 'capacity_retained_targeted030_runtime_evidence.json'},
        keyword_names={
            'reward_exploration_physics_reset_and_control_unchanged':
                'exploration_physics_reset_and_control_unchanged',
            'sampler_RNG_draws_intentionally_added_only_for_mature_nonfixed_training':
                'retained_sampler_RNG_draws_only_for_mature_nonfixed_training'})
    module.FACTORY_METADATA = family.PLANT_METADATA + evidence
    module.install_sampling_field_restore()
    return module, evidence


def private_runtime(source=None):
    module, evidence = sampler_factory()
    result = module.private_runtime(source)
    # The original retained runner resolves these names from this private
    # module's globals. Its load still validates the exact production source.
    result.profile = profile
    result.require_source = family.require_source
    result.capacity_count_assertion_evidence = evidence
    return result


def native(output, arguments):
    from auto_train_mine import write_json
    from train_joint_pd_standing_seed import saved_updates
    import joint_pd_review_interval200 as interval
    import joint_pd_source2201_yaw025_runtime as template

    request = json.loads(MANIFEST.read_text())
    source = Path(arguments[arguments.index('--checkpoint_path') + 1]).resolve()
    actual = saved_updates(source)
    if (sha(source) != request['source_checkpoint_sha256']
            or int(arguments[arguments.index('--num_envs') + 1]) != NUM_ENVS
            or int(arguments[arguments.index('--target_iteration') + 1]) != actual + UPDATES
            or NUM_ENVS not in (4096, 8192) or UPDATES != 5
            or any(x in arguments for x in ('--reset_optimizer', '--reset_exploration_std',
                                           '--save_initial_checkpoint'))):
        raise ValueError('Capacity run must retain the source and exactly five complete updates')
    family.require_source(source)
    family._sampler_api.install_sampling_field_restore()
    # This changes only the original observation interval counters inside
    # this separate process, never the production interval or PPO math.
    interval.UPDATES = UPDATES
    _, evidence = sampler_factory()
    write_json(output.parent / 'capacity_runtime_contract.json', dict(
        capacity_only=True, num_envs=NUM_ENVS, standing_envs=NUM_ENVS // 4,
        rollout_steps=48, original_adam_minibatches_per_update=20,
        probe_updates=UPDATES, source_checkpoint_sha256=sha(source),
        original_sampler_factory_source=str(family.previous.SCRIPT),
        observer_count_assertion_adapter=evidence,
        profile=profile(), rewards_controller_physics_or_acceptance_changed=False,
        production_files_changed=False, capacity_weights_may_be_promoted=False))
    family.bind(template.native, SCRIPT=SCRIPT, FLAG=family.FLAG,
        repair_ready=family.repair_ready, require_source=family.require_source,
        private_runtime=private_runtime, install_leg_exploration=family.install_leg_exploration)(
            output, arguments)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-request', type=Path, required=True)
    parser.add_argument('--observation-output', type=Path, required=True)
    parser.add_argument('--capacity-num-envs', type=int, required=True)
    boundary = sys.argv.index('--')
    options = parser.parse_args(sys.argv[1:boundary])
    MANIFEST = options.capacity_request.resolve()
    NUM_ENVS = options.capacity_num_envs
    UPDATES = json.loads(MANIFEST.read_text())['probe_updates']
    native(options.observation_output.resolve(), sys.argv[boundary + 1:])
