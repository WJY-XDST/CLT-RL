"""Capacity-only five-update PPO from a reviewed4096 or8192 production source.

Use the already qualified ordinary rollout, count adapter and original
optimizer. Neither a capacity checkpoint nor an unreviewed prefix can be
selected through this entry. This module never admits production training.
"""
import copy
import json
from pathlib import Path
import sys

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
import joint_pd_AF7001_leg005_parallel8192_runtime as production
import joint_pd_yawcost200_capacity_runtime as factory

SCRIPT = Path(__file__).resolve()
FLAG = '--capacity_population_change'
NUM_ENVS = None
MANIFEST = None


def require_source(path, transition=False, *, initial_transfer=False):
    # A count boundary is not a reward/controller transfer.
    if initial_transfer:
        raise ValueError('Capacity tests never transfer the original model200')
    return production.require_source(path)


def request_ready():
    request = json.loads(MANIFEST.read_text())
    source = Path(request['source_checkpoint']).resolve()
    _, cfg = require_source(source)
    final = json.loads(Path(request['source_final_verification']).read_text())
    if (request['probe_updates'] != 5 or request['counts'] != [8192,4096]
            or NUM_ENVS not in request['counts'] or not request['block_id']
            or request['source_checkpoint_sha256'] != sha(source)
            or request['source_actual_updates'] != saved_updates(source)
            or request['source_all17_Adam_step'] != saved_updates(source)*20
            or request['source_num_envs'] != cfg['scene']['num_envs']
            or not final['verified'] or final['checkpoint_sha256'] != sha(source)
            or final['actual_ppo_updates'] != saved_updates(source)
            or final['all17_Adam_step'] != saved_updates(source)*20
            or not request['source_is_completely_reviewed_production_checkpoint']
            or not request['required_complete_reports']
            or any(sha(p) != h for p,h in request['source_pins'].items())):
        raise ValueError('Require a fresh exact-source capacity request after full production review')
    for item in request['required_complete_reports']:
        report=json.loads(Path(item['path']).read_text())
        if report.get(item['completion_key']) is not True:
            raise ValueError('Finish every selected-source review before any next-block capacity PPO')
    return request


def profile():
    value=copy.deepcopy(production.profile())
    value.update(profile='plain_AF7001_yawcost200_leg005_capacity_v1',
        capacity_only_probe=True,capacity_num_envs=NUM_ENVS,capacity_probe_updates=5,
        native_training_environment_count=NUM_ENVS,fixed_standing_environment_count=NUM_ENVS//4,
        samples_per_original_PPO_update=NUM_ENVS*48,capacity_runtime_sha256=sha(SCRIPT),
        capacity_request_sha256=sha(MANIFEST),capacity_weights_may_not_be_promoted=True)
    return value


def private_runtime(source=None):
    if source is not None:
        require_source(source)
    sampler,evidence=production.bind(factory.sampler_factory,NUM_ENVS=NUM_ENVS,profile=profile)()
    sampler.profile,sampler.require_source=profile,require_source
    module=sampler.private_runtime()
    module.profile,module.require_source=profile,require_source
    module.capacity_count_assertion_evidence=evidence
    return module


def install_leg_exploration(output):
    return production.bind(production.install_leg_exploration,profile=profile)(output)


def native(output, arguments):
    import joint_pd_review_interval200 as interval
    import joint_pd_source2201_yaw025_runtime as template

    request=request_ready()
    source=Path(arguments[arguments.index('--checkpoint_path')+1]).resolve()
    if (str(source) != request['source_checkpoint']
            or int(arguments[arguments.index('--num_envs')+1]) != NUM_ENVS
            or int(arguments[arguments.index('--target_iteration')+1]) != saved_updates(source)+5
            or ((FLAG in arguments) != (request['source_num_envs'] != NUM_ENVS))
            or any(flag in arguments for flag in ('--reset_optimizer','--reset_exploration_std','--save_initial_checkpoint'))):
        raise ValueError('Keep exactly five capacity updates and explicitly declare only the count boundary')
    interval.UPDATES=5
    production.family._sampler_api.install_sampling_field_restore()
    write_json(output.parent/'capacity_runtime_contract.json',dict(capacity_only=True,
        num_envs=NUM_ENVS,standing_envs=NUM_ENVS//4,source_num_envs=request['source_num_envs'],
        source_checkpoint_sha256=sha(source),probe_updates=5,rollout_steps=48,
        original_adam_minibatches_per_update=20,profile=profile(),
        source_model_all17_Adam_std_and_actual_LR_not_reset=True,
        rewards_controller_physics_and_acceptance_unchanged=True,
        capacity_weights_may_be_promoted=False,policy_accepted=False))
    return production.bind(template.native,SCRIPT=SCRIPT,FLAG=FLAG,
        repair_ready=request_ready,require_source=require_source,private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)(output,arguments)


if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--capacity-request',type=Path,required=True)
    cli.add_argument('--observation-output',type=Path,required=True)
    cli.add_argument('--capacity-num-envs',type=int,required=True)
    boundary=sys.argv.index('--');args=cli.parse_args(sys.argv[1:boundary])
    MANIFEST=args.capacity_request.resolve();NUM_ENVS=args.capacity_num_envs
    native(args.observation_output.resolve(),sys.argv[boundary+1:])
