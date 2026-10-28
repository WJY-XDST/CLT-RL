"""Isolated five-real-update capacity for the selected AF7001 yaw width."""
import copy,json,sys
from pathlib import Path
from auto_train_mine import write_json
from joint_pd_continuation import sha
from train_joint_pd_standing_seed import saved_updates
import joint_pd_AF7001_yawsigma0025_runtime as production

SCRIPT=Path(__file__).resolve()
FLAG='--capacity_population_change'
NUM_ENVS=None
MANIFEST=None

def require_source(path,transition=False,*,initial_transfer=False):
    if initial_transfer:raise ValueError('No original200 transfer')
    if sha(path)!=production.ANCHOR_SHA:raise ValueError('Only the selected exact AF7001 source')
    return production.require_source(path)

def request_ready():
    production.repair_ready()
    r=json.loads(MANIFEST.read_text());source=Path(r['source_checkpoint'])
    _,cfg=require_source(source)
    if (r['counts']!=[8192,4096] or NUM_ENVS not in r['counts'] or r['probe_updates']!=5
            or r['source_checkpoint_sha256']!=sha(source) or r['source_actual_updates']!=7001
            or r['source_all17_Adam_step']!=140020 or cfg['scene']['num_envs']!=r['source_num_envs']
            or r['proposed_reward_overrides']!={'tracking_sigma_ang':.0025}
            or any(sha(p)!=h for p,h in r['source_pins'].items())):
        raise ValueError('Require the exact-source new-recipe capacity request')
    return r

def profile():
    r=copy.deepcopy(production.profile_for_count(NUM_ENVS))
    r.update(capacity_only_probe=True,capacity_num_envs=NUM_ENVS,capacity_probe_updates=5,
        capacity_runtime_sha256=sha(SCRIPT),capacity_request_sha256=sha(MANIFEST),
        capacity_weights_may_not_be_promoted=True)
    return r

def private_runtime(source=None):
    if source is not None:require_source(source)
    m=production.private_for_count(NUM_ENVS,profile)
    m.profile,m.require_source=profile,require_source
    return m

def install_leg_exploration(output):
    return production.bind(production.family.install_leg_exploration,profile=profile)(output)

def native(output,args):
    import joint_pd_review_interval200 as interval
    import joint_pd_source2201_yaw025_runtime as template
    r=request_ready();source=Path(args[args.index('--checkpoint_path')+1]).resolve()
    if (str(source)!=r['source_checkpoint'] or int(args[args.index('--num_envs')+1])!=NUM_ENVS
            or int(args[args.index('--target_iteration')+1])!=7006
            or ((FLAG in args)!=(NUM_ENVS!=r['source_num_envs']))
            or any(x in args for x in ('--reset_optimizer','--reset_exploration_std','--save_initial_checkpoint'))):
        raise ValueError('Exactly five capacity-only retained-state updates')
    interval.UPDATES=5;production.family._sampler_api.install_sampling_field_restore()
    write_json(Path(output).parent/'capacity_runtime_contract.json',dict(capacity_only=True,
        num_envs=NUM_ENVS,probe_updates=5,rollout_steps=48,Adam_minibatches_per_update=20,
        only_proposed_reward_change=production.DELTA['rewards'],profile=profile(),
        source_model_all17_Adam_std_and_actual_LR_not_reset=True,
        capacity_weights_may_be_promoted=False,policy_accepted=False))
    return production.bind(template.native,SCRIPT=SCRIPT,FLAG=FLAG,
        repair_ready=request_ready,require_source=require_source,private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)(output,args)

if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--capacity-request',type=Path,required=True)
    cli.add_argument('--observation-output',type=Path,required=True)
    cli.add_argument('--capacity-num-envs',type=int,required=True)
    boundary=sys.argv.index('--');args=cli.parse_args(sys.argv[1:boundary])
    MANIFEST=args.capacity_request.resolve();NUM_ENVS=args.capacity_num_envs
    native(args.observation_output.resolve(),sys.argv[boundary+1:])
