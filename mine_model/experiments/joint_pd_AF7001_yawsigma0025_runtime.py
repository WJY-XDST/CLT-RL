"""One existing yaw-kernel width change after the complete floor005 review.

Retain AF7001 network, seventeen Adam states, learned six std, actual LR,
full command sampler and plant. Narrow only tracking_sigma_ang .01→.0025.
Production remains gated on fresh exact-recipe capacity, CPU and frozen27
qualifications; capacity weights are never production sources.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
import joint_pd_preserved6801_yawcost200_runtime as family
import joint_pd_yawcost200_capacity_runtime as count_factory

SCRIPT=Path(__file__).resolve()
RESULTS=ROOT/'mine_model/results'
DECISION=RESULTS/'joint_pd_AF7001_yawsigma0025_trial_decision_20261023.json'
ADMISSION=RESULTS/'joint_pd_AF7001_yawsigma0025_admission_20261023.json'
SOURCE_REPORT=RESULTS/'joint_pd_optimization_20261022_135230_239804_preserved6801_yawcost200_200/final_checkpoint_verification_readonly_20261019.json'
ANCHOR_SHA='af47077587044230d5814f5e0e76a633366260698471da6162f3411494e64f49'
KEY,base,criteria_fix,bind=family.KEY,family.base,family.criteria_fix,family.bind
FLAG='--repair_AF7001_yaw_kernel0025'
REWARD_KEY,OLD_VALUE,REWARD_VALUE='tracking_sigma_ang',.01,.0025
DELTA=dict(rewards={REWARD_KEY:[OLD_VALUE,REWARD_VALUE]},commands={},exploration={},controller={},physics={})
NUM_ENVS=4096
if ADMISSION.exists():
    NUM_ENVS=json.loads(ADMISSION.read_text())['selected_num_envs']
if NUM_ENVS not in (4096,8192):raise ValueError('Invalid admitted native population')

@lru_cache(maxsize=1)
def baseline():
    report=json.loads(SOURCE_REPORT.read_text())
    source=Path(report['checkpoint'])
    saved,cfg=family.require_source(source)
    if (not report['verified'] or sha(source)!=ANCHOR_SHA
            or report['actual_ppo_updates']!=7001 or report['all17_Adam_step']!=140020
            or cfg['rewards'][REWARD_KEY]!=OLD_VALUE or cfg['rewards']['yaw_rate_error_sq']!=-200.):
        raise ValueError('Require the completely reviewed AF7001 and unchanged dense yaw cost')
    return source,saved,cfg

def profile_for_count(count):
    if count not in (4096,8192):raise ValueError('Require an admitted native count')
    result=copy.deepcopy(family.profile())
    result.update(profile='plain_AF7001_yaw_kernel0025_v1',tracking_sigma_ang=REWARD_VALUE,
        minimum_leg_std=.01,minimum_wheel_std=.01,global_minimum_std=.01,
        native_runtime_sha256=sha(SCRIPT),parent_runtime_sha256=sha(family.SCRIPT),
        native_training_environment_count=count,fixed_standing_environment_count=count//4,
        samples_per_original_PPO_update=count*48,original_Adam_minibatches_per_update=20,
        capacity_only_probe=False,authority='complete_all_scope_review_and_fresh_exact_recipe_capacity')
    return result

def profile():return profile_for_count(NUM_ENVS)

def recipe(rewards):
    return (rewards.get(REWARD_KEY)==REWARD_VALUE
        and family.recipe(dict(rewards,tracking_sigma_ang=OLD_VALUE)))

def require_source(path,transition=False,*,initial_transfer=False):
    if initial_transfer:raise ValueError('Never reload originalmodel200 or reset Adam')
    path=Path(path).resolve()
    if sha(path)==ANCHOR_SHA:return family.require_source(path)
    if transition:raise ValueError('Only AF7001 crosses this declared kernel boundary')
    _,before,cfg=baseline()
    saved=torch.load(path,map_location='cpu',weights_only=True)
    current=validate_checkpoint_robot(path)
    expected=copy.deepcopy(cfg)
    expected['log_dir']=current.get('log_dir')
    expected['rewards'][REWARD_KEY]=REWARD_VALUE
    for key in ('scene','terrain'):expected[key]['num_envs']=NUM_ENVS
    actual=saved_updates(path);states=saved['optimizer_state_dict']['state']
    if (not tensor_finite(saved) or current!=expected or actual<7001
            or saved.get('infos',{}).get(KEY)!=profile() or len(states)!=17
            or any(float(v['step'])!=actual*20 for v in states.values())
            or saved['model_state_dict'].keys()!=before['model_state_dict'].keys()
            or any(saved['model_state_dict'][k].shape!=v.shape for k,v in before['model_state_dict'].items())):
        raise ValueError('Require this exact ordinary-PPO descendant, never capacity weights')
    return saved,current

def private_for_count(count,profile_function=profile):
    # Preserve the established sampler and execution. Override its existing
    # reward-recipe assertion, not action/reward operations or optimization.
    sampler,evidence=bind(count_factory.sampler_factory,NUM_ENVS=count,profile=profile_function)()
    def plant(source=None):
        if source is not None:require_source(source)
        return bind(family._plant.private_runtime,recipe=recipe,profile=profile_function)()
    sampler._provider=SimpleNamespace(private_runtime=plant)
    sampler.profile,sampler.require_source=profile_function,require_source
    sampler.DELTA,sampler.ANCHOR_SHA=DELTA,ANCHOR_SHA
    module=sampler.private_runtime()
    module.profile,module.require_source=profile_function,require_source
    module.population_assertion_evidence=evidence
    return module

def private_runtime(source=None):
    if source is not None:require_source(source)
    return private_for_count(NUM_ENVS)

def install_leg_exploration(output):
    return bind(family.install_leg_exploration,profile=profile)(output)

def repair_ready():
    d=json.loads(DECISION.read_text())
    if (not d['all_required_review_scopes_complete'] or not d['selected_single_yaw_kernel_trial']
            or d['source_checkpoint_sha256']!=ANCHOR_SHA or d['source_actual_updates']!=7001
            or d['source_all17_Adam_step']!=140020 or d['planned_actual_updates']!=200
            or d['next_configuration_changes']!=DELTA or d['relative_error_target']!=.1
            or any(d[k] for k in ('optimizer_reset','std_reset','actual_LR_reset','original_model200_reload','acceptance_changed'))
            or any(sha(p)!=h for p,h in d['source_pins'].items())):
        raise ValueError('Complete every owned review and pin the single kernel-width trial')
    for item in d['required_complete_review_receipts']:
        if json.loads(Path(item['path']).read_text()).get(item['completion_key']) is not True:
            raise ValueError('Every owned review must finish before the next trial')
    return d

def review_ready(source):
    repair_ready()
    d=json.loads(ADMISSION.read_text())
    from check_plain27_yawsigma0025_environment_capacity import require_capacity_selection
    if (d['source_checkpoint_sha256']!=sha(source) or d['selected_num_envs']!=NUM_ENVS
            or d['proposal_decision_sha256']!=sha(DECISION) or not d['CPU_qualified']
            or not d['frozen27_qualified'] or d['next_configuration_changes']!=DELTA
            or any(sha(p)!=h for p,h in d['source_pins'].items())):
        raise ValueError('Require exact-source CPU, frozen27 and fresh capacity admission')
    require_capacity_selection(d['capacity_verified_summary'],block_id=d['capacity_block_id'],
        source=source,num_envs=NUM_ENVS)
    return d

def command(source,cfg,label,updates,commands,stop_file,**kwargs):
    source=Path(source).resolve();saved,current=require_source(source)
    if updates!=200 or cfg!=current or commands!=current['commands']:
        raise ValueError('Exactly200 ordinary updates with the complete saved command distribution')
    overrides=kwargs.pop('reward_overrides',{})
    if any(k!='rewards.'+REWARD_KEY or v!=REWARD_VALUE for k,v in overrides.items()):
        raise ValueError('Only the selected existing yaw width may change')
    if (kwargs.pop('minimum_wheel_std',.01) not in (.01,.03)
            or kwargs.pop('joint_mean_guard_coef',0.)!=0.
            or kwargs.pop('joint_height_feedback_coef',0.)!=0. or kwargs):
        raise ValueError('Retain ordinary PPO and every saved exploration entry')
    anchor,_,original=baseline()
    argv,_,_=family.command(anchor,original,label,200,original['commands'],Path(stop_file))
    argv[1]=str(SCRIPT)
    option='env.rewards.tracking_sigma_ang=0.01'
    if argv.count(option)!=1:raise ValueError('Requalify the selected source CLI')
    argv[argv.index(option)]='env.rewards.tracking_sigma_ang=0.0025'
    actual=saved_updates(source)
    for flag,value in {'--checkpoint_path':source,'--target_iteration':actual+200,
        '--num_envs':NUM_ENVS,'--observation-output':Path(stop_file).parent/'native_learning_signal',
        '--run_name':label,'--stop_file':Path(stop_file)}.items():argv[argv.index(flag)+1]=str(value)
    if sha(source)==ANCHOR_SHA:argv.append(FLAG)
    if any(x in argv for x in ('--reset_optimizer','--reset_exploration_std','--save_initial_checkpoint')):
        raise ValueError('Preserve the exact saved source state')
    return argv,actual+199,saved

def native(output,arguments):
    import joint_pd_source2201_yaw025_runtime as template
    source=Path(arguments[arguments.index('--checkpoint_path')+1]).resolve()
    d=review_ready(source)
    if (int(arguments[arguments.index('--num_envs')+1])!=NUM_ENVS
            or int(arguments[arguments.index('--target_iteration')+1])!=saved_updates(source)+200
            or ((FLAG in arguments)!=(sha(source)==ANCHOR_SHA))):
        raise ValueError('Declare one retained-state reward transfer and exact admitted count')
    family._sampler_api.install_sampling_field_restore()
    write_json(Path(output).parent/'yaw_kernel_native_admission.json',dict(
        source_checkpoint_sha256=sha(source),num_envs=NUM_ENVS,
        only_reward_change=DELTA['rewards'],capacity_receipt=d['capacity_verified_summary'],
        all17_Adam_six_std_and_actual_LR_preserved=True,policy_accepted=False))
    return bind(template.native,SCRIPT=SCRIPT,FLAG=FLAG,repair_ready=lambda:d,
        require_source=require_source,private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)(output,arguments)

evaluator_tree=family.evaluator_tree

def replay_command(source,output):
    require_source(source)
    anchor,_,_=baseline()
    argv=family.replay_command(anchor,output)
    argv[argv.index(str(family.SCRIPT))]=str(SCRIPT)
    argv[argv.index('--checkpoint')+1]=str(Path(source).resolve())
    if sha(source)==ANCHOR_SHA:
        argv.extend([FLAG,'--overrides_json',str(DECISION.parent/'joint_pd_AF7001_yawsigma0025_frozen_overrides.json')])
    return argv

def replay(arguments):
    arguments=list(arguments);transition=FLAG in arguments
    if transition:arguments.remove(FLAG);repair_ready()
    source=Path(arguments[arguments.index('--checkpoint')+1]);require_source(source,transition=transition)
    if transition:
        if ('--overrides_json' not in arguments or json.loads(Path(arguments[arguments.index('--overrides_json')+1]).read_text())!={'rewards.'+REWARD_KEY:REWARD_VALUE}):
            raise ValueError('Frozen transfer changes only the existing yaw reward width')
    elif sha(source)==ANCHOR_SHA or '--overrides_json' in arguments:
        raise ValueError('Descendant replay uses its exact saved configuration')
    family._sampler_api.install_sampling_field_restore()
    import gymnasium as gym
    original_make=gym.make;gym.make=family._sampler_api._wrap_make(original_make)
    tree,path=evaluator_tree();sys.argv=[str(SCRIPT),*base.tires.arguments(arguments)]
    try:
        exec(compile(tree,str(path),'exec'),dict(__name__='__main__',__file__=str(path),
            _install_quality_criteria=criteria_fix.install,
            _configure_sampling=family._sampler_api._configure,
            _observe_source200=private_runtime().observe_environment))
    finally:gym.make=original_make

if __name__=='__main__':
    mode=sys.argv.pop(1)
    if mode=='native':
        boundary=sys.argv.index('--');native(Path(sys.argv[2]),sys.argv[boundary+1:])
    elif mode=='replay':replay(sys.argv[1:])
    else:raise ValueError('Only ordinary native PPO and frozen replay are supported')
