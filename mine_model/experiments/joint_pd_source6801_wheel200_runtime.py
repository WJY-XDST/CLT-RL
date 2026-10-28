"""Single wheel-cap trial from reviewed6801; retain ordinary PPO and Adam."""
import ast
import copy
from functools import lru_cache
import inspect
import json
from pathlib import Path
import sys

import torch

from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
import plain27_sampler_metadata_adapter as adapter
import joint_pd_source6601_targeted_turn030_runtime as previous

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT/'mine_model/results'
TRAINING = RESULTS/'joint_pd_optimization_20261021_032504_524539_source6601_targeted_turn030_200'
DECISION = RESULTS/'joint_pd_source6801_wheel200_trial_decision_20261021.json'
ANCHOR_SHA = '82d733cd85618ca78365be77a490eb3e85a196ee87862a8821fe60caef0108e0'
FLAG = '--begin_verified_source6801_wheel200'
LIMIT = 2.
DELTA = dict(rewards={},commands={},exploration={},controller={'wheel_action_limit':[1.25,LIMIT]},physics={})
KEY,base,criteria_fix,bind,recipe = previous.KEY,previous.base,previous.criteria_fix,previous.bind,previous.recipe
read = lambda path:json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def anchor():
    final = read(TRAINING/'final_checkpoint_verification_readonly_20261019.json')
    source = Path(final['checkpoint'])
    saved,cfg = previous.require_source(source)
    if (not final['verified'] or final['actual_ppo_updates']!=6801 or final['all17_Adam_step']!=136020
            or sha(source)!=ANCHOR_SHA or cfg['wheel_action_limit']!=1.25 or cfg['action_scale_vel']!=15.
            or cfg['commands'][previous.sampler.CONFIG_KEY]!=.3
            or cfg['rewards']['tracking_sigma_ang']!=.01 or cfg['rewards']['orientation']!=-60.):
        raise ValueError('Require the exact reviewed6801 and existing full command/reward contract')
    return source,saved,cfg


def profile():
    value=copy.deepcopy(previous.profile())
    value.update(profile='plain_source6801_targeted030_wheel200_v1',native_runtime_sha256=sha(SCRIPT),
        parent_targeted030_runtime_sha256=sha(previous.SCRIPT),wheel_clamp_transfer_anchor_sha256=ANCHOR_SHA,
        wheel_action_limit=LIMIT,action_scale_vel=15.,wheel_reference_max_rad_s=30.,
        authority='paired_identified_reset_seed_headroom_diagnosis; no_hardware_or_goal_acceptance')
    return value


def require_source(path,transition=False,*,initial_transfer=False):
    if initial_transfer:raise ValueError('Never reload originalmodel200 or reset the optimizer')
    _,before,cfg=anchor()
    path=Path(path).resolve()
    if sha(path)==ANCHOR_SHA:return previous.require_source(path)
    if transition:raise ValueError('Only exact6801 may start this controller transfer')
    saved=torch.load(path,map_location='cpu',weights_only=True)
    current=validate_checkpoint_robot(path)
    expected=copy.deepcopy(cfg);expected['wheel_action_limit']=LIMIT;expected['log_dir']=current.get('log_dir')
    actual=saved_updates(path);states=saved['optimizer_state_dict']['state']
    if (not tensor_finite(saved) or current!=expected or saved.get('infos',{}).get(KEY)!=profile()
            or actual<6801 or len(states)!=17 or any(float(s['step'])!=20*actual for s in states.values())
            or saved['model_state_dict'].keys()!=before['model_state_dict'].keys()
            or any(saved['model_state_dict'][k].shape!=v.shape for k,v in before['model_state_dict'].items())):
        raise ValueError('Require exact normal-cap2 direct27 policy and continuous17Adam')
    return saved,current


def repair_ready():
    if not DECISION.exists():raise ValueError('Select the source-bound single-controller trial before native work')
    d=read(DECISION)
    full=read(d['source_full_review']);long=read(d['source_original_sustained120'])
    headroom=read(d['identified_failed_seed_headroom']);coverage=read(d['paired_final4096_reset_diagnosis'])
    if (not d['selected_single_controller_clamp_trial'] or d['source_checkpoint_sha256']!=ANCHOR_SHA
            or d['source_actual_updates']!=6801 or d['source_all17_Adam_step']!=136020
            or d['planned_actual_updates']!=200 or d['target_actual_updates']!=7001
            or d['next_configuration_changes']!=DELTA or d['relative_error_target']!=.1
            or not full['all_original_and_retained_reviews_complete'] or full['actual_updates']!=6801
            or not long['raw_integrity_verified'] or long['original_gates_and_nonzero_relative10_both_passed']!=54
            or not headroom['completed'] or not headroom['original3_native_bitwise_fixture_verified']
            or not headroom['all_initial27_inputs_bitwise_paired'] or not coverage['completed']
            or not d['all_failed_and_regressed_trials_preserved'] or d['acceptance_changed']
            or d['optimizer_reset'] or d['std_reset'] or d['original_model200_reload']
            or d['faster_curriculum_authorized']
            or any(sha(p)!=digest for p,digest in d['source_pins'].items())):
        raise ValueError('Require completed own reviews and controlled headroom evidence without resets or extra changes')
    return d


_api=adapter.private_module(previous.SCRIPT,'_source6801_wheel200_sampler')
_provider=adapter.private_module(Path(previous._provider.__file__),'_source6801_wheel200_provider')
tree=ast.parse(inspect.getsource(_provider.private_runtime));original_tree=ast.dump(tree)
changes=[];counts={}
for node in ast.walk(tree):
    if isinstance(node,ast.Constant):
        new=node.value
        if type(node.value)is int and node.value==6201:new=6801
        elif type(node.value)is float and node.value in (1.25,18.75):new={1.25:2.,18.75:30.}[node.value]
        elif isinstance(node.value,str):new=node.value.replace('lin300_runtime_evidence.json','wheel200_base_invariants.json')
        if new!=node.value:
            counts[repr(node.value)]=counts.get(repr(node.value),0)+1
            changes.append((node,node.value));node.value=new
if counts.get('1.25')!=1 or counts.get('18.75')!=2:
    raise ValueError('Requalify only declared wheel-bound metadata and the source clock')
modified=copy.deepcopy(tree)
for node,value in reversed(changes):node.value=value
if ast.dump(tree)!=original_tree:raise ValueError('Original normal-map/physics observer operations changed')
exec(compile(ast.fix_missing_locations(modified),inspect.getsourcefile(_provider.private_runtime),'exec'),_provider.__dict__)
FACTORY_METADATA=[dict(function='private_runtime',declared_source_and_wheel_bound_metadata=counts,restored_AST_identical=True)]
SAMPLER_METADATA=adapter.retarget(_api,('private_runtime',),integers={'*':{6601:6801}},
    strings={'targeted_turn_runtime_evidence.json':'wheel200_targeted030_runtime_evidence.json'},
    keyword_names={'reward_exploration_physics_reset_and_control_unchanged':'inherited_rewards_exploration_physics_reset_retained',
        'sampler_RNG_draws_intentionally_added_only_for_mature_nonfixed_training':'retained_sampler_draws_only_on_mature_nonfixed_training'})
for module in (_api,_provider):
    for name in ('SCRIPT','ANCHOR_SHA','FLAG','DELTA','LIMIT','profile','recipe','require_source','repair_ready','anchor','TRAINING','DECISION','FACTORY_METADATA'):
        setattr(module,name,globals()[name])
_api._provider=_provider
private_runtime=_api.private_runtime
install_leg_exploration=_provider.install_leg_exploration
_api.install_leg_exploration=install_leg_exploration
install_sampling_field_restore=_api.install_sampling_field_restore
evaluator_tree=_api.evaluator_tree


def native(output,arguments):return _api.native(output,arguments)


def command(source,cfg,label,updates,commands,stop_file,**kwargs):
    source=Path(source).resolve();saved,actual_cfg=require_source(source);anchor_path,_,old_cfg=anchor()
    if updates!=200 or cfg!=actual_cfg or commands!=cfg['commands']:
        raise ValueError('Keep exact200 actual updates and saved source/commands')
    overrides=kwargs.get('reward_overrides',{})
    if any(not k.startswith('rewards.') or old_cfg['rewards'].get(k[8:])!=v for k,v in overrides.items()):
        raise ValueError('No reward may change in this controller-only trial')
    argv,_,_=previous.command(anchor_path,old_cfg,label,updates,old_cfg['commands'],stop_file,**kwargs)
    option='env.wheel_action_limit=1.25'
    if argv.count(option)!=1 or any(a.startswith('--repair_') for a in argv):
        raise ValueError('Requalify the original full200 CLI')
    argv[argv.index(str(previous.SCRIPT))]=str(SCRIPT)
    argv[argv.index(option)]='env.wheel_action_limit=2.0'
    actual=saved_updates(source)
    argv[argv.index('--checkpoint_path')+1]=str(source)
    argv[argv.index('--target_iteration')+1]=str(actual+200)
    if sha(source)==ANCHOR_SHA:argv.append(FLAG)
    if any(x in argv for x in ('--reset_optimizer','--reset_exploration_std','--save_initial_checkpoint')):
        raise ValueError('Do not reset or overwrite the source')
    return argv,actual+199,saved


def replay_command(source,output):
    source=Path(source).resolve();require_source(source)
    if sha(source)==ANCHOR_SHA:raise ValueError('Baseline replay uses its saved previous runtime; normal2.0 requires a declared saved fixture or descendant')
    anchor_path,_,_=anchor();argv=previous.replay_command(anchor_path,output)
    argv[argv.index(str(previous.SCRIPT))]=str(SCRIPT)
    argv[argv.index('--checkpoint')+1]=str(source)
    return argv


def replay(arguments):
    arguments=list(arguments);source=Path(arguments[arguments.index('--checkpoint')+1]);require_source(source)
    if sha(source)==ANCHOR_SHA or FLAG in arguments or '--overrides_json' in arguments or '--wheel-reference-limit-probe' in arguments:
        raise ValueError('Normal replay needs an exact saved2.0 configuration without interventions')
    install_sampling_field_restore()
    import gymnasium as gym
    original_make=gym.make;gym.make=_api._wrap_make(original_make)
    tree,path=evaluator_tree();sys.argv=[str(SCRIPT),*base.tires.arguments(arguments)]
    try:
        exec(compile(tree,str(path),'exec'),dict(__name__='__main__',__file__=str(path),
            _install_quality_criteria=criteria_fix.install,_configure_sampling=_api._configure,
            _observe_source200=private_runtime().observe_environment))
    finally:gym.make=original_make


if __name__=='__main__':
    mode=sys.argv.pop(1)
    if mode=='replay':replay(sys.argv[1:])
    elif mode=='native':
        boundary=sys.argv.index('--')
        if sys.argv[1:boundary]!=['--observation-output',sys.argv[2]]:raise ValueError('Preserve native observation-output contract')
        native(Path(sys.argv[2]),sys.argv[boundary+1:])
    else:raise ValueError('Only ordinary PPO and normal frozen replay')
