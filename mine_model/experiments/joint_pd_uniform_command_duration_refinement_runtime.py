"""One declared global training command period over the verified source runtime.

The continuous value sampler, role formulas, physical control and rewards are
unchanged. Original fixed-command replay retains its own unchanged callback.
No height, speed, yaw, case label or failure list selects a rule here.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

import joint_pd_uniform_source_refinement_runtime as base

SCRIPT=Path(__file__).resolve()
BASE_SCRIPT=base.SCRIPT
ORIGINAL_CONFIGURE=base.configure
ORIGINAL_EXPECTED=base.expected_configuration
ORIGINAL_PROFILE=base.profile_for_count
ORIGINAL_IMMEDIATE=base.immediate_changes
ORIGINAL_COMMAND=base.command
ORIGINAL_EVALUATOR=base.evaluator_tree
ORIGINAL_QUALIFY=base.qualify
_path=None
_changes=None

def __getattr__(name):return getattr(base,name)

def configure(path):
    global _path,_changes
    path=Path(path).resolve()
    if _path is not None:
        assert path==_path,'One immutable recipe per process'
        return base.recipe()
    specification=json.loads(path.read_text())['uniform_source_refinement']
    changes=specification['global_training_command_changes']
    assert set(changes)=={'resampling_time'}
    pair=changes['resampling_time']
    assert len(pair)==2 and all(type(x) in (int,float) and math.isfinite(x) and x>0 for x in pair)
    assert pair[0]<pair[1]
    ORIGINAL_CONFIGURE(path)
    assert base.baseline()[2]['commands']['resampling_time']==pair[0]
    _path,_changes=path,changes
    for namespace in {id(v):v for v in [base.__dict__,*base.NAMESPACES]}.values():
        namespace.update(SCRIPT=SCRIPT,expected_configuration=expected_configuration,profile_for_count=profile_for_count)
    base.configure=configure
    base.immediate_changes=immediate_changes
    base.command=command
    base.evaluator_tree=evaluator_tree
    base.parent.configure=configure
    base.parent.command=command
    base.parent.evaluator_tree=evaluator_tree
    return base.recipe()

def expected_configuration(cfg,count):
    normalized=copy.deepcopy(cfg)
    for key,pair in _changes.items():
        assert normalized['commands'][key] in pair
        normalized['commands'][key]=pair[0]
    result=ORIGINAL_EXPECTED(normalized,count)
    for key,pair in _changes.items():
        assert result['commands'][key]==pair[0]
        result['commands'][key]=pair[1]
    return result

def immediate_changes(cfg,source_floors=None):
    result=ORIGINAL_IMMEDIATE(cfg,source_floors)
    target=expected_configuration(cfg,8192)
    result['commands']={key:[cfg['commands'][key],target['commands'][key]]
                        for key in _changes if cfg['commands'][key]!=target['commands'][key]}
    return result

def profile_for_count(count):
    result=ORIGINAL_PROFILE(count)
    result.update(profile='plain27_uniform_saved_source_command_duration_refinement_v1',
        global_training_command_changes=copy.deepcopy(immediate_changes(base.baseline()[2])['commands']),
        global_command_overrides={key:pair[1] for key,pair in _changes.items()},
        source_physics_reward_commands_leg_PD_and_action_maps_retained=False,
        source_physics_rewards_leg_PD_and_action_maps_retained=True,
        original_continuous_command_value_distribution_and_role_formulas_retained=True,
        common_period_for_every_environment_height_velocity_sign_and_role=True,
        original_fixed_command_replay_callback_retained=True,
        command_duration_base_runtime_sha256=base.sha(BASE_SCRIPT))
    return result

def command(source,cfg,label,updates,commands,stop_file):
    argv,final,saved=ORIGINAL_COMMAND(source,cfg,label,updates,commands,stop_file)
    for key,pair in _changes.items():
        prefix='env.commands.'+key+'='
        indices=[i for i,value in enumerate(argv) if value.startswith(prefix)]
        assert len(indices)==1
        argv[indices[0]]=prefix+str(pair[1])
    return argv,final,saved

def evaluator_tree():
    # The original evaluator sets its own resampling_time and overrides
    # _update_commands for exact requested commands. Do not insert a training
    # period override into that function or alter its numerical reader.
    tree,source,proof=ORIGINAL_EVALUATOR()
    proof.update(training_only_command_period_changes=copy.deepcopy(_changes),
                 no_additional_replay_AST_transformation=True,
                 fixed_command_replay_time_seed_command_and_metric_contract_retained=True)
    return tree,source,proof

def qualify(path,output):
    configure(path)
    result=ORIGINAL_QUALIFY(path,output)
    result.update(global_training_command_changes=copy.deepcopy(_changes),
        same_period_for_all_training_commands=True,original_fixed_command_replay_unchanged=True,
        base_runtime_sha256=base.sha(BASE_SCRIPT),parameter_change_selected=False,training_launched=False)
    result['source_pins'][str(BASE_SCRIPT)]=base.sha(BASE_SCRIPT)
    (output/'qualification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result

def train(request_path):
    import train_saved_joint_pd_review50 as workflow
    request=json.loads(Path(request_path).read_text())
    configure(request['recipe'])
    original_write=workflow.write_json
    def write(path,value):
        if Path(path).name=='experiment_plan.json':
            value=copy.deepcopy(value)
            value.update(only_runtime_sampling_change={'configured_global_resampling_period':copy.deepcopy(_changes)},
                original_continuous_command_value_distribution_and_role_formulas_retained=True,
                training_command_period_changed_uniformly=True,
                original_fixed_command_replay_callback_retained=True)
        return original_write(path,value)
    workflow.write_json=write
    try:return base.train(request_path)
    finally:workflow.write_json=original_write

if __name__=='__main__':
    mode=sys.argv.pop(1)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    if mode=='native':
        parser.add_argument('--observation-output',type=Path,required=True)
        at=sys.argv.index('--');a=parser.parse_args(sys.argv[1:at]);configure(a.recipe)
        base.native(a.observation_output,sys.argv[at+1:])
    elif mode=='replay':
        a,rest=parser.parse_known_args(sys.argv[1:]);configure(a.recipe);base.replay(rest)
    elif mode=='qualify':
        parser.add_argument('--output',type=Path,required=True)
        a=parser.parse_args();v=qualify(a.recipe.resolve(),a.output.resolve())
        print(json.dumps(dict(qualified=v['qualified'],checkpoint_sha256=v['checkpoint_sha256'],
                             actual_ppo_updates=v['actual_ppo_updates'],training_launched=False)))
    elif mode=='compare':
        parser.add_argument('directory',type=Path);parser.add_argument('--through',type=int,required=True)
        a=parser.parse_args();configure(a.recipe);base.compare(a.directory.resolve(),a.through)
    elif mode=='train':
        parser.add_argument('--request',type=Path,required=True)
        a=parser.parse_args();configure(a.recipe);train(a.request.resolve())
    else:raise ValueError(mode)
