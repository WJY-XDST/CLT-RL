"""Matched AF7001/D075 trial: lower only the future wheel exploration floor.

Keep the fully saved source NN/17 Adam/std/LR. D075 and8192 preference match
the completed comparison trial; the only new factor versus that trial is the
wheel floor. Leg floor remains.01, all six heads train, and PPO is unchanged.
Historical runtime files and all rejected candidates remain immutable.
"""
import ast
import copy
import json
from pathlib import Path
import sys
from types import FunctionType

import joint_pd_AF7001_wheelD075_runtime as previous
import joint_pd_source200_leg05_runtime as projection
import plain27_sampler_metadata_adapter as adapter
from joint_pd_continuation import ROOT,sha


SCRIPT=Path(__file__).resolve()
RESULTS=ROOT/'mine_model/results'
QUALIFICATION=RESULTS/'joint_pd_AF7001_wheelD075_wheel005_qualification_20261024'
DECISION=QUALIFICATION/'proposal.json'
ADMISSION=QUALIFICATION/'native_admission.json'
FLAG='--repair_AF7001_wheelD075_wheel005'
NUM_ENVS=8192
LEG_STD_FLOOR=.01
GLOBAL_STD_FLOOR=WHEEL_STD_FLOOR=.005
ANCHOR_SHA=previous.ANCHOR_SHA
SOURCE_REPORT=previous.SOURCE_REPORT
DELTA=copy.deepcopy(previous.DELTA)
DELTA['exploration']={'minimum_wheel_std':[.01,.005],'global_minimum_std':[.01,.005]}
api=adapter.private_module(previous.SCRIPT,'_AF7001_D075_wheel005_isolated_original_runtime')
for key in ('SCRIPT','DECISION','ADMISSION','FLAG','NUM_ENVS','DELTA'):
    setattr(api,key,globals()[key])
_require=api.require_source
_private=api.private_runtime
_private_for_count=api.private_for_count
_command=api.command
_native=api.native
_replay=api.replay
_replay_command=api.replay_command
_frozen_transfer=api.frozen_transfer_command
_evaluator_tree=api.evaluator_tree
baseline=api.baseline
expected_configuration=api.expected_configuration
family,base,bind,KEY=api.family,api.base,api.bind,api.KEY


def sync():
    api.NUM_ENVS=NUM_ENVS


def profile_for_count(count):
    value=previous.profile_for_count(count)
    value.update(profile='plain_AF7001_wheelD075_wheel005_v1',
        minimum_leg_std=LEG_STD_FLOOR,minimum_wheel_std=WHEEL_STD_FLOOR,
        global_minimum_std=GLOBAL_STD_FLOOR,native_runtime_sha256=sha(SCRIPT),
        inherited_D075_runtime_sha256=sha(previous.SCRIPT),
        unchanged_projection_implementation_sha256=sha(projection.__file__),
        only_new_factor_vs_completed_same_source_D075_trial='future_wheel_std_floor_.01_to_.005',
        authority='completed_full_D075_reviews_and_preserved50_100_diagnostics_then_fresh_capacity')
    return value


def profile():
    return profile_for_count(NUM_ENVS)


def require_source(path,transition=False,*,initial_transfer=False):
    sync()
    return _require(path,transition=transition,initial_transfer=initial_transfer)


def repair_ready():
    proof=json.loads(DECISION.read_text())
    assert proof['selected_matched_wheel_floor_trial'] and proof['all_required_review_scopes_complete']
    assert proof['source_checkpoint_sha256']==ANCHOR_SHA
    assert proof['source_actual_updates']==7001 and proof['source_all17_Adam_step']==140020
    assert proof['planned_actual_updates']==200 and proof['next_configuration_changes']==DELTA
    assert proof['relative_error_target']==.1
    assert not any(proof[key] for key in ('optimizer_reset','std_reset','actual_LR_reset',
        'original_model200_reload','acceptance_changed'))
    assert all(sha(path)==digest for path,digest in proof['source_pins'].items())
    for entry in proof['required_complete_review_receipts']:
        assert json.loads(Path(entry['path']).read_text())[entry['completion_key']] is True
    return proof


def review_ready(source):
    repair_ready();decision=json.loads(ADMISSION.read_text())
    assert decision['source_checkpoint_sha256']==sha(source)==ANCHOR_SHA
    assert decision['selected_num_envs']==NUM_ENVS and decision['next_configuration_changes']==DELTA
    assert decision['CPU_qualified'] and decision['frozen27_qualified']
    assert decision['proposal_decision_sha256']==sha(DECISION)
    assert all(sha(path)==digest for path,digest in decision['source_pins'].items())
    # The exact-source, five-real-update resource validator is bound by the
    # dedicated capacity entry. Its numerical/resource gates are unchanged.
    from check_plain27_wheelD075_wheel005_environment_capacity import require_capacity_selection
    require_capacity_selection(decision['capacity_verified_summary'],
        block_id=decision['capacity_block_id'],source=source,num_envs=NUM_ENVS)
    return decision


def private_for_count(count,profile_api=None):
    sync()
    selected=(lambda:profile_for_count(count)) if profile_api is None else profile_api
    return _private_for_count(count,selected)


def private_runtime(source=None):
    sync()
    return _private(source)


def install_leg_exploration(output,profile_api=None):
    selected=profile if profile_api is None else profile_api
    def log(*values,**kwargs):
        if values and isinstance(values[0],str) and 'User-requested leg-only exploration floor' in values[0]:
            values=('[INFO]: Future leg floor .01; wheel floor .005; saved initial std and all17 Adam retained.',*values[1:])
        print(*values,**kwargs)
    namespace=dict(projection.__dict__,LEG_STD_FLOOR=LEG_STD_FLOOR,
        GLOBAL_STD_FLOOR=GLOBAL_STD_FLOOR,WHEEL_STD_FLOOR=WHEEL_STD_FLOOR,
        profile=selected,print=log)
    install=FunctionType(projection.install_leg_exploration.__code__,namespace,
        projection.install_leg_exploration.__name__)
    return install(output)


def command(*args,**kwargs):
    sync()
    requested=kwargs.pop('minimum_wheel_std',WHEEL_STD_FLOOR)
    assert requested in (.005,.01,.03)
    # The old reward carrier passes its historical option; the actual command
    # is explicitly rewritten to the selected persistent future floor.
    argv,final,saved=_command(*args,minimum_wheel_std=.01,**kwargs)
    for option,value in (('--min_exploration_std',GLOBAL_STD_FLOOR),
                         ('--min_wheel_exploration_std',WHEEL_STD_FLOOR)):
        assert argv.count(option)==1 and float(argv[argv.index(option)+1])==.01
        argv[argv.index(option)+1]=str(value)
    assert not any(flag in argv for flag in ('--reset_optimizer','--reset_exploration_std','--wheel_exploration_std'))
    return argv,final,saved


def evaluator_tree():
    tree,path=_evaluator_tree();before=ast.dump(tree)
    main=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    index=next(i for i,node in enumerate(main.body) if isinstance(node,ast.Assign)
        and ast.unparse(node)=='cfg = WheelLeggedVMCFlatEnvCfg()')+1
    added=ast.parse("assert saved_config['wheel_damping'] == saved_config['robot']['actuators']['wheels']['damping']\n"
        "cfg.wheel_damping = saved_config['wheel_damping']\n"
        "cfg.robot.actuators['wheels'].damping = saved_config['robot']['actuators']['wheels']['damping']").body
    main.body[index:index]=added;modified=copy.deepcopy(tree)
    del main.body[index:index+len(added)]
    assert ast.dump(tree)==before
    return ast.fix_missing_locations(modified),path


def replay_command(source,output):
    sync();return _replay_command(source,output)


def frozen_transfer_command(source,output):
    sync();return _frozen_transfer(source,output)


def native(output,arguments):
    sync();return _native(output,arguments)


def replay(arguments):
    sync();return _replay(arguments)


for name in ('profile_for_count','profile','require_source','repair_ready','review_ready',
             'private_runtime','private_for_count','install_leg_exploration','evaluator_tree'):
    setattr(api,name,globals()[name])


def __getattr__(name):
    return getattr(api,name)


if __name__=='__main__':
    mode=sys.argv.pop(1)
    if mode=='native':
        boundary=sys.argv.index('--');native(Path(sys.argv[2]),sys.argv[boundary+1:])
    elif mode=='replay':replay(sys.argv[1:])
    else:raise ValueError('Only matched ordinary PPO or frozen replay')
