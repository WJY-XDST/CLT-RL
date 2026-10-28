"""Original full51 and static readers bound to the prospective leg020 trial."""
import argparse
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path

import review_leg035_full_height_and_retention as template
import review_unchanged_yaw025_full_height as waiter
import joint_pd_source3201_leg020_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import ROOT,sha

SCRIPT=Path(__file__).resolve()
STATIC_QUALIFICATION=ROOT/'mine_model/results/joint_pd_leg020_static_retention_qualification_20261019.json'


def family():
    spec=importlib.util.spec_from_file_location('_leg020_original_full51_static_family',template.__file__)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCRIPT,module.STATIC_QUALIFICATION,module.runtime=SCRIPT,STATIC_QUALIFICATION,runtime
    tree=ast.parse(inspect.getsource(module.adapted))
    before,edits,counts=ast.dump(tree),[],{}
    for node in ast.walk(tree):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):
            value=node.value.replace("{'minimum_leg_std':[.05,.035]}","{'minimum_leg_std':[.035,.02]}")
            value=value.replace('leg035','leg020')
            if value!=node.value:
                edits.append((node,node.value));node.value=value
                counts['floor_declaration_or_label_metadata']=counts.get('floor_declaration_or_label_metadata',0)+1
        elif isinstance(node,ast.List) and ast.unparse(node)=='[0.05, 0.035]':
            for index,value in [(0,.035),(1,.02)]:
                edits.append((node.elts[index],node.elts[index].value));node.elts[index].value=value
            counts['declared_floor_result_metadata']=counts.get('declared_floor_result_metadata',0)+1
    if counts.get('declared_floor_result_metadata')!=1 or counts.get('floor_declaration_or_label_metadata')!=2:
        raise ValueError('Requalify the exact original floor-declaration and reader-label adapter')
    changed=copy.deepcopy(tree)
    for node,value in reversed(edits):
        node.value=value
    if ast.dump(tree)!=before:
        raise ValueError('Original launch/source guards or numeric acceptance predicates changed')
    exec(compile(ast.fix_missing_locations(changed),template.__file__,'exec'),module.__dict__)
    module.outer_adapter_evidence=dict(metadata_changes=counts,restored_AST_identical=True,
        original_template_sha256=sha(template.__file__))
    return module


def adapted(launch,qualification):
    retained=family()
    module,launch,evidence=retained.adapted(launch,qualification)
    return module,launch,dict(original=evidence,outer=retained.outer_adapter_evidence)


def retention():
    return family().retention()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-launch',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--qualify-checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--static-retention-output',type=Path)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args()
    module,launch,evidence=adapted(args.training_launch,args.qualification)
    adapter=args.qualification.with_name(args.qualification.stem+'_adapter.json')
    if args.qualify_checkpoint:
        result=module.qualify(args.qualify_checkpoint,args.qualification)
        static=retention().qualify(args.qualify_checkpoint,module.EXPECTED_TRAINING,STATIC_QUALIFICATION)
        write_json(adapter,dict(qualified=result['qualified'] and static['qualified'],
            entry_sha256=sha(SCRIPT),launch_sha256=sha(args.training_launch),runtime_sha256=sha(runtime.SCRIPT),
            template_sha256=sha(template.__file__),adaptations=evidence,PPO_updates_performed=0,
            native_simulations_started=0,original_thresholds_changed=False,policy_accepted=False))
    else:
        proof=json.loads(adapter.read_text())
        if (not proof['qualified'] or proof['entry_sha256']!=sha(SCRIPT)
                or proof['launch_sha256']!=sha(args.training_launch)
                or proof['runtime_sha256']!=sha(runtime.SCRIPT)
                or proof['template_sha256']!=sha(template.__file__) or proof['original_thresholds_changed']):
            raise ValueError('Preserve the exact qualified reader, launch, profile and original gates')
        if args.output is None or args.static_retention_output is None:
            parser.error('Require both full51 and own static-retention outputs')
        if args.wait:
            waiter.wait_completed(module,args.output.with_name(args.output.name+'_waiter.json'))
        module.run(args.output)
        result=retention().run(module.EXPECTED_TRAINING,args.output,args.static_retention_output)
    print(json.dumps(dict(qualified=result.get('qualified'),
        static_retention_verified=result.get('static_retention_verified'),policy_accepted=False)))
