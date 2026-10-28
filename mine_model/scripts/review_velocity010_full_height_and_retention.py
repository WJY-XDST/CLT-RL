"""Original full51/static readers for the single linear-width reward trial."""
import argparse
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path

import review_motion025_orientation80_full_height as template
import review_unchanged_yaw025_full_height as waiter
import validate_yaw025_static_retention10 as retention_template
import joint_pd_source2801_velocity010_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import ROOT,sha

SCRIPT=Path(__file__).resolve()
STATIC_QUALIFICATION=ROOT/'mine_model/results/joint_pd_velocity010_static_retention_qualification_20261019.json'


def adapted(launch_path,qualification):
    launch_path,qualification=Path(launch_path).resolve(),Path(qualification).resolve()
    launch=json.loads(launch_path.read_text());decision=json.loads(Path(launch['decision']).read_text())
    if (launch['source_actual_updates']!=2801 or launch['target_actual_updates']!=3001
            or launch['target_all17_Adam_step']!=60020
            or launch['next_configuration_changes']!=dict(rewards={'tracking_sigma':[.025,.01]},
                commands={},exploration={},controller={},physics={})
            or launch['controller_sha256']!=sha(launch['controller'])
            or launch['decision_sha256']!=sha(launch['decision'])):
        raise ValueError('Bind to the exact one-width source2801-to3001 trial')
    baseline=Path(decision['full51_output'])/'candidate'
    spec=importlib.util.spec_from_file_location('_velocity010_original_full51_reader',template.__file__)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.SCRIPT,module.QUALIFICATION,module.BASELINE=SCRIPT,qualification,baseline
    module.EXPECTED_TRAINING,module.EXPECTED_UNIT=Path(launch['output']),launch['unit']

    def dependencies():
        result=list(template.dependencies());result[2]=runtime;return tuple(result)

    def baseline_readonly(api):
        original,corrected,_,_,_,digest,_=api
        path=baseline/'motion025_full_height_diagnosis.json';report=json.loads(path.read_text())
        source=Path(report['checkpoint'])
        if report['checkpoint_sha256']!=runtime.ANCHOR_SHA or digest(source)!=runtime.ANCHOR_SHA:
            raise ValueError('Baseline differs from the complete reviewed source2801')
        _,cfg=runtime.require_source(source,transition=True)
        pins={**report['source_files'],str(path):digest(path),
            str(original.SCRIPT):digest(original.SCRIPT),str(corrected.SCRIPT):digest(corrected.SCRIPT)}
        if any(digest(p)!=h for p,h in pins.items()):
            raise ValueError('Retained native baseline changed')
        function,writes=corrected.corrected_reader(),[]
        function.__globals__['write_json']=lambda p,value:writes.append((Path(p),value))
        rebuilt=function(baseline,source)
        if len(writes)!=1 or writes[0][0]!=path or writes[0][1]!=rebuilt or rebuilt!=report:
            raise ValueError('Original full51 reader does not reproduce the retained raw baseline')
        if any(digest(p)!=h for p,h in pins.items()):
            raise ValueError('Read-only baseline reconstruction changed evidence')
        return source,cfg,report,pins

    module.dependencies,module.baseline_readonly=dependencies,baseline_readonly
    evidence=[]
    for name in ('qualify','run'):
        tree=ast.parse(inspect.getsource(getattr(module,name)));before=ast.dump(tree);edits=[];counts={}
        def change(node,field,value,label):
            edits.append((node,field,copy.deepcopy(getattr(node,field))))
            setattr(node,field,value);counts[label]=counts.get(label,0)+1
        class Adapter(ast.NodeVisitor):
            def visit_UnaryOp(self,node):
                if (name=='run' and isinstance(node.op,ast.USub)
                        and isinstance(node.operand,ast.Constant)
                        and type(node.operand.value) is float and node.operand.value in (40.,80.)):
                    value=.025 if node.operand.value==40. else .01
                    change(node,'op',ast.UAdd(),'declared_reward_sign_metadata')
                    change(node.operand,'value',value,'declared_reward_value_metadata')
                    return
                self.generic_visit(node)
            def visit_Constant(self,node):
                if type(node.value) is int and node.value in (1801,36020):
                    change(node,'value',3001 if node.value==1801 else 60020,'target_clock_metadata')
                elif isinstance(node.value,str):
                    value=node.value.replace('orientation80','velocity010').replace(
                        'source1801','source3001').replace('source1601','source2801').replace(
                        'candidate1801','candidate3001')
                    if value in ('orientation','rewards.orientation'):
                        value=value.replace('orientation','tracking_sigma')
                    if value!=node.value:change(node,'value',value,'reward_key_or_source_label_metadata')
        Adapter().visit(tree)
        if name=='run' and (counts.get('declared_reward_sign_metadata')!=6
                or counts.get('declared_reward_value_metadata')!=6):
            raise ValueError('Requalify exact single-width comparison metadata')
        modified=copy.deepcopy(tree)
        for node,field,value in reversed(edits):setattr(node,field,value)
        if ast.dump(tree)!=before:
            raise ValueError('Original source guards, metrics or acceptance predicates changed')
        exec(compile(ast.fix_missing_locations(modified),template.__file__,'exec'),module.__dict__)
        evidence.append(dict(function=name,metadata_changes=counts,restored_AST_identical=True))
    return module,launch,evidence


def retention():
    module=retention_template.adapted()
    module.runtime,module.SCRIPT,module.QUALIFICATION=runtime,SCRIPT,STATIC_QUALIFICATION
    return module


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-launch',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--qualify-checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--static-retention-output',type=Path)
    parser.add_argument('--wait',action='store_true')
    args=parser.parse_args();module,launch,evidence=adapted(args.training_launch,args.qualification)
    adapter=args.qualification.with_name(args.qualification.stem+'_adapter.json')
    if args.qualify_checkpoint:
        result=module.qualify(args.qualify_checkpoint,args.qualification)
        static=retention().qualify(args.qualify_checkpoint,module.EXPECTED_TRAINING,STATIC_QUALIFICATION)
        write_json(adapter,dict(qualified=result['qualified'] and static['qualified'],
            entry_sha256=sha(SCRIPT),launch_sha256=sha(args.training_launch),
            runtime_sha256=sha(runtime.SCRIPT),template_sha256=sha(template.__file__),
            adaptations=evidence,original_thresholds_changed=False,
            native_simulations_started=0,PPO_updates_performed=0,policy_accepted=False))
    else:
        proof=json.loads(adapter.read_text())
        if (not proof['qualified'] or proof['entry_sha256']!=sha(SCRIPT)
                or proof['launch_sha256']!=sha(args.training_launch)
                or proof['runtime_sha256']!=sha(runtime.SCRIPT)
                or proof['template_sha256']!=sha(template.__file__)
                or proof['original_thresholds_changed']):
            raise ValueError('Preserve exact qualified reader, launch, profile and original gates')
        if args.output is None or args.static_retention_output is None:
            parser.error('Require full51 and independent static output paths')
        if args.wait:waiter.wait_completed(module,args.output.with_name(args.output.name+'_waiter.json'))
        result=module.run(args.output)
        result=retention().run(module.EXPECTED_TRAINING,args.output,args.static_retention_output)
    print(json.dumps(dict(qualified=result.get('qualified'),
        static_retention_verified=result.get('static_retention_verified'),policy_accepted=False)))
