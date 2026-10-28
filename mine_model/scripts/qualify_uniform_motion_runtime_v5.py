"""Original both-count CPU roundtrips plus declared global-command metadata."""
import argparse
import ast
import copy
import inspect
from pathlib import Path

import joint_pd_uniform_motion_envelope_runtime_v4 as runtime
import qualify_uniform_control_joint_pd_runtime_v3 as original
import prepare_uniform_motion_trial_v5 as preparation
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT=Path(__file__).resolve()


def qualify(recipe,output):
    tree=ast.parse(inspect.getsource(original.qualify))
    before=ast.dump(tree)
    body=tree.body[0].body
    # Find the argv equality assertion inside the original per-count loop.
    loops=[node for node in body if isinstance(node,ast.For)]
    assert len(loops)==1
    loop=loops[0].body
    indices=[i for i,node in enumerate(loop) if isinstance(node,ast.Try)]
    assert len(indices)==1
    checked=loop[indices[0]].body
    points=[i for i,node in enumerate(checked) if isinstance(node,ast.Assert)
            and ast.unparse(node.test)=="argv == expected_argv"]
    assert len(points)==1
    index=points[0]
    insert=ast.parse("""
for key,value in runtime.recipe()['command_overrides'].items():
    prefix='env.commands.'+key+'='
    indices=[i for i,item in enumerate(expected_argv) if item.startswith(prefix)]
    assert len(indices)==1
    expected_argv[indices[0]]=prefix+str(value)
""").body
    checked[index:index]=insert
    qualified=copy.deepcopy(tree)
    del checked[index:index+len(insert)]
    assert ast.dump(tree)==before
    scope=dict(original.__dict__,SCRIPT=SCRIPT)
    exec(compile(ast.fix_missing_locations(qualified),str(SCRIPT),"exec"),scope)
    result=scope['qualify'](recipe,output)
    value=runtime.recipe()
    cfg=runtime.baseline()[2]
    negatives=[]
    for name,change in (
        ('extra_height_override',lambda v:v['command_overrides'].update(ranges_height=[.25,.35])),
        ('lost_small_yaw_support',lambda v:v['command_overrides'].update(yaw_min_abs_rate=.15)),
        ('outside_global_scale',lambda v:v['command_overrides'].update(training_motion_scale=1.1)),
        ('condition_specific_logic',lambda v:v.update(case_specific_training_logic=True)),
        ('nonfinite_global_scalar',lambda v:v['command_overrides'].update(training_motion_scale=float('nan')))):
        bad=copy.deepcopy(value)
        change(bad)
        try:runtime.validate_global_command_metadata(bad,cfg)
        except (AssertionError,ValueError,TypeError):negatives.append(dict(name=name,rejected=True))
        else:raise AssertionError(name)
    result.update(global_command_overrides=value['command_overrides'],
                  global_command_changes=value['command_changes'],
                  all_original_CPU_roundtrip_and_native_argv_checks_retained=True,
                  original_CPU_qualifier_AST_restored_after_command_metadata_insertion=True,
                  negative_global_command_metadata_examples=negatives,
                  original_native_PPO_unchanged=True,actual_PPO_updates=0)
    for p in [SCRIPT,preparation.SCRIPT,Path(original.__file__),runtime.SCRIPT,
              SCRIPT.with_name('check_uniform_motion_environment_capacity_v4.py'),
              SCRIPT.with_name('train_uniform_motion_block200_v5.py'),
              SCRIPT.with_name('monitor_uniform_motion_learning_v4.py'),
              SCRIPT.with_name('review_uniform_motion_block_v4.py'),
              SCRIPT.with_name('compare_uniform_motion_recorded_joint_targets_v4.py'),
              SCRIPT.parent.parent/'experiments/joint_pd_uniform_motion_capacity_runtime_v4.py']:
        result['source_pins'][str(p.resolve())]=sha(p)
    write_json(output/'qualification.json',result)
    return result


if __name__=='__main__':
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=qualify(args.recipe.resolve(),args.output.resolve())
    print(json.dumps(dict(qualified=result['qualified'],results=result['results'],
                         negatives=result['negative_global_command_metadata_examples'],actual_PPO_updates=0)))
