"""Original fixed-input comparison with the declared1.25-to2.0 wheel bound."""
import argparse
import ast
import json
from pathlib import Path

from joint_pd_continuation import sha
from audit_saved_action_limits import transform
from legacy_wheel_control_schema import differences_with_wheel_default
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
import compare_plain27_recorded_joint_targets as original
import joint_pd_source6801_wheel200_runtime as runtime

SCRIPT=Path(__file__).resolve()


def compare(directory,through):
    directory=Path(directory).resolve();plan=json.loads((directory/'experiment_plan.json').read_text())
    health=json.loads((directory/f'policy_update_health_prefix_{through:04d}_readonly.json').read_text())
    before_path=Path(plan['source_checkpoint']);after_path=Path(health['snapshots'][-1]['checkpoint_after'])
    _,before=runtime.require_source(before_path);_,after=runtime.require_source(after_path)
    if (plan['only_controller_change']!=runtime.DELTA['controller'] or plan['next_configuration_changes']!=runtime.DELTA
            or sha(before_path)!=runtime.ANCHOR_SHA or checkpoint_wheel_action_limit(before)!=1.25
            or checkpoint_wheel_action_limit(after)!=2. or before['action_scale_vel']!=15. or after['action_scale_vel']!=15.):
        raise ValueError('Require the exact6801 single wheel-cap transfer')
    tree=transform(original.compare,[("{'/log_dir'}","{'/log_dir', '/wheel_action_limit'}"),
        ('np.clip(r[..., (2, 5)], -1., 1.)','np.clip(r[..., (2, 5)], -saved_limit(cfg), saved_limit(cfg))')])
    scope=dict(original.__dict__,SCRIPT=SCRIPT,saved_limit=checkpoint_wheel_action_limit,differences=differences_with_wheel_default)
    exec(compile(ast.fix_missing_locations(tree),str(original.SCRIPT),'exec'),scope)
    path,result=scope['compare'](directory,through)
    result.update(only_controller_change=runtime.DELTA['controller'],source_and_candidate_saved_wheel_action_limits=[1.25,2.],
        original_four_leg_mapping_all17_Adam_RNG_and_fixed_input_checks_preserved=True,
        original_AST_restored_identical=True,wheel_target_difference_includes_declared_controller_and_policy_changes=True)
    result['source_pins'].update({str(p):sha(p) for p in [SCRIPT,original.SCRIPT,runtime.SCRIPT,
        Path(transform.__code__.co_filename),Path(differences_with_wheel_default.__code__.co_filename)]})
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return path,result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--through',type=int,required=True);args=parser.parse_args()
    path,result=compare(args.directory,args.through)
    print(json.dumps(dict(report=str(path),fixed_input_only=True,original_AST_restored_identical=True)))
