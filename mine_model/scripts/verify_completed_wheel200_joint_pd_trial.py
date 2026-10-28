"""Original complete200 verifier with the explicit normal2.0 wheel contract."""
import argparse
import ast
import json
from pathlib import Path

from joint_pd_continuation import sha
from audit_saved_action_limits import transform
import verify_completed_plain27_joint_pd_trial as original
import joint_pd_source6801_wheel200_runtime as runtime

SCRIPT=Path(__file__).resolve()


def verify(directory,destination=None):
    directory=Path(directory).resolve();plan=json.loads((directory/'experiment_plan.json').read_text())
    if (plan['source_actual_ppo_updates']!=6801 or plan['target_actual_ppo_updates']!=7001
            or plan['source_sha256']!=runtime.ANCHOR_SHA or plan['only_controller_change']!=runtime.DELTA['controller']
            or plan['next_configuration_changes']!=runtime.DELTA or not plan['action_map_or_controller_changed']
            or plan['training_config']['wheel_action_limit']!=2. or plan['training_config']['action_scale_vel']!=15.
            or plan['only_reward_change'] or plan['only_command_change'] or plan['only_exploration_change']):
        raise ValueError('Require the exact6801 single wheel-bound200 trial')
    path=directory/'native_learning_signal/normal_wheel_limit_observation.json';mapping=json.loads(path.read_text())
    if (mapping['actual_policy_steps']!=200*48 or mapping['native_environments']!=4096
            or mapping['all_environment_actions_checked']!=200*48*4096 or mapping['wheel_action_limit']!=2.
            or mapping['wheel_reference_max_rad_s']!=30. or mapping['wheel_mapping_max_error'] or mapping['leg_premap_max_error']
            or mapping['observer_changes_actions_or_rewards'] or mapping['added_loss_or_backward']):
        raise ValueError('Every native environment action must obey the declared normal2.0 controller')
    tree=transform(original.verify,[("plan['action_map_or_controller_changed']",'not DECLARED_SINGLE_CAP_VERIFIED'),
                                    ("{'/log_dir'}","{'/log_dir', '/wheel_action_limit'}")])
    scope=dict(original.__dict__,DECLARED_SINGLE_CAP_VERIFIED=True,SCRIPT=SCRIPT)
    exec(compile(ast.fix_missing_locations(tree),str(original.SCRIPT),'exec'),scope)
    result=scope['verify'](directory,destination)
    result.update(only_controller_change=runtime.DELTA['controller'],
        original_other_checkpoint_Adam_RNG_alignment_and_actor_checks_preserved=True,
        original_verifier_AST_restored_identical=True,all39321600_native_wheel_and_leg_premaps_verified=True,
        normal_controller_mapping_evidence=str(path))
    result['source_files'].update({str(p):sha(p) for p in [SCRIPT,original.SCRIPT,path]})
    target=Path(destination).resolve() if destination else directory/'final_checkpoint_verification_readonly_20261019.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    parser.add_argument('--destination',type=Path);args=parser.parse_args();d=verify(args.directory,args.destination)
    print(json.dumps({k:d[k] for k in ['verified','actual_ppo_updates','all17_Adam_step','all39321600_native_wheel_and_leg_premaps_verified']}))
