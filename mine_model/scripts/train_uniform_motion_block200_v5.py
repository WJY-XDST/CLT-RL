"""Original200 PPO controller with a declared global command-envelope change."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import joint_pd_uniform_motion_envelope_runtime_v4 as runtime
import train_uniform_control_joint_pd_block200_v3 as original
import monitor_uniform_motion_learning_v4 as monitor
import compare_uniform_motion_recorded_joint_targets_v4
import review_uniform_motion_block_v4 as review


SCRIPT=Path(__file__).resolve()


def run(pointer):
    previous_subprocess,previous_write,previous_script=original.subprocess,original.write_json,original.SCRIPT
    def route(argv):
        result=list(argv)
        if len(result)>1:
            name=Path(result[1]).name
            if name=="monitor_uniform_control_joint_pd_learning_v3.py":result[1]=str(monitor.SCRIPT)
            elif name=="review_uniform_control_joint_pd_block_v3.py":result[1]=str(review.SCRIPT)
        return result
    def write(path,value):
        if Path(path).name=="experiment_plan.json":
            value["next_configuration_changes"]["commands"]=runtime.recipe()["command_changes"]
            value["only_global_command_envelope_change"]=runtime.recipe()["command_changes"]
            value["original_class_sampler_and_roles_unchanged"]=True
            for key in list(value):
                if "commands_controller_physics_and_reset_configurations_unchanged" in key or key.endswith("_unchanged_recipe_continuation"):
                    value.pop(key,None)
        elif Path(path).name=="final_checkpoint_verification_readonly.json":
            value["declared_global_command_scalar_changes"]=runtime.recipe()["command_changes"]
            value["physical_controller_reward_and_class_sampler_unchanged"]=True
        return previous_write(path,value)
    original.subprocess=SimpleNamespace(
        Popen=lambda argv,*a,**kw:previous_subprocess.Popen(route(argv),*a,**kw),
        run=lambda argv,*a,**kw:previous_subprocess.run(route(argv),*a,**kw))
    original.write_json,original.SCRIPT=write,SCRIPT
    try:return original.run(Path(pointer).resolve())
    finally:original.subprocess,original.write_json,original.SCRIPT=previous_subprocess,previous_write,previous_script


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer",type=Path,required=True)
    args=parser.parse_args()
    try:run(args.pointer)
    except BaseException as error:
        request=original.read(args.pointer)
        output=Path(request["planned_production_output"])
        output.mkdir(exist_ok=True)
        original.write_json(output/"controller_inspection_required.json",dict(error=repr(error),
            checkpoint_files_preserved=True,training_not_automatically_restarted=True,
            goal_remains_active=True,policy_accepted=False))
        raise
