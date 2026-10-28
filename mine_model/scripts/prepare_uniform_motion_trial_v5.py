"""Fresh both5 admission and a single ordinary200 global-envelope trial."""
import argparse
from pathlib import Path

import joint_pd_uniform_motion_envelope_runtime_v4 as runtime
import check_uniform_motion_environment_capacity_v4 as capacity
import prepare_uniform_control_trial_v3 as original
import train_uniform_motion_block200_v5 as controller


SCRIPT=Path(__file__).resolve()


def prepare(args):
    old_script,old_systemd,old_writer,old_capacity=original.SCRIPT,original.systemd,original.write_json,original.capacity
    def systemd(unit,log,command,*,training=False):
        command=list(command)
        if training:
            assert Path(command[1]).name=="train_uniform_control_joint_pd_block200_v3.py"
            command[1]=controller.SCRIPT
        return old_systemd(unit,log,command,training=training)
    def write(path,value):
        if Path(path).name=="request.json":
            value["command_overrides"]=runtime.recipe()["command_overrides"]
            value["command_changes"]=runtime.recipe()["command_changes"]
            value["global_motion_envelope_only"]=True
        return old_writer(path,value)
    original.SCRIPT,original.systemd,original.write_json,original.capacity=SCRIPT,systemd,write,capacity
    try:return original.prepare(args)
    finally:original.SCRIPT,original.systemd,original.write_json,original.capacity=old_script,old_systemd,old_writer,old_capacity


if __name__=="__main__":
    import json
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe",type=Path,required=True)
    parser.add_argument("--qualification",type=Path,required=True)
    parser.add_argument("--source-final-verification",type=Path)
    parser.add_argument("--source-mapping-replay",type=Path)
    parser.add_argument("--source-same51-replay",type=Path)
    print(json.dumps(prepare(parser.parse_args())))
