"""Repair subprocess constant forwarding before any formal PPO is started.

The immutable v5 controller, runtime, source and fresh both5 admission are
reused. The facade delegates missing subprocess attributes such as STDOUT.
"""
import argparse
from pathlib import Path
import subprocess
from types import SimpleNamespace

import train_uniform_motion_block200_v5 as implementation


SCRIPT=Path(__file__).resolve()


class SubprocessFacade(SimpleNamespace):
    def __getattr__(self,name):
        return getattr(subprocess,name)


def run(pointer):
    old_class,old_script=implementation.SimpleNamespace,implementation.SCRIPT
    implementation.SimpleNamespace,implementation.SCRIPT=SubprocessFacade,SCRIPT
    try:return implementation.run(pointer)
    finally:implementation.SimpleNamespace,implementation.SCRIPT=old_class,old_script


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer',type=Path,required=True)
    args=parser.parse_args()
    try:run(args.pointer)
    except BaseException as error:
        request=implementation.original.read(args.pointer)
        output=Path(request['planned_production_output'])
        output.mkdir(exist_ok=True)
        implementation.original.write_json(output/'controller_inspection_required.json',
            dict(error=repr(error),checkpoint_files_preserved=True,
                 training_not_automatically_restarted=True,goal_remains_active=True,
                 policy_accepted=False))
        raise
