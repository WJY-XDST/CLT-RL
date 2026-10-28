"""Five real isolated PPO updates using the exact selected wheel005 recipe."""
import argparse
import json
from pathlib import Path
import sys

import joint_pd_AF7001_wheelD075_capacity_runtime as original
import joint_pd_AF7001_wheelD075_wheel005_runtime as production

SCRIPT=Path(__file__).resolve()
original.SCRIPT=SCRIPT
original.production=production
_request_ready=original.request_ready


def request_ready():
    request=_request_ready()
    assert request['proposed_exploration_floors']==[.01,.01,.005,.01,.01,.005]
    args=request['prototype_command']
    assert float(args[args.index('--min_exploration_std')+1])==.005
    assert float(args[args.index('--min_wheel_exploration_std')+1])==.005
    return request


def install_leg_exploration(output):
    return production.install_leg_exploration(output,profile_api=original.profile)


original.request_ready=request_ready
original.install_leg_exploration=install_leg_exploration


def __getattr__(name):
    return getattr(original,name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capacity-request',type=Path,required=True)
    parser.add_argument('--observation-output',type=Path,required=True)
    parser.add_argument('--capacity-num-envs',type=int,required=True)
    boundary=sys.argv.index('--');args=parser.parse_args(sys.argv[1:boundary])
    original.MANIFEST=args.capacity_request.resolve()
    original.NUM_ENVS=args.capacity_num_envs
    original.native(args.observation_output.resolve(),sys.argv[boundary+1:])
