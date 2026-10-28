"""Original real-five capacity runner with declared global command metadata."""
import argparse
from pathlib import Path
import sys

import joint_pd_uniform_motion_envelope_runtime_v4 as production
import joint_pd_uniform_control_capacity_runtime_v3 as implementation


SCRIPT = Path(__file__).resolve()
implementation.production = production
implementation.SCRIPT = SCRIPT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-request", type=Path, required=True)
    parser.add_argument("--observation-output", type=Path, required=True)
    parser.add_argument("--capacity-num-envs", type=int, required=True)
    boundary = sys.argv.index("--")
    args = parser.parse_args(sys.argv[1:boundary])
    implementation.MANIFEST = args.capacity_request.resolve()
    implementation.NUM_ENVS = args.capacity_num_envs
    implementation.native(args.observation_output.resolve(), sys.argv[boundary+1:])
else:
    sys.modules[__name__] = implementation
