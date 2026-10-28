"""Restore the declared global command configuration for unchanged replay."""
import argparse
from pathlib import Path

import joint_pd_uniform_command_runtime
from review_uniform_control_global_stage_v5 import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    args = parser.parse_args()
    run(args.output.resolve(), args.recipe.resolve())
