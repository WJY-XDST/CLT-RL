"""Register the recorded training-only entropy profile, then run native training."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--leg-entropy-coefficient', type=float, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('training_arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.training_arguments
    if arguments[:1] == ['--']:
        arguments = arguments[1:]
    root = Path(__file__).resolve().parents[2]
    script = root/'wheel_legged_isaaclab/scripts/rsl_rl/train.py'
    from wheel_legged_gym_isaaclab.joint_pd_exploration import register_runner_policy
    register_runner_policy(args.leg_entropy_coefficient, args.evidence)
    sys.argv = [str(script), *arguments]
    # The ordinary entry imports cli_args from its own directory.
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
