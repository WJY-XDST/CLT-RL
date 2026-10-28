"""Fresh both-count admission, routing one ordinary200 block through v7 metadata fix."""
import argparse
from pathlib import Path

import prepare_uniform_motion_trial_v5 as implementation
import train_uniform_motion_block200_v7 as controller


SCRIPT = Path(__file__).resolve()


def prepare(args):
    previous_controller, previous_script = implementation.controller, implementation.SCRIPT
    implementation.controller, implementation.SCRIPT = controller, SCRIPT
    try:
        return implementation.prepare(args)
    finally:
        implementation.controller, implementation.SCRIPT = previous_controller, previous_script


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--source-final-verification', type=Path)
    parser.add_argument('--source-mapping-replay', type=Path)
    parser.add_argument('--source-same51-replay', type=Path)
    print(json.dumps(prepare(parser.parse_args())))
