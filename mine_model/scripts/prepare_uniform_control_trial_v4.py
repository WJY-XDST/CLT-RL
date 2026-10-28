"""Use unchanged v3 both5 capacity admission with the v4 metadata finalizer."""
import argparse
from pathlib import Path

import prepare_uniform_control_trial_v3 as original
import train_uniform_control_joint_pd_block200_v4 as controller


SCRIPT = Path(__file__).resolve()


def prepare(args):
    previous_script, previous_systemd = original.SCRIPT, original.systemd

    def systemd(unit, log, command, *, training=False):
        command = list(command)
        if training:
            assert Path(command[1]).name == "train_uniform_control_joint_pd_block200_v3.py"
            command[1] = controller.SCRIPT
        return previous_systemd(unit, log, command, training=training)

    original.SCRIPT, original.systemd = SCRIPT, systemd
    try:
        return original.prepare(args)
    finally:
        original.SCRIPT, original.systemd = previous_script, previous_systemd


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--source-final-verification", type=Path)
    parser.add_argument("--source-mapping-replay", type=Path)
    parser.add_argument("--source-same51-replay", type=Path)
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=False))
