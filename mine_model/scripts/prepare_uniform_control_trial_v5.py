"""Use the unchanged v3 both5 admission, with the v5 reader-routing controller."""
import argparse
import json
from pathlib import Path

import prepare_uniform_control_trial_v3 as original
import train_uniform_control_joint_pd_block200_v5 as controller


SCRIPT = Path(__file__).resolve()
CAPACITY_GATE = SCRIPT.with_name('train_uniform_control_after_capacity.py')


def prepare(args):
    old_script, old_systemd = original.SCRIPT, original.systemd

    def systemd(unit, log, command, *, training=False):
        command = list(command)
        if training:
            assert Path(command[1]).name == 'train_uniform_control_joint_pd_block200_v3.py'
            command[1] = CAPACITY_GATE
        return old_systemd(unit, log, command, training=training)

    original.SCRIPT, original.systemd = SCRIPT, systemd
    try:
        return original.prepare(args)
    finally:
        original.SCRIPT, original.systemd = old_script, old_systemd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--source-final-verification', type=Path)
    parser.add_argument('--source-mapping-replay', type=Path)
    parser.add_argument('--source-same51-replay', type=Path)
    print(json.dumps(prepare(parser.parse_args()), ensure_ascii=False))
