"""Start the ordinary block only after its fresh capacity receipt is complete."""
import argparse
import json
from pathlib import Path
import os
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def run(pointer):
    pointer = Path(pointer).resolve()
    spec = json.loads(pointer.read_text())
    receipt = Path(spec['capacity']) / 'complete.json'
    while not receipt.exists():
        raw = subprocess.run(['systemctl', '--user', 'show', spec['capacity_unit'],
                              '-p', 'MainPID', '-p', 'SubState', '-p', 'Result'],
                             capture_output=True, text=True, check=True).stdout
        state = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
        if state.get('MainPID') == '0' or state.get('SubState') in {'failed', 'dead'}:
            raise RuntimeError('Capacity owner ended before receipt: ' + repr(state))
        time.sleep(10.)
    capacity = json.loads(receipt.read_text())
    if capacity.get('selected_num_envs') not in (4096, 8192):
        raise RuntimeError('Fresh capacity checks did not select a production environment count')
    command = [str(ROOT / 'run_python.sh'), str(Path(__file__).with_name('train_uniform_control_joint_pd_block200_v5.py')),
               '--pointer', str(pointer)]
    os.execv(command[0], command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pointer', type=Path, required=True)
    run(parser.parse_args().pointer)
