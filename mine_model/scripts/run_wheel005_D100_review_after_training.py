"""Keep the review parent lightweight until the exact owned200 job exits."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / 'mine_model/results/joint_pd_wheel005_source7201_D100_qualification_20261024'
UNIT = 'clt-joint-pd-wheel005-source7201-D100-200-20261024.service'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    proof = json.loads((Q / 'preproduction_entry_CPU_verification.json').read_text())
    assert proof['qualified'] and proof['source_actual_updates'] == 7201
    assert proof['target_actual_updates'] == 7401 and proof['native_num_envs'] == 8192
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', UNIT,
            '-p', 'MainPID', '-p', 'SubState', '-p', 'Result'],
            capture_output=True, text=True, check=True)
        owner = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if owner['MainPID'] == '0':
            assert owner['SubState'] == 'exited' and owner['Result'] == 'success', owner
            break
        path = Q / 'waiting_for_owned200_before_review_import.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),
            state='waiting_on_live_owned200', unit=UNIT, unit_state=owner,
            heavy_simulator_import_deferred=True, policy_accepted=False), indent=2) + '\n')
        temp.replace(path)
        time.sleep(15.)
    assert all(sha(path) == digest for path, digest in proof['source_pins'].items())
    executable = ROOT / 'run_python.sh'
    script = ROOT / 'mine_model/scripts/review_wheel005_source7201_D100_block200.py'
    os.execv(str(executable), [str(executable), str(script)])
