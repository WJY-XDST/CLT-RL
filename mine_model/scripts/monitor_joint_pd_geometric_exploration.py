"""Follow reviewed blocks and read executed motor/FK exploration at40/100/200."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'mine_model/results'
AUDITOR = ROOT / 'mine_model/scripts/audit_joint_target_geometric_exploration.py'
STATUS = RESULTS / 'joint_pd_geometric_exploration_monitor_status_20261019.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tick():
    pointer = json.loads((RESULTS / 'joint_pd_active_joint_tracking.json').read_text())
    output = Path(pointer['directory']).resolve()
    if output.parent != RESULTS:
        raise ValueError('Require an owned current training output')
    plan = json.loads((output / 'experiment_plan.json').read_text())
    if (plan['bounded_training_updates'] != 200 or not plan['ordinary_full_policy_PPO']
            or plan['training_config']['leg_control_mode'] != 'implicit_joint_position'):
        raise ValueError('Only reviewed direct-joint ordinary200-update PPO blocks')
    completed = []
    for prefix in (40, 100, 200):
        health = output / f'learning_health_combined_prefix_{prefix:04d}_readonly.json'
        if not health.is_file():
            continue
        if json.loads(health.read_text())['committed_prefix'] != prefix:
            raise ValueError('Wrong committed learning-health prefix')
        target = output / f'joint_target_geometric_exploration_prefix_{prefix:04d}_readonly.json'
        if not target.exists():
            environment = dict(os.environ,
                PYTHONPATH=':'.join(str(ROOT / p) for p in ('wheel_legged_isaaclab', 'mine_model/scripts')),
                OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
            with (output / f'learning_health_geometric_exploration_{prefix:04d}.log').open('a') as log:
                subprocess.run([str(ROOT / 'run_python.sh'), str(AUDITOR), str(output),
                    '--through', str(prefix)], cwd=ROOT, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
        report = json.loads(target.read_text())
        if (report['committed_updates'] != prefix or report['training_output'] != str(output)
                or report['source_pins'].get(str(AUDITOR)) != sha(AUDITOR)
                or report['original_native_motor_target_parity_max_error_rad'] > 1.e-7
                or report['original_native_FK_reference_parity_max_error_m'] > 2.e-7
                or report['RNG_draws_added'] or not report['RNG_state_unchanged']
                or report['training_modified'] or report['optimizer_executed'] or report['simulator_started']):
            raise ValueError('Require native-parity-qualified, read-only target exploration evidence')
        completed.append(prefix)
    status = dict(observed_at=datetime.now().astimezone().isoformat(), monitor_pid=os.getpid(),
        auditor_sha256=sha(AUDITOR), directory=str(output), unit=pointer['unit'],
        state='observing', completed_prefixes=completed,
        training_modified=False, simulator_started=False, optimizer_executed=False,
        acceptance_changed=False, policy_accepted=False)
    STATUS.write_text(json.dumps(status, indent=2) + '\n')
    return status


if __name__ == '__main__':
    while True:
        try:
            tick()
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        except BaseException as error:
            (RESULTS / 'joint_pd_geometric_exploration_monitor_error_20261019.json').write_text(
                json.dumps(dict(observed_at=datetime.now().astimezone().isoformat(),
                    error=repr(error), requires_agent_inspection=True, training_modified=False)) + '\n')
            raise
        time.sleep(15.)
