"""Finish the interrupted height3 block from its retained98-update save.

Only the observer segment length changes to102; model, Adam, actual LR,
exploration floors, rewards and control remain the qualified ordinary PPO.
The controller saves every update and stops a diagnosed native assertion rather
than leaving a dialog to wait forever. It does not ignore engine assertions.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
REMAINING = 102


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def native(output, arguments):
    import source1201_orientation40
    import joint_pd_source1801_height3_runtime as repair
    import joint_pd_review_interval200 as interval
    from train_joint_pd_standing_seed import saved_updates
    from joint_pd_continuation import sha
    source = Path(arguments[arguments.index('--checkpoint_path') + 1]).resolve()
    saved, cfg = repair.require_source(source)
    plan = json.loads((Path(output).parent / 'recovery_plan.json').read_text())
    if (saved_updates(source) != 1899 or sha(source) != plan['resume_checkpoint_sha256']
            or int(arguments[arguments.index('--target_iteration') + 1]) != 2001
            or repair.FLAG in arguments or cfg['rewards']['base_height'] != 3.
            or cfg['rewards']['orientation'] != -40. or cfg['rewards']['tracking_sigma'] != .025
            or cfg['commands']['training_motion_scale'] != .25
            or cfg['scene']['num_envs'] != 4096
            or len(saved['optimizer_state_dict']['state']) != 17
            or any(float(x['step']) != 37980. for x in saved['optimizer_state_dict']['state'].values())):
        raise ValueError('Recover only the exact complete98-update height3 checkpoint')
    interval.UPDATES = REMAINING
    repair.native(Path(output), arguments)


def controller(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = plan_path.parent
    import hashlib
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if (digest(SCRIPT) != plan['recovery_entry_sha256']
            or any(digest(path) != value for path, value in plan['source_pins'].items())):
        raise ValueError('The qualified retained checkpoint or recovery implementation changed')
    import train_source200_drift25_measured_memory200 as memory
    memory.wait_memory(output)
    status = dict(state='training_initialization', training_active=True,
        source_actual_updates=1899, live_ppo_updates=1899,
        retained_original_block_updates=98, segment_updates=REMAINING,
        target_actual_ppo_updates=2001, checkpoint=plan['resume_checkpoint'],
        policy_accepted=False, goal_remains_active=True)
    write(output / 'status.json', status)
    log_path = output / 'training.log'
    if log_path.exists():
        raise ValueError('Inspect the existing recovery log instead of repeating this segment')
    with log_path.open('x') as log:
        child = subprocess.Popen(plan['native_command'], cwd=ROOT, env=dict(os.environ,
            PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts')+':'+str(ROOT/'mine_model/experiments'),
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            PYTHONUNBUFFERED='1'), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        status['child_pid'] = child.pid
        write(output/'status.json', status)
        while child.poll() is None:
            contents = log_path.read_text(errors='replace')
            if 'Assertion Failed!' in contents or 'Recursion not allowed' in contents:
                status.update(state='native_assertion_requires_diagnosis',
                    training_active=False, assertion_log_preserved=True)
                write(output/'status.json', status)
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                raise RuntimeError('Native assertion captured; preserve every checkpoint before diagnosis')
            import re
            iterations = re.findall(r'Learning iteration\s+(\d+)/(\d+)', contents)
            if iterations:
                actual = int(iterations[-1][0]) + 1
                status.update(state='training', live_ppo_updates=actual,
                    retained_original_block_updates=actual-1801,
                    updated_at=datetime.now().astimezone().isoformat())
                write(output/'status.json', status)
            time.sleep(2.)
        code = child.returncode
    status.update(training_active=False, child_exit_code=code,
        state='segment_native_exited' if code == 0 else 'segment_exit_requires_diagnosis',
        updated_at=datetime.now().astimezone().isoformat())
    write(output/'status.json', status)
    if code:
        raise RuntimeError('Recovery native process exited with code ' + str(code))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'native':
        boundary = sys.argv.index('--')
        if sys.argv[2:boundary] != ['--observation-output', sys.argv[3]]:
            raise ValueError('Keep the exact owned observation output')
        native(Path(sys.argv[3]), sys.argv[boundary+1:])
    elif mode == 'controller':
        controller(Path(sys.argv[2]))
    else:
        raise ValueError('Only the explicit recovery native segment or its controller')
