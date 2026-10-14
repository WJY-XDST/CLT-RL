"""Arm a scoped completion gate without interrupting a running PPO rollout.

The supervisor owns training and evaluation. This watcher only finalizes an
accepted, already packaged experiment, after the supervisor releases its lock.
It never treats reward, an iteration count, or an accepted label as evidence
enough to power off: complete_mine_training replays every acceptance gate.
"""
import argparse
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
LOG_ROOT = ROOT / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat'


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def eligible(state, completion, request):
    """Fail closed on a stale result, wrong execution chain or looser limits."""
    if not request.get('armed') or state.get('state') != 'accepted':
        return False
    if state.get('error') or state.get('child_pid') is not None:
        return False
    if state.get('overrides', {}).get('leg_control_mode') != request['leg_control_mode']:
        return False
    relative = state.get('acceptance_criteria', {}).get('relative_error')
    if (not isinstance(relative, (int, float)) or not math.isfinite(relative)
            or not 0 < relative <= request['relative_error_target']):
        return False
    if not completion.get('replays_passed') or not completion.get('documentation_saved'):
        return False
    return bool(state.get('checkpoint') and completion.get('checkpoint_sha256'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=30.)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error('interval must be finite and at least one second')
    output = args.output.resolve()
    request_path = output / 'completion_request.json'
    status_path = output / 'shutdown_monitor.json'
    if not output.is_relative_to(ROOT / 'mine_model/results'):
        parser.error('watcher output must be a mine-model experiment in this project')
    with (output / '.completion_watch.lock').open('a') as watcher_lock:
        fcntl.flock(watcher_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            record = dict(state='waiting_for_acceptance',
                          updated_at=datetime.now().astimezone().isoformat(),
                          experiment=str(output), poweroff_command_submitted=False)
            try:
                request = json.loads(request_path.read_text())
                if request.get('experiment') != str(output):
                    raise ValueError('Completion request belongs to another experiment')
                if request.get('leg_control_mode') != 'explicit_vmc':
                    raise ValueError('This request requires the explicit VMC experiment')
                target = request.get('relative_error_target')
                if not isinstance(target, (int, float)) or not math.isfinite(target) or not 0 < target <= .2:
                    raise ValueError('Invalid 20-percent completion target')
                if (output / 'CANCEL_SHUTDOWN').exists() or not request.get('armed'):
                    record['state'] = 'cancelled'
                    atomic_json(status_path, record)
                    return
                state = json.loads((output / 'status.json').read_text())
                record['training_state'] = state.get('state')
                completion_path = output / 'completion.json'
                completion = json.loads(completion_path.read_text()) if completion_path.exists() else {}
                if eligible(state, completion, request):
                    # Never race a live training/evaluation/finalization job.
                    with (LOG_ROOT / '.auto_training.lock').open('a') as training_lock:
                        try:
                            fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            record['state'] = 'waiting_for_supervisor_exit'
                        else:
                            # Re-read after acquiring the lock, including cancellation.
                            state = json.loads((output / 'status.json').read_text())
                            request = json.loads(request_path.read_text())
                            completion = json.loads(completion_path.read_text())
                            if (output / 'CANCEL_SHUTDOWN').exists() or not eligible(state, completion, request):
                                continue
                            record['state'] = 'rechecking_full_delivery_before_poweroff'
                            atomic_json(status_path, record)
                            command = [sys.executable, str(ROOT / 'mine_model/scripts/complete_mine_training.py'),
                                       '--state', str(output / 'status.json'),
                                       '--require-leg-control-mode', 'explicit_vmc', '--shutdown']
                            with (output / 'completion_watch.log').open('a') as stream:
                                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                            # A failed artifact/acceptance gate must not retry poweroff
                            # in a tight loop or turn a failed run into an accepted one.
                            record.update(state='finalizer_returned' if result.returncode == 0 else 'finalizer_failed',
                                          finalizer_exit_code=result.returncode)
                            latest_completion=json.loads(completion_path.read_text())
                            record['poweroff_command_submitted']=bool(
                                result.returncode==0 and latest_completion.get('shutdown_requested')
                                and not latest_completion.get('shutdown_cancelled')
                                and not latest_completion.get('shutdown_error'))
                            atomic_json(status_path, record)
                            if result.returncode:
                                raise RuntimeError('Completion gate failed; see completion_watch.log; no forced shutdown')
                            return
                atomic_json(status_path, record)
            except (OSError, ValueError, KeyError) as error:
                record.update(state='waiting_for_valid_evidence', error=str(error))
                atomic_json(status_path, record)
            time.sleep(args.interval)


if __name__ == '__main__':
    main()
