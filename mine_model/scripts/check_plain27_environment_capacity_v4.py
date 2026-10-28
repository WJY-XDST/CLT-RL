"""Run a capacity probe in an idle native-simulation slot.

Production200 and its monitor must finish first. The frozen source7001 is
already fully reviewed, so its capacity probes can share the existing GPU
mutex with later7201 replay jobs. Reserve that mutex while measuring host
headroom and launching the diagnostic unit, then release it for the child
to take the same lock through run_python.sh. No replay process is stopped.
"""
import fcntl
import os
from pathlib import Path
import subprocess
import time
import types

import check_plain27_environment_capacity_v3 as retained
import check_plain27_environment_capacity as original

SCRIPT = Path(__file__).resolve()
LOCK = Path(os.environ.get('XDG_RUNTIME_DIR', '/run/user/' + str(os.getuid()))) / (
    'clt-rl-native-simulation-' + str(os.getuid()) + '.lock')
_original_probe = original.run_probe


def run_probe(request, request_path, output, count, policy):
    with LOCK.open('a') as handle:
        locked = False
        while not locked:
            original.intact(request)
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            except BlockingIOError:
                original.write(output / ('native_slot_wait_' + str(count) + '.json'), dict(
                    observed_at=original.now(), state='waiting_for_existing_native_simulation_slot',
                    num_envs=count, production_or_replay_process_stopped=False,
                    capacity_simulation_started=False))
                time.sleep(2.)
        original.write(output / ('native_slot_reservation_' + str(count) + '.json'), dict(
            observed_at=original.now(), state='idle_slot_reserved_before_resource_snapshot',
            num_envs=count, resources=original.resources(), production_or_replay_process_stopped=False))

        def launch_then_release(command, *args, **kwargs):
            nonlocal locked
            result = subprocess.run(command, *args, **kwargs)
            if command and command[0] == 'systemd-run' and locked:
                # Type=exec confirms the launched run_python.sh process.
                # That child uses its own global flock; release ours before
                # monitoring it so the two locks can never deadlock.
                fcntl.flock(handle, fcntl.LOCK_UN)
                locked = False
            return result

        namespace = dict(_original_probe.__globals__,
            subprocess=types.SimpleNamespace(run=launch_then_release))
        probe = types.FunctionType(_original_probe.__code__, namespace,
            _original_probe.__name__, _original_probe.__defaults__, _original_probe.__closure__)
        try:
            return probe(request, request_path, output, count, policy)
        finally:
            if locked:
                fcntl.flock(handle, fcntl.LOCK_UN)


original.run_probe = run_probe
original.SCRIPT = SCRIPT
require_capacity_selection = original.require_capacity_selection
main = original.main


if __name__ == '__main__':
    main()
