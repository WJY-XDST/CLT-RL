"""Stop only the identified training process after its target checkpoint is saved."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument("--pid", type=int, required=True)
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--iteration", type=int, required=True)
args = parser.parse_args()
proc = Path(f"/proc/{args.pid}")
identity = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
command = (proc / "cmdline").read_bytes()
if b"scripts/rsl_rl/train.py" not in command or b"mine_wheel_legged_vmc_flat" not in command:
    raise RuntimeError("PID does not identify the requested mine training")

def alive():
    try:
        return (proc / "stat").read_text().rsplit(")", 1)[1].split()[19] == identity
    except FileNotFoundError:
        return False

target = args.run / f"model_{args.iteration}.pt"
status = args.run / "trial_stop_status.json"
status.write_text(json.dumps({"state": "waiting", "target_checkpoint": str(target), "pid": args.pid}, indent=2))
print(f"Watching PID {args.pid}; stop after {target}", flush=True)
while alive():
    if target.exists() and target.stat().st_size > 0:
        before = target.stat()
        time.sleep(2)
        after = target.stat()
        if (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns):
            # The periodic checkpoint is already complete; no progress is lost.
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                if not alive():
                    break
                os.kill(args.pid, sig)
                for _ in range(20):
                    if not alive():
                        break
                    time.sleep(.5)
            status.write_text(json.dumps({"state": "stopped_at_checkpoint", "target_checkpoint": str(target), "pid": args.pid}, indent=2))
            print(f"Stopped training after saving {target}", flush=True)
            break
    time.sleep(5)
else:
    status.write_text(json.dumps({"state": "process_exited_before_target", "target_checkpoint": str(target), "pid": args.pid}, indent=2))
    print("Training exited before target checkpoint", flush=True)
