"""Monitor PPO health and run a recorded evaluation plan after normal completion.

Uses console episode statistics, not total reward, to detect prolonged collapse.
The process identity is checked before every signal. Completed checkpoints remain
on disk if an unhealthy run is stopped.
"""

import argparse
import json
import re
import signal
import statistics
import subprocess
import time
from pathlib import Path


def inspect_log(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    matches = list(re.finditer(r"Learning iteration (\d+)/(\d+)", text))
    records = []
    for index, match in enumerate(matches):
        block = text[match.end():matches[index + 1].start() if index + 1 < len(matches) else len(text)]
        record = {"iteration": int(match[1]), "total_iterations": int(match[2])}
        for label, key in (
            ("Mean episode length", "episode_steps"),
            ("Episode_Termination/unsafe_contact_or_orientation", "unsafe"),
            ("Episode_Termination/time_out", "timeout"),
        ):
            value = re.search(re.escape(label) + r":\s*([0-9.]+)", block)
            if value:
                record[key] = float(value[1])
        if all(key in record for key in ("episode_steps", "unsafe", "timeout")):
            records.append(record)
    if not records:
        return {"state": "starting", "unhealthy": False}
    result = dict(records[-1], state="training", unhealthy=False)
    if len(records) >= 80:
        recent = records[-80:]
        old_length = statistics.mean(row["episode_steps"] for row in recent[:40])
        new_length = statistics.mean(row["episode_steps"] for row in recent[40:])
        result.update(
            recent_episode_steps=new_length,
            previous_episode_steps=old_length,
            recent_unsafe=statistics.mean(row["unsafe"] for row in recent),
            recent_timeout=statistics.mean(row["timeout"] for row in recent),
        )
        result["unhealthy"] = (
            result["iteration"] >= 300
            and new_length < 75  # below 1.5 s with this task's 50 Hz controller
            and new_length <= old_length * 1.15
            and result["recent_unsafe"] >= 0.99
            and result["recent_timeout"] <= 0.01
        )
    return result


def matches_process(pid, run_name):
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\0")
    except FileNotFoundError:
        return False
    return any(command[i:i + 2] == ["--run_name", run_name] for i in range(len(command) - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--inspect", action="store_true", help="Read log only; never signal a process.")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--final-checkpoint", type=Path)
    parser.add_argument("--status", type=Path)
    parser.add_argument("--evaluation-plan", type=Path, help="JSON list of command/log/cwd entries.")
    args = parser.parse_args()
    if args.inspect:
        print(json.dumps(inspect_log(args.log), indent=2))
        return
    if not all((args.pid, args.run_name, args.final_checkpoint, args.status)):
        parser.error("Monitoring requires --pid, --run-name, --final-checkpoint and --status.")
    args.status.parent.mkdir(parents=True, exist_ok=True)

    def save(status):
        status["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        args.status.write_text(json.dumps(status, indent=2) + "\n")

    while matches_process(args.pid, args.run_name):
        status = inspect_log(args.log) if args.log.exists() else {"state": "starting"}
        save(status)
        if status.get("unhealthy"):
            if matches_process(args.pid, args.run_name):
                import os

                os.kill(args.pid, signal.SIGTERM)
                for _ in range(20):
                    if not matches_process(args.pid, args.run_name):
                        break
                    time.sleep(1)
                if matches_process(args.pid, args.run_name):
                    os.kill(args.pid, signal.SIGKILL)
            status["state"] = "stopped_unhealthy"
            save(status)
            print("Stopped: persistently short episodes and nearly all unsafe terminations.", flush=True)
            return
        time.sleep(30)
    status = inspect_log(args.log)
    if not args.final_checkpoint.exists() or status.get("iteration", -1) < status.get("total_iterations", 1) - 1:
        status["state"] = "exited_before_completion"
        save(status)
        return
    status["state"] = "training_complete"
    save(status)
    if args.evaluation_plan:
        for index, job in enumerate(json.loads(args.evaluation_plan.read_text())):
            status.update(state="evaluating", evaluation_job=index)
            save(status)
            log = Path(job["log"])
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("w") as stream:
                result = subprocess.run(job["command"], cwd=job["cwd"], stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                status.update(state="evaluation_error", returncode=result.returncode)
                save(status)
                return
        status["state"] = "evaluation_complete_needs_review"
        save(status)


if __name__ == "__main__":
    main()
