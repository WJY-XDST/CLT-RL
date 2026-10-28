"""Lightweight owner-aware scheduler for CPU-only recorded-state diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run(args):
    root = Path(__file__).resolve().parents[2]
    script = Path(__file__).with_name("audit_recorded_uniform_policy_response.py")
    pins = {str(path): digest(path) for path in (Path(__file__), script, args.recipe)}
    completed = []
    status = args.production / "recorded195_response_monitor_status.json"
    for prefix in args.prefixes:
        health = args.production / f"policy_update_health_prefix_{prefix:04d}_readonly.json"
        output = args.production / f"all195_recorded_policy_command_response_prefix{prefix:04d}_readonly.json"
        while not health.exists():
            owner = subprocess.run(["systemctl", "--user", "show", args.training_unit,
                "-p", "MainPID", "-p", "ActiveState", "-p", "SubState", "-p", "Result"],
                capture_output=True, text=True, check=True).stdout
            state = dict(line.split("=", 1) for line in owner.splitlines() if "=" in line)
            write(status, dict(state="waiting_for_original_prefix_report", prefix=prefix,
                owner=state, completed_prefixes=completed, optimizer_executed=False,
                training_modified=False, policy_accepted=False))
            if state.get("MainPID") == "0" and state.get("SubState") in ("exited", "dead", "failed"):
                raise RuntimeError("Owned controller ended without its prefix artifact; diagnose, never restart PPO here")
            time.sleep(5.)
        assert all(digest(Path(path)) == value for path, value in pins.items())
        if not output.exists():
            argv = [str(root / "run_python.sh"), str(script), "--recipe", str(args.recipe),
                "--health", str(health), "--source-replays", *map(str, args.source_replays),
                "--output", str(output)]
            env = os.environ.copy()
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
            with (args.production / f"recorded195_response_prefix{prefix:04d}.log").open("x") as stream:
                subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        report = json.loads(output.read_text())
        assert report["verified"] and report["candidate_actual_updates"] == report["source_actual_updates"] + prefix
        assert sum(group["trial_count"] for group in report["groups"]) == 195
        assert not report["training_modified"] and not report["backward_or_optimizer_executed"]
        completed.append(prefix)
        write(status, dict(state="prefix_report_complete", completed_prefixes=completed,
            source_pins=pins, optimizer_executed=False, training_modified=False, policy_accepted=False))
    write(status, dict(state="complete", completed_prefixes=completed, source_pins=pins,
        optimizer_executed=False, training_modified=False, policy_accepted=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--source-replays", nargs="+", type=Path, required=True)
    parser.add_argument("--prefixes", type=int, nargs="+", required=True)
    run(parser.parse_args())
