"""Wait for an owned frozen review, then compare every paired trial read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def run(args):
    root = Path(__file__).resolve().parents[2]
    reader = Path(__file__).with_name("compare_uniform_joint_pd_replay_retention.py")
    assert len(args.source) == len(args.scope)
    pins = {str(path): digest(path) for path in (Path(__file__), reader)}
    status = args.production / "paired_retention_monitor_status.json"
    completion = args.production / "independent_review/complete_comparison_agent_decision_required.json"
    while not completion.exists():
        owner = subprocess.run(["systemctl", "--user", "show", args.training_unit,
            "-p", "MainPID", "-p", "ActiveState", "-p", "SubState", "-p", "Result"],
            capture_output=True, text=True, check=True).stdout
        state = dict(line.split("=", 1) for line in owner.splitlines() if "=" in line)
        write(status, dict(state="waiting_for_complete_owned_frozen_review", owner=state,
            optimizer_executed=False, training_modified=False, policy_accepted=False))
        if state.get("MainPID") == "0" and state.get("SubState") in ("exited", "dead", "failed"):
            raise RuntimeError("Owned controller ended without a complete review; diagnose without restarting PPO")
        time.sleep(5.)
    assert all(digest(path) == value for path, value in pins.items())
    review = json.loads(completion.read_text())
    assert review["complete"] and not review["policy_accepted"]
    destination = args.production / "independent_review/all_matched195_failed_and_passing_retention_readonly.json"
    assert not destination.exists()
    candidates = [args.production / "independent_review" / scope for scope in args.scope]
    argv = [str(root / "run_python.sh"), str(reader), "--source", *map(str, args.source),
        "--candidate", *map(str, candidates), "--scope", *args.scope, "--output", str(destination)]
    env = os.environ.copy()
    modules = [root / "wheel_legged_isaaclab", root / "mine_model/scripts", root / "mine_model/experiments"]
    env["PYTHONPATH"] = os.pathsep.join(map(str, modules)) + os.pathsep + env.get("PYTHONPATH", "")
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    with (args.production / "paired_retention_reader.log").open("x") as stream:
        subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
    report = json.loads(destination.read_text())
    assert report["verified"] and not report["training_modified"] and not report["optimizer_executed"]
    assert all(digest(path) == value for path, value in pins.items())
    write(status, dict(state="complete", report=str(destination), source_pins=pins,
        optimizer_executed=False, training_modified=False, policy_accepted=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--training-unit", required=True)
    parser.add_argument("--source", nargs="+", type=Path, required=True)
    parser.add_argument("--scope", nargs="+", required=True)
    run(parser.parse_args())
