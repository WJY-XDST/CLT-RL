"""Read stage-relative coverage after the owned original50/100/200 readers.

This lightweight waiter imports no simulator or policy library. It invokes the
unchanged stage reader after each immutable original report is complete, and
never starts, stops, restarts, selects, or accepts training.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
READER = SCRIPT.with_name("audit_stage_normalized_training_yaw.py")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def owner(unit):
    result = subprocess.run(["systemctl", "--user", "show", unit,
        "--property=MainPID,SubState,Result"], capture_output=True, text=True, check=True)
    return dict(row.split("=", 1) for row in result.stdout.splitlines() if "=" in row)


def run(output, unit):
    output = output.resolve()
    assert output.parent == ROOT / "mine_model/results"
    plan = read(output / "experiment_plan.json")
    assert plan["bounded_training_updates"] == 200
    assert plan["native_num_envs"] in (4096, 8192)
    assert plan["ordinary_full_policy_PPO"] and not plan["auxiliary_losses_enabled"]
    assert plan["training_config"]["leg_control_mode"] == "implicit_joint_position"
    pins = {str(p): sha(p) for p in [SCRIPT, READER, output / "experiment_plan.json"]}
    completed = []
    for prefix in (50, 100, 200):
        source = output / f"sampled_yaw_reference_components_prefix_{prefix:04d}_readonly.json"
        while not source.exists():
            state = owner(unit)
            assert state["MainPID"] != "0", (
                "Owned controller ended before the required original report; inspect without restarting PPO", state)
            write(output / "stage_diagnostics_readonly_status.json", dict(
                observed_at=datetime.now().astimezone().isoformat(), state="verified_wait_for_original_report",
                waiting_prefix=prefix, completed_prefixes=completed, live_owner=state,
                output=str(output), unit=unit, training_modified=False, policy_accepted=False))
            time.sleep(15.)
        assert all(sha(path) == digest for path, digest in pins.items())
        destination = output / f"stage_relative_training_yaw_prefix_{prefix:04d}_readonly.json"
        if not destination.exists():
            with (output / f"stage_relative_training_yaw_prefix_{prefix:04d}.log").open("x") as log:
                subprocess.run([sys.executable, str(READER), str(output), "--through", str(prefix),
                    "--output", str(destination)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        report = read(destination)
        assert report["verified"] and report["through"] == prefix
        assert report["actual_num_envs"] == plan["native_num_envs"]
        assert not report["masks_enter_training"] and not report["training_modified"]
        assert not report["optimizer_executed"] and not report["acceptance_changed"]
        assert all(sha(path) == digest for path, digest in report["source_pins"].items())
        completed.append(prefix)
        print(json.dumps(dict(verified=True, completed_prefix=prefix, report=str(destination),
                              training_modified=False, policy_accepted=False)), flush=True)
    write(output / "stage_diagnostics_readonly_complete.json", dict(
        verified=True, observed_at=datetime.now().astimezone().isoformat(), completed_prefixes=completed,
        source_pins=pins, original_reader_unmodified=True, masks_enter_training=False,
        training_modified=False, optimizer_executed=False, policy_accepted=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    try:
        run(args.output, args.unit)
    except BaseException as error:
        write(args.output / "stage_diagnostics_readonly_inspection_required.json", dict(
            observed_at=datetime.now().astimezone().isoformat(), error=repr(error),
            training_modified=False, PPO_not_automatically_restarted=True, policy_accepted=False))
        raise
