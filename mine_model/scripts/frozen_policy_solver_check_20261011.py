"""Finite solver regression with unchanged checkpoint, gains and control clock.

No training, configuration edits, asset edits or automatic restart. Each replay
runs in a fresh simulator, and pre-existing results are never overwritten.
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "mine_model/results/frozen_policy_solver_check_20261011"
SOURCE = ROOT / (
    "IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/"
    "2026-10-10_23-23-52_mcu_tgs_vmc_env8192_stand25_to1999/model_800.pt"
)
OVERRIDES = ROOT / (
    "mine_model/results/mcu_standing_restart_20261010/"
    "protected_stops_replay_overrides.json"
)
CASES = ((200, 64), (200, 128), (400, 128), (800, 128))


def main():
    from summarize_mine_trial import assess

    checkpoint_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    asset_overrides = json.loads(OVERRIDES.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        started_at=datetime.now().astimezone().isoformat(),
        checkpoint=str(SOURCE), checkpoint_sha256=checkpoint_hash,
        controller_hz=200, policy_hz=50, seeds=[83, 84, 85],
        seconds_per_condition=20, command_scale=1.,
        cases=[dict(physics_hz=hz, position_iterations=iterations)
               for hz, iterations in CASES],
        asset_overrides=asset_overrides,
        policy_training_phase="iteration 800: standing-only curriculum until 1000",
        boundary="Frozen-policy numerical diagnosis, not acceptance of untrained motion",
    )
    manifest_path = OUTPUT / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if any(saved[key] != manifest[key] for key in manifest if key != "started_at"):
            raise RuntimeError("Existing regression has a different contract")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "wheel_legged_isaaclab"),
                       PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
    reports = []
    for hz, iterations in CASES:
        name = f"physics{hz}_control200_policy50_p{iterations}"
        directory = OUTPUT / name
        if not (directory / "completed.json").exists():
            log = OUTPUT / (name + ".log")
            if log.exists() or directory.exists():
                raise RuntimeError("Inspect incomplete replay before retrying: " + name)
            path = OUTPUT / (name + "_overrides.json")
            overrides = dict(asset_overrides)
            overrides["robot.spawn.articulation_props.solver_position_iteration_count"] = iterations
            path.write_text(json.dumps(overrides, indent=2) + "\n")
            command = [str(ROOT / "run_python.sh"),
                       str(ROOT / "wheel_legged_isaaclab/scripts/evaluate_mine_policy.py"),
                       "--checkpoint", str(SOURCE), "--output", str(directory),
                       "--overrides_json", str(path), "--seconds", "20",
                       "--seeds", "83", "84", "85", "--physics-hz", str(hz),
                       "--control-hz", "200", "--record_policy_inputs",
                       "--headless", "--device", "cuda:0"]
            print("FROZEN_REPLAY_START " + name, flush=True)
            with log.open("x") as stream:
                subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                               stderr=subprocess.STDOUT, timeout=420, check=True)
        report = assess(directory)
        standing = next(c for c in report["conditions"] if c["name"] == "stand")
        result = dict(name=name, physics_hz=hz, position_iterations=iterations,
                      checkpoint_sha256=report["checkpoint_sha256"],
                      survivors=report["survivors"], total_trials=report["total_trials"],
                      resets=report["resets"], all_tracking_passed=report["passed"],
                      standing=standing, assessment=str(directory / "assessment.json"))
        reports.append(result)
        print("FROZEN_REPLAY_RESULT " + json.dumps(result), flush=True)
        if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != checkpoint_hash:
            raise RuntimeError("Checkpoint changed during frozen regression")
    (OUTPUT / "summary.json").write_text(json.dumps(dict(
        completed=True, checkpoint_unchanged=True, reports=reports,
        training_started=False, production_config_modified=False,
    ), indent=2) + "\n")
    print("FROZEN_SOLVER_REGRESSION_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        print("Frozen regression aborted; training remains stopped.", file=sys.stderr, flush=True)
        raise
