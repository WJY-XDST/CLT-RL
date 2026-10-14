"""Resume bounded full-policy PPO only after the frozen solver tests finish.

The controller watches its own child, stops on physical/numerical regression,
and frees training GPU memory before each checkpoint replay. It never retunes
rewards, changes hardware data, exports to hardware, or shuts down the host.
"""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from guard_mcu_vmc_training import anomaly, completed_updates, replay_gate

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "mine_model/results/vmc_p128_guarded_resume_20261011"
REGRESSION = ROOT / "mine_model/results/frozen_policy_solver_check_20261011"
QUALIFICATION = ROOT / "mine_model/results/tgs_numerical_convergence_20261011/qualification_p128_v2"
SOURCE = ROOT / (
    "IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/"
    "2026-10-10_23-23-52_mcu_tgs_vmc_env8192_stand25_to1999/model_800.pt"
)
LOG_ROOT = ROOT / "IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat"
ASSET_OVERRIDES = ROOT / "mine_model/results/mcu_standing_restart_20261010/protected_stops_replay_overrides.json"
OLD_UNIT = "clt-vmc8192-stand25-20261010.service"
MILESTONES = (1000, 1500, 1998)


def save(state, **values):
    data = dict(state=state, updated_at=datetime.now().astimezone().isoformat(), **values)
    path = OUTPUT / "status.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
    print(json.dumps(data, ensure_ascii=False), flush=True)


def qualified(qualification, strict, regression, checkpoint_hash):
    expected = dict(physics_hz=200, control_hz=200, policy_hz=50,
                    position_iterations=128, velocity_iterations=8,
                    external_forces_every_iteration=True)
    if not qualification.get("training_permitted") or qualification.get("candidate") != expected:
        return "fixed_controller_qualification_not_passed"
    if not qualification.get("checks") or not all(
        all(checks.values()) for checks in qualification["checks"].values()
    ):
        return "mechanical_or_control_clock_check_failed"
    if not strict.get("comparisons") or not all(c.get("passed") for c in strict["comparisons"]):
        return "fixed_controller_waveform_convergence_failed"
    # Retain the original strict failure. Only the separately investigated FP32
    # gravity summation residual may use the supplemental qualification.
    failures = [(name, check) for name, case in strict["per_case"].items()
                for check, passed in case["checks"].items() if not passed]
    if failures and failures != [("true_static_800hz_p128", "gravity_velocity")]:
        return "unexplained_strict_numerical_failure"
    gravity = qualification.get("gravity_accumulation", {})
    if not gravity or any(not g.get("passed") or g["fp32_model_max_difference_m_s"] != 0.
                          for g in gravity.values()):
        return "gravity_accumulation_diagnosis_not_verified"
    if not regression.get("completed") or not regression.get("checkpoint_unchanged"):
        return "frozen_policy_regression_incomplete"
    reports = regression.get("reports", [])
    expected_pairs = {(200, 64), (200, 128), (400, 128), (800, 128)}
    if {(r["physics_hz"], r["position_iterations"]) for r in reports} != expected_pairs:
        return "frozen_policy_regression_cases_missing"
    for report in reports:
        if report["checkpoint_sha256"] != checkpoint_hash:
            return "frozen_policy_checkpoint_mismatch"
        # Early full-motion precision is not an admission gate: iteration 800
        # was trained on standing only. Survival and standing balance still gate.
        if replay_gate(dict(conditions=[report["standing"]],
                            survivors=report["survivors"], total_trials=report["total_trials"])):
            return "frozen_policy_support_regression"
    return None


def training_command(checkpoint, milestone, label, overrides):
    command = [str(ROOT / "run_python.sh"),
               str(ROOT / "wheel_legged_isaaclab/scripts/rsl_rl/train.py"),
               "--task", "WheelLeggedVMC-Flat-v0", "--require_leg_control_mode", "explicit_vmc",
               "--checkpoint_path", str(checkpoint), "--num_envs", "8192",
               "--seed", "43", "--target_iteration", str(milestone + 1),
               "--run_name", label, "--headless", "--device", "cuda:0"]
    command.extend("env." + key + "=" + json.dumps(value) for key, value in overrides.items())
    return command


def stop_child(child):
    # A launcher shell can exit while its native Python child still owns CUDA.
    # Remember group ownership at creation; do not rely on the wrapper's poll.
    group = getattr(child, '_owned_process_group_id', None)
    if group != child.pid:
        raise RuntimeError("Refusing to signal an unowned process group")
    for sig, wait in ((signal.SIGINT, 15), (signal.SIGTERM, 10), (signal.SIGKILL, 10)):
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            child.poll()
            return
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            child.poll()  # Reap the wrapper, but still check native descendants.
            try:
                os.killpg(group, 0)
            except ProcessLookupError:
                return
            time.sleep(.2)
    raise RuntimeError('Owned native process group did not exit after signal ladder')


def run_child(command, log, environment, training=False, reporter=None, cooperative_stop=False):
    with log.open("x") as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        if os.getpgid(child.pid) != child.pid:
            raise RuntimeError('Child did not create its own process group')
        child._owned_process_group_id = child.pid
        previous_iteration = None
        last_progress = time.monotonic()
        started = last_progress
        stop_requested_at = None
        try:
            while child.poll() is None:
                if (OUTPUT / "STOP").exists():
                    if not (training and cooperative_stop):
                        raise RuntimeError("user_stop_requested")
                    if stop_requested_at is None:
                        stop_requested_at = time.monotonic()
                        (reporter or save)('saving_checkpoint_before_stop', child_pid=child.pid, log=str(log))
                    if time.monotonic() - stop_requested_at > 120:
                        raise RuntimeError('cooperative_checkpoint_stop_exceeded_120_seconds')
                content = log.read_text(errors="replace")
                updates = completed_updates(content)
                if training:
                    reason = anomaly(updates, content)
                    if reason:
                        if cooperative_stop:
                            marker = OUTPUT / 'STOP'
                            if not marker.exists():
                                marker.write_text(json.dumps(dict(reason='agent_diagnosis_'+reason,
                                    checkpoint_before_exit=True))+'\n')
                            if stop_requested_at is None:
                                stop_requested_at = time.monotonic()
                                (reporter or save)('saving_checkpoint_for_diagnosis', reason=reason,
                                                   child_pid=child.pid, log=str(log))
                        else:
                            raise RuntimeError(reason)
                    latest = updates[-1]["iteration"] if updates else None
                    if latest != previous_iteration:
                        last_progress = time.monotonic()
                        previous_iteration = latest
                        (reporter or save)("training", child_pid=child.pid, log=str(log),
                             latest_update=updates[-1] if updates else None,
                             full_policy_training=True, num_envs=8192)
                    # Initial 8192-robot asset loading is slower than one update.
                    if updates and time.monotonic() - last_progress > 600:
                        raise RuntimeError("training_stalled_600_seconds")
                    if not updates and time.monotonic() - started > 900:
                        raise RuntimeError("training_initialization_exceeded_900_seconds")
                elif time.monotonic() - started > 420:
                    raise RuntimeError("checkpoint_replay_exceeded_420_seconds")
                time.sleep(5)
            if child.returncode:
                raise RuntimeError("child_failed_exit_" + str(child.returncode))
            if training:
                content = log.read_text(errors="replace")
                reason = anomaly(completed_updates(content), content)
                if reason:
                    raise RuntimeError(reason)
            if (OUTPUT / 'STOP').exists():
                raise RuntimeError('user_stop_requested')
        finally:
            stop_child(child)


def main():
    from summarize_mine_trial import assess

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / ".supervisor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / "launch.json").exists():
            raise RuntimeError("Inspect the existing launch instead of starting a duplicate")
        source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        deadline = time.monotonic() + 1800
        save("waiting_for_frozen_solver_regression", source_checkpoint=str(SOURCE),
             source_iteration=800, scheduled_milestones=list(MILESTONES),
             weights_updating=False)
        while not (REGRESSION / "summary.json").exists():
            if (OUTPUT / "STOP").exists():
                raise RuntimeError("user_stop_requested_before_training")
            if time.monotonic() > deadline:
                raise RuntimeError("frozen_solver_regression_not_completed_within_1800_seconds")
            alive = subprocess.run(["pgrep", "-f", "^[^ ]*python.*frozen_policy_solver_check_20261011[.]py"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if alive.returncode:
                raise RuntimeError("frozen_solver_regression_exited_without_complete_summary")
            time.sleep(10)
        reason = qualified(json.loads((QUALIFICATION / "qualified_training_screen.json").read_text()),
                           json.loads((QUALIFICATION / "assessment.json").read_text()),
                           json.loads((REGRESSION / "summary.json").read_text()), source_hash)
        if reason:
            raise RuntimeError(reason)
        state = subprocess.check_output(["systemctl", "--user", "show", OLD_UNIT,
                                         "-p", "ActiveState", "--value"], text=True).strip()
        if state == "active":
            raise RuntimeError("old_training_service_still_active")
        other = subprocess.run(["pgrep", "-f", "^[^ ]*python.*scripts/rsl_rl/train[.]py"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if other.returncode == 0:
            raise RuntimeError("another_training_process_is_active")
        overrides = json.loads(ASSET_OVERRIDES.read_text())
        overrides.update({
            "robot.spawn.articulation_props.solver_position_iteration_count": 128,
            "robot.spawn.articulation_props.solver_velocity_iteration_count": 8,
            "enable_external_forces_every_iteration": True,
            "commands.fixed_standing_groups": True,
            "commands.standing_env_fraction": .25,
        })
        overrides_path = OUTPUT / "replay_overrides.json"
        overrides_path.write_text(json.dumps(overrides, indent=2) + "\n")
        (OUTPUT / "launch.json").write_text(json.dumps(dict(
            source_checkpoint=str(SOURCE), source_sha256=source_hash,
            milestones=list(MILESTONES), num_envs=8192, overrides=overrides,
            changed_physics_parameter="position solver iterations 64 -> 128",
            rewards_changed=False, hardware_changed=False,
            full_policy_training=True, vmc_hz=200, policy_hz=50,
            strict_qualification_passed=False,
            supplemental_qualification=str(QUALIFICATION / "qualified_training_screen.json"),
            frozen_policy_regression=str(REGRESSION / "summary.json"),
        ), indent=2) + "\n")
        environment = dict(os.environ, PYTHONPATH=str(ROOT / "wheel_legged_isaaclab"),
                           PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1")
        checkpoint = SOURCE
        for milestone in MILESTONES:
            segment = OUTPUT / ("to_" + str(milestone))
            segment.mkdir()
            label = "vmc_p128_guarded_to_" + str(milestone) + "_20261011"
            if list(LOG_ROOT.glob("*_" + label)):
                raise RuntimeError("Existing training run label: " + label)
            command = training_command(checkpoint, milestone, label, overrides)
            (segment / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            save("training_initialization", source_checkpoint=str(checkpoint),
                 milestone=milestone, log=str(segment / "training.log"))
            run_child(command, segment / "training.log", environment, training=True)
            runs = list(LOG_ROOT.glob("*_" + label))
            if len(runs) != 1:
                raise RuntimeError("Cannot uniquely identify training output")
            checkpoint = runs[0] / ("model_" + str(milestone) + ".pt")
            if not checkpoint.exists():
                raise RuntimeError("Training exited without its target checkpoint")
            # The PPO child has exited, so its GPU allocation is released before
            # a new simulator is launched. No competing frozen 8192-robot process.
            save("checkpoint_evaluation", checkpoint=str(checkpoint), milestone=milestone)
            replay = segment / "replay"
            scale = .25 if milestone <= 1000 else (.5 if milestone <= 1500 else 1.)
            command = [str(ROOT / "run_python.sh"),
                       str(ROOT / "wheel_legged_isaaclab/scripts/evaluate_mine_policy.py"),
                       "--checkpoint", str(checkpoint), "--overrides_json", str(overrides_path),
                       "--output", str(replay), "--seconds", "30", "--seeds", "83", "84", "85",
                       "--command-scale", str(scale), "--record_policy_inputs", "--headless", "--device", "cuda:0"]
            run_child(command, segment / "evaluation.log", environment)
            assessment = assess(replay)
            reason = replay_gate(assessment)
            save("checkpoint_evaluated", checkpoint=str(checkpoint), milestone=milestone,
                 survivors=assessment["survivors"], total_trials=assessment["total_trials"],
                 tracking_passed=assessment["passed"], command_scale=scale,
                 assessment=str(replay / "assessment.json"), physical_stop_reason=reason)
            if reason:
                raise RuntimeError(reason)
        save("bounded_training_and_replay_complete", checkpoint=str(checkpoint),
             assessment=str(replay / "assessment.json"), tracking_passed=assessment["passed"],
             all_errors_under_10_percent=assessment["passed"],
             next_action="accepted_simulation_only" if assessment["passed"] else "inspect_full_command_errors_before_next_optimization")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            save("stopped_for_diagnosis", reason=str(error), automatic_restart=False)
        raise
