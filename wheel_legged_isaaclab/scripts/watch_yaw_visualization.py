"""Run one finite GUI yaw comparison after the currently recorded training job.

This companion lives outside scripts/rsl_rl so arming it does not change the
running optimizer's frozen simulation sources. It pauses only the verified
optimizer supervisor, after its training child has exited normally, and lets
any already-started evaluation finish before opening the GUI.
"""

import argparse
import csv
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
from assess_yaw import assess_yaw, yaw_sequence
from monitor_training import inspect_log
from optimize_training import source_hashes, write_json, sha256


def identity(pid):
    try:
        proc = Path(f"/proc/{pid}")
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {"start": fields[19], "argv": (proc / "cmdline").read_bytes().split(b"\0")[:-1]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def children(pid):
    try:
        return [int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
                if identity(int(value)) is not None]
    except FileNotFoundError:
        return []


def completed_training(checkpoint, log, training_pid, expected_identity):
    if identity(training_pid) == expected_identity:
        return False
    health = inspect_log(log)
    return (checkpoint.is_file() and health.get("iteration", -1) >= health.get("total_iterations", 1) - 1
            and not health.get("unhealthy", False))


def save_comparison(trace, baseline, directory):
    candidate = assess_yaw(trace)
    reference = assess_yaw(baseline)
    report = {"candidate": candidate, "baseline5740": reference,
              "comparison_scope": "same seed 53; one GUI yaw matrix, not multi-seed final acceptance"}
    write_json(directory / "comparison.json", report)
    lines = ["# 本轮训练后的 yaw 可视化对比", "",
             "相同 seed 53、指令和初始扰动。误差取各阶段后两秒；图中保留切换瞬态。", "",
             "| 工况 vx / yaw rate | 5740 yaw P95相对误差 | 本轮 yaw P95相对误差 | 本轮速度MAE |",
             "| --- | --- | --- | --- |"]
    for old, new in zip(reference["phases"], candidate["phases"]):
        if new["yaw"]:
            lines.append(f"| {new['speed']:+.1f} m/s / {new['yaw']:+.1f} rad/s | "
                         f"{old['ratios']['yaw']*5:.2f}% | {new['ratios']['yaw']*5:.2f}% | {new['speed_mae']:.4f} m/s |")
    lines += ["", f"本轮重置次数：{candidate['resets']}；本次矩阵全部指标通过：{candidate['passed']}。",
              "非零 yaw rate 使用 5% 门槛；保向阶段使用 0.03 rad/s 绝对门槛。",
              "原地转检查水平漂移和两轮平均表面速度各 <0.02 m/s；姿态和腿摆角差沿用绝对 3°。",
              "累计转角参考为角速度指令积分，不是额外输入的 yaw 角目标。此单种子测试不替代正式验收。"]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def load(path):
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}

    data, old = load(trace), load(baseline)
    report["turn_angle_error_deg"] = {"final": float(data["yaw_turn_error_deg"][-1]),
                                      "max_abs": float(np.abs(data["yaw_turn_error_deg"]).max())}
    write_json(directory / "comparison.json", report)
    with (directory / "REPORT.md").open("a") as stream:
        stream.write(f"\n累计转角相对指令积分的末尾偏差：{report['turn_angle_error_deg']['final']:.2f}°；"
                     f"全程绝对峰值：{report['turn_angle_error_deg']['max_abs']:.2f}°。\n")
    t = data["sim_time_s"]
    reset = (data["terminated"] + data["time_out"]) > 0
    fig, axes = plt.subplots(6, 1, figsize=(15, 17), sharex=True, constrained_layout=True)

    def line(ax, values, label, **kwargs):
        values = values.copy()
        values[reset] = np.nan
        ax.plot(t, values, label=label, linewidth=1, **kwargs)

    line(axes[0], data["cmd_yaw"], "Command", color="black", linestyle="--")
    line(axes[0], old["yaw_rate_body"], "5740 feedback", color="gray")
    line(axes[0], data["yaw_rate_body"], "New feedback", color="tab:blue")
    axes[0].set_ylabel("Yaw rate (rad/s)")
    rate_error = np.where(data["heading_hold"] > 0, data["yaw_rate_body"], data["yaw_rate_body"] - data["cmd_yaw"])
    line(axes[1], rate_error, "Rate error / residual in hold", color="tab:red")
    band = np.where(data["heading_hold"] > 0, .03, .05 * np.abs(data["cmd_yaw"]))
    axes[1].fill_between(t, -band, band, alpha=.15, color="green", label="5% turning / 0.03 holding")
    axes[1].set_ylabel("Yaw error (rad/s)")
    for key, label in (("yaw_turn_reference_deg", "Integrated command"), ("yaw_turn_measured_deg", "Measured turn")):
        line(axes[2], data[key], label)
    line(axes[2], data["yaw_turn_error_deg"], "Turn error", linestyle="--", color="tab:red")
    axes[2].set_ylabel("Yaw turn (deg)")
    line(axes[3], data["cmd_x_target"], "Speed target", color="black", linestyle="--")
    line(axes[3], data["vel_x_heading"], "Forward speed")
    line(axes[3], data["vel_y_heading"], "Lateral speed")
    line(axes[3], np.hypot(data["vel_x_heading"], data["vel_y_heading"]), "Horizontal speed", color="gray")
    axes[3].set_ylabel("Speed / drift (m/s)")
    line(axes[4], data["wheel_vel_left"], "Left wheel")
    line(axes[4], data["wheel_vel_right"], "Right wheel")
    axes[4].set_ylabel("Wheel speed (rad/s)")
    line(axes[5], np.degrees(data["pitch_est_rad"]), "Pitch")
    line(axes[5], np.degrees(data["roll_est_rad"]), "Roll")
    line(axes[5], np.degrees(data["theta_left"] - data["theta_right"]), "Leg angle difference")
    axes[5].set_ylabel("Body / legs (deg)")
    for ax in axes:
        for when in range(5, 75, 5):
            ax.axvline(when, color="gray", alpha=.2, linewidth=.7)
        for when in t[reset]:
            ax.axvline(when, color="red", linewidth=1.3)
        ax.grid(alpha=.2)
        ax.legend(loc="upper left", ncol=4, fontsize=8)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle(f"Yaw comparison after training | seed 53 | resets={candidate['resets']} | passed={candidate['passed']}")
    fig.savefig(directory / "yaw_comparison.png", dpi=140)
    plt.close(fig)
    return candidate


def main():
    def terminate(signum, frame):
        raise InterruptedError(f"Watcher received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--check", action="store_true", help="Validate the active target without scheduling or signaling anything")
    args = parser.parse_args()
    state_dir = args.state_dir.resolve()
    initial = json.loads((state_dir / "state.json").read_text())
    if initial["state"] != "training":
        raise RuntimeError("Arm this watcher while the intended training round is active")
    checkpoint, log = Path(initial["expected_checkpoint"]), Path(initial["log"])
    training_pid, supervisor_pid = initial["training_pid"], initial["pid"]
    training_identity, supervisor_identity = identity(training_pid), identity(supervisor_pid)
    if training_identity is None or initial["run_name"].encode() not in training_identity["argv"]:
        raise RuntimeError("Training process identity does not match")
    if supervisor_identity is None or str(state_dir).encode() not in supervisor_identity["argv"]:
        raise RuntimeError("Optimizer identity does not match")
    if source_hashes() != initial["source_hashes"]:
        raise RuntimeError("Simulation sources changed while training")
    baseline = state_dir / "baseline5740_yaw/seed53.csv"
    assess_yaw(baseline)
    directory = state_dir / f"visual_after_round{initial['round']:04d}"
    if args.check:
        print(json.dumps({"checkpoint": str(checkpoint), "output": str(directory), "training_active": True}, indent=2))
        return
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"watcher_pid": os.getpid(), "checkpoint": str(checkpoint), "training_pid": training_pid,
              "supervisor_pid": supervisor_pid, "round": initial["round"]}

    def save(state, **extra):
        status.update(state=state, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), **extra)
        write_json(directory / "watch_state.json", status)

    def check():
        if (directory / "STOP").exists():
            raise InterruptedError("Visualization watcher STOP requested")
        if source_hashes() != initial["source_hashes"]:
            raise RuntimeError("Simulation sources changed; refusing a mismatched replay")

    paused = False
    process = None
    try:
        save("waiting_for_training")
        while identity(training_pid) == training_identity:
            check()
            time.sleep(5)
        if not completed_training(checkpoint, log, training_pid, training_identity):
            raise RuntimeError("Training did not complete normally; no GUI launched")
        check()
        if identity(supervisor_pid) == supervisor_identity:
            os.kill(supervisor_pid, signal.SIGSTOP)
            paused = True
        save("waiting_for_current_evaluation", supervisor_paused=paused)
        deadline = time.monotonic() + 1800
        while paused and children(supervisor_pid):
            check()
            if time.monotonic() > deadline:
                raise RuntimeError("Existing optimizer child did not finish in time")
            time.sleep(2)
        sequence = yaw_sequence()
        trace = directory / "seed53_gui.csv"
        command = [str(ROOT / "play_wheel.sh"), "--device", "cuda:0", "--num_envs", "1", "--seed", "53",
                   "--checkpoint_path", str(checkpoint), "--fixed_command", "0", "0", ".18",
                   "--velocity_cycle", *[str(s[0]) for s in sequence],
                   "--yaw_cycle", *[str(s[1]) for s in sequence],
                   "--height_cycle", *[str(s[2]) for s in sequence],
                   "--max_steps", str(len(sequence)*250), "--trace_csv", str(trace),
                   "--live_stats", "--real-time", "--camera_mode", "orbit",
                   "env.episode_length_s=1000.0", "env.robot.init_state.pos=[0.0,0.0,0.18]",
                   "env.reset_velocity_initial=0.5", "env.reset_velocity_final=0.5"]
        write_json(directory / "command.json", command)
        with (directory / "play.log").open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                       env=dict(os.environ, DISPLAY=":0", XAUTHORITY="/run/user/1000/gdm/Xauthority"))
            save("visualizing", gui_wrapper_pid=process.pid, checkpoint_sha256=sha256(checkpoint))
            deadline = time.monotonic() + 900
            while process.poll() is None:
                check()
                if time.monotonic() > deadline:
                    raise RuntimeError("GUI replay exceeded 15 minutes")
                time.sleep(2)
        if process.returncode:
            raise RuntimeError(f"GUI replay exited {process.returncode}; see play.log")
        save("analyzing")
        report = save_comparison(trace, baseline, directory)
        save("complete", passed=report["passed"], resets=report["resets"], worst_ratios=report["worst_ratios"])
        with (ROOT / "wheel_legged_isaaclab/checkpoints/TRAINING_NOTES.md").open("a") as stream:
            stream.write(f"\n### 训练结束后的 yaw 可视化测试 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                         f"- 模型 `{checkpoint.name}`，seed 53，15 阶段；重置 {report['resets']}，本次矩阵通过={report['passed']}。\n"
                         f"- 数据、对比曲线和误差报告：`{directory}`。该单种子可视化不代替正式多种子验收。\n")
    except BaseException as exc:
        save("needs_attention", error=repr(exc))
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        if paused and identity(supervisor_pid) == supervisor_identity:
            os.kill(supervisor_pid, signal.SIGCONT)
        if paused:
            status["supervisor_paused"] = False
            write_json(directory / "watch_state.json", status)


if __name__ == "__main__":
    main()
