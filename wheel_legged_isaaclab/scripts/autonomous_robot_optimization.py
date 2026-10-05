"""Unattended measured optimization of wheel-leg startup, tracking and heading.

Lives outside the simulator source lock. Takes over an existing supervisor
without interrupting its current training, then uses explicit gain/reward
overrides for every training and replay. GUI tests precede each next trial.
"""

import argparse
import csv
import json
import math
import os
import signal
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
from analyze_response import settling_time
from assess_tracking import percentile
from assess_yaw import yaw_sequence
from monitor_training import inspect_log, matches_process
from optimize_training import ROOT, REPLAY_OVERRIDES, write_json, sha256
from optimize_yaw import ACCEPTED, YawOptimizer
from watch_yaw_visualization import identity, children

GAIN_CHOICES = ((50., 6.), (60., 8.), (70., 9.), (55., 7.), (50., 8.))
STARTUP_LIMITS = {"nominal_settle_s": .4, "perturbed_settle_s": 1.,
                  "nominal_peak_deg": 2., "perturbed_peak_deg": 10.,
                  "nominal_variation_deg": 5., "perturbed_variation_deg": 25.}


def read_trace(path, count):
    with Path(path).open(newline="") as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if len(rows) != count:
        raise ValueError(f"Incomplete trace: {path}: {len(rows)} != {count}")
    for i, row in enumerate(rows):
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError("Nonfinite replay sample")
        if row["step"] != i or abs(row["sim_time_s"] - i * .02) > 1e-5 or row["env_id"] != 0:
            raise ValueError("Missing, mistimed or mixed-environment sample")
        if any(row[k] not in (0, 1) for k in ("terminated", "time_out")):
            raise ValueError("Invalid reset flag")
    return rows


def startup_report(path, perturbation):
    rows = read_trace(path, 250)
    if any(abs(r["cmd_x_target"]) > 1e-6 or abs(r["height_cmd"] - .18) > 1e-5 for r in rows):
        raise ValueError("Startup trial requires standing at height 0.18")
    disturbed = perturbation > 0
    settle_limit = STARTUP_LIMITS["perturbed_settle_s" if disturbed else "nominal_settle_s"]
    peak_limit = STARTUP_LIMITS["perturbed_peak_deg" if disturbed else "nominal_peak_deg"]
    variation_limit = STARTUP_LIMITS["perturbed_variation_deg" if disturbed else "nominal_variation_deg"]
    settling, peaks = [], []
    for key in ("theta_left", "theta_right"):
        angle = [math.degrees(r[key]) for r in rows]
        steady = statistics.mean(angle[-100:])
        settling.append(settling_time([abs(v - steady) for v in angle], .5))
        peaks.append(max(map(abs, angle)))
    mean_angle = [math.degrees((r["theta_left"] + r["theta_right"]) / 2) for r in rows]
    variation = sum(abs(mean_angle[i] - mean_angle[i - 1]) for i in range(1, 100))
    resets = sum(r["terminated"] + r["time_out"] for r in rows)
    ratios = {"settling": max(t / settle_limit if t is not None else 10. for t in settling),
              "peak": max(peaks) / peak_limit, "variation": variation / variation_limit,
              "speed": percentile([math.hypot(r["vel_x_heading"], r["vel_y_heading"]) for r in rows[-100:]]) / .02,
              "body": max(math.degrees(max(abs(r["pitch_est_rad"]), abs(r["roll_est_rad"]))) for r in rows) / 10.,
              "symmetry": max(math.degrees(abs(r["theta_left"] - r["theta_right"])) for r in rows[-100:]) / 1.}
    return {"trace": str(path), "perturbation": perturbation, "settling_s": settling,
            "peak_leg_angle_deg": peaks, "variation_deg": variation, "ratios": ratios,
            "resets": resets, "passed": resets == 0 and all(v <= 1 for v in ratios.values())}


def heading_sequence(extended=False):
    if not extended:
        return yaw_sequence()
    sequence = [(0., 0., .18)]
    for height in (.16, .18, .20):
        for speed in (0., .8, -.8):
            for rate in (.3, -.3):
                sequence.extend(((speed, rate, height), (speed, 0., height)))
    return sequence


def heading_report(path, extended=False):
    sequence = heading_sequence(extended)
    rows = read_trace(path, len(sequence) * 250)
    reference, errors = 0., []
    for i, row in enumerate(rows):
        speed, rate, height = sequence[i // 250]
        if row.get("heading_feedback") != 1 or row.get("heading_hold") != 0:
            raise ValueError("Heading feedback missing or overwritten")
        if (abs(row["yaw_rate_requested"] - rate) > 1e-5 or
                abs(row["cmd_x_target"] - speed) > 1e-5 or abs(row["height_cmd"] - height) > 1e-5):
            raise ValueError("Incorrect heading test commands")
        reference += math.degrees(rate * .02)
        if abs(row["yaw_turn_reference_deg"] - reference) > 1e-4 or abs(row["yaw_rate_applied"]) > .500001:
            raise ValueError("Heading reference shifted or corrected rate exceeded learned bounds")
        errors.append(row["yaw_turn_measured_deg"] - reference)
    phase_p95 = [percentile([abs(v) for v in errors[i * 250 + 150:(i + 1) * 250]])
                 for i in range(len(sequence))]
    resets = sum(r["terminated"] + r["time_out"] for r in rows)
    ratios = {"steady": max(phase_p95) / 1., "peak": max(map(abs, errors)) / 3.}
    return {"trace": str(path), "resets": resets, "phase_p95_deg": phase_p95,
            "final_error_deg": errors[-1], "max_error_deg": max(map(abs, errors)),
            "ratios": ratios, "passed": resets == 0 and all(v < 1 for v in ratios.values()),
            "extended": extended}


def bundle_reports(straight, yaw, startup, heading):
    ratios = {**{f"straight_{k}": v for k, v in straight["worst_ratios"].items()},
              **{f"yaw_{k}": v for k, v in yaw["worst_ratios"].items()},
              "startup": startup["score"], "heading": heading["score"]}
    reports = (straight, yaw, startup, heading)
    return {"straight": straight, "yaw": yaw, "startup": startup, "heading": heading,
            "worst_ratios": ratios, "score": max(ratios.values()),
            "resets": sum(r["resets"] for r in reports),
            "passed": all(r["passed"] for r in reports),
            "traces": [t for r in reports for t in r["traces"]]}


def can_promote(candidate, incumbent):
    # Strong straight protection applies even to a provisional improvement.
    if candidate["resets"] or not candidate["straight"]["passed"]:
        return False
    for task in ("straight", "yaw"):
        old, new = incumbent[task]["traces"], candidate[task]["traces"]
        if len(old) != len(new):
            raise ValueError("Cannot compare incomplete or different replay suites")
        for before, after in zip(old, new):
            if len(before["phases"]) != len(after["phases"]):
                raise ValueError("Replay phase count changed")
            for a, b in zip(before["phases"], after["phases"]):
                if (a["speed"], a["height"], a.get("yaw")) != (b["speed"], b["height"], b.get("yaw")):
                    raise ValueError("Replay phase commands changed")
                if any(b["ratios"][k] > max(1., v * 1.10) for k, v in a["ratios"].items()):
                    return False
    safe = all(v <= max(1., incumbent["worst_ratios"][k] * 1.10)
               for k, v in candidate["worst_ratios"].items())
    return safe and (candidate["passed"] or candidate["score"] < incumbent["score"] * .90)


def next_plan(rewards, gains, report, attempts):
    changed, angular_gains = dict(rewards), gains
    ranked = sorted(report["worst_ratios"], key=report["worst_ratios"].get, reverse=True)
    failed = [k for k in ranked if report["worst_ratios"][k] >= 1] or ranked
    # Rotate when the same objective keeps failing; do not repeat saturated weights forever.
    key = failed[(attempts - 1) % len(failed)]
    field = None
    if key == "startup":
        angular_gains = GAIN_CHOICES[attempts % len(GAIN_CHOICES)]
        changed["action_smooth"] = -min(.06, abs(changed.get("action_smooth", -.01)) * 1.4)
    elif key == "heading" or "yaw_yaw" == key:
        field = "yaw_rate_error_sq"
    elif "spin_center" in key:
        field = "spin_center_velocity"
    elif "speed" in key:
        # Distinguish a stationary-turn drift from moving speed error.
        phases = [p for t in report["yaw"]["traces"] for p in t["phases"]]
        stationary = max((p["ratios"]["speed"] for p in phases if p["speed"] == 0), default=0)
        moving = max((p["ratios"]["speed"] for p in phases if p["speed"] != 0), default=0)
        field = "standing_velocity" if stationary >= max(1., moving) else "lin_vel_error_sq"
    elif "height" in key:
        field = "base_height_error_sq"
    elif "symmetry" in key:
        field = "nominal_state"
    else:
        field = "orientation"
    caps = {"yaw_rate_error_sq": 180., "spin_center_velocity": 400., "standing_velocity": 400.,
            "lin_vel_error_sq": 180., "base_height_error_sq": 3500., "nominal_state": 300., "orientation": 500.}
    if field:
        old = abs(changed[field]); cap = caps[field]
        changed[field] = -(min(cap, old * 1.4) if old < cap else old * .8)
    return {"fresh": False, "iterations": 1000, "learning_rate": .0001,
            "rewards": changed, "gains": angular_gains,
            "purpose": f"针对未达标项 {key} 调整奖励/阻尼，同时保护全部已达标工况"}


class AutonomousOptimizer(YawOptimizer):
    def __init__(self, args):
        super().__init__(args)
        self.gains = (50., 3.)
        self.script_hash = sha256(Path(__file__))
        self.old_supervisor = None
        self.old_identity = None
        self.old_paused = False
        self.save(orchestrator_sha256=self.script_hash, startup_limits=STARTUP_LIMITS,
                  heading_limits={"steady_p95_deg": 1., "peak_deg": 3.},
                  scope="Robot control/tracking/heading acceptance; navigation path accuracy is not evaluated")

    def check(self):
        super().check()
        if sha256(Path(__file__)) != self.script_hash:
            raise RuntimeError("Autonomous orchestration source changed during execution")

    def run_job(self, command, log):
        try:
            return super().run_job(command, log)
        finally:
            self.save(child_pid=None, child_command=None, child_log=None)

    def set_gains(self, gains):
        self.gains = tuple(gains)
        REPLAY_OVERRIDES[:] = [v for v in REPLAY_OVERRIDES if not v.startswith(("env.kp_theta=", "env.kd_theta="))]
        REPLAY_OVERRIDES.extend((f"env.kp_theta={gains[0]}", f"env.kd_theta={gains[1]}"))

    def publish(self, checkpoint, rewards, report, label, training_run):
        # Keep the simulator's source lock exact. Record this companion's
        # version in the archived assessment without changing that lock.
        report["orchestrator_sha256"] = self.script_hash
        report["orchestrator_path"] = str(Path(__file__).relative_to(ROOT))
        super().publish(checkpoint, rewards, report, label, training_run)

    def command(self, checkpoint, seed, trace, sequence, gui=False, heading=False, perturb=.5):
        command = [str(ROOT / "play_wheel.sh"), "--device", "cuda:0", "--num_envs", "1",
                   "--seed", str(seed), "--checkpoint_path", str(checkpoint), "--fixed_command", "0", "0", ".18",
                   "--velocity_cycle", *[str(s[0]) for s in sequence],
                   "--yaw_cycle", *[str(s[1]) for s in sequence],
                   "--height_cycle", *[str(s[2]) for s in sequence],
                   "--max_steps", str(len(sequence) * 250), "--trace_csv", str(trace),
                   "env.episode_length_s=1000.0", "env.robot.init_state.pos=[0.0,0.0,0.18]",
                   f"env.reset_velocity_initial={perturb}", f"env.reset_velocity_final={perturb}",
                   f"env.kp_theta={self.gains[0]}", f"env.kd_theta={self.gains[1]}"]
        command += ["--live_stats", "--real-time", "--camera_mode", "orbit"] if gui else ["--headless"]
        if heading:
            command += ["--heading_feedback"]
        return command

    def visualize(self, checkpoint, label):
        self.save(state="visualizing", checkpoint=str(checkpoint), gains=self.gains)
        directory = self.directory / label
        directory.mkdir()
        trace = directory / "seed53.csv"
        command = self.command(checkpoint, 53, trace, heading_sequence(), gui=True, heading=True)
        write_json(directory / "command.json", command)
        os.environ.update(DISPLAY=":0", XAUTHORITY="/run/user/1000/gdm/Xauthority")
        try:
            self.run_job(command, directory / "play.log")
            report = heading_report(trace)
        except RuntimeError as exc:
            # A closed GUI does not count as completed testing. Finish the same
            # matrix headlessly so technical display failure cannot masquerade as acceptance.
            self.note(f"- 可视化未完整完成：{exc}；保留日志并完整补测相同指令。")
            command = self.command(checkpoint, 53, trace, heading_sequence(), heading=True)
            write_json(directory / "fallback_command.json", command)
            self.run_job(command, directory / "fallback.log")
            report = heading_report(trace)
        write_json(directory / "assessment.json", report)
        self.note(f"- 可视化/完整回放 `{label}`：航向最终误差 {report['final_error_deg']:.3f}°，"
                  f"最大角度误差 {report['max_error_deg']:.3f}°，重置 {report['resets']}。")
        return trace

    def evaluate_heading(self, checkpoint, label, seeds=(53, 54, 55), extended=False, gui_trace=None):
        self.save(state="evaluating_heading", evaluation_label=label, checkpoint=str(checkpoint))
        directory = self.directory / label
        directory.mkdir()
        reports = []
        for seed in seeds:
            trace = directory / f"seed{seed}.csv"
            if seed == 53 and gui_trace and not extended:
                import shutil
                shutil.copy2(gui_trace, trace)
            else:
                command = self.command(checkpoint, seed, trace, heading_sequence(extended), heading=True)
                write_json(directory / f"seed{seed}_command.json", command)
                self.run_job(command, directory / f"seed{seed}.log")
            reports.append(heading_report(trace, extended))
        result = {"traces": reports, "score": max(v for r in reports for v in r["ratios"].values()),
                  "resets": sum(r["resets"] for r in reports), "passed": all(r["passed"] for r in reports)}
        write_json(directory / "assessment.json", result)
        self.note(f"- 航向 `{label}`：门槛倍数 {result['score']:.3f}，通过={result['passed']}。")
        return result

    def evaluate_startup(self, checkpoint, label, seeds=(53, 54, 55)):
        self.save(state="evaluating_startup", evaluation_label=label, checkpoint=str(checkpoint))
        directory = self.directory / label
        directory.mkdir()
        reports = []
        for seed in seeds:
            for perturb in (0., .5):
                trace = directory / f"seed{seed}_reset{perturb:g}.csv"
                command = self.command(checkpoint, seed, trace, [(0., 0., .18)], heading=True, perturb=perturb)
                write_json(trace.with_suffix(".command.json"), command)
                self.run_job(command, trace.with_suffix(".log"))
                reports.append(startup_report(trace, perturb))
        result = {"traces": reports, "limits": STARTUP_LIMITS,
                  "score": max(v for r in reports for v in r["ratios"].values()),
                  "resets": sum(r["resets"] for r in reports), "passed": all(r["passed"] for r in reports)}
        write_json(directory / "assessment.json", result)
        self.note(f"- 启动 `{label}`：门槛倍数 {result['score']:.3f}，通过={result['passed']}。")
        return result

    def suite(self, checkpoint, label, gui_trace=None, extra=False):
        yaw = self.evaluate_yaw(checkpoint, label + "_yaw", seeds=(56, 57, 58) if extra else (53, 54, 55), extended=extra)
        straight = self.evaluate(checkpoint, label + "_straight", seeds=(46, 47, 48) if extra else (43, 44, 45), reverse=extra)
        startup = self.evaluate_startup(checkpoint, label + "_startup", seeds=(56, 57, 58) if extra else (53, 54, 55))
        heading = self.evaluate_heading(checkpoint, label + "_heading", seeds=(56, 57, 58) if extra else (53, 54, 55), extended=extra, gui_trace=gui_trace)
        result = bundle_reports(straight, yaw, startup, heading)
        write_json(self.directory / f"{label}_assessment.json", result)
        return result

    def control_preflight(self, checkpoint, reference, candidates=None, fallback=(50., 3.)):
        """Check fixed-policy dynamics before choosing a new training controller."""
        candidates = candidates or ((50., 6.), (60., 6.5), (70., 7.))
        for gains in dict.fromkeys(tuple(g) for g in candidates):
            self.set_gains(gains)
            label = f"control_preflight{time.time_ns()}"
            yaw = self.evaluate_yaw(checkpoint, label + "_yaw", seeds=(53,))
            straight = self.evaluate(checkpoint, label + "_straight", seeds=(43, 44, 45))
            # Complete timing/oscillation checks are included even though a
            # controller's preflight does not replace post-training acceptance.
            startup = self.evaluate_startup(checkpoint, label + "_startup", seeds=(53,))
            heading = self.evaluate_heading(checkpoint, label + "_heading", seeds=(53,))
            report = bundle_reports(straight, yaw, startup, heading)
            report.update(gains=gains, fixed_policy=True)
            write_json(self.directory / (label + "_assessment.json"), report)
            safe = (report["resets"] == 0 and straight["passed"]
                    and yaw["worst_ratios"]["body"] < 1
                    and yaw["worst_ratios"]["symmetry"] < 1
                    and yaw["worst_ratios"]["transient_body"] < 1
                    and report["worst_ratios"]["heading"] <= max(1., reference["worst_ratios"]["heading"] * 1.10))
            if safe:
                self.note(f"- 固定策略控制器检查通过：Kp/Kd={gains}，采用此组进行后续学习；最终仍需多种子完整验收。")
                return gains
        self.set_gains(fallback)
        self.note(f"- 候选增益未通过固定策略保护检查，保持已保留增益{fallback}并通过奖励优化恢复行为。")
        return fallback

    def attach(self):
        state_dir = self.args.attach_state.resolve()
        current = json.loads((state_dir / "state.json").read_text())
        self.old_supervisor = current["pid"]
        self.old_identity = identity(self.old_supervisor)
        if self.old_identity is not None:
            if str(state_dir).encode() not in self.old_identity["argv"]:
                raise RuntimeError("Existing supervisor identity mismatch")
            os.kill(self.old_supervisor, signal.SIGSTOP)
            self.old_paused = True
        checkpoint, log = Path(current["expected_checkpoint"]), Path(current["log"])
        pid, name = current["training_pid"], current["run_name"]
        self.note(f"- 接管当前训练 `{name}`：让其完整结束，原调度器暂停，训练进程继续。")
        if not self.wait_training(pid, name, log, checkpoint):
            raise RuntimeError("Attached training was unhealthy")
        self.save(state="waiting_for_existing_evaluation", old_supervisor=self.old_supervisor,
                  old_supervisor_paused=self.old_paused, inherited_checkpoint=str(checkpoint))
        deadline = time.monotonic() + 1800
        while self.old_paused and children(self.old_supervisor):
            self.check()
            if time.monotonic() > deadline:
                raise RuntimeError("Existing evaluation child did not exit")
            time.sleep(2)
        if self.old_paused and identity(self.old_supervisor) == self.old_identity:
            os.kill(self.old_supervisor, signal.SIGTERM)
            os.kill(self.old_supervisor, signal.SIGCONT)
            for _ in range(30):
                if identity(self.old_supervisor) != self.old_identity:
                    break
                time.sleep(1)
            else:
                raise RuntimeError("Existing supervisor did not terminate")
        self.old_paused = False
        self.save(old_supervisor_paused=False)
        current.update(state="superseded_after_training", continued_in=str(self.directory),
                       reason="User requested unattended startup, tracking and yaw optimization after GUI testing")
        write_json(state_dir / "state.json", current)
        return checkpoint, checkpoint.parent

    def main(self):
        checkpoint, run = self.attach()
        self.set_gains((50., 3.))
        gui = self.visualize(checkpoint, "attached_gui")
        baseline = self.suite(ACCEPTED, "baseline5740")
        best_checkpoint, best_report, best_gains = ACCEPTED, baseline, (50., 3.)
        current = self.suite(checkpoint, "attached_candidate", gui_trace=gui)
        rewards = {"nominal_state": -90., "lin_vel_error_sq": -90., "standing_velocity": -150.,
                   "base_height_error_sq": -1000., "orientation": -150., "yaw_rate_error_sq": -45.,
                   "tracking_sigma_ang": .0025, "tracking_sigma_precise": .0004,
                   "spin_center_velocity": -150., "action_smooth": -.015}
        if can_promote(current, baseline):
            best_checkpoint, best_report = checkpoint, current
            parent_rewards = json.loads((self.args.attach_state / "state.json").read_text())["plan"]["rewards"]
            current.update(gains=self.gains)
            provisional = dict(current, base_passed=current["passed"], passed=False, acceptance="provisional")
            self.publish(checkpoint, parent_rewards, provisional, "attached_improved", run)
            if current["passed"]:
                extra = self.suite(checkpoint, "attached_extra", extra=True)
                if extra["passed"]:
                    final = dict(current, passed=True, additional_validation=extra, acceptance="accepted")
                    final["traces"] = current["traces"] + extra["traces"]
                    self.publish(checkpoint, parent_rewards, final, "attached_accepted", run)
                    self.save(state="complete", accepted_checkpoint=str(checkpoint), accepted_gains=self.gains)
                    self.note("- 接管模型已通过全部基础和额外种子验收，停止训练。")
                    return
        gains = self.control_preflight(best_checkpoint, best_report)
        plan = {"fresh": False, "iterations": 1000, "learning_rate": .0001,
                "rewards": rewards, "gains": gains,
                "purpose": f"使用已验证Kp/Kd={gains}，收紧速度/yaw精度，同时学习启动抗扰恢复"}
        for number in range(1, self.args.max_rounds + 1):
            if number > 1 and tuple(plan["gains"]) != self.gains:
                plan["gains"] = self.control_preflight(best_checkpoint, best_report,
                                                      candidates=(plan["gains"], best_gains), fallback=best_gains)
            self.set_gains(plan["gains"])
            iteration = int(best_checkpoint.stem.split("_")[-1]) if best_checkpoint.stem.startswith("model_") else 5740
            plan["env_overrides"] = {"kp_theta": self.gains[0], "kd_theta": self.gains[1],
                                     "commands.heading_command": False, "commands.grouped_training": True,
                                     "commands.ranges_ang_vel_yaw": "[-0.5,0.5]",
                                     "commands.yaw_start_steps": iteration * 48 if best_checkpoint == ACCEPTED else 0,
                                     "commands.yaw_ramp_steps": 4800 if best_checkpoint == ACCEPTED else 1}
            self.save(round=number, best_checkpoint=str(best_checkpoint), best_gains=best_gains, plan=plan)
            checkpoint, run = self.train(best_checkpoint, plan, number)
            if checkpoint is None:
                raise RuntimeError("Training unhealthy; retained model preserved")
            gui = self.visualize(checkpoint, f"round{number:04d}_gui")
            report = self.suite(checkpoint, f"round{number:04d}", gui_trace=gui)
            report.update(gains=self.gains, training_plan=plan)
            improved = can_promote(report, best_report)
            self.status["history"].append({"round": number, "checkpoint": str(checkpoint),
                                           "gains": self.gains, "score": report["score"],
                                           "passed": report["passed"], "selected": improved})
            if improved:
                best_checkpoint, best_report, best_gains = checkpoint, report, self.gains
                provisional = dict(report, base_passed=report["passed"], passed=False, acceptance="provisional")
                self.publish(checkpoint, plan["rewards"], provisional, f"round{number:04d}_improved", run)
            if report["passed"] and improved:
                extra = self.suite(checkpoint, f"round{number:04d}_extra", extra=True)
                if extra["passed"]:
                    report.update(additional_validation=extra, passed=True, acceptance="accepted")
                    report["traces"] += extra["traces"]
                    self.publish(checkpoint, plan["rewards"], report, f"round{number:04d}_accepted", run)
                    self.save(state="complete", accepted_checkpoint=str(checkpoint), accepted_gains=self.gains)
                    self.note("- 启动振荡/恢复、速度、高度、腿角差、姿态、yaw-rate及航向角通过额外种子验收，停止训练。导航路径精度仍需另行验收。")
                    return
                # A failed extra matrix must influence tuning, never disappear.
                report = extra
            plan = next_plan(plan["rewards"], self.gains, report, number)
            self.save(best_checkpoint=str(best_checkpoint), next_plan=plan)
        self.save(state="needs_attention", reason="Trial limit reached; no full acceptance; all data and best model preserved")
        self.note("- 自动试验达到上限，尚未全部达标；保留最佳模型和失败项，未宣称完成。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--attach-state", required=True, type=Path)
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.max_rounds < 1:
        parser.error("--max-rounds must be positive")
    args.attach_state = args.attach_state.resolve()
    cached_baseline = args.attach_state / "baseline5740_yaw"
    args.baseline_cache = cached_baseline if (cached_baseline / "assessment.json").is_file() else None
    optimizer = AutonomousOptimizer(args)
    def stop(signum, frame):
        raise InterruptedError(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    try:
        optimizer.main()
    except BaseException as exc:
        child = optimizer.training_process
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=20)
            except Exception:
                os.killpg(child.pid, signal.SIGKILL)
        optimizer.save(state="stopped" if isinstance(exc, InterruptedError) else "needs_attention", error=repr(exc))
        optimizer.note(f"- 自动流程异常停止：{exc!r}；未宣称达标。")
        raise
    finally:
        if optimizer.old_paused and identity(optimizer.old_supervisor) == optimizer.old_identity:
            os.kill(optimizer.old_supervisor, signal.SIGCONT)


if __name__ == "__main__":
    main()
