"""Recorded train/evaluate/tune loop. Headless only; acceptance uses measured errors.

Attach to one existing training run, then evaluate, retain improvements and run
further experiments. Config changes are explicit Hydra overrides, never hidden
edits to a running simulator. Stop with the STOP file in --state-dir. A stopped
or failed experiment cannot be mistaken for acceptance.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from assess_tracking import HEIGHTS, SPEEDS, assess_suite, clearly_better
from monitor_training import inspect_log, matches_process

ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "wheel_legged_isaaclab/scripts/rsl_rl"
LOG_ROOT = ROOT / "IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat"
NOTES = ROOT / "wheel_legged_isaaclab/checkpoints/TRAINING_NOTES.md"
BASELINE = ROOT / "wheel_legged_isaaclab/checkpoints/corrected_vmc_height_model_3742.pt"
BASE_REWARDS = {"lin_vel_error_sq": -30.0, "standing_velocity": -50.0,
                "base_height_error_sq": -1000.0, "nominal_state": -60.0, "orientation": -150.0}
TUNING = {"speed": ("lin_vel_error_sq", 120.0), "height": ("base_height_error_sq", 3000.0),
          "symmetry": ("nominal_state", 240.0), "body": ("orientation", 450.0)}
# Physical/controller settings are fixed across comparisons. Rewards do not affect replay actions.
REPLAY_OVERRIDES = ["env.episode_length_s=300.0", "env.robot.init_state.pos=[0.0,0.0,0.18]",
                    "env.reset_velocity_initial=0.5", "env.reset_velocity_final=0.5"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = list((ROOT / "wheel_legged_isaaclab/wheel_legged_gym_isaaclab").rglob("*.py"))
    paths += list(SCRIPTS.glob("*.py"))
    paths += [ROOT / "train_wheel.sh", ROOT / "play_wheel.sh"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def choose_trial(rewards, report, failures):
    """Rotate failing objectives; periodically test fresh initialization after stagnation."""
    severity = dict(report["worst_ratios"])
    severity["height"] = max(severity["height"], severity.get("height_overshoot", 0.0))
    severity["body"] = max(severity["body"], severity.get("transient_body", 0.0))
    ranked = sorted(TUNING, key=lambda k: severity[k], reverse=True)
    failing = [key for key in ranked if severity[key] >= 1] or ranked
    # First permit adaptation with existing weights. Subsequent experiments alter one term.
    changed = dict(rewards)
    fresh = failures > 0 and failures % 4 == 0
    if failures == 0 or fresh:
        purpose = "完整配置继续学习" if not fresh else "连续候选未改进，检验从零训练能否摆脱旧策略偏好"
    else:
        attempt = failures - 1 - (failures - 1) // 4  # fresh runs do not consume a weight trial
        key = failing[attempt % len(failing)]
        weight, cap = TUNING[key]
        multiplier = 1.5 if (attempt // len(failing)) % 2 == 0 else 0.8
        changed[weight] = -min(cap, max(abs(BASE_REWARDS[weight]) * 0.5, abs(changed[weight]) * multiplier))
        purpose = f"针对 {key} 检验权重平衡：{weight} {rewards[weight]} → {changed[weight]}"
        if key == "speed":
            changed["standing_velocity"] = -min(150.0, abs(rewards["standing_velocity"]) * multiplier)
    return {"rewards": changed, "fresh": fresh, "iterations": 2000 if fresh else 1000,
            "learning_rate": 0.0003 if fresh else 0.0001, "purpose": purpose}


class Optimizer:
    def __init__(self, args):
        self.args = args
        self.directory = args.state_dir.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / "optimizer.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (self.directory / "state.json").exists():
            raise RuntimeError("State directory already exists; inspect it and use a new directory to avoid replaying old jobs")
        self.code = source_hashes()
        self.training_process = None
        self.training_identity = None
        self.status = {"state": "starting", "round": 0, "source_hashes": self.code,
                       "publish_requested": args.publish, "history": []}
        self.save()

    def save(self, **fields):
        self.status.update(fields, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"), pid=os.getpid())
        write_json(self.directory / "state.json", self.status)

    def check(self):
        if (self.directory / "STOP").exists():
            raise InterruptedError("STOP file requested termination of the optimization loop")
        if source_hashes() != self.code:
            raise RuntimeError("Source changed during optimization; review configuration before continuing")

    def note(self, message):
        with NOTES.open("a") as stream:
            stream.write(f"\n### 自动优化 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n{message}\n")
        print(message, flush=True)

    def run_job(self, command, log):
        self.check()
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            self.save(child_pid=process.pid, child_command=command, child_log=str(log))
            started = time.monotonic()
            try:
                while process.poll() is None:
                    self.check()
                    if time.monotonic() - started > 1800:
                        raise RuntimeError(f"Job exceeded 30 minutes: {log}")
                    time.sleep(5)
            except BaseException:
                # Own process group, never a global name-based kill.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            if process.returncode:
                raise RuntimeError(f"Job exited {process.returncode}: {log}")

    def wait_training(self, pid, run_name, log, checkpoint, owned=None):
        self.training_identity = (pid, run_name)
        self.save(state="training", training_pid=pid, run_name=run_name, log=str(log),
                  expected_checkpoint=str(checkpoint))
        last_iteration = None
        last_progress = time.monotonic()
        while matches_process(pid, run_name):
            self.check()
            health = inspect_log(log) if log.exists() else {"state": "starting"}
            self.save(health=health)
            if health.get("iteration") != last_iteration:
                last_iteration = health.get("iteration")
                last_progress = time.monotonic()
            if time.monotonic() - last_progress > 600:
                raise RuntimeError(f"No training iteration progress for ten minutes: {log}")
            if health.get("unhealthy"):
                if matches_process(pid, run_name):
                    os.kill(pid, signal.SIGTERM)
                for _ in range(20):
                    if not matches_process(pid, run_name):
                        break
                    time.sleep(1)
                if matches_process(pid, run_name):
                    os.kill(pid, signal.SIGKILL)
                if owned:
                    owned.wait(timeout=30)
                self.training_process = None
                self.training_identity = None
                self.note(f"- 本轮持续短回合且近乎全部异常终止，已停止。保留 checkpoint；不能算验收通过。\n- 日志：`{log}`")
                return False
            time.sleep(15)
        if owned and owned.wait(timeout=30) != 0:
            raise RuntimeError(f"Training process failed: {log}")
        health = inspect_log(log)
        if not checkpoint.is_file() or health.get("iteration", -1) < health.get("total_iterations", 1) - 1:
            raise RuntimeError(f"Training exited before completion: {log}")
        self.training_process = None
        self.training_identity = None
        return True

    def evaluate(self, checkpoint, label, seeds=(43, 44, 45), reverse=False):
        self.save(state="evaluating", evaluation_label=label, checkpoint=str(checkpoint))
        directory = self.directory / label
        directory.mkdir(parents=True, exist_ok=True)
        speeds = tuple(reversed(SPEEDS)) if reverse else SPEEDS
        heights = tuple(reversed(HEIGHTS)) if reverse else HEIGHTS
        paths = []
        for seed in seeds:
            trace = directory / f"seed{seed}.csv"
            command = [str(ROOT / "play_wheel.sh"), "--headless", "--device", "cuda:0", "--num_envs", "1",
                       "--seed", str(seed), "--checkpoint_path", str(checkpoint), "--fixed_command", "0", "0", ".18",
                       "--velocity_cycle", *[str(v) for v in speeds for _ in heights],
                       "--height_cycle", *map(str, heights), "--max_steps", "8250", "--trace_csv", str(trace),
                       *REPLAY_OVERRIDES]
            write_json(directory / f"seed{seed}_command.json", command)
            self.run_job(command, directory / f"seed{seed}.log")
            paths.append(trace)
        result = assess_suite(paths, reverse=reverse)
        result.update(seeds=list(seeds), checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
        write_json(directory / "assessment.json", result)
        self.run_job(["python3", str(SCRIPTS / "plot_trace.py"), str(paths[0])], directory / "plot.log")
        self.note(f"- 回放 `{label}`：异常/超时总次数 {result['resets']}；通过={result['passed']}。\n"
                  f"- 各项最差阈值倍数（<1 为达标）：`{json.dumps(result['worst_ratios'], ensure_ascii=False)}`。\n"
                  f"- 完整数据：`{directory}`。")
        return result

    def publish(self, checkpoint, rewards, report, label, training_run):
        directory = ROOT / "wheel_legged_isaaclab/checkpoints/optimized" / f"{self.directory.name}_{label}"
        directory.mkdir(parents=True, exist_ok=False)
        import shutil

        shutil.copy2(checkpoint, directory / "model.pt")
        for name in ("env.yaml", "agent.yaml"):
            shutil.copy2(training_run / "params" / name, directory / name)
        write_json(directory / "assessment.json", report)
        write_json(directory / "manifest.json", {"checkpoint_source": str(checkpoint),
                   "sha256": sha256(checkpoint), "reward_overrides": rewards,
                   "source_hashes": self.code, "evaluation_traces": [r["trace"] for r in report["traces"]],
                   "provisional_angle_limit_deg": 3.0})
        self.note(f"- 与保留参照相比有明确改善，留存模型与精确配置：`{directory.relative_to(ROOT)}`。\n"
                  "- 此记录不代表所有目标已达标；以 assessment.json 的 passed 和额外种子验证为准。")
        if self.args.publish:
            paths = [str(directory.relative_to(ROOT)), str(NOTES.relative_to(ROOT))]
            self.run_job(["git", "add", "--", *paths], self.directory / f"{label}_git_add.log")
            self.run_job(["git", "commit", "--only", "-m", f"留存轮腿训练改进模型与逐工况验收：{label}", "--", *paths],
                         self.directory / f"{label}_git_commit.log")
            try:
                self.run_job(["git", "push", "origin", "HEAD:main"], self.directory / f"{label}_git_push.log")
            except RuntimeError as exc:
                if source_hashes() != self.code:
                    raise
                self.save(publish_error=str(exc))
                self.note(f"- 本地模型和中文提交已留存，但 GitHub 推送失败：{exc}。继续训练，下次提升时再次推送。")
            else:
                self.save(last_published=label, publish_error=None)

    def train(self, checkpoint, plan, number):
        self.check()
        name = f"autotune_r{number:04d}_{time.strftime('%Y%m%d_%H%M%S')}"
        directory = self.directory / name
        directory.mkdir()
        command = [str(ROOT / "train_wheel.sh"), "--headless", "--device", "cuda:0", "--num_envs", "12288",
                   "--seed", str(42 + number), "--max_iterations", str(plan["iterations"]),
                   "--learning_rate", str(plan["learning_rate"]), "--run_name", name,
                   "agent.algorithm.schedule=fixed", "agent.save_interval=100"]
        if not plan["fresh"]:
            command += ["--checkpoint_path", str(checkpoint), "--reset_optimizer"]
        command += [f"env.rewards.{key}={value}" for key, value in plan["rewards"].items()]
        command += [f"env.{key}={value}" for key, value in plan.get("env_overrides", {}).items()]
        write_json(directory / "plan.json", dict(plan, command=command, parent_checkpoint=str(checkpoint), source_hashes=self.code))
        self.note(f"- **训练前目的**：{plan['purpose']}。\n- **修改内容**：奖励配置 `{plan['rewards']}`；"
                  f"额外环境配置 `{plan.get('env_overrides', {})}`；其余配置保持当前代码版本。\n- **起点**：{'随机初始化' if plan['fresh'] else str(checkpoint)}；"
                  f"{'新优化器' if not plan['fresh'] else '不加载模型'}，12288 环境、{plan['iterations']} 次 PPO 迭代、"
                  f"固定学习率 {plan['learning_rate']}。\n- **验收重点**：完整速度/高度矩阵、静止、姿态、腿角差和切换，"
                  "与保留参照比较，不接受以一项改善换取其他未达标项明显退步。")
        log = directory / "train.log"
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        self.training_process = process
        # The shell wrapper creates the Python child. Match the unique run name and train.py.
        pid = None
        deadline = time.monotonic() + 180
        run = None
        while time.monotonic() < deadline:
            self.check()
            if process.poll() is not None:
                raise RuntimeError(f"Training startup failed: {log}")
            for entry in Path("/proc").iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    argv = (entry / "cmdline").read_bytes().decode().split("\0")
                except (FileNotFoundError, PermissionError, ProcessLookupError):
                    continue
                if argv and "python" in Path(argv[0]).name and any(arg.endswith("/train.py") for arg in argv):
                    if matches_process(int(entry.name), name):
                        pid = int(entry.name)
            runs = list(LOG_ROOT.glob(f"*_{name}"))
            health = inspect_log(log)
            if pid and len(runs) == 1 and "total_iterations" in health:
                run = runs[0]
                break
            time.sleep(5)
        if run is None:
            os.killpg(process.pid, signal.SIGTERM)
            raise RuntimeError(f"Training did not initialize: {log}")
        health = inspect_log(log)
        target = run / f"model_{health['total_iterations'] - 1}.pt"
        if not self.wait_training(pid, name, log, target, owned=process):
            return None, run
        return target, run

    def main(self):
        complete = self.wait_training(self.args.attach_pid, self.args.attach_run, self.args.attach_log,
                                      self.args.attach_checkpoint)
        best_checkpoint = BASELINE
        best_rewards = dict(BASE_REWARDS)
        best_report = self.evaluate(best_checkpoint, "baseline3742")
        failures = 0
        candidate = self.args.attach_checkpoint if complete else None
        candidate_run = self.args.attach_checkpoint.parent
        candidate_rewards = dict(BASE_REWARDS)
        number = 0
        while True:
            self.check()
            label = f"round{number:04d}"
            self.save(round=number, best_checkpoint=str(best_checkpoint), consecutive_non_improvements=failures)
            if candidate:
                report = self.evaluate(candidate, label)
                improved = clearly_better(report, best_report)
                self.status["history"].append({"round": number, "checkpoint": str(candidate),
                                               "score": report["score"], "resets": report["resets"], "selected": improved})
                if improved:
                    best_checkpoint, best_rewards, best_report = candidate, candidate_rewards, report
                    failures = 0
                    self.publish(candidate, candidate_rewards, report, label, candidate_run)
                else:
                    failures += 1
                    self.note("- 本轮未满足明确改善且无明显退步的选择条件，保留原参照，候选及失败数据仍留存。")
                if report["passed"]:
                    validation = self.evaluate(candidate, label + "_validation", seeds=(46, 47, 48), reverse=True)
                    if validation["passed"]:
                        write_json(self.directory / "accepted_validation.json", validation)
                        self.note("- 训练验收完成：三个开发种子及三个额外种子、正反命令顺序均通过。"
                                  "速度/高度为 5% 门槛，零速度和角度采用记录中的绝对门槛；这是测试范围内达标，不是所有环境下完美。")
                        self.publish(candidate, candidate_rewards, dict(report, additional_validation=validation),
                                     label + "_accepted", candidate_run)
                        self.save(state="complete", accepted_checkpoint=str(candidate), validation=validation["worst_ratios"])
                        return
                    self.note("- 常规矩阵通过，但额外种子/反向顺序未通过；继续优化，不能宣布完成。")
                    # Use the held-out failure to choose the next objective, while keeping
                    # comparisons between candidates on the same development seeds/order.
                    self.status["validation_worst_ratios"] = validation["worst_ratios"]
                    failures += 1
            tuning_report = best_report
            if best_report["passed"] and self.status.get("validation_worst_ratios"):
                tuning_report = dict(best_report, worst_ratios=self.status["validation_worst_ratios"])
            plan = choose_trial(best_rewards, tuning_report, failures)
            number += 1
            candidate_rewards = plan["rewards"]
            candidate, candidate_run = self.train(best_checkpoint, plan, number)
            if candidate is None:
                failures += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attach-pid", type=int, required=True)
    parser.add_argument("--attach-run", required=True)
    parser.add_argument("--attach-log", type=Path, required=True)
    parser.add_argument("--attach-checkpoint", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--publish", action="store_true", help="Push selected improvements; requires user authorization.")
    args = parser.parse_args()
    optimizer = Optimizer(args)
    def stop(signum, frame):
        raise InterruptedError(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    try:
        optimizer.main()
    except BaseException as exc:
        process = optimizer.training_process
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        elif optimizer.training_identity:
            pid, name = optimizer.training_identity
            if matches_process(pid, name):
                os.kill(pid, signal.SIGTERM)
        optimizer.save(state="stopped" if isinstance(exc, InterruptedError) else "needs_attention", error=str(exc))
        optimizer.note(f"- 自动流程停止，未宣称达标：{exc}")
        raise


if __name__ == "__main__":
    main()
