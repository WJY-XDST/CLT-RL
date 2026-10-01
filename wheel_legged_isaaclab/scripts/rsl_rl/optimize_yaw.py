"""Fine-tune centered spins and turning; protect the accepted straight-motion model.

Each trial is followed by yaw and the original straight matrix. An optional
first run trains from random initialization for 2000 iterations; continuation runs use 1000.
Only promote a candidate that improves turning without losing straight acceptance.
All-seed acceptance is required before declaring the new task complete.
"""

import argparse
import json
import shutil
from pathlib import Path

from assess_yaw import assess_yaw, yaw_sequence
from optimize_training import Optimizer, ROOT, REPLAY_OVERRIDES, write_json, sha256

ACCEPTED = ROOT / "wheel_legged_isaaclab/checkpoints/optimized/continuous_optimization_20261001_round0002_accepted/model.pt"


class YawOptimizer(Optimizer):
    def evaluate_yaw(self, checkpoint, label, seeds=(53, 54, 55), extended=False):
        self.save(state="evaluating_yaw", evaluation_label=label, checkpoint=str(checkpoint))
        directory = self.directory / label
        directory.mkdir(parents=True, exist_ok=True)
        sequence = yaw_sequence(extended)
        reports = []
        cache = getattr(self.args, "baseline_cache", None) if label == "baseline5740_yaw" else None
        if cache:
            cache = cache.resolve()
            metadata = json.loads((cache / "assessment.json").read_text())
            previous_code = json.loads((cache.parent / "state.json").read_text())["source_hashes"]
            if metadata["checkpoint_sha256"] != sha256(checkpoint) or metadata["seeds"] != list(seeds):
                raise ValueError("Cached baseline checkpoint or seeds do not match")
            # Orchestration changes may alter this file; simulation and analysis sources must match.
            changed = [p for p, digest in self.code.items() if not p.endswith("/optimize_yaw.py")
                       and previous_code.get(p) != digest]
            if changed:
                raise ValueError(f"Cached baseline source mismatch: {changed}")
        for seed in seeds:
            trace = directory / f"seed{seed}.csv"
            if cache:
                shutil.copy2(cache / f"seed{seed}.csv", trace)
                shutil.copy2(cache / f"seed{seed}_command.json", directory / f"seed{seed}_source_command.json")
                reports.append(assess_yaw(trace, extended))
                continue
            command = [str(ROOT / "play_wheel.sh"), "--headless", "--device", "cuda:0", "--num_envs", "1",
                       "--seed", str(seed), "--checkpoint_path", str(checkpoint), "--fixed_command", "0", "0", ".18",
                       "--velocity_cycle", *[str(s[0]) for s in sequence],
                       "--yaw_cycle", *[str(s[1]) for s in sequence],
                       "--height_cycle", *[str(s[2]) for s in sequence],
                       "--max_steps", str(len(sequence) * 250), "--trace_csv", str(trace),
                       *[v for v in REPLAY_OVERRIDES if not v.startswith("env.episode_length_s=")],
                       "env.episode_length_s=1000.0"]
            write_json(directory / f"seed{seed}_command.json", command)
            self.run_job(command, directory / f"seed{seed}.log")
            reports.append(assess_yaw(trace, extended))
        worst = {k: max(r["worst_ratios"][k] for r in reports) for k in reports[0]["worst_ratios"]}
        result = {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "seeds": list(seeds),
                  "traces": reports, "worst_ratios": worst, "score": max(worst.values()),
                  "resets": sum(r["resets"] for r in reports), "passed": all(r["passed"] for r in reports)}
        write_json(directory / "assessment.json", result)
        if cache:
            write_json(directory / "reused_baseline.json", {"source": str(cache), "checkpoint_sha256": sha256(checkpoint),
                                                         "source_checked": True, "traces_reassessed": True})
        self.note(f"- 转向回放 `{label}`：重置 {result['resets']}，通过={result['passed']}；最坏门槛倍数 `{worst}`。")
        return result

    def main(self):
        best_checkpoint = ACCEPTED
        best_iteration = 5740
        speed_penalty = getattr(self.args, "initial_speed_penalty", 30.)
        if not 0 < speed_penalty <= 120:
            raise ValueError("initial_speed_penalty must be in (0, 120]")
        standing_penalty = getattr(self.args, "initial_standing_penalty", 100.)
        spin_penalty = getattr(self.args, "initial_spin_penalty", 100.)
        if not all(0 < v <= 300 for v in (standing_penalty, spin_penalty)):
            raise ValueError("Standing and spin penalty magnitudes must be in (0, 300]")
        rewards = {"nominal_state": -90., "lin_vel_error_sq": -speed_penalty, "standing_velocity": -standing_penalty,
                   "base_height_error_sq": -1000., "orientation": -150., "yaw_rate_error_sq": -30.,
                   "tracking_sigma_ang": .01, "spin_center_velocity": -spin_penalty}
        best_yaw = self.evaluate_yaw(best_checkpoint, "baseline5740_yaw")
        failures = 0
        for number in range(1, self.args.max_rounds + 1):
            fresh = bool(getattr(self.args, "from_scratch", False)) and number == 1
            plan = {"fresh": fresh, "iterations": 2000 if fresh else 1000,
                    "learning_rate": .0003 if fresh else .0001,
                    "rewards": dict(rewards),
                    "purpose": ("按用户要求从随机初始化重练六类工况，与封存5740比较" if fresh else
                                "学习原地双轮反向转与行进转弯，降低角速度误差，同时保持直行、腿长和姿态"),
                    "env_overrides": {"commands.heading_command": False,
                                      "commands.grouped_training": True,
                                      "commands.ranges_ang_vel_yaw": "[-0.5,0.5]",
                                      "commands.yaw_env_fraction": .5,
                                      "commands.yaw_start_steps": best_iteration * 48 if not fresh and best_checkpoint == ACCEPTED else 0,
                                      "commands.yaw_ramp_steps": 9600}}
            self.save(round=number, best_checkpoint=str(best_checkpoint), plan=plan)
            checkpoint, run = self.train(best_checkpoint, plan, number)
            if checkpoint is None:
                self.save(state="needs_attention", reason="Training became unstable; original model preserved")
                return
            yaw = self.evaluate_yaw(checkpoint, f"round{number:04d}_yaw")
            straight = self.evaluate(checkpoint, f"round{number:04d}_straight")
            # Protect every original straight criterion, plus every yaw criterion that already passed.
            no_regression = all(v <= max(1., best_yaw["worst_ratios"][k] * 1.10)
                                for k, v in yaw["worst_ratios"].items())
            improved = (yaw["resets"] == 0 and straight["passed"] and no_regression
                        and (yaw["passed"] or yaw["score"] < best_yaw["score"] * .90))
            self.status["history"].append({"round": number, "checkpoint": str(checkpoint),
                                           "yaw": yaw["score"], "straight_passed": straight["passed"],
                                           "selected": improved})
            combined = {"passed": False, "traces": straight["traces"] + yaw["traces"],
                        "straight": straight, "yaw": yaw, "training_plan": plan}
            if improved:
                best_checkpoint, best_yaw = checkpoint, yaw
                best_iteration = int(checkpoint.stem.split("_")[-1])
                failures = 0
                self.publish(checkpoint, rewards, combined, f"round{number:04d}_improved", run)
            else:
                failures += 1
                self.note("- 候选未同时满足转向改善和原直行验收，不替换保留模型。")
            if yaw["passed"] and straight["passed"] and improved:
                extra_yaw = self.evaluate_yaw(checkpoint, f"round{number:04d}_yaw_extended", seeds=(56, 57, 58), extended=True)
                extra_straight = self.evaluate(checkpoint, f"round{number:04d}_straight_extra", seeds=(46, 47, 48), reverse=True)
                if extra_yaw["passed"] and extra_straight["passed"]:
                    combined.update(passed=True, yaw_extended=extra_yaw, straight_extra=extra_straight)
                    combined["traces"] += extra_yaw["traces"] + extra_straight["traces"]
                    self.publish(checkpoint, rewards, combined, f"round{number:04d}_accepted", run)
                    self.save(state="complete", accepted_checkpoint=str(checkpoint))
                    self.note("- 转向、原地旋转中心及原直行矩阵通过额外种子与扩展范围验收，停止训练。")
                    return
            # After insufficient progress, change the failing objective rather than repeating unchanged trials.
            if failures:
                severity = yaw["worst_ratios"]
                if not straight["passed"]:
                    worst_straight = max((k for k in ("speed", "height", "body", "symmetry")),
                                         key=lambda k: straight["worst_ratios"][k])
                    key = {"speed": "lin_vel_error_sq", "height": "base_height_error_sq",
                           "body": "orientation", "symmetry": "nominal_state"}[worst_straight]
                    cap = {"lin_vel_error_sq": 120., "base_height_error_sq": 3000.,
                           "orientation": 450., "nominal_state": 240.}[key]
                    rewards[key] = -min(cap, abs(rewards[key]) * 1.5)
                elif severity["yaw"] >= max(severity["speed"], severity["spin_center"]):
                    rewards["yaw_rate_error_sq"] = -min(120., abs(rewards["yaw_rate_error_sq"]) * 1.5)
                else:
                    rewards["standing_velocity"] = -min(300., abs(rewards["standing_velocity"]) * 1.5)
                    rewards["spin_center_velocity"] = -min(300., abs(rewards["spin_center_velocity"]) * 1.5)
            self.save(best_checkpoint=str(best_checkpoint), consecutive_non_improvements=failures)
        self.save(state="needs_attention", reason="Trial limit reached; not accepted; inspect reports before further changes")
        self.note("- 本批转向优化已达到轮数上限，尚未通过完整验收，不宣称训练完成。保留全部结果供进一步诊断。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--from-scratch", action="store_true", help="Train the first candidate from random initialization for 2000 iterations.")
    parser.add_argument("--baseline-cache", type=Path, help="Reuse completed baseline traces after verifying checkpoint, seeds and simulation sources.")
    parser.add_argument("--initial-speed-penalty", type=float, default=30.,
                        help="Initial positive magnitude of the linear-speed squared-error penalty (0, 120].")
    parser.add_argument("--initial-standing-penalty", type=float, default=100.)
    parser.add_argument("--initial-spin-penalty", type=float, default=100.)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    if args.max_rounds < 1:
        parser.error("--max-rounds must be positive")
    if not 0 < args.initial_speed_penalty <= 120:
        parser.error("--initial-speed-penalty must be in (0, 120]")
    if not all(0 < v <= 300 for v in (args.initial_standing_penalty, args.initial_spin_penalty)):
        parser.error("Standing and spin penalty magnitudes must be in (0, 300]")
    optimizer = YawOptimizer(args)
    try:
        optimizer.main()
    except (Exception, KeyboardInterrupt) as exc:
        optimizer.save(state="needs_attention", error=repr(exc))
        optimizer.note(f"- 转向优化停止，需要检查：{exc!r}。未宣称验收通过。")
        raise
