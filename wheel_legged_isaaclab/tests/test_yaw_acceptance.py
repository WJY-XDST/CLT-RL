"""Reject misleading yaw results, incomplete coverage and single-wheel pivots."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from assess_yaw import assess_yaw, yaw_sequence
from optimize_yaw import YawOptimizer


class SimulatedYawOptimizer(YawOptimizer):
    """Exercise promotion decisions without launching a simulator."""
    def __init__(self, straight_passed=True, extra_passed=True):
        self.args = SimpleNamespace(max_rounds=1)
        self.status = {"history": []}
        self.straight_passed, self.extra_passed = straight_passed, extra_passed
        self.published = []

    def save(self, **fields): self.status.update(fields)
    def note(self, message): pass
    def train(self, checkpoint, plan, number):
        self.training_plan = plan
        return Path("model_1999.pt" if plan["fresh"] else "model_6739.pt"), Path("run")
    def publish(self, checkpoint, rewards, report, label, run): self.published.append(label)
    def evaluate_yaw(self, checkpoint, label, seeds=(), extended=False):
        passed = "baseline" not in label and (not extended or self.extra_passed)
        value = .5 if passed else 2.
        return {"passed": passed, "score": value, "resets": 0, "traces": [],
                "worst_ratios": {k:value for k in ("yaw", "speed", "spin_center")}}
    def evaluate(self, *args, **kwargs):
        return {"passed": self.straight_passed, "traces": [],
                "worst_ratios": {k:2 for k in ("speed", "height", "body", "symmetry")}}


class TestYawAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "yaw.csv"
        self.rows = []
        for index, (speed, yaw, height) in enumerate(yaw_sequence()):
            for offset in range(250):
                step = index * 250 + offset
                self.rows.append(dict(step=step, sim_time_s=step*.02, env_id=0,
                                      cmd_x_target=speed, cmd_yaw=yaw, height_cmd=height,
                                      heading_hold=int(yaw == 0), terminated=0, time_out=0,
                                      vel_x_heading=speed, vel_y_heading=0, yaw_rate_body=yaw,
                                      base_height=height, pitch_est_rad=0, roll_est_rad=0,
                                      theta_left=0, theta_right=0,
                                      wheel_vel_left=(speed-.341*yaw/2)/.0675,
                                      wheel_vel_right=(speed+.341*yaw/2)/.0675))

    def tearDown(self):
        self.temp.cleanup()

    def assess(self):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader(); writer.writerows(self.rows)
        return assess_yaw(self.path)

    def test_ideal_centered_spins_and_travel_pass(self):
        self.assertTrue(self.assess()["passed"])

    def test_single_wheel_pivot_fails_even_if_body_feedback_looks_ideal(self):
        for row in self.rows[250:500]:
            row["wheel_vel_left"], row["wheel_vel_right"] = -1.6, 0
        self.assertFalse(self.assess()["passed"])

    def test_yaw_oscillation_is_not_cancelled(self):
        for i, row in enumerate(self.rows[250:500]):
            row["yaw_rate_body"] += .05 if i % 2 else -.05
        self.assertFalse(self.assess()["passed"])

    def test_reset_and_missing_coverage_cannot_pass(self):
        self.rows[255]["terminated"] = 1
        self.assertFalse(self.assess()["passed"])
        self.rows.pop()
        with self.assertRaises(ValueError): self.assess()

    def test_heading_override_cannot_mask_missing_turn(self):
        self.rows[250]["heading_hold"] = 1
        with self.assertRaises(ValueError): self.assess()

    def test_turn_improvement_cannot_replace_model_when_straight_regresses(self):
        optimizer = SimulatedYawOptimizer(straight_passed=False)
        optimizer.main()
        self.assertEqual(optimizer.published, [])
        self.assertEqual(optimizer.status["state"], "needs_attention")

    def test_extended_failure_cannot_be_declared_complete(self):
        optimizer = SimulatedYawOptimizer(extra_passed=False)
        optimizer.main()
        self.assertEqual(optimizer.published, ["round0001_improved"])
        self.assertEqual(optimizer.status["state"], "needs_attention")

    def test_requested_scratch_run_does_not_resume_or_skip_curriculum(self):
        optimizer = SimulatedYawOptimizer(straight_passed=False)
        optimizer.args.from_scratch = True
        optimizer.main()
        self.assertTrue(optimizer.training_plan["fresh"])
        self.assertEqual(optimizer.training_plan["iterations"], 2000)
        self.assertEqual(optimizer.training_plan["env_overrides"]["commands.yaw_start_steps"], 0)

    def test_recovery_applies_selected_speed_penalty_without_fresh_start(self):
        optimizer = SimulatedYawOptimizer(straight_passed=False)
        optimizer.args.initial_speed_penalty = 45.
        optimizer.args.initial_standing_penalty = 150.
        optimizer.args.initial_spin_penalty = 150.
        optimizer.main()
        self.assertEqual(optimizer.training_plan["rewards"]["lin_vel_error_sq"], -45.)
        self.assertEqual(optimizer.training_plan["rewards"]["standing_velocity"], -150.)
        self.assertEqual(optimizer.training_plan["rewards"]["spin_center_velocity"], -150.)
        self.assertFalse(optimizer.training_plan["fresh"])
        self.assertEqual(optimizer.training_plan["env_overrides"]["commands.yaw_start_steps"], 5740 * 48)

    def test_invalid_initial_speed_penalty_is_rejected_before_training(self):
        for value in (0, -45, 121, float("nan"), float("inf")):
            optimizer = SimulatedYawOptimizer()
            optimizer.args.initial_speed_penalty = value
            with self.assertRaises(ValueError):
                optimizer.main()
            self.assertFalse(hasattr(optimizer, "training_plan"))


if __name__ == "__main__":
    unittest.main()
