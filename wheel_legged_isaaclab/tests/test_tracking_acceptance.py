"""Regression tests for false acceptance and unsafe model promotion (no simulator)."""

import copy
import csv
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from assess_tracking import assess_trace, assess_suite, clearly_better, sequence
from optimize_training import choose_trial, BASE_REWARDS, BASELINE, Optimizer


class TestTrackingAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.rows = []
        for phase, (speed, height) in enumerate(sequence()):
            for offset in range(250):
                step = phase * 250 + offset
                self.rows.append(dict(step=step, sim_time_s=step * 0.02, env_id=0,
                    cmd_x_target=speed, height_cmd=height, terminated=0, time_out=0,
                    vel_x_heading=speed, vel_y_heading=0, base_height=height,
                    pitch_est_rad=0, roll_est_rad=0, theta_left=0, theta_right=0,
                    length_left=.23, length_right=.23, target_length_left=.23, target_length_right=.23))

    def tearDown(self):
        self.temp.cleanup()

    def trace(self, name="trace.csv"):
        path = self.directory / name
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows)
        return path

    def test_complete_ideal_suite_passes(self):
        self.assertTrue(assess_suite([self.trace(f"seed{s}.csv") for s in (43, 44, 45)])["passed"])

    def test_truncated_trace_cannot_pass(self):
        self.rows.pop()
        with self.assertRaises(ValueError):
            assess_trace(self.trace())

    def test_wrong_commands_cannot_pass(self):
        self.rows[1000]["cmd_x_target"] = -.8
        with self.assertRaises(ValueError):
            assess_trace(self.trace())

    def test_nan_and_missing_sample_cannot_pass(self):
        self.rows[10]["base_height"] = math.nan
        with self.assertRaises(ValueError):
            assess_trace(self.trace())
        self.rows[10]["base_height"] = .16
        self.rows[10]["step"] = 11
        with self.assertRaises(ValueError):
            assess_trace(self.trace())

    def test_reset_in_settling_window_cannot_be_hidden(self):
        self.rows[5]["terminated"] = 1
        result = assess_trace(self.trace())
        self.assertEqual(result["resets"], 1)
        self.assertFalse(result["passed"])

    def test_zero_mean_signed_error_cannot_hide_oscillation(self):
        for i in range(250):
            self.rows[750 + i]["vel_x_heading"] = .2 + (.025 if i % 2 else -.025)
        result = assess_trace(self.trace())
        self.assertGreater(result["phases"][3]["speed_relative_mae"], .05)
        self.assertFalse(result["passed"])

    def test_lateral_standing_drift_fails(self):
        for row in self.rows[:250]:
            row["vel_y_heading"] = .03
        self.assertFalse(assess_trace(self.trace())["passed"])

    def test_steady_body_spike_and_transition_overshoot_fail(self):
        self.rows[200]["pitch_est_rad"] = math.radians(4)
        self.assertFalse(assess_trace(self.trace())["passed"])
        self.rows[200]["pitch_est_rad"] = 0
        self.rows[260]["base_height"] = .2  # overshoots target .18 during settling
        result = assess_trace(self.trace())
        self.assertGreater(result["phases"][1]["ratios"]["height_overshoot"], 1)
        self.assertFalse(result["passed"])

    def test_large_improvement_with_posture_regression_is_rejected(self):
        report = assess_suite([self.trace(f"seed{s}.csv") for s in (43, 44, 45)])
        before, after = copy.deepcopy(report), copy.deepcopy(report)
        before.update(score=3, passed=False)
        after.update(score=1.5, passed=False)
        before["traces"][0]["phases"][0]["ratios"]["body"] = 1.1
        after["traces"][0]["phases"][0]["ratios"]["body"] = 2
        self.assertFalse(clearly_better(after, before))
        after["traces"][0]["phases"][0]["ratios"]["body"] = 1.05
        self.assertTrue(clearly_better(after, before))
        after["resets"] = 1
        self.assertFalse(clearly_better(after, before))

    def test_tuning_is_bounded_and_restarts_after_stagnation(self):
        report = {"worst_ratios": {"speed": 3, "height": .3, "body": 2, "symmetry": .9}}
        plan = choose_trial(BASE_REWARDS, report, failures=1)
        self.assertEqual(plan["rewards"]["lin_vel_error_sq"], -45)
        self.assertFalse(plan["fresh"])
        self.assertEqual(choose_trial(BASE_REWARDS, report, failures=2)["rewards"]["orientation"], -225)
        self.assertEqual(choose_trial(BASE_REWARDS, report, failures=3)["rewards"]["lin_vel_error_sq"], -24)
        self.assertEqual(choose_trial(BASE_REWARDS, report, failures=5)["rewards"]["orientation"], -120)
        plan = choose_trial(BASE_REWARDS, report, failures=4)
        self.assertTrue(plan["fresh"])
        self.assertEqual(plan["iterations"], 2000)
        huge = dict(BASE_REWARDS, lin_vel_error_sq=-1000)
        self.assertEqual(choose_trial(huge, report, failures=1)["rewards"]["lin_vel_error_sq"], -120)

    def harness(self, reports):
        class Harness(Optimizer):
            def check(self):
                pass
            def save(self, **fields):
                self.status.update(fields)
            def note(self, message):
                pass
            def wait_training(self, *args, **kwargs):
                return True
            def evaluate(self, *args, **kwargs):
                return next(self.reports)
            def publish(self, checkpoint, rewards, report, label, training_run):
                self.published.append((checkpoint, label))
            def train(self, checkpoint, plan, number):
                self.next_trial = (checkpoint, plan, number)
                raise InterruptedError("Test stops at next planned training")
        harness = Harness.__new__(Harness)
        harness.args = SimpleNamespace(attach_pid=123, attach_run="test", attach_log=Path("unused"),
                                       attach_checkpoint=Path("candidate/model_1999.pt"))
        harness.status = {"history": []}
        harness.directory = self.directory
        harness.reports = iter(reports)
        harness.published = []
        return harness

    def test_loop_rejects_failed_candidate_and_trains_from_preserved_baseline(self):
        baseline = assess_suite([self.trace(f"seed{s}.csv") for s in (43, 44, 45)])
        baseline.update(passed=False, score=2)
        baseline["worst_ratios"]["symmetry"] = 4
        candidate = copy.deepcopy(baseline)
        candidate["resets"] = 3
        harness = self.harness([baseline, candidate])
        with self.assertRaises(InterruptedError):
            harness.main()
        checkpoint, plan, number = harness.next_trial
        self.assertEqual(checkpoint, BASELINE)
        self.assertEqual(plan["rewards"]["nominal_state"], -90)
        self.assertEqual(harness.published, [])
        self.assertNotEqual(harness.status["state"] if "state" in harness.status else None, "complete")

    def test_loop_requires_additional_validation_before_completion(self):
        ideal = assess_suite([self.trace(f"seed{s}.csv") for s in (43, 44, 45)])
        baseline = copy.deepcopy(ideal)
        baseline.update(passed=False, score=2)
        bad_validation = copy.deepcopy(ideal)
        bad_validation.update(passed=False, resets=1)
        bad_validation["worst_ratios"]["body"] = 4
        harness = self.harness([baseline, ideal, bad_validation])
        with self.assertRaises(InterruptedError):
            harness.main()
        self.assertEqual(harness.next_trial[1]["rewards"]["orientation"], -225)
        self.assertNotEqual(harness.status.get("state"), "complete")
        harness = self.harness([baseline, ideal, ideal])
        harness.main()
        self.assertEqual(harness.status["state"], "complete")
        self.assertEqual(harness.published[-1][1], "round0000_accepted")


if __name__ == "__main__":
    unittest.main()
