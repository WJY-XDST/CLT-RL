"""Meaningful acceptance checks for unattended startup and heading trials."""

import copy
import csv
import math
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from autonomous_robot_optimization import (startup_report, heading_report, heading_sequence,
                                         bundle_reports, can_promote, next_plan, AutonomousOptimizer)


def fake_bundle(score=2., passed=False):
    phase = {"speed": .4, "height": .18, "yaw": .3, "ratios": {"speed": .5, "yaw": .5}}
    raw = {"worst_ratios": {"speed": .5, "yaw": .5}, "passed": True, "resets": 0,
           "traces": [{"trace": "dummy", "phases": [phase]}]}
    other = {"score": score, "passed": passed, "resets": 0, "traces": []}
    return bundle_reports(copy.deepcopy(raw), copy.deepcopy(raw), copy.deepcopy(other), copy.deepcopy(other))


class TestAutonomousAcceptance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "trace.csv"

    def tearDown(self):
        self.temp.cleanup()

    def write(self, rows):
        with self.path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader(); writer.writerows(rows)

    def test_startup_rejects_oscillation_even_when_final_pose_is_good(self):
        rows = [dict(step=i, sim_time_s=i*.02, env_id=0, terminated=0, time_out=0,
                     cmd_x_target=0, height_cmd=.18, theta_left=0, theta_right=0,
                     vel_x_heading=0, vel_y_heading=0, pitch_est_rad=0, roll_est_rad=0)
                for i in range(250)]
        self.write(rows)
        self.assertTrue(startup_report(self.path, 0.)["passed"])
        for i in range(1, 60):
            rows[i]["theta_left"] = rows[i]["theta_right"] = math.radians(2 if i % 2 else -2)
        self.write(rows)
        self.assertFalse(startup_report(self.path, 0.)["passed"])

    def test_heading_requires_original_target_and_no_resets(self):
        rows, reference = [], 0.
        for speed, rate, height in heading_sequence():
            for _ in range(250):
                i = len(rows); reference += math.degrees(rate*.02)
                rows.append(dict(step=i, sim_time_s=i*.02, env_id=0, terminated=0, time_out=0,
                                 heading_feedback=1, heading_hold=0, yaw_rate_requested=rate,
                                 cmd_x_target=speed, height_cmd=height, yaw_rate_applied=rate,
                                 yaw_turn_reference_deg=reference, yaw_turn_measured_deg=reference))
        self.write(rows)
        self.assertTrue(heading_report(self.path)["passed"])
        rows[-1]["terminated"] = 1
        self.write(rows)
        self.assertFalse(heading_report(self.path)["passed"])
        rows[-1]["yaw_turn_reference_deg"] += .1
        self.write(rows)
        with self.assertRaises(ValueError): heading_report(self.path)

    def test_missing_samples_cannot_pass(self):
        self.write([dict(step=0, sim_time_s=0, env_id=0, terminated=0, time_out=0)])
        with self.assertRaises(ValueError): startup_report(self.path, 0.)

    def test_all_components_required_and_phase_regression_blocks_promotion(self):
        baseline = fake_bundle()
        candidate = fake_bundle(.5, True)
        self.assertTrue(can_promote(candidate, baseline))
        candidate["straight"]["traces"][0]["phases"][0]["ratios"]["speed"] = 1.2
        self.assertFalse(can_promote(candidate, baseline))
        candidate = fake_bundle(.5, True)
        candidate["heading"]["passed"] = False
        combined = bundle_reports(candidate['straight'], candidate['yaw'], candidate['startup'], candidate['heading'])
        self.assertFalse(combined['passed'])

    def test_tuning_changes_saturated_weight_and_never_starts_fresh(self):
        report = fake_bundle()
        report['worst_ratios']['yaw_yaw'] = 3.
        rewards = dict(yaw_rate_error_sq=-180., lin_vel_error_sq=-90., standing_velocity=-150.,
                       spin_center_velocity=-150., base_height_error_sq=-1000., nominal_state=-90., orientation=-150.)
        plan = next_plan(rewards, (50., 6.), report, 1)
        self.assertNotEqual(plan['rewards']['yaw_rate_error_sq'], -180.)
        self.assertFalse(plan['fresh'])
        self.assertEqual(plan['iterations'], 1000)

    def test_different_matrix_cannot_be_compared(self):
        before, after = fake_bundle(), fake_bundle(.5, True)
        after['yaw']['traces'][0]['phases'][0]['yaw'] = -.3
        with self.assertRaises(ValueError): can_promote(after, before)

    def test_failed_extra_heading_validation_cannot_mark_complete(self):
        class SimulatedOptimizer(AutonomousOptimizer):
            def __init__(self):
                self.args = SimpleNamespace(max_rounds=1)
                self.status = {'history': []}
                self.gains = (50., 3.)
                self.published = []
            def save(self, **fields): self.status.update(fields)
            def note(self, message): pass
            def attach(self): return Path('model_6739.pt'), Path('run')
            def set_gains(self, gains): self.gains = gains
            def visualize(self, checkpoint, label): return Path('gui.csv')
            def train(self, *args): return Path('model_7739.pt'), Path('run')
            def control_preflight(self, *args, **kwargs): return (50., 6.)
            def publish(self, checkpoint, rewards, report, label, run):
                self.published.append((label, report['passed']))
            def suite(self, checkpoint, label, **kwargs):
                if label.startswith('round') and not kwargs.get('extra'):
                    return fake_bundle(.5, True)
                return fake_bundle(2., False)
        optimizer = SimulatedOptimizer()
        optimizer.main()
        self.assertEqual(optimizer.status['state'], 'needs_attention')
        self.assertEqual(optimizer.published, [('round0001_improved', False)])


if __name__ == "__main__":
    unittest.main()
