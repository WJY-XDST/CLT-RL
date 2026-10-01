import math
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from heading_feedback import HeadingFeedback
from assess_heading import assess
from assess_yaw import yaw_sequence


class TestHeadingFeedback(unittest.TestCase):
    def test_turn_then_hold_corrects_bias_without_moving_target(self):
        controller = HeadingFeedback()
        angle = rate = 0.
        for step in range(2000):
            command = controller.step(.3 if step < 250 else 0., angle, .02)
            rate += .02 / .12 * (command - .015 - rate)
            angle += rate * .02
        self.assertAlmostEqual(controller.reference, 1.5)
        self.assertLess(abs(angle - 1.5), math.radians(.05))

    def test_wrap_and_saturation_do_not_accumulate_integral(self):
        controller = HeadingFeedback()
        for _ in range(1000):
            self.assertEqual(controller.step(.5, -1., .02), .5)
            controller.reference = 0.
        self.assertEqual(controller.integral_rate, 0.)
        controller.reset()
        self.assertLess(abs(controller.step(0., 2 * math.pi - .001, .02)), .01)

    def test_reset_discards_old_target_and_integral(self):
        controller = HeadingFeedback()
        controller.step(.3, -.1, .02)
        controller.reset()
        self.assertEqual(controller.step(0., 0., .02), 0.)

    def test_invalid_samples_rejected(self):
        for sample in ((.6, 0., .02), (0., float('nan'), .02), (0., 0., 0.)):
            with self.assertRaises(ValueError):
                HeadingFeedback().step(*sample)

    def test_assessment_uses_requested_reference_and_rejects_partial_or_moving_target(self):
        rows, reference = [], 0.
        for speed, rate, height in yaw_sequence():
            for _ in range(250):
                i = len(rows); reference += math.degrees(rate * .02)
                rows.append(dict(step=i, sim_time_s=.02*i, env_id=0, cmd_x_target=speed,
                                 height_cmd=height, heading_feedback=1, yaw_rate_requested=rate,
                                 yaw_turn_reference_deg=reference, yaw_rate_applied=rate+.01,
                                 yaw_turn_measured_deg=reference, terminated=0, time_out=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'trace.csv'
            def write():
                with path.open('w') as stream:
                    writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                    writer.writeheader(); writer.writerows(rows)
            write()
            self.assertTrue(assess(path, True)[0]['passed'])
            rows[-1]['yaw_turn_reference_deg'] += 1.
            write()
            with self.assertRaises(ValueError): assess(path, True)
            rows.pop(); write()
            with self.assertRaises(ValueError): assess(path, True)
