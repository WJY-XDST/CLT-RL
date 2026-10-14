import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from heading_feedback import HeadingFeedback


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
