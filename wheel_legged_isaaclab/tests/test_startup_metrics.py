import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl"))
from startup_metrics import StartupCapture


class TestStartupCapture(unittest.TestCase):
    def test_synchronous_oscillation_is_visible_and_first_window_stays_pinned(self):
        capture = StartupCapture(.02)
        for i in range(249):
            angle = math.radians(2 if i < 30 and i % 2 else -2 if i < 30 else .2)
            capture.add(angle, angle)
        self.assertIsNone(capture.report())
        capture.add(math.radians(.2), math.radians(.2))
        report = capture.report()
        self.assertAlmostEqual(report['left']['settling_s'], .6)
        self.assertAlmostEqual(report['left']['peak_abs_deg'], 2.)
        self.assertGreater(report['left']['variation_first_2s_deg'], 100.)
        self.assertEqual(report['left'], report['right'])
        capture.add(math.radians(99), math.radians(99))
        self.assertEqual(capture.report(), report)

    def test_persistent_motion_does_not_claim_settled(self):
        capture = StartupCapture(.02)
        for i in range(250):
            capture.add(math.radians(2 if i % 2 else -2), 0.)
        self.assertIsNone(capture.report()['left']['settling_s'])
        self.assertEqual(capture.report()['right']['settling_s'], 0.)


if __name__ == '__main__':
    unittest.main()
