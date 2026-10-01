"""Guard settling-time decisions against overshoot and incomplete stabilization."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/rsl_rl"))
from analyze_response import settling_time


class TestSettlingTime(unittest.TestCase):
    def test_uses_last_reentry_after_overshoot(self):
        errors = [.1] * 5 + [0] * 20 + [.06] * 10 + [0] * 30
        self.assertEqual(settling_time(errors, .05), .7)

    def test_requires_half_second_of_continuous_hold(self):
        self.assertIsNone(settling_time([.1] * 100 + [0] * 24, .05))
        self.assertEqual(settling_time([.1] * 100 + [0] * 25, .05), 2.0)

    def test_already_settled_and_never_settled(self):
        self.assertEqual(settling_time([.01] * 250, .05), 0)
        self.assertIsNone(settling_time([.1] * 250, .05))

    def test_nonfinite_data_cannot_settle(self):
        with self.assertRaises(ValueError):
            settling_time([0, math.nan] + [0] * 100, .05)


if __name__ == "__main__":
    unittest.main()
