import ast
from pathlib import Path
import unittest

from wheel_legged_gym_isaaclab.diagnostic_timing import control_stride, is_control_update


class TestDiagnosticTiming(unittest.TestCase):
    def test_vmc_update_times_do_not_change_with_physics_timestep(self):
        for frequency in (200, 400, 800):
            stride = control_stride(frequency, 200)
            times = [i / frequency for i in range(frequency)
                     if is_control_update(i, stride)]
            self.assertEqual(times, [i / 200 for i in range(200)])

    def test_policy_updates_remain_at_fifty_hz(self):
        for frequency in (200, 400, 800):
            self.assertEqual(frequency // 50 // control_stride(frequency, 200), 4)

    def test_noninteger_frequency_ratios_are_rejected(self):
        for frequencies in ((300, 200), (200, 400), (400, 75), (0, 200), (-200, 200), (True, 200)):
            with self.subTest(frequencies=frequencies), self.assertRaises(ValueError):
                control_stride(*frequencies)

    def test_default_historical_timing_is_not_silently_changed(self):
        self.assertEqual(control_stride(500, 500), 1)

    def test_first_step_updates_then_holds(self):
        self.assertEqual([is_control_update(i, 4) for i in range(9)],
                         [True, False, False, False, True, False, False, False, True])

    def test_diagnostic_substep_is_the_state_clock_not_postintegration_time(self):
        source = Path(__file__).resolve().parents[1] / 'scripts/check_mcu_vmc_support.py'
        tree = ast.parse(source.read_text())
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_apply_action')
        text = ast.unparse(method)
        self.assertIn("'state_time_s': self.diagnostic_substep * self.cfg.sim.dt", text)
        # Holding the target does not call the parent VMC on intervening steps.
        call = next(n for n in ast.walk(method) if isinstance(n, ast.If)
                    and ast.unparse(n.test) == 'update_control')
        self.assertIn('super()._apply_action()', ast.unparse(call.body[0]))
        self.assertNotIn('super()._apply_action()', '\n'.join(ast.unparse(n) for n in call.orelse))


if __name__ == '__main__':
    unittest.main()
