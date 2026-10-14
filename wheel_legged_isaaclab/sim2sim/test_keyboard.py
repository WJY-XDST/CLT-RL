"""Replay keyboard regressions; pure event tests need neither Tk nor Isaac Sim."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from keyboard import KeyboardConfig, KeyboardCommandState, KeyboardInput, MujocoKeyboardController


class IdleScheduler:
    def __init__(self):
        self.callbacks = {}
        self.next_token = 0

    def after_idle(self, callback):
        self.next_token += 1
        self.callbacks[self.next_token] = callback
        return self.next_token

    def after_cancel(self, token):
        self.callbacks.pop(token, None)

    def flush(self):
        callbacks, self.callbacks = list(self.callbacks.values()), {}
        for callback in callbacks:
            callback()


class KeyboardTests(unittest.TestCase):
    def setUp(self):
        self.state = KeyboardCommandState(KeyboardConfig())
        self.scheduler = IdleScheduler()
        self.input = KeyboardInput(self.state, self.scheduler)

    def test_no_takeover_on_untouched_focus_loss_or_close(self):
        self.assertIsNone(self.state.advance(.02))
        self.input.focus_lost()
        self.input.close()
        self.assertIsNone(self.state.advance(.02))

    def test_hold_release_and_opposite_axes(self):
        self.input.press('w')
        self.input.press('a')
        self.assertEqual(self.state.advance(.02), (.8, .6, .3))
        self.input.press('s')
        self.input.press('d')
        self.assertEqual(self.state.advance(.02), (0., 0., .3))
        for key in ('W', 'S', 'A', 'D'):
            self.input.release(key)
        self.scheduler.flush()
        self.assertEqual(self.state.advance(.02), (0., 0., .3))

    def test_x11_autorepeat_does_not_add_height_taps(self):
        self.input.press('E')
        for _ in range(10):
            self.input.release('E')
            self.input.press('E')
            self.scheduler.flush()
        self.assertAlmostEqual(self.state.target_height, .305)
        self.assertEqual(self.state.pressed, {'E'})
        self.input.release('E')
        self.scheduler.flush()
        for _ in range(13):
            command = self.state.advance(.02)
        self.assertAlmostEqual(command[2], .305)

    def test_stop_does_not_rearm_a_physically_held_motion_key(self):
        self.input.press('W')
        self.input.press('ESCAPE')
        self.input.release('W')  # X11 auto-repeat pair, not a real key release.
        self.input.press('W')
        self.scheduler.flush()
        self.assertEqual(self.state.advance(.02)[:2], (0., 0.))
        self.input.release('W')
        self.scheduler.flush()
        self.input.press('W')
        self.assertEqual(self.state.advance(.02)[0], .8)

    def test_focus_loss_stops_motion_and_height_hold(self):
        self.input.press('W')
        self.input.press('E')
        self.state.advance(.02)
        goal = self.state.target_height
        self.input.focus_lost()
        self.input.press('W')  # Held key repeat cannot resume after focus returns.
        command = self.state.advance(.02)
        self.assertEqual(command[:2], (0., 0.))
        self.assertAlmostEqual(self.state.target_height, goal)
        self.assertLessEqual(abs(command[2] - .3004), .0004 + 1e-12)

    def test_space_is_single_request_without_a_jump_action(self):
        self.input.press('W')
        self.input.press('SPACE')
        for _ in range(5):
            self.input.release('SPACE')
            self.input.press('SPACE')
        self.scheduler.flush()
        self.assertEqual(self.state.jump_requests, 1)
        self.assertTrue(self.state.take_jump_request())
        self.assertFalse(self.state.take_jump_request())
        self.assertEqual(self.state.advance(.02)[:2], (0., 0.))

    def test_modifiers_are_not_movement_but_release_still_clears(self):
        for mask in (4, 8, 64, 128):
            self.assertFalse(self.input.press('W', mask))
        self.assertIsNone(self.state.advance(.02))
        self.assertTrue(self.input.press('W', 1 | 2))
        self.input.release('W')
        self.scheduler.flush()
        self.assertEqual(self.state.advance(.02)[0], 0.)
        self.assertFalse(self.input.press('F1'))

    def test_panel_close_keeps_manual_zero_instead_of_resuming_demo(self):
        self.input.press('W')
        self.input.close()
        self.assertEqual(self.state.advance(.02), (0., 0., .3))
        self.assertFalse(self.input.press('W'))
        self.assertFalse(self.input.release('W'))

    def test_height_stays_inside_checkpoint_bounds(self):
        self.input.press('E')
        for _ in range(200):
            self.assertLessEqual(self.state.advance(.02)[2], .32)
        self.input.release('E')
        self.scheduler.flush()
        self.input.press('Q')
        for _ in range(200):
            self.assertGreaterEqual(self.state.advance(.02)[2], .28)

    def controller_with_mock_window(self):
        controller = MujocoKeyboardController.__new__(MujocoKeyboardController)
        controller.state, controller.input = self.state, self.input
        controller.window, controller.closed = Mock(), False
        controller._tk = SimpleNamespace(TclError=RuntimeError)
        return controller

    def test_late_takeover_uses_current_demo_height_and_slews(self):
        controller = self.controller_with_mock_window()
        controller.window.update.side_effect = lambda: self.input.press('Q')
        command = controller.advance(.02, .32)
        self.assertAlmostEqual(command[2], .3196)
        self.assertAlmostEqual(self.state.target_height, .3146)
        self.assertEqual(command[:2], (0., 0.))

    def test_closed_panel_keeps_height_ramp_and_does_not_pump_tk(self):
        controller = self.controller_with_mock_window()
        self.input.press('E')
        controller.close()
        command = controller.advance(.02, .28)
        self.assertAlmostEqual(command[2], .3004)
        controller.window.update.assert_not_called()
        self.assertEqual(command[:2], (0., 0.))

    def test_focus_transfer_to_button_is_not_focus_loss(self):
        controller = self.controller_with_mock_window()
        self.input.press('W')
        controller.window.focus_displayof.return_value = object()
        controller._check_focus()
        self.assertEqual(self.state.advance(.02)[0], .8)
        controller.window.focus_displayof.return_value = None
        controller._check_focus()
        self.assertEqual(self.state.advance(.02)[0], 0.)

    def test_nominal_height_is_a_ramp_not_a_command_step(self):
        controller = self.controller_with_mock_window()
        self.state.height, self.state.target_height = .32, .32
        controller._nominal_height()
        self.assertEqual(self.state.height, .32)
        self.assertEqual(self.state.target_height, .3)
        self.assertAlmostEqual(controller.advance(.02, .32)[2], .3196)


if __name__ == '__main__':
    unittest.main()
