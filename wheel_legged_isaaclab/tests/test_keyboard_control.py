"""CPU-only tests; fake UI/input avoids launching a second GPU simulator."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'wheel_legged_isaaclab/scripts/rsl_rl'))
sys.path.insert(0, str(PROJECT / 'mine_model/scripts'))
from keyboard_control import (
    KeyboardConfig, KeyboardCommandState, IsaacKeyboardController, apply_keyboard_command, supports_keyboard_task,
)
from play_keyboard import replay_command
from wheel_legged_gym_isaaclab.reward_signals import DIRECT_JOINT_REWARD_PROFILE


class TestKeyboardState(unittest.TestCase):
    def setUp(self):
        self.state = KeyboardCommandState(KeyboardConfig())

    def test_fixed_replay_untouched_until_first_key(self):
        self.assertIsNone(self.state.advance(.02))
        self.assertFalse(self.state.handle_key('F', 'press'))
        self.state.handle_key('W', 'release')
        self.assertIsNone(self.state.advance(.02))
        self.state.handle_key('W', 'press')
        self.assertEqual(self.state.advance(.02), (.8, 0., .30))

    def test_press_repeat_release_combination_and_opposing_keys(self):
        for key in ('W', 'A'):
            self.state.handle_key(key, 'press')
            self.state.handle_key(key, 'repeat')
            self.state.handle_key(key, 'press')
        self.assertEqual(self.state.advance(.02), (.8, .6, .30))
        for key in ('S', 'D'):
            self.state.handle_key(key, 'press')
        self.assertEqual(self.state.advance(.02), (0., 0., .30))
        for key in ('W', 'A'):
            self.state.handle_key(key, 'release')
        self.assertEqual(self.state.advance(.02), (-.8, -.6, .30))
        for key in ('S', 'D'):
            self.state.handle_key(key, 'release')
        self.assertEqual(self.state.advance(.02), (0., 0., .30))

    def test_height_adjustment_is_timestep_based_bounded_and_retained(self):
        self.state.handle_key('E', 'press')
        self.assertEqual(self.state.target_height, .305)
        self.assertEqual(self.state.height, .30)
        self.assertEqual(self.state.advance(1.)[2], .32)
        self.assertEqual(self.state.advance(100.)[2], .32)
        self.state.handle_key('E', 'release')
        self.assertEqual(self.state.advance(.02)[2], .32)
        self.state.handle_key('Q', 'press')
        self.assertEqual(self.state.advance(100.)[2], .28)

    def test_height_tap_changes_goal_only_repeat_ignored_release_finishes_ramp(self):
        self.state.handle_key('E', 'press')
        self.assertAlmostEqual(self.state.target_height, .305)
        self.assertEqual(self.state.height, .30)
        self.state.handle_key('E', 'repeat')
        self.state.handle_key('E', 'press')
        self.assertAlmostEqual(self.state.target_height, .305)
        self.state.handle_key('E', 'release')
        heights = [.30] + [self.state.advance(.02)[2] for _ in range(13)]
        self.assertAlmostEqual(heights[1], .3004)
        self.assertEqual(heights[-1], self.state.target_height)
        for previous, current in zip(heights, heights[1:]):
            self.assertGreaterEqual(current, previous)
            self.assertLessEqual(current - previous, .0004 + 1e-12)
        self.assertEqual(self.state.advance(.02)[2], .305)
        self.state.handle_key('Q', 'press')
        self.assertAlmostEqual(self.state.target_height, .30)
        self.state.handle_key('Q', 'release')
        self.assertAlmostEqual(self.state.advance(.1)[2], .303)
        self.assertAlmostEqual(self.state.advance(.1)[2], .301)
        self.assertEqual(self.state.advance(.1)[2], .30)

    def test_height_taps_clamp_and_opposing_key_does_not_add_step(self):
        self.state.height = .319
        self.state.set_height_target(.319)
        self.state.handle_key('E', 'press')
        self.assertEqual(self.state.target_height, .32)
        self.assertEqual(self.state.height, .319)
        self.state.handle_key('Q', 'press')
        self.assertEqual(self.state.advance(1.)[2], .32)
        self.state.handle_key('E', 'release')
        self.assertAlmostEqual(self.state.advance(.5)[2], .31)
        self.state.handle_key('Q', 'release')
        self.state.height = .281
        self.state.set_height_target(.281)
        self.state.handle_key('Q', 'press')
        self.assertEqual(self.state.target_height, .28)
        self.state.handle_key('E', 'press')
        self.assertEqual(self.state.advance(100.)[2], .28)

    def test_ramp_timestep_independence_and_long_hold_release(self):
        coarse = KeyboardCommandState(KeyboardConfig(), active=True)
        fine = KeyboardCommandState(KeyboardConfig(), active=True)
        coarse.handle_key('E', 'press'); fine.handle_key('E', 'press')
        coarse.advance(.25)
        for _ in range(25):
            fine.advance(.01)
        self.assertAlmostEqual(coarse.height, .305)
        self.assertAlmostEqual(coarse.target_height, .31)
        self.assertAlmostEqual(coarse.height, fine.height)
        self.assertAlmostEqual(coarse.target_height, fine.target_height)
        for state in (coarse, fine):
            state.handle_key('E', 'release')
            self.assertAlmostEqual(state.advance(.25)[2], .31)
            self.assertEqual(state.height, state.target_height)

    def test_goal_reversal_and_nominal_reset_never_jump_command(self):
        self.state.active = True
        self.state.set_height_target(.32)
        self.assertAlmostEqual(self.state.advance(.1)[2], .302)
        self.state.set_height_target(.28)
        self.assertAlmostEqual(self.state.height, .302)
        self.assertAlmostEqual(self.state.advance(.1)[2], .30)
        self.assertAlmostEqual(self.state.advance(.1)[2], .298)
        self.state.stop()
        self.state.set_height_target(.30)
        self.assertAlmostEqual(self.state.height, .298)
        self.assertAlmostEqual(self.state.advance(.02)[2], .2984)
        self.assertEqual(self.state.advance(.2)[2], .30)

    def test_space_is_single_pending_request_and_clears_motion(self):
        for key in ('W', 'A', 'E', 'SPACE'):
            self.state.handle_key(key, 'press')
        self.state.handle_key('SPACE', 'repeat')
        self.state.handle_key('SPACE', 'press')
        self.assertEqual(self.state.jump_requests, 1)
        self.assertEqual(self.state.advance(1.), (0., 0., .305))
        self.assertTrue(self.state.take_jump_request())
        self.assertFalse(self.state.take_jump_request())
        self.state.handle_key('SPACE', 'release')
        self.state.handle_key('SPACE', 'press')
        self.assertEqual(self.state.jump_requests, 2)

    def test_stop_keys_clear_all_axes(self):
        for stop_key in ('L', 'ESCAPE'):
            self.state = KeyboardCommandState(KeyboardConfig())
            self.state.handle_key('S', 'press')
            self.state.handle_key('Q', 'press')
            self.state.handle_key(stop_key, 'press')
            command = self.state.advance(.1)
            self.assertEqual(command[:2], (0., 0.))
            self.assertAlmostEqual(command[2], .298)
            self.assertEqual(self.state.target_height, .295)
            self.assertEqual(self.state.advance(.5)[2], .295)

    def test_invalid_config_and_timestep(self):
        for kwargs in ({'speed': float('nan')}, {'height_rate': 0}, {'initial_height': .40},
                       {'height_max': .20}, {'yaw_rate': float('inf')}, {'height_step': 0},
                       {'height_slew_rate': 0}, {'height_slew_rate': float('nan')}):
            with self.assertRaises(ValueError):
                KeyboardConfig(**kwargs)
        for dt in (0, -.1, float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                self.state.advance(dt)
        with self.assertRaises(ValueError):
            self.state.set_height_target(float('nan'))

    def test_command_observation_sync_preserves_speed_ramp_and_other_features(self):
        env = SimpleNamespace(cfg=SimpleNamespace(commands=SimpleNamespace(heading_command=True)),
                              _command_ranges=np.zeros((3, 2)), _commands=np.array([[.12, .1, .30]]),
                              _target_lin_vel_x=np.zeros(1), _commands_scale=np.array([2., .25, 5.]))
        policy = np.arange(27, dtype=float).reshape(1, 27)
        original = policy.copy()
        obs = {'policy': policy, 'critic': np.full((1, 27), 99.)}
        apply_keyboard_command(env, obs, (.5, -.3, .31))
        self.assertFalse(env.cfg.commands.heading_command)
        np.testing.assert_allclose(env._command_ranges, [[.5, .5], [-.3, -.3], [.31, .31]])
        self.assertEqual(env._target_lin_vel_x[0], .5)
        self.assertEqual(env._commands[0, 0], .12)  # actual command still ramps in env
        np.testing.assert_allclose(policy[0, 6:9], [.24, -.075, 1.55])
        np.testing.assert_array_equal(policy[:, :6], original[:, :6])
        np.testing.assert_array_equal(policy[:, 9:], original[:, 9:])
        np.testing.assert_array_equal(obs['critic'], np.full((1, 27), 99.))
        before = policy.copy()
        apply_keyboard_command(env, policy, None)
        np.testing.assert_array_equal(before, policy)
        apply_keyboard_command(env, policy, (0., 0., .30))
        self.assertEqual(env._target_lin_vel_x[0], 0.)

    def test_real_torch_tensordict_batch_matches_replay_wrapper(self):
        import torch
        from tensordict import TensorDict
        env = SimpleNamespace(cfg=SimpleNamespace(commands=SimpleNamespace(heading_command=True)),
                              _command_ranges=torch.zeros(3, 2),
                              _commands=torch.tensor([[.12, .1, .30, 1.], [.16, .1, .30, -1.]]),
                              _target_lin_vel_x=torch.zeros(2), _commands_scale=torch.tensor([2., .25, 5.]))
        obs = TensorDict({'policy': torch.ones(2, 27)}, batch_size=[2])
        with torch.inference_mode():
            apply_keyboard_command(env, obs, (-.5, .3, .31))
        torch.testing.assert_close(obs['policy'][:, 6:9],
                                   torch.tensor([[.24, .075, 1.55], [.32, .075, 1.55]]))
        torch.testing.assert_close(env._target_lin_vel_x, torch.full((2,), -.5))
        torch.testing.assert_close(env._commands[:, 3], torch.tensor([1., -1.]))
        self.assertTrue((obs['policy'][:, 9:] == 1.).all())

    def test_stationary_jump_task_never_receives_wasd_motion_commands(self):
        env = SimpleNamespace(cfg=SimpleNamespace(jump=True, commands=SimpleNamespace(heading_command=False)),
                              _command_ranges=np.zeros((3, 2)), _commands=np.array([[0., 0., .30]]),
                              _target_lin_vel_x=np.zeros(1), _commands_scale=np.array([2., .25, 5.]))
        obs = np.ones((1, 39))
        apply_keyboard_command(env, obs, (.8, .6, .30))
        np.testing.assert_array_equal(env._command_ranges[:2], np.zeros((2, 2)))
        np.testing.assert_array_equal(obs[:, 6:9], [[0., 0., 1.5]])
        np.testing.assert_array_equal(obs[:, 27:], np.ones((1, 12)))


class TestNativeKeyboardAdapter(unittest.TestCase):
    def test_subscription_consumes_control_keys_focus_loss_and_cleanup(self):
        carb = ModuleType('carb'); carb_input = ModuleType('carb.input'); carb.input = carb_input
        carb_input.KeyboardEventType = SimpleNamespace(KEY_PRESS=1, KEY_RELEASE=2, KEY_REPEAT=3)
        carb_input.DeviceType = SimpleNamespace(KEYBOARD=1, MOUSE=2)
        carb_input.KEYBOARD_MODIFIER_FLAG_CONTROL = 1
        carb_input.KEYBOARD_MODIFIER_FLAG_ALT = 2
        carb_input.KEYBOARD_MODIFIER_FLAG_SUPER = 4
        input_api = MagicMock(); input_api.subscribe_to_input_events.return_value = 42
        carb_input.acquire_input_interface = lambda: input_api
        omni = ModuleType('omni'); appwindow = ModuleType('omni.appwindow'); ui = ModuleType('omni.ui')
        omni.appwindow = appwindow; omni.ui = ui
        app = MagicMock(); app.is_focused.return_value = True
        appwindow.get_default_app_window = lambda: app
        for name in ('Window', 'VStack', 'HStack', 'Label', 'Button'):
            setattr(ui, name, MagicMock())
        ui.Label.side_effect = lambda *args, **kwargs: MagicMock()
        modules = {'carb': carb, 'carb.input': carb_input, 'omni': omni,
                   'omni.appwindow': appwindow, 'omni.ui': ui}
        with patch.dict(sys.modules, modules):
            controller = IsaacKeyboardController(KeyboardConfig(), active=True)
            input_api.subscribe_to_input_events.assert_called_once_with(
                controller._on_input_event, device=app.get_keyboard(), order=0)
            def send(key, kind=1, device=1, modifiers=0):
                return controller._on_input_event(SimpleNamespace(deviceType=device,
                    event=SimpleNamespace(type=kind, modifiers=modifiers, input=SimpleNamespace(name=key))))
            self.assertFalse(send('W'))
            self.assertFalse(send('W', 3))
            self.assertEqual(controller.advance(.02)[0], .8)
            self.assertTrue(send('F'))
            self.assertTrue(send('W', device=2))
            self.assertFalse(send('W', 2))
            self.assertEqual(controller.advance(.02)[0], 0.)
            self.assertTrue(send('S', modifiers=1))
            self.assertEqual(controller.advance(.02)[0], 0.)
            send('W')
            self.assertFalse(send('W', 2, modifiers=1))
            self.assertEqual(controller.advance(.02)[0], 0.)
            send('W'); send('E')
            app.is_focused.return_value = False
            command = controller.advance(.02)
            self.assertEqual(command[:2], (0., 0.))
            self.assertAlmostEqual(command[2], .3004)
            self.assertTrue(send('W'))
            app.is_focused.return_value = True
            self.assertEqual(controller.advance(.02)[0], 0.)
            self.assertFalse(send('SPACE'))
            command = controller.advance(.02)
            self.assertEqual(command[:2], (0., 0.))
            self.assertAlmostEqual(command[2], .3012)
            self.assertIn('NOT executed', controller.jump_status.text)
            self.assertIn('Height goal 0.305', controller.height_status.text)
            before_reset = controller.state.height
            controller._reset_height()
            self.assertEqual(controller.state.height, before_reset)
            self.assertEqual(controller.state.target_height, .30)
            self.assertAlmostEqual(controller.advance(.02)[2], before_reset - .0004)
            window = controller.window
            controller.close(); controller.close()
            input_api.unsubscribe_to_input_events.assert_called_once_with(42)
            window.destroy.assert_called_once()
            handler = MagicMock()
            controller = IsaacKeyboardController(KeyboardConfig(), active=True, jump_handler=handler)
            controller.state.handle_key('SPACE', 'press')
            controller.advance(.02)
            controller.state.handle_key('SPACE', 'repeat')
            controller.advance(.02)
            handler.assert_called_once_with()
            self.assertIn('queued', controller.jump_status.text)
            controller.close()


class TestReplayLauncher(unittest.TestCase):
    def test_both_registered_flat_tasks_accept_keyboard_commands(self):
        for task in ('WheelLeggedVMC-Flat-v0', 'WheelLeggedJointPD-Flat-v0'):
            self.assertTrue(supports_keyboard_task(task))
        for task in ('', 'Other-Flat-v0', 'WheelLeggedJointPD-NotRegistered-v0'):
            self.assertFalse(supports_keyboard_task(task))

    def test_joint_pd_launcher_task_is_supported_by_real_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'model_999.pt'; checkpoint.touch()
            overrides = root / 'overrides.json'; overrides.write_text('{}')
            saved = {'leg_control_mode': 'implicit_joint_position',
                     'rewards': {'profile': DIRECT_JOINT_REWARD_PROFILE}}
            with patch('play_keyboard.checkpoint_config', return_value=saved):
                command = replay_command(root, checkpoint=checkpoint, overrides=overrides)
            task = command[command.index('--task') + 1]
            self.assertEqual(task, 'WheelLeggedJointPD-Flat-v0')
            self.assertTrue(supports_keyboard_task(task))
            self.assertIn('--keyboard', command)

    def test_latest_assessed_model_uses_matching_overrides_not_training_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); output = root / 'output'
            replay = output / 'round_001/replay'; replay.mkdir(parents=True)
            model = root / 'model_199.pt'; model.touch()
            overrides = replay.parent / 'overrides.json'; overrides.write_text('{}')
            assessment = replay / 'assessment.json'
            assessment.write_text(json.dumps({'checkpoint': str(model)}))
            (output / 'status.json').write_text(json.dumps({'last_assessment': str(assessment),
                                                          'checkpoint': str(root / 'model_999.pt')}))
            with patch('play_keyboard.checkpoint_config', return_value={'leg_control_mode': 'explicit_vmc'}):
                command = replay_command(root, output, extra=['--keyboard_speed', '.3'])
            self.assertIn(str(model), command)
            self.assertIn(str(overrides), command)
            self.assertIn('--keyboard', command)
            self.assertNotIn(str(root / 'model_999.pt'), command)
            self.assertEqual(command[-2:], ['--keyboard_speed', '.3'])

    def test_explicit_model_requires_explicit_overrides_and_missing_files_rejected(self):
        with self.assertRaises(ValueError):
            replay_command(PROJECT, PROJECT, checkpoint=Path('model.pt'))
        with self.assertRaises(FileNotFoundError):
            replay_command(PROJECT, PROJECT, checkpoint=Path('not-present.pt'), overrides=Path('not-present.json'))


if __name__ == '__main__':
    unittest.main()
