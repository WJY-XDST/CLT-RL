"""Command-to-policy integration and opt-in native Tk/viewer smoke tests."""

import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import torch

from core import Simulation
from test_interface import vehicle_fixture
from keyboard import KeyboardConfig, KeyboardCommandState, KeyboardInput, MujocoKeyboardController
from test_keyboard import IdleScheduler
import run


class KeyboardIntegrationTests(unittest.TestCase):
    def test_commands_reach_observation_without_replacing_other_features(self):
        _, cfg, mjcf = vehicle_fixture()
        with tempfile.TemporaryDirectory() as directory:
            sim = Simulation(cfg, mujoco.MjModel.from_xml_path(str(mjcf)))
            height = cfg['rewards']['base_height_target']
            sim.command[:] = [0, 0, height]
            before = sim.observation().clone()
            low, high = cfg['commands']['ranges_height']
            state = KeyboardCommandState(KeyboardConfig(height_min=low, height_max=high, initial_height=height))
            state.handle_key('W', 'press')
            state.handle_key('A', 'press')
            state.handle_key('E', 'press')
            dt = cfg['sim']['dt'] * cfg['decimation']
            target = state.advance(dt)
            limit = cfg['commands']['linear_acceleration_limit'] * dt
            sim.command[0] += np.clip(target[0] - sim.command[0], -limit, limit)
            sim.command[1:] = target[1:]
            after = sim.observation()
            scales = cfg['obs_scales']
            expected = torch.tensor(sim.command * [scales['lin_vel'], scales['ang_vel'], scales['height_measurements']], dtype=torch.float32)
            torch.testing.assert_close(after[0, 6:9], expected)
            torch.testing.assert_close(after[:, :6], before[:, :6], rtol=0, atol=0)
            torch.testing.assert_close(after[:, 9:], before[:, 9:], rtol=0, atol=0)
            self.assertAlmostEqual(sim.command[0], limit)
            self.assertAlmostEqual(sim.command[2], height + .0004)

    def test_replay_takeover_release_and_panel_close_do_not_restore_cycle(self):
        class ScriptedPanel(MujocoKeyboardController):
            def __init__(self, config):
                self.state = KeyboardCommandState(config)
                self.input = KeyboardInput(self.state, IdleScheduler())
                self.window, self.closed = Mock(), False
                self._tk = SimpleNamespace(TclError=RuntimeError)
                self.step = -1

            def advance(self, dt, fallback_height):
                self.step += 1
                if self.step == 1:
                    self.input.press('W')
                    self.input.press('E')
                elif self.step == 2:
                    self.input.release('W')
                    self.input.release('E')
                    self.input.scheduler.flush()
                elif self.step == 3:
                    self.input.press('A')
                elif self.step == 4:
                    self.input.release('A')
                    self.input.scheduler.flush()
                elif self.step == 5:
                    self.close()
                return super().advance(dt, fallback_height)

            def show_state(self, *args):
                pass

        viewer = Mock()
        viewer.__enter__ = Mock(return_value=viewer)
        viewer.__exit__ = Mock(return_value=False)
        viewer.is_running.return_value = True
        viewer.cam = SimpleNamespace(distance=0, elevation=0, azimuth=0, lookat=np.zeros(3))
        viewer.opt = SimpleNamespace(geomgroup=np.zeros(6))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, cfg, mjcf = vehicle_fixture()
            argv = ['run.py', '--checkpoint', str(checkpoint), '--mjcf', str(mjcf),
                    '--fixed-command', '0', '0', str(cfg['rewards']['base_height_target']), '--velocity-cycle', '.4',
                    '--yaw-cycle', '.2', '--max-steps', '10', '--output', directory]
            with patch.object(sys, 'argv', argv), patch('mujoco.viewer.launch_passive', return_value=viewer), \
                    patch.object(run, 'MujocoKeyboardController', ScriptedPanel), patch.object(run.time, 'sleep'):
                self.assertEqual(run.main(), 0)
            with (Path(directory) / 'trace.csv').open() as trace:
                rows = list(csv.DictReader(trace))
            self.assertEqual(rows[0]['command_source'], 'cycle')
            self.assertEqual(float(rows[0]['cmd_x_target']), .4)
            self.assertEqual(float(rows[1]['cmd_x_target']), .8)
            self.assertEqual(float(rows[2]['cmd_x_target']), 0.)
            self.assertEqual(float(rows[3]['cmd_yaw']), .6)
            self.assertEqual(float(rows[4]['cmd_yaw']), 0.)  # No implicit heading hold.
            for row in rows[5:]:
                self.assertEqual(row['command_source'], 'keyboard')
                self.assertEqual(float(row['cmd_x_target']), 0.)
                self.assertEqual(float(row['cmd_yaw']), 0.)
            heights = np.array([float(row['cmd_height']) for row in rows[1:]])
            self.assertTrue(np.all(np.abs(np.diff(heights)) <= .0004 + 1e-12))
            self.assertTrue(json.loads((Path(directory) / 'metadata.json').read_text())['keyboard_enabled'])
            self.assertFalse(json.loads((Path(directory) / 'summary.json').read_text())['conditions'][0]['constant_targets'])


@unittest.skipUnless(os.environ.get('MUJOCO_KEYBOARD_GUI_TEST') == '1', 'Opt in to opening a short-lived Tk/viewer window')
class NativeGuiTests(unittest.TestCase):
    def test_native_key_events_and_viewer_lifecycle(self):
        from mujoco import viewer
        panel = MujocoKeyboardController(KeyboardConfig())
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
        try:
            panel.window.update()
            panel.window.focus_force()
            panel.window.update()
            with viewer.launch_passive(model, mujoco.MjData(model)) as handle:
                panel.window.focus_force()
                panel.window.update()
                panel.window.event_generate('<KeyPress>', keysym='w')
                self.assertEqual(panel.advance(.02, .3)[:2], (.8, 0.))
                panel.window.event_generate('<KeyRelease>', keysym='w')
                self.assertEqual(panel.advance(.02, .3)[:2], (0., 0.))
                panel.window.event_generate('<KeyPress>', keysym='e')
                self.assertAlmostEqual(panel.advance(.02, .3)[2], .3004)
                panel.window.event_generate('<KeyRelease>', keysym='e')
                panel.advance(.02, .3)
                goal = panel.state.target_height
                for _ in range(15):
                    panel.advance(.02, .3)
                self.assertAlmostEqual(panel.state.height, goal)
                panel.window.event_generate('<KeyPress>', keysym='space')
                panel.advance(.02, .3)
                self.assertEqual(panel.state.jump_requests, 1)
                self.assertTrue(handle.is_running())  # SPACE did not pause/close the viewer.
                handle.sync()
                panel.show_state((0, 0, panel.state.height), 0, .3, 0, 0)
                panel.window.update()
                panel.close()
                self.assertEqual(panel.advance(.02, .3)[:2], (0., 0.))
        finally:
            panel.close()


if __name__ == '__main__':
    unittest.main()
