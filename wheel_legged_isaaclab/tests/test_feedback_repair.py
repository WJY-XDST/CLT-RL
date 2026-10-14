"""CPU regression for reward repair, frozen feedback and log publication."""
import importlib.util
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mine_model/scripts'))
sys.path.insert(0, str(ROOT / 'wheel_legged_isaaclab'))
from diagnose_policy_feedback import diagnose_arrays, wheel_spectrum
from wheel_legged_gym_isaaclab.reward_signals import action_second_difference_penalty, merge_step_diagnostics
from wheel_legged_gym_isaaclab.yaw_training import curriculum_replay_cases


def fixture():
    o = np.zeros((500, 6, 27))
    o[:, :, 5] = -1
    o[:, 3:, 6] = .3 * 2
    manifest = dict(policy_dt=.02, seconds_per_case=10, seeds=[83, 84, 85],
                    cases=[dict(name='stand', speed=0.), dict(name='forward', speed=.3)])
    cfg = dict(obs_scales=dict(lin_vel=2.), action_scale_vel=15., rewards=dict(
        lin_vel_error_sq=-30., lin_vel_penalty_clip=80., orientation=-150.,
        orientation_roll_multiplier=4., orientation_penalty_clip=30.))
    return o, np.zeros((500, 6, 6)), manifest, cfg, dict(completed=True, samples=3000, resets=0)


class FeedbackRepairTests(unittest.TestCase):
    def test_replay_main_speed_does_not_start_at_scaled_zero(self):
        cfg = SimpleNamespace(ranges_lin_vel_x=(.2, .8), ranges_reverse_lin_vel_x=(-.8, -.2), ranges_height=(.28, .32))
        cases = curriculum_replay_cases(.3, cfg, .1)
        self.assertAlmostEqual(cases[1][1], .21)
        self.assertAlmostEqual(cases[2][1], .24)
        self.assertAlmostEqual(cases[3][1], -.21)
        self.assertAlmostEqual(cases[5][2], .03)
        self.assertAlmostEqual(cases[7][3], .298)

    def test_full_curriculum_retains_the_original_acceptance_commands(self):
        cfg = SimpleNamespace(ranges_lin_vel_x=(.2, .8), ranges_reverse_lin_vel_x=(-.8, -.2), ranges_height=(.28, .32))
        cases = curriculum_replay_cases(.3, cfg, 1.)
        self.assertEqual([c[1] for c in cases[:5]], [0., .3, .6, -.3, -.6])
        self.assertEqual(cases[5][2], .3)

    def test_constant_wheel_reference_remains_unpenalized(self):
        a = torch.tensor([[0., 0., .6, 0., 0., -.1]])
        self.assertEqual(action_second_difference_penalty(a, a, a, 5.).item(), 0.)

    def test_linear_wheel_ramp_has_zero_second_difference(self):
        a = torch.tensor([[0., 0., .2, 0., 0., .1]])
        self.assertLess(action_second_difference_penalty(3*a, 2*a, a, 5.).item(), 1.e-12)

    def test_both_wheel_columns_are_penalized(self):
        a = torch.tensor([[0., 0., 1., 0., 0., -1.]])
        self.assertEqual(action_second_difference_penalty(a, -a, a, 5.).item(), 160.)
        self.assertEqual(action_second_difference_penalty(a, -a, a, 0.).item(), 0.)

    def test_invalid_history_and_weight_refused(self):
        a = torch.zeros(2, 6)
        for weight in (-1, float('nan')):
            with self.assertRaises(ValueError):
                action_second_difference_penalty(a, a, a, weight)
        with self.assertRaises(ValueError):
            action_second_difference_penalty(a, a[:1], a, 1.)

    def test_diagnostics_survive_reset_log_replacement_without_aliasing(self):
        cached = {'Tracking/speed': .1}
        reset_log = {'Episode_Reward/total': 1.}
        before = merge_step_diagnostics(reset_log, cached)
        before['Tracking/speed'] = torch.tensor(.9)  # RSL logger mutates values
        after = merge_step_diagnostics({'Episode_Reward/total': 2.}, cached)
        self.assertEqual(after['Tracking/speed'], .1)
        self.assertEqual(after['Episode_Reward/total'], 2.)
        self.assertNotIn('Tracking/speed', reset_log)

    def test_small_numerical_spectrum_is_not_chatter(self):
        values = fixture()
        values[1][:, :, 2] = 1.e-4 * (-1.) ** np.arange(500)[:, None]
        report = diagnose_arrays(*values)
        self.assertIsNone(report['stop_reason'])

    def test_repeated_reverse_motion_is_diagnosed(self):
        values = fixture()
        values[0][:, 3:, 17] = -.4
        report = diagnose_arrays(*values)
        self.assertEqual(report['stop_reason'], 'opposite_motion')

    def test_balancing_chatter_without_motion_is_diagnosed(self):
        values = fixture()
        values[1][:, :, 5] = .3 * (-1.) ** np.arange(500)[:, None]
        report = diagnose_arrays(*values)
        self.assertEqual(report['stop_reason'], 'wheel_reference_chatter')
        self.assertEqual(report['trials'][0]['metrics']['wheel_spectra'][1]['peak_hz'], 25.)

    def test_old_caps_saturate_while_new_caps_keep_error_discrimination(self):
        speed_cost = 30 * np.array([.3, .6, 1.2]) ** 2
        self.assertTrue(np.all(np.minimum(speed_cost, 1) == 1))
        self.assertTrue(np.all(np.diff(np.minimum(speed_cost, 80)) > 0))
        pitch_cost = 150 * np.sin(np.deg2rad([8., 11., 17.])) ** 2
        self.assertTrue(np.all(np.diff(np.minimum(pitch_cost, 30)) > 0))

    def test_production_repair_and_post_reset_publication_are_wired(self):
        directory = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat'
        cfg = (directory / 'wheel_legged_vmc_flat_env_cfg.py').read_text()
        for assignment in ('lin_vel_penalty_clip = 8.0', 'orientation_penalty_clip = 30.0',
                           'action_smooth_wheel_multiplier = 5.0', 'termination = -20.0'):
            self.assertIn(assignment, cfg)
        from env_source import environment_source
        env = environment_source()
        self.assertIn("self.extras['log'] = merge_step_diagnostics(", env)
        self.assertIn('self._tracking_diagnostics = diagnostics', env)


if __name__ == '__main__':
    unittest.main()
