import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[2] / 'mine_model/scripts/guard_mcu_vmc_training.py'
spec = importlib.util.spec_from_file_location('vmc_training_guard', path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def update(**metrics):
    return dict(iteration=800, metrics={
        'Mean episode length': 999.,
        'Episode_Termination/unsafe_contact_or_orientation': 0.,
        'Episode_Termination/standing_unsafe': 0., **metrics})


class TestTrainingAnomalyGuard(unittest.TestCase):
    def test_normal_standing_is_not_paused(self):
        self.assertIsNone(guard.anomaly([update()] * 3, ''))

    def test_low_reward_is_not_a_physics_failure(self):
        self.assertIsNone(guard.anomaly([update(**{'Mean reward': -50.})] * 3, ''))

    def test_missing_motion_tracking_at_curriculum_start_is_not_fatal(self):
        self.assertIsNone(guard.anomaly([update(**{'Episode_Reward/track_lin_vel': .01})] * 3, ''))

    def test_numerical_error_is_immediate(self):
        self.assertEqual(guard.anomaly([update(**{'Episode_Termination/numerical': .0001})], ''), 'numerical_termination')

    def test_nonfinite_loss_is_immediate(self):
        self.assertEqual(guard.anomaly([update(**{'Mean value_function loss': float('nan')})], ''), 'nonfinite_training_metric')

    def test_three_short_failed_updates_pause_support_failure(self):
        failed = update(**{'Mean episode length': 80., 'Episode_Termination/unsafe_contact_or_orientation': .8})
        self.assertIsNone(guard.anomaly([failed] * 2, ''))
        self.assertEqual(guard.anomaly([failed] * 3, ''), 'support_collapse_three_updates')

    def test_fixed_standing_regression_is_not_masked_by_moving_success(self):
        failed = update(**{'Episode_Termination/standing_unsafe': .3})
        self.assertEqual(guard.anomaly([failed] * 3, ''), 'standing_survival_regression_three_updates')

    def test_physx_warning_is_not_confused_with_error(self):
        self.assertIsNone(guard.anomaly([update()], '[Warning] [omni.physx.plugin] TGS iterations changed'))
        self.assertEqual(guard.anomaly([], '[Error] [omni.physx.plugin] invalid articulation'), 'native_or_cuda_error')

    def test_native_mutex_assertion_cannot_hide_as_a_stalled_training_loop(self):
        for message in ('The following assertion condition failed in the process 143920:',
                        'Recursion not allowed', 'Fatal Python error: Segmentation fault'):
            with self.subTest(message=message):
                self.assertEqual(guard.anomaly([], message), 'native_or_cuda_error')

    def test_incomplete_log_block_is_not_used(self):
        log = ('\x1b[1mLearning iteration 800/1999\x1b[0m\n Mean episode length: 999.00\n ETA: 02:00:00\n'
               'Learning iteration 801/1999\n Mean reward: 10\n')
        values = guard.completed_updates(log)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0]['iteration'], 800)
        self.assertEqual(values[0]['metrics']['Mean episode length'], 999.)

    def test_dead_standing_policy_fails_checkpoint_gate(self):
        result = dict(conditions=[dict(name='stand', trials=[dict(survived=False)])])
        self.assertEqual(guard.replay_gate(result), 'frozen_policy_cannot_stand')

    def test_sustained_opposite_motion_is_not_treated_as_survival_success(self):
        failed = update(**{'Tracking/forward_fraction': .2, 'Tracking/forward_command_m_s': .3,
                           'Tracking/forward_opposite_direction_fraction': .8})
        self.assertIsNone(guard.anomaly([failed] * 2, ''))
        self.assertEqual(guard.anomaly([failed] * 3, ''), 'forward_opposite_motion_three_updates')

    def test_small_untrained_commands_do_not_trigger_reverse_alarm(self):
        early = update(**{'Tracking/forward_fraction': .2, 'Tracking/forward_command_m_s': .06,
                          'Tracking/forward_opposite_direction_fraction': .8})
        self.assertIsNone(guard.anomaly([early] * 3, ''))

    def test_sustained_large_wheel_chatter_is_anomaly(self):
        failed = update(**{'Tracking/wheel_reference_second_difference_rms_rad_s': 12.,
                           'Tracking/wheel_reference_delta_rms_rad_s': 7.})
        self.assertEqual(guard.anomaly([failed] * 3, ''), 'wheel_reference_chatter_three_updates')


if __name__ == '__main__':
    unittest.main()
