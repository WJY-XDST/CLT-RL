"""No subprocesses: validate repair supervisor's scope and training contract."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mine_model/scripts'))
import optimize_vmc_feedback as supervisor
sys.path.insert(0, str(ROOT / 'wheel_legged_isaaclab'))
from wheel_legged_gym_isaaclab.training_state import restore_adaptive_learning_rate


class FeedbackSupervisorTests(unittest.TestCase):
    def test_adaptive_scheduler_restores_loaded_adam_rate(self):
        alg = SimpleNamespace(learning_rate=3.e-4,
            optimizer=SimpleNamespace(param_groups=[{'lr': 3.375e-5}]))
        self.assertEqual(restore_adaptive_learning_rate(alg), 3.375e-5)
        self.assertEqual(alg.learning_rate, 3.375e-5)

    def test_invalid_loaded_learning_rate_does_not_change_scheduler(self):
        for rates in ([0.], [float('nan')], [1.e-4, 2.e-4], []):
            alg = SimpleNamespace(learning_rate=3.e-4,
                optimizer=SimpleNamespace(param_groups=[{'lr': r} for r in rates]))
            with self.assertRaises(ValueError):
                restore_adaptive_learning_rate(alg)
            self.assertEqual(alg.learning_rate, 3.e-4)

    def test_all_heads_and_standing_identity_are_retained(self):
        cmd = supervisor.training_command(Path('/tmp/source.pt'), 1004, 'test',
            {'commands.fixed_standing_groups': True, 'commands.standing_env_fraction': .25}, first=True)
        self.assertEqual(cmd[cmd.index('--num_envs') + 1], '8192')
        self.assertEqual(cmd[cmd.index('--require_leg_control_mode') + 1], 'explicit_vmc')
        self.assertNotIn('--wheel_head_only_finetune', cmd)
        self.assertNotIn('--wheel_exploration_std', cmd)
        self.assertIn('--reset_optimizer', cmd)
        self.assertEqual(cmd[cmd.index('--reset_exploration_std') + 1], '.05')
        self.assertIn('env.commands.standing_env_fraction=0.25', cmd)

    def test_only_first_segment_resets_optimizer_and_exploration(self):
        cmd = supervisor.training_command(Path('/tmp/source.pt'), 1100, 'test', {}, first=False)
        self.assertNotIn('--reset_optimizer', cmd)
        self.assertNotIn('--reset_exploration_std', cmd)

    def test_replays_match_progress_and_full_acceptance_not_inferred_early(self):
        self.assertEqual(supervisor.replay_scale(1004), .004)
        self.assertEqual(supervisor.replay_scale(1500), .5)
        self.assertEqual(supervisor.replay_scale(2000), 1.)

    def test_repairs_do_not_modify_hardware_or_joint_mapping(self):
        self.assertTrue(all(key.startswith('rewards.') for key in supervisor.REPAIR))
        self.assertEqual(supervisor.MILESTONES[0], 1004)
        self.assertEqual(supervisor.MILESTONES[-1], 2000)

    def test_signal_ladder_refuses_unowned_process(self):
        child = Mock(pid=321)
        child._owned_process_group_id = None
        with patch.object(supervisor.runtime.os, 'killpg') as signal:
            with self.assertRaises(RuntimeError):
                supervisor.runtime.stop_child(child)
            signal.assert_not_called()

    def test_wrapper_exit_does_not_skip_native_child_shutdown(self):
        child = Mock(pid=321)
        child._owned_process_group_id = 321
        child.poll.return_value = 0
        with patch.object(supervisor.runtime.os, 'killpg',
                          side_effect=[None, None, ProcessLookupError]) as signal:
            supervisor.runtime.stop_child(child)
            self.assertEqual(signal.call_args_list[1].args, (321, supervisor.runtime.signal.SIGINT))

    def test_motion_reset_cannot_be_hidden_by_overall_survival_fraction(self):
        metrics = dict(upright_support=True, body_p95_deg=.5, speed_mae_m_s=.001, yaw_mae_rad_s=.001)
        result = dict(conditions=[dict(name='stand', trials=[dict(survived=True, metrics=metrics)])],
                      survivors=26, total_trials=27, resets=1)
        self.assertEqual(supervisor.feedback_replay_gate(result, dict(stop_reason=None)),
                         'frozen_replay_reset_requires_diagnosis')


if __name__ == '__main__':
    unittest.main()
