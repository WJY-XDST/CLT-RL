import unittest
from types import SimpleNamespace
from wheel_legged_gym_isaaclab.config_overrides import (
    RETIRED_CONFIG_FIELDS, RETIRED_SAMPLING_FIELDS, apply_config_overrides, strip_retired_config_fields,
)


class TestReplayOverrides(unittest.TestCase):
    def test_actuator_dictionary_and_root_fields_restore_together(self):
        cfg=SimpleNamespace(wheel_damping=.5,robot=SimpleNamespace(actuators={'wheels':SimpleNamespace(damping=.5)}))
        apply_config_overrides(cfg,{'wheel_damping':1.25,'robot.actuators.wheels.damping':1.25})
        self.assertEqual(cfg.wheel_damping,1.25)
        self.assertEqual(cfg.robot.actuators['wheels'].damping,1.25)

    def test_typo_cannot_silently_create_an_unused_gain(self):
        with self.assertRaisesRegex(ValueError,'Unknown override'):
            apply_config_overrides(SimpleNamespace(wheel_damping=.5),{'wheel_dampng':2.})

    def test_old_saved_fields_are_removed_without_changing_input_or_active_scales(self):
        old = {'obs_scales': {'dof_acc': .0025, 'torque': .05, 'dof_vel': .05},
               'commands': {'num_commands': 3, 'ranges_height': [.28, .32]},
               'rewards': {'max_contact_force': 100., 'dof_acc': -2.5e-7}}
        result = strip_retired_config_fields(old)
        self.assertEqual(result, {'obs_scales': {'dof_vel': .05},
                                  'commands': {'ranges_height': [.28, .32]},
                                  'rewards': {'dof_acc': -2.5e-7}})
        self.assertEqual(old['obs_scales']['dof_acc'], .0025)
        result['commands']['ranges_height'][0] = 0.
        self.assertEqual(old['commands']['ranges_height'][0], .28)

    def test_only_exact_retired_override_keys_are_ignored_and_reported(self):
        cfg = SimpleNamespace(wheel_damping=.5)
        with self.assertWarnsRegex(UserWarning, 'retired no-op'):
            apply_config_overrides(cfg, {**dict.fromkeys(RETIRED_CONFIG_FIELDS, 123),
                                         'wheel_damping': 1.25})
        self.assertEqual(vars(cfg), {'wheel_damping': 1.25})
        for key in ('obs_scales.dof_accel', 'rewards.dof_acc', 'env.commands.num_commands'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Unknown override'):
                apply_config_overrides(cfg, {key: 1})

    def test_inactive_old_sampling_fields_do_not_change_grouped_or_fixed_replay(self):
        for commands in ({'grouped_training': True, 'ranges_lin_vel_x': [.2, .8]},
                         {'grouped_training': False, 'ranges_lin_vel_x': [0., 0.]}):
            old = {'commands': {**commands, **dict.fromkeys(RETIRED_SAMPLING_FIELDS, .25)}}
            self.assertEqual(strip_retired_config_fields(old), {'commands': commands})
            self.assertTrue(RETIRED_SAMPLING_FIELDS <= set(old['commands']))

    def test_active_retired_sampler_is_rejected_not_reinterpreted(self):
        with self.assertRaisesRegex(ValueError, 'non-grouped random sampler'):
            strip_retired_config_fields({'commands': {'grouped_training': False,
                'ranges_lin_vel_x': [.2, .8], 'transition_env_fraction': .25}})
        with self.assertRaisesRegex(ValueError, 'Unknown override'):
            apply_config_overrides(SimpleNamespace(commands=SimpleNamespace()),
                                   {'commands.yaw_env_fraction': .5})
