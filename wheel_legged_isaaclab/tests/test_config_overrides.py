import unittest
from types import SimpleNamespace
from wheel_legged_gym_isaaclab.config_overrides import apply_config_overrides


class TestReplayOverrides(unittest.TestCase):
    def test_actuator_dictionary_and_root_fields_restore_together(self):
        cfg=SimpleNamespace(wheel_damping=.5,robot=SimpleNamespace(actuators={'wheels':SimpleNamespace(damping=.5)}))
        apply_config_overrides(cfg,{'wheel_damping':1.25,'robot.actuators.wheels.damping':1.25})
        self.assertEqual(cfg.wheel_damping,1.25)
        self.assertEqual(cfg.robot.actuators['wheels'].damping,1.25)

    def test_typo_cannot_silently_create_an_unused_gain(self):
        with self.assertRaisesRegex(ValueError,'Unknown override'):
            apply_config_overrides(SimpleNamespace(wheel_damping=.5),{'wheel_dampng':2.})
