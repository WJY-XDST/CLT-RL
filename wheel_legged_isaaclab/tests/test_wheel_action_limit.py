import unittest

import torch

from wheel_legged_gym_isaaclab.action_bias_probe import wheel_reference_probe
from wheel_legged_gym_isaaclab.wheel_control import (
    apply_wheel_action_limit, checkpoint_wheel_action_limit, validated_wheel_action_limit,
)


class WheelActionLimitTests(unittest.TestCase):
    def test_historical_limit_keeps_existing_action_tensor(self):
        raw = torch.tensor([[-3., 2., -1.4, .2, -.1, 1.8]])
        applied = raw.clamp(-1., 1.)
        self.assertIs(apply_wheel_action_limit(applied, raw, 1.), applied)
        self.assertEqual(checkpoint_wheel_action_limit({}), 1.)

    def test_saved_limit_matches_frozen_intervention_and_preserves_legs(self):
        values = torch.tensor([-3., -1.25, -1., -.3, 0., .3, 1., 1.25, 3.])
        raw = values.repeat(6, 1).T.contiguous()
        applied = raw.clamp(-1., 1.)
        original = applied.clone()
        rng = torch.random.get_rng_state().clone()
        actual = apply_wheel_action_limit(applied, raw, 1.25)
        self.assertTrue(torch.equal(actual, wheel_reference_probe(applied, raw, 1.25)))
        self.assertTrue(torch.equal(actual[:, (0, 1, 3, 4)], applied[:, (0, 1, 3, 4)]))
        self.assertTrue(torch.equal(applied, original))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(float((actual[:, (2, 5)] * 15.).abs().max()), 18.75)

    def test_saved_configuration_and_explicit_transfer(self):
        self.assertEqual(checkpoint_wheel_action_limit({'wheel_action_limit': 1.25}), 1.25)
        self.assertEqual(checkpoint_wheel_action_limit({}, {'wheel_action_limit': 1.25}), 1.25)
        self.assertEqual(checkpoint_wheel_action_limit({'wheel_action_limit': 1.25}, {}), 1.25)
        for value in (True, '1.25', None, 0., .9, 2.01, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validated_wheel_action_limit(value)


if __name__ == '__main__':
    unittest.main()
