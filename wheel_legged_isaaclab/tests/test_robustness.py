"""Causal latency, reset isolation and non-invasive disabled-mode checks."""
import unittest
import torch

from wheel_legged_gym_isaaclab.robustness import DelayHistory, RobustnessSpec, RobustnessSession


class RobustnessTests(unittest.TestCase):
    def test_zero_and_heterogeneous_delay_use_only_existing_samples(self):
        queue = DelayHistory(3, 1, 3, device='cpu')
        queue.reset(torch.arange(3), torch.full((3, 1), -1.))
        ages = torch.tensor([0, 1, 3])
        for step in range(8):
            queue.push(torch.full((3, 1), float(step)))
            actual = queue.read(ages)[:, 0]
            expected = torch.tensor([float(step), float(max(-1, step - 1)), float(max(-1, step - 3))])
            self.assertTrue(torch.equal(actual, expected))

    def test_reset_clears_every_slot_only_for_selected_robots(self):
        queue = DelayHistory(2, 1, 4, device='cpu')
        for step in range(9):
            queue.push(torch.tensor([[float(step)], [float(step + 10)]]))
        retained = queue.values[:, 1].clone()
        queue.reset(torch.tensor([0]), torch.tensor([[-3.]]))
        self.assertTrue(torch.equal(queue.values[:, 1], retained))
        for age in range(5):
            self.assertEqual(queue.read(torch.tensor([age, age]))[0, 0].item(), -3.)

    def test_disabled_does_not_require_or_mutate_an_environment_or_RNG(self):
        before = torch.random.get_rng_state().clone()
        value = RobustnessSession(None, RobustnessSpec())
        self.assertFalse(value.applied)
        self.assertEqual(value.evidence['hooks_installed'], 0)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_unknown_fields_and_invalid_distributions_fail_closed(self):
        for item in ({'height_specific_gain': 1.}, {'mass_scale': [0., 1.]},
                     {'friction_scale': [1., .5]}, {'action_delay_steps': [0, 1.5]},
                     {'sensor_delay_steps': [-1, 1]}, {'gyro_bias_rad_s': float('nan')},
                     {'enabled': 1}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                RobustnessSpec.from_dict(item)


if __name__ == '__main__':
    unittest.main()
