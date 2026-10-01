"""Prevent silently omitting stationary turns or rewarding single-wheel pivots."""

import unittest
import torch
from wheel_legged_gym_isaaclab.yaw_training import sample_yaw_rates, spin_center_velocity_penalty


class TestYawTraining(unittest.TestCase):
    def sample(self, count=20000, **kwargs):
        options = dict(device="cpu", low=-.5, high=.5, fraction=.5, minimum=.15,
                       boundary_fraction=.2, progress=1.)
        options.update(kwargs)
        return sample_yaw_rates(count, **options)

    def test_standing_and_moving_groups_both_keep_straight_and_bidirectional_turns(self):
        torch.manual_seed(52)
        rates = self.sample()
        # The first quarter represents the zero-speed population; yaw sampling must not omit it.
        for group in (rates[:5000], rates[5000:]):
            self.assertGreater((group == 0).float().mean().item(), .45)
            self.assertGreater((group > 0).float().mean().item(), .20)
            self.assertGreater((group < 0).float().mean().item(), .20)
            self.assertLessEqual(group.abs().max().item(), .5)
            self.assertGreaterEqual(group[group != 0].abs().min().item(), .15 - 1e-6)

    def test_curriculum_and_disabled_sampler(self):
        self.assertEqual(self.sample(progress=0).count_nonzero().item(), 0)
        self.assertLessEqual(self.sample(progress=.2).abs().max().item(), .1 + 1e-6)
        state = torch.random.get_rng_state()
        self.assertEqual(self.sample(low=0, high=0).count_nonzero().item(), 0)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))

    def test_spin_penalty_rejects_pivot_but_allows_centered_turn_and_travel(self):
        wheels = torch.tensor([[-1., 1.], [-2., 0.], [4., 6.], [-2., 0.], [-2., 0.]])
        actual = spin_center_velocity_penalty(
            wheels, .0675, torch.tensor([0., 0., .4, .05, 0.]),
            torch.tensor([0., 0., .4, 0., 0.]), torch.tensor([.3, .3, .3, .3, 0.]),
        )
        self.assertEqual(actual[0], 0)
        self.assertGreater(actual[1], 0)
        self.assertTrue(torch.equal(actual[2:], torch.zeros(3)))


if __name__ == "__main__":
    unittest.main()
