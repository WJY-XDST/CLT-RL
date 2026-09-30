"""Check virtual work against independent derivatives of the leg coordinates."""

import unittest

import torch

from wheel_legged_gym_isaaclab.vmc import leg_coordinates, virtual_leg_torques


class TestVirtualWork(unittest.TestCase):
    def test_mirrored_joint_torques_match_coordinate_gradients(self):
        generator = torch.Generator().manual_seed(27)
        sign = torch.tensor([1.0, -1.0], dtype=torch.float64)
        hip = ((torch.rand(64, 2, generator=generator, dtype=torch.float64) + 0.1) * sign).requires_grad_()
        knee = ((torch.rand(64, 2, generator=generator, dtype=torch.float64) * 1.5 - 0.5) * sign).requires_grad_()
        theta1, theta2 = hip * sign, knee * sign + torch.pi / 2.0
        length, angle = leg_coordinates(theta1, theta2, l1=0.15, l2=0.25, offset=0.054)
        force = torch.randn(64, 2, generator=generator, dtype=torch.float64) * 50.0
        torque = torch.randn(64, 2, generator=generator, dtype=torch.float64) * 5.0
        # Virtual work: tau.dq = F.dL + T.dtheta for arbitrary F, T, and dq.
        expected = torch.autograd.grad((force * length + torque * angle).sum(), (hip, knee))
        actual = virtual_leg_torques(theta1, theta2, length, angle, force, torque, l1=0.15, l2=0.25)
        for reference, canonical in zip(expected, actual):
            torch.testing.assert_close(canonical * sign, reference, rtol=1e-10, atol=1e-10)

    def test_nominal_hip_angular_coefficient(self):
        q1 = torch.tensor(0.5, dtype=torch.float64)
        q2 = torch.tensor(0.35 + torch.pi / 2.0, dtype=torch.float64)
        length, angle = leg_coordinates(q1, q2, l1=0.15, l2=0.25, offset=0.054)
        actual, _ = virtual_leg_torques(q1, q2, length, angle, 0.0, 1.0, l1=0.15, l2=0.25)
        epsilon = 1e-6
        _, plus = leg_coordinates(q1 + epsilon, q2, l1=0.15, l2=0.25, offset=0.054)
        _, minus = leg_coordinates(q1 - epsilon, q2, l1=0.15, l2=0.25, offset=0.054)
        self.assertAlmostEqual(actual.item(), ((plus - minus) / (2 * epsilon)).item(), places=8)
        legacy, _ = virtual_leg_torques(
            q1, q2, length, angle, 0.0, 1.0, l1=0.15, l2=0.25, legacy_angular_mapping=True
        )
        self.assertAlmostEqual(legacy.item(), 0.4052898352496305, places=10)
        self.assertGreater(abs(actual.item() - legacy.item()), 0.5)


if __name__ == "__main__":
    unittest.main()
