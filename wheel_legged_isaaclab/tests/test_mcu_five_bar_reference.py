"""Compare simulation math against the deployed control/vmc.c equations.

The MCU has front/back rows, inward-positive F0 and different polar axes.
Simulation has back/front rows and outward-positive F. Compare after those
explicit frame conversions, never by copying unconverted MCU motor signs.
"""
import math
import unittest

import torch
from wheel_legged_gym_isaaclab.five_bar_vmc import (
    five_bar_state, five_bar_torques,
)
from wheel_legged_gym_isaaclab.robot_model import MAPPED_FIVE_BAR
from wheel_legged_gym_isaaclab.mcu_hardware import GRAVITY_ANGLE_BIAS


def mcu_geometry(phi):
    """Literal get_l_theta geometry, without the IMU angle correction."""
    theta1, theta2 = phi.unbind()
    xb, yb = .21 * theta1.cos(), .21 * theta1.sin()
    xd, yd = .21 * theta2.cos(), .21 * theta2.sin()
    lbd_sq = (xd - xb).square() + (yd - yb).square()
    a0, b0, c0 = 2 * .25 * (xd - xb), 2 * .25 * (yd - yb), lbd_sq
    u2 = 2 * torch.atan2(b0 + (a0.square() + b0.square() - c0.square()).sqrt(),
                         a0 + c0)
    u3 = torch.atan2(yb - yd + .25 * u2.sin(), xb - xd + .25 * u2.cos())
    xc, yc = xb + .25 * u2.cos(), yb + .25 * u2.sin()
    return (xc.square() + yc.square()).sqrt(), torch.atan2(yc, xc), u2, u3


def mcu_mapping(phi):
    """Literal F_Calc J11/J12/J21/J22, arranged by front/back motor."""
    theta1, theta2 = phi.unbind()
    length, u0, u2, u3 = mcu_geometry(phi)
    denom = (u3 - u2).sin()
    j11 = .21 * (u0 - u3).sin() * (theta1 - u2).sin() / denom
    j12 = .21 * (u0 - u3).cos() * (theta1 - u2).sin() / (length * denom)
    j21 = .21 * (u0 - u2).sin() * (u3 - theta2).sin() / denom
    j22 = .21 * (u0 - u2).cos() * (u3 - theta2).sin() / (length * denom)
    return torch.stack((torch.stack((j11, j12)), torch.stack((j21, j22))))


class McuFiveBarReferenceTests(unittest.TestCase):
    def test_mapped_geometry_is_the_deployed_large_five_bar(self):
        for value in MAPPED_FIVE_BAR['upper']:
            self.assertAlmostEqual(value, .21)
        for value in MAPPED_FIVE_BAR['lower']:
            self.assertAlmostEqual(value, .25)
        self.assertEqual(MAPPED_FIVE_BAR['scale'], 1.)

    def test_fk_jacobian_and_motor_efforts_match_deployed_equations(self):
        for b_ref in (-.25, 0., .25):
            for l_ref in (-.25, 0., .25):
                qb = torch.tensor([b_ref, -b_ref], dtype=torch.float64)
                ql = torch.tensor([l_ref, -l_ref], dtype=torch.float64)
                sim_l, sim_angle, sim_j, valid_fk = five_bar_state(qb, ql, **MAPPED_FIVE_BAR)
                self.assertTrue(valid_fk.all())
                for side, motor_sign in enumerate((1., -1.)):
                    with self.subTest(back=b_ref, front=l_ref, side=side):
                        theta_b = MAPPED_FIVE_BAR['offsets'][0] + motor_sign * qb[side]
                        theta_l = MAPPED_FIVE_BAR['offsets'][1] + motor_sign * ql[side]
                        # MCU axes x=-X_base, y=-Z_base; front=q_L, back=q_B.
                        phi = torch.remainder(torch.stack((-theta_l - math.pi/2,
                                                          -theta_b - math.pi/2)), 2*math.pi)
                        fw_l, fw_u, _, _ = mcu_geometry(phi)
                        # CAD import/mapping retains nanometre-scale link
                        # rounding; MCU stores the design lengths as floats.
                        self.assertAlmostEqual(float(fw_l), float(sim_l[side]), delta=1e-7)
                        self.assertAlmostEqual(float(math.pi/2-fw_u), float(sim_angle[side]), delta=1e-7)
                        for pitch in (-.1, 0., .1):
                            fw_world_angle = -(fw_u-1.57) + pitch + math.radians(2.)
                            self.assertAlmostEqual(float(fw_world_angle),
                                float(sim_angle[side]) + pitch + GRAVITY_ANGLE_BIAS, delta=1e-7)
                        fw_jt = mcu_mapping(phi)
                        numerical_jt = torch.autograd.functional.jacobian(
                            lambda x: torch.stack(mcu_geometry(x)[:2]), phi).T
                        torch.testing.assert_close(fw_jt, numerical_jt, rtol=1e-11, atol=1e-12)
                        for force_n, swing_nm in ((-40., -5.), (60.5277, 0.), (150., 3.)):
                            force = torch.full_like(sim_l, force_n)
                            torque = torch.full_like(sim_l, swing_nm)
                            sim_b, sim_front = five_bar_torques(sim_j, force, torque)
                            fw_fb = motor_sign * (fw_jt @ torch.stack((-force[side], torque[side])))
                            torch.testing.assert_close(fw_fb.flip(0),
                                torch.stack((sim_b[side], sim_front[side])), rtol=1e-7, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
