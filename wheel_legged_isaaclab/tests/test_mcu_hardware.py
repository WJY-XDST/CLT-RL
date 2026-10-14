"""Firmware caps, gravity, conversion boundaries and replay compatibility."""
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from wheel_legged_gym_isaaclab import mcu_hardware as hw
from wheel_legged_gym_isaaclab.robot_model import MODEL, MAPPED_FIVE_BAR, validate_robot_contract


class McuHardwareTests(unittest.TestCase):
    def test_deployed_torque_calibration_is_kept_separate_from_motor_rating(self):
        self.assertEqual(hw.LEG_MOTOR_MODEL, "翎控 MG8016-i6")
        self.assertIn("owner_confirmed", hw.LEG_TORQUE_CALIBRATION_AUTHORITY)
        self.assertEqual(hw.LEG_TORQUE_PER_AMP, 1.078)

    def test_output_shaft_caps_are_not_multiplied_by_gears_or_upstream_gain(self):
        self.assertAlmostEqual(hw.LEG_EFFORT_LIMIT, 35.574)
        self.assertAlmostEqual(hw.WHEEL_EFFORT_LIMIT, 4.80944)
        self.assertAlmostEqual(hw.GRAVITY_FORCE, 60.5277)

    def test_unspecified_angle_and_speed_limits_are_not_invented(self):
        for name in ('LEG_POSITION_LIMIT_RAD', 'LEG_SPEED_LIMIT_RAD_S', 'WHEEL_SPEED_LIMIT_RAD_S'):
            self.assertIsNone(getattr(hw, name))
        for key in ('leg_position_rad', 'leg_speed_rad_s', 'wheel_speed_rad_s'):
            self.assertIsNone(hw.MCU_LIMIT_EVIDENCE[key]['value'])

    def test_firmware_evidence_matches_the_active_outer_build(self):
        root = Path(__file__).resolve().parents[2] / hw.MCU_SOURCE
        project = (root / 'MDK-ARM/project_h7.uvprojx').read_text(errors='replace')
        for source in ('torque.c', 'vmc.c', 'LK_motor.c', 'wheel.c'):
            self.assertIn(source, project)
        lk = (root / 'cubotMiddleware/Devices/LK_motor.c').read_text(errors='replace')
        wheel = (root / 'control/torque.c').read_text(errors='replace')
        self.assertIn('(torque / 1.078) * 2048 / 33', lk)
        self.assertIn('output > 2048', lk)
        self.assertIn('output < -2048', lk)
        self.assertIn('Data.Output>16000', wheel)
        self.assertIn('Data.Output<-16000', wheel)
        self.assertIn('vmc->feed_torque>150', wheel)
        self.assertIn('vmc->feed_torque<-150', wheel)

    def test_gravity_matches_literal_mcu_formula(self):
        leg = torch.tensor([[-.3, .2], [.4, -.1]], dtype=torch.float64)
        pitch = torch.tensor([[.03], [-.12]], dtype=torch.float64)
        angle = leg + pitch + hw.GRAVITY_ANGLE_BIAS
        result = hw.gravity_feedforward(leg, pitch, mode="mcu_pitch_compensated",
            force=hw.GRAVITY_FORCE, angle_bias=hw.GRAVITY_ANGLE_BIAS)
        torch.testing.assert_close(result, hw.GRAVITY_FORCE / angle.cos())
        self.assertFalse(torch.equal(result[:, 0], result[:, 1]))

    def test_singularity_is_finite_and_capped(self):
        angles = torch.tensor([[math.pi/2, -math.pi/2, 0., .8, 2.]], dtype=torch.float64)
        result = hw.gravity_feedforward(angles, torch.zeros(1, 1),
            mode="mcu_pitch_compensated", force=hw.GRAVITY_FORCE)
        self.assertTrue(torch.isfinite(result).all())
        self.assertLessEqual(result.abs().max(), 150.)
        self.assertAlmostEqual(abs(float(result[0, 0])), 150.)
        self.assertAlmostEqual(float(result[0, 2]), hw.GRAVITY_FORCE)

    def test_constant_mode_preserves_historical_behavior(self):
        result = hw.gravity_feedforward(torch.ones(3, 2), torch.zeros(3, 1),
                                         mode="constant", force=46.857465)
        torch.testing.assert_close(result, torch.full_like(result, 46.857465))

    def test_bad_gravity_configuration_is_rejected(self):
        for kwargs in ({"mode": "wrong", "force": 60.},
                       {"mode": "constant", "force": float("nan")},
                       {"mode": "mcu_pitch_compensated", "force": 60., "force_limit": 0.}):
            with self.assertRaises(ValueError):
                hw.gravity_feedforward(torch.zeros(1, 2), torch.zeros(1, 1), **kwargs)

    def test_software_and_nested_limits_are_synchronized(self):
        cfg = SimpleNamespace(leg_effort_limit=hw.LEG_EFFORT_LIMIT,
            wheel_effort_limit=hw.WHEEL_EFFORT_LIMIT,
            robot=SimpleNamespace(actuators={name: SimpleNamespace(effort_limit_sim=10., effort_limit=10.)
                                             for name in ("legs", "wheels")}))
        hw.synchronize_actuator_limits(cfg)
        for group, expected in (("legs", hw.LEG_EFFORT_LIMIT), ("wheels", hw.WHEEL_EFFORT_LIMIT)):
            self.assertEqual(cfg.robot.actuators[group].effort_limit_sim, expected)
            self.assertIsNone(cfg.robot.actuators[group].effort_limit)

    def test_raw_rpm_conversion_uses_executed_calibration(self):
        actual = hw.wheel_rpm_to_output_speed(torch.tensor([-1000., 1000.]))
        torch.testing.assert_close(actual, torch.full_like(actual, 6.643))
        # Conflicting unused 1:19 declaration is retained as evidence, not used.
        self.assertNotAlmostEqual(hw.WHEEL_RPM_TO_OUTPUT_RAD_S, 2*math.pi/60*hw.WHEEL_HEADER_REDUCTION_RATE)

    def test_historical_and_mcu_profiles_both_validate(self):
        cfg = {"leg_model": "mine_five_bar", "five_bar_geometry": copy.deepcopy(MAPPED_FIVE_BAR),
               **{key: copy.deepcopy(MODEL[key]) for key in ("leg_joint_names", "wheel_joint_names",
                  "wheel_radius", "wheel_track_width", "leg_effort_limit", "wheel_effort_limit")}}
        with self.assertRaises(ValueError):
            validate_robot_contract(cfg)
        cfg.update(hardware_profile=hw.HARDWARE_PROFILE, leg_effort_limit=hw.LEG_EFFORT_LIMIT,
                   wheel_effort_limit=hw.WHEEL_EFFORT_LIMIT)
        validate_robot_contract(cfg)
        cfg["leg_effort_limit"] = 10.
        with self.assertRaises(ValueError):
            validate_robot_contract(cfg)
        cfg["hardware_profile"] = "unknown"
        with self.assertRaises(ValueError):
            validate_robot_contract(cfg)

    def test_robot_contract_can_load_without_package_context(self):
        path = Path(__file__).resolve().parents[1] / "wheel_legged_gym_isaaclab/robot_model.py"
        spec = importlib.util.spec_from_file_location("offline_robot_contract", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual(module.HARDWARE_PROFILE, hw.HARDWARE_PROFILE)


if __name__ == "__main__":
    unittest.main()
