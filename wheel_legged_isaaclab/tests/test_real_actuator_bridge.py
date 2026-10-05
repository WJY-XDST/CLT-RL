"""Check units, signs and limits without a simulator or motor connection."""
import math
from pathlib import Path
import unittest

from deployment.actuator_bridge import (
    ActuatorBridge,LegPDConfig,WheelPIDConfig,WheelSpeedCurrentPID,
    leg_joint_pd,load_bridge_config,
)


def wheel_cfg(**overrides):
    values=dict(kp=2.0,ki=0.0,kd=0.0,torque_per_amp=0.5,current_limit_a=8.0,
                torque_limit_nm=10.0,current_slew_a_s=10000.0,
                derivative_filter_s=0.0,max_dt_s=0.01)
    values.update(overrides)
    return WheelPIDConfig(**values)


def leg_cfg():
    return LegPDConfig((300.0,)*4,(3.0,)*4,(10.0,)*4,(-1.57,)*4,(1.57,)*4)


class TestRealActuatorBridge(unittest.TestCase):
    def test_nominal_pd_and_current_units_match_reference_drive_law(self):
        torque=leg_joint_pd(leg_cfg(),[0.01]*4,[0.0]*4,[0.2]*4)
        for value in torque:self.assertAlmostEqual(value,2.4)
        result=WheelSpeedCurrentPID(wheel_cfg()).step(2.0,1.0,0.005)
        self.assertAlmostEqual(result['torque_nm'],2.0)
        self.assertAlmostEqual(result['current_a'],4.0)

    def test_physical_current_and_torque_caps_both_apply(self):
        for cfg,expected in ((wheel_cfg(),4.0),
                             (wheel_cfg(torque_limit_nm=1.0),1.0)):
            result=WheelSpeedCurrentPID(cfg).step(100.0,0.0,0.005)
            self.assertAlmostEqual(result['torque_nm'],expected)
            self.assertTrue(result['limited'])
        self.assertEqual(leg_joint_pd(leg_cfg(),[0.5]*4,[0.0]*4,[0.0]*4),(10.0,)*4)

    def test_mirrored_joint_axis_and_driver_polarity_are_separate(self):
        bridge=ActuatorBridge(leg_cfg(),[wheel_cfg(),wheel_cfg()],[1,-1],[1,-1])
        result=bridge.step([0.0]*4,[0.0]*4,[0.0]*4,[1.0,1.0],[0.0,0.0],0.005)
        # Raw right-axis torque is negative. Driver polarity maps that torque
        # to a positive current command for this synthetic calibration.
        self.assertEqual(result['wheel_currents_a'],(4.0,4.0))
        self.assertEqual(result['wheel_diagnostics'][1]['torque_nm'],-2.0)

    def test_integral_cannot_wind_up_against_saturated_current(self):
        pid=WheelSpeedCurrentPID(wheel_cfg(ki=4.0))
        for _ in range(1000):pid.step(100.0,0.0,0.005)
        self.assertEqual(pid.integral_torque,0.0)
        self.assertEqual(pid.step(0.0,0.0,0.005)['current_a'],0.0)

    def test_integral_unwinds_when_error_changes_direction(self):
        pid=WheelSpeedCurrentPID(wheel_cfg(kp=0.0,ki=1.0))
        for _ in range(100):pid.step(1.0,0.0,0.005)
        self.assertAlmostEqual(pid.integral_torque,0.5)
        for _ in range(100):pid.step(-1.0,0.0,0.005)
        self.assertAlmostEqual(pid.integral_torque,0.0)

    def test_slew_limit_and_derivative_on_feedback(self):
        pid=WheelSpeedCurrentPID(wheel_cfg(current_slew_a_s=20.0))
        self.assertAlmostEqual(pid.step(100.0,0.0,0.005)['current_a'],0.1)
        self.assertAlmostEqual(pid.step(100.0,0.0,0.005)['current_a'],0.2)
        derivative=WheelSpeedCurrentPID(wheel_cfg(kp=0.0,kd=1.0))
        derivative.step(0.0,0.0,0.005)
        self.assertEqual(derivative.step(100.0,0.0,0.005)['current_a'],0.0)
        self.assertLess(derivative.step(100.0,0.01,0.005)['current_a'],0.0)

    def test_disabled_and_invalid_feedback_clear_controller_state(self):
        pid=WheelSpeedCurrentPID(wheel_cfg(ki=1.0))
        pid.step(1.0,0.0,0.005)
        self.assertEqual(pid.step(1.0,0.0,0.005,enabled=False)['current_a'],0.0)
        for args in ((math.nan,0.0,0.005),(1.0,0.0,0.02)):
            with self.assertRaises(ValueError):pid.step(*args)
            self.assertEqual(pid.current_a,0.0)
            self.assertEqual(pid.integral_torque,0.0)
        with self.assertRaises(ValueError):pid.step(1.0,0.0,0.005,feedback_age_s=0.02)

    def test_invalid_leg_state_is_rejected_and_resets_both_wheels(self):
        bridge=ActuatorBridge(leg_cfg(),[wheel_cfg(),wheel_cfg()],[1,-1],[1,1])
        bridge.step([0.0]*4,[0.0]*4,[0.0]*4,[1.0,1.0],[0.0,0.0],0.005)
        with self.assertRaises(ValueError):
            bridge.step([2.0]*4,[0.0]*4,[0.0]*4,[1.0,1.0],[0.0,0.0],0.005)
        self.assertTrue(all(w.current_a==0.0 for w in bridge.wheels))

    def test_uncalibrated_template_and_invalid_constants_are_rejected(self):
        path=Path(__file__).resolve().parents[2]/'mine_model/config/real_actuator_template.json'
        with self.assertRaisesRegex(ValueError,'calibration'):load_bridge_config(path)
        for changes in ({'torque_per_amp':0.0},{'current_limit_a':-1.0},{'ki':-1.0}):
            with self.assertRaises(ValueError):wheel_cfg(**changes)


if __name__=='__main__':unittest.main()
