"""Selected Actor contract and time/feedback behavior, without physical UART."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from wheelleg_deploy.numpy_policy import NumpyPolicy
from wheelleg_deploy.nx_runtime import NxControlSession, run_serial
from wheelleg_deploy.observation import build_observation, CommandRamp
from wheelleg_deploy.schema import make_schema
from wheelleg_deploy.serial_interface import RobotSerialInterface
from wheelleg_deploy.torque_control import TorqueController, gravity_effort
from wheelleg_deploy import wire as w

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("CLT_NX_TEST_BUNDLE", str(ROOT/"bundles/selected_nx_policy")))


class Clock:
    now = 0.
    def __call__(self):
        return self.now


class Link:
    def __init__(self):
        self.incoming, self.sent = b"", []
    def read(self):
        result, self.incoming = self.incoming, b""
        return result
    def write(self, data):
        self.sent.append(data)
        return len(data)
    def close(self):
        pass


class FakePolicy:
    def __init__(self, cfg, action=None):
        self.cfg = copy.deepcopy(cfg)
        self.schema = make_schema(cfg)
        self.metadata = {"policy_period_s": .02}
        self.action = [0.]*6 if action is None else action
        self.inputs = []
        self.slow = None
    def __call__(self, obs):
        self.inputs.append(list(obs))
        if self.slow:
            self.slow()
        return list(self.action)


def sample(policy, raw=None, command=None):
    if raw is None:
        raw = policy.cfg["leg_joint_position_offset"] + [0.]*6
    if command is None:
        command = [0., 0., sum(policy.cfg["commands"]["ranges_height"])/2]
    physical = [0., 0., 0., 0., 0., -1.] + raw[:4] + raw[4:8] + [0., 0.] + raw[8:10] + [0.]*6
    obs = build_observation(physical, command, policy.cfg, policy.schema)
    return obs, raw


@unittest.skipUnless((BUNDLE/"manifest.json").is_file(), "Selected portable bundle is required")
class NxRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((BUNDLE/"runtime_config.json").read_text())

    def setUp(self):
        self.policy = FakePolicy(self.config)
        self.clock, self.link = Clock(), Link()
        self.interface = RobotSerialInterface(self.link, "11"*32, session=7, clock=self.clock)
        self.session = NxControlSession(self.policy, self.interface, clock=self.clock)

    def feed(self, seq=1, ack=0, flags=3, raw=None, command=None, obs_change=None):
        obs, raw = sample(self.policy, raw=raw, command=command)
        if obs_change:
            obs[obs_change[0]] = obs_change[1]
        tick = 1000 + round(self.clock.now*1e6)
        self.link.incoming += w.Packet(w.STATUS, 7, seq, tick,
                                      w.status_payload("11"*32, flags, ack, obs, raw)).encode()

    def test_actor_is_50hz_while_pd_updates_at_200hz(self):
        self.policy.action = [.1, -.1, .2, -.1, .1, .2]
        outputs = []
        for step in range(20):
            self.clock.now = step*.005
            raw = self.config["leg_joint_position_offset"] + [0.]*6
            raw[0] += step*.001
            self.feed(step+1, raw=raw, command=[.8, .2, .3])
            outputs.append(self.session.step())
        self.assertEqual(len(self.policy.inputs), 5)
        self.assertEqual(sum(o["inference_performed"] for o in outputs), 5)
        self.assertAlmostEqual(self.session.command[0], .08)
        self.assertNotEqual(outputs[0]["torque_nm"][0], outputs[1]["torque_nm"][0])
        self.assertEqual(outputs[0]["position_target_rad"], outputs[1]["position_target_rad"])

    def test_no_first_state_or_stale_state_latches_stop(self):
        self.clock.now = 1.01
        with self.assertRaisesRegex(RuntimeError, "No compatible"):
            self.session.step()
        self.assertEqual(w.BinaryCodec().feed(self.link.sent[-1])[0].kind, w.STOP)
        self.setUp()
        self.feed()
        self.session.step()
        self.clock.now = .01
        self.session.step()
        self.clock.now = .031
        with self.assertRaisesRegex(RuntimeError, "freshness"):
            self.session.step()
        self.feed(2)
        with self.assertRaisesRegex(RuntimeError, "latched"):
            self.session.step()

    def test_scheduling_stall_stops_even_if_new_state_is_waiting(self):
        self.feed()
        self.session.step()
        self.clock.now = .04
        self.feed(2)
        with self.assertRaisesRegex(RuntimeError, "scheduling gap"):
            self.session.step()

    def test_raw_versus_scaled_state_and_sensor_flags_must_match(self):
        for kwargs, message in [
                (dict(obs_change=(9, .4)), "units/order/scale"),
                (dict(flags=w.SENSOR_VALID), "invalid"),
                (dict(flags=3|w.FAULT), "MCU reported")]:
            self.setUp()
            self.feed(**kwargs)
            with self.assertRaisesRegex(RuntimeError, message):
                self.session.step()

    def test_slow_inference_sends_stop_without_a_torque_packet(self):
        self.feed()
        self.policy.slow = lambda: setattr(self.clock, "now", self.clock.now+.006)
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            self.session.step()
        self.assertEqual([w.BinaryCodec().feed(p)[0].kind for p in self.link.sent], [w.HELLO, w.STOP])

    def test_execute_requires_permission_and_monotonic_known_ack(self):
        self.interface.execute = True
        self.feed()
        with self.assertRaisesRegex(RuntimeError, "calibrated"):
            self.session.step()
        self.setUp()
        self.interface.execute = True
        flags = 3|w.CALIBRATION_VERIFIED|w.EXECUTION_AVAILABLE
        for step, ack in [(0, 0), (1, 1), (2, 2)]:
            self.clock.now = step*.005
            self.feed(step+1, ack=ack, flags=flags)
            self.session.step()
        self.clock.now = .015
        self.feed(4, ack=1, flags=flags)
        with self.assertRaisesRegex(RuntimeError, "backwards"):
            self.session.step()

    def test_repeated_ack_does_not_keep_an_unapplied_stream_alive(self):
        self.interface.execute = True
        flags = 3|w.CALIBRATION_VERIFIED|w.EXECUTION_AVAILABLE
        for step in range(7):
            self.clock.now = step*.005
            self.feed(step+1, ack=0 if step==0 else 1, flags=flags)
            self.session.step()
        self.clock.now = .041
        self.feed(9, ack=1, flags=flags)
        with self.assertRaisesRegex(RuntimeError, "acknowledgement timeout"):
            self.session.step()

    def test_reference_hold_and_ack_history_are_different_state(self):
        self.policy.action = [.5, -.5, 1.2, -.5, .5, -1.2]
        controller = TorqueController(self.policy)
        obs, raw = sample(self.policy)
        before = controller.step(obs, raw, infer=True)
        held = controller.step(obs, raw, infer=False, previous_action=[0.]*6)
        self.assertEqual(held["effective_action"], before["effective_action"])
        self.assertEqual(held["position_target_rad"], before["position_target_rad"])
        self.assertEqual(held["observation"][21:27], [0.]*6)

    def test_raw_velocity_not_actor_clipped_velocity_drives_pd(self):
        self.policy.action = [1., 0., 0., 0., 0., 0.]
        raw = self.config["leg_joint_position_offset"] + [0.]*6
        raw[0] -= .2
        raw[4] = 60.  # Actor input clips at 50; actuator feedback must remain 60.
        obs, raw = sample(self.policy, raw)
        result = TorqueController(self.policy).step(obs, raw, infer=True)
        cfg = self.config
        expected = cfg["leg_joint_stiffness"]*.5-cfg["leg_joint_damping"]*60+result["gravity_effort_nm"][0]
        self.assertAlmostEqual(result["leg_total_before_clamp_nm"][0], expected, places=9)
        self.assertAlmostEqual(obs[13], 50*cfg["obs_scales"]["dof_vel"])
        self.assertLess(result["torque_nm"][0], 0.)

    def test_final_sum_limits_and_nan_angle_rejection(self):
        self.policy.action = [1., -1., 10., -1., 1., -10.]
        obs, raw = sample(self.policy)
        result = TorqueController(self.policy).step(obs, raw, infer=True)
        for j in [0, 1, 3, 4]:
            self.assertLessEqual(abs(result["torque_nm"][j]), self.config["leg_effort_limit"])
        for j in [2, 5]:
            self.assertEqual(abs(result["torque_nm"][j]), self.config["wheel_effort_limit"])
        self.assertEqual(result["wheel_target_rad_s"], [18.75, -18.75])
        for value in [float("nan"), 5.]:
            raw[0] = value
            with self.assertRaises(ValueError):
                TorqueController(self.policy).step(obs, raw, infer=True)

    def test_invalid_args_do_not_open_a_device(self):
        with patch("wheelleg_deploy.nx_runtime.open_robot_serial") as open_port:
            for kwargs in [dict(control_hz=201), dict(duration_s=float("nan")),
                           dict(command_override=[9., 0., .3]), dict(state_timeout_s=True)]:
                with self.assertRaises(ValueError):
                    run_serial(BUNDLE, "/dev/never-opened", **kwargs)
            open_port.assert_not_called()

    def test_command_ramp_only_moves_on_explicit_policy_ticks(self):
        ramp = CommandRamp(self.config)
        first = ramp.step([.8, 0., .3], .02)
        self.assertAlmostEqual(first[0], .016)
        for _ in range(4):
            self.assertAlmostEqual(ramp.applied_vx, .016)
        self.assertAlmostEqual(ramp.step([.8, 0., .3], .02)[0], .032)
        with self.assertRaises(ValueError):
            ramp.step([.9, 0., .3], .02)


@unittest.skipUnless((BUNDLE/"manifest.json").is_file(), "Selected portable bundle is required")
class SelectedPolicyTests(unittest.TestCase):
    def test_no_uart_benchmark_records_actual_machine_and_budget(self):
        from wheelleg_deploy.benchmark import benchmark
        with patch("wheelleg_deploy.nx_runtime.open_robot_serial") as open_port:
            result=benchmark(BUNDLE,duration_s=.025)
            self.assertGreater(result["samples"],0)
            self.assertGreater(result["inference_samples"],0)
            self.assertIn("p99_ms",result["compute"])
            self.assertFalse(result["serial_opened"])
            self.assertFalse(result["real_hardware_motion_validated"])
            open_port.assert_not_called()
    def test_exported_actor_matches_the_saved_torch_golden(self):
        policy = NumpyPolicy(BUNDLE)
        with np.load(BUNDLE/"synthetic_golden.npz", allow_pickle=False) as golden:
            actual = np.array([policy(row.tolist()) for row in golden["obs"]])
            self.assertLess(float(np.max(np.abs(actual-golden["action"]))), 2e-5)
        self.assertEqual(policy.metadata["actual_updates"], 7476)
        self.assertEqual(policy.cfg["wheel_damping"], 4.5)
        self.assertFalse(policy.metadata["hardware_approved"])
        self.assertFalse(policy.metadata["policy_precision_accepted"])

    def test_hash_and_execution_topology_mismatch_rejected(self):
        for change in ["weight", "config", "execution"]:
            with tempfile.TemporaryDirectory() as temp:
                target = Path(temp)/"bundle"
                shutil.copytree(BUNDLE, target)
                if change == "weight":
                    path = target/"actor_fp32.npz"
                    data = bytearray(path.read_bytes())
                    data[50] ^= 1
                    path.write_bytes(data)
                elif change == "config":
                    path = target/"runtime_config.json"
                    cfg = json.loads(path.read_text())
                    cfg["wheel_damping"] += 1.
                    path.write_text(json.dumps(cfg))
                else:
                    path = target/"manifest.json"
                    cfg = json.loads(path.read_text())
                    cfg["control_contract"]["execution"] = "MCU computes an additional PD"
                    path.write_text(json.dumps(cfg))
                with self.assertRaises(ValueError):
                    NumpyPolicy(target)

    def test_runtime_import_and_inference_do_not_load_torch_or_isaac(self):
        script = ("import sys; from wheelleg_deploy.numpy_policy import NumpyPolicy;"
                  "p=NumpyPolicy(sys.argv[1]);obs=[0.]*27;obs[5]=-1.;obs[8]=1.5;p(obs);"
                  "assert not any(n.split('.')[0] in ('torch','isaaclab','pxr','yaml') for n in sys.modules)")
        subprocess.run([sys.executable, "-c", script, str(BUNDLE)], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
