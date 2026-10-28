import unittest

from wheelleg_deploy.protocol import OfflineJsonCodec, MAX_FRAME_BYTES
from wheelleg_deploy.schema import make_schema, validate_observation, finite_vector
from wheelleg_deploy.session import PolicySession
from wheelleg_deploy.targets import action_to_targets
from wheelleg_deploy.transport import MemoryTransport, SerialTransport


def config():
    return {
        "leg_control_mode": "implicit_joint_position", "leg_joint_gravity_feedforward": True,
        "leg_joint_position_offset": [-.07, -.03, .07, .03],
        "leg_joint_position_scale": [.3, .3, .3, .3],
        "obs_scales": {"ang_vel": .25, "lin_vel": 2., "height_measurements": 5.,
                       "dof_pos": 1., "dof_vel": .05, "l0": 5., "l0_dot": .25},
        "obs_clip": {"ang_vel": 20., "projected_gravity": 1., "theta": 3.1415926536,
                     "theta_dot": 50., "leg_length": .5, "leg_length_dot": 5.,
                     "lin_vel": 5., "wheel_vel": 100., "action": 1.},
        "commands": {"ranges_lin_vel_x": [.2, .8], "ranges_reverse_lin_vel_x": [-.8, -.2],
                     "ranges_ang_vel_yaw": [-.3, .3], "ranges_height": [.28, .32]},
        "action_scale_theta": .25, "theta0_ref_min": -.2, "theta0_ref_max": .2,
        "action_scale_l0": .09, "l0_offset": .27, "action_scale_vel": 15.,
        "height_reference_blend": .95, "leg_length_height_offset": -.035,
        "height_feedback_gain": .25, "height_feedback_max_adjustment": .03,
        "l0_ref_min": .18, "l0_ref_max": .36,
        "forward_support_speed_threshold": .15, "forward_support_min_leg_length": .3,
        "forward_support_height_margin": -.035,
    }


def observation():
    obs = [0.] * 27
    obs[5], obs[8] = -1., 1.5
    return obs


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now


class FakePolicy:
    def __init__(self, action=None):
        self.cfg, self.schema = config(), make_schema(config())
        self.metadata = {"execution_contract": "direct_joint_position_pd_gravity_and_wheel_velocity"}
        self.action = [0.] * 6 if action is None else action

    def __call__(self, obs):
        return self.action


def packet(policy, seq=0, tick=0):
    return dict(kind="observation", seq=seq, device_time_us=tick,
                schema_sha256=policy.schema["sha256"], obs=observation())


class SchemaTests(unittest.TestCase):
    def test_exact_order_and_scaling(self):
        schema = make_schema(config())
        self.assertEqual([f["index"] for f in schema["fields"]], list(range(27)))
        self.assertEqual(schema["fields"][17]["name"], "vx")
        self.assertEqual(schema["fields"][19]["name"], "wheel_left")
        self.assertEqual(schema["fields"][21]["name"], "previous_LB_position")
        self.assertEqual([f['name'] for f in schema['fields'][9:13]], ['q_LB_offset','q_LL_offset','q_RB_offset','q_RL_offset'])
        self.assertEqual([f['name'] for f in schema['fields'][13:17]], ['dq_LB','dq_LL','dq_RB','dq_RL'])
        self.assertEqual(schema["fields"][8]["scale"], 5.)
        self.assertEqual(validate_observation(observation(), schema, config()), observation())

    def test_schema_digest_changes_with_scales(self):
        cfg = config()
        cfg["obs_scales"]["ang_vel"] = .5
        self.assertNotEqual(make_schema(cfg)["sha256"], make_schema(config())["sha256"])

    def test_wrong_dim_nonfinite_bool_or_nested_rejected(self):
        for values in ([0.] * 26, [0.] * 30, [float("nan")] * 27,
                       [float("inf")] * 27, [True] * 27, [[0.]] * 27):
            with self.subTest(values=values), self.assertRaises(ValueError):
                finite_vector(values, 27, "obs")

    def test_gravity_units_and_double_scaling_rejected(self):
        for change in ((5, -9.81), (5, 0.), (0, 6.), (19, 10.)):
            obs = observation()
            obs[change[0]] = change[1]
            with self.assertRaises(ValueError):
                validate_observation(obs, make_schema(config()), config())

    def test_commands_out_of_distribution_rejected(self):
        for index, value in ((6, 2.), (7, .15), (8, .9)):
            obs = observation()
            obs[index] = value
            with self.assertRaises(ValueError):
                validate_observation(obs, make_schema(config()), config())


class ReferenceTests(unittest.TestCase):
    def test_action_clamp_scale_and_pair_order(self):
        target = action_to_targets([2., 1., .5, -2., -1., -.5], observation(), config())
        for actual, expected in zip(target['joint_position_ref_rad'], [.23, .27, -.23, -.27]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(target['joint_order'], ['LB','LL','RB','RL'])
        self.assertEqual(target["wheel_speed_ref_rad_s"], [7.5, -7.5])
        self.assertEqual(target["height_command_m"], .3)

    def test_saved_wheel_limit_keeps_headroom_and_unit_leg_bounds(self):
        cfg = config()
        cfg['wheel_action_limit'] = 1.25
        target = action_to_targets([2., -2., 1.2, -2., 2., -2.], observation(), cfg)
        self.assertEqual(target['wheel_speed_ref_rad_s'], [18., -18.75])
        for actual, expected in zip(target['joint_position_ref_rad'], [.23, -.33, -.23, .33]):
            self.assertAlmostEqual(actual, expected)

    def test_invalid_saved_wheel_limit_is_rejected(self):
        for bound in [True, 0.5, 2.01, float('nan'), float('inf'), '1.25']:
            with self.subTest(bound=bound), self.assertRaises(ValueError):
                action_to_targets([0.] * 6, observation(), dict(config(), wheel_action_limit=bound))

    def test_zero_action_does_not_mean_motor_stop(self):
        target = action_to_targets([0.] * 6, observation(), config())
        self.assertEqual(target["joint_position_ref_rad"], config()['leg_joint_position_offset'])

    def test_height_command_does_not_change_direct_motor_action_map(self):
        obs = observation()
        obs[6] = .6  # Scaled 0.3 m/s command.
        target = action_to_targets([0., -1., 0., 0., -1., 0.], obs, config())
        obs[8] = 1.6
        other = action_to_targets([0., -1., 0., 0., -1., 0.], obs, config())
        self.assertEqual(target['joint_position_ref_rad'], other['joint_position_ref_rad'])

    def test_retired_action_and_observation_contracts_are_rejected(self):
        cfg = config()
        cfg['leg_control_mode'] = 'implicit_joint_reference'
        with self.assertRaises(ValueError):
            action_to_targets([0.] * 6, observation(), cfg)
        with self.assertRaises(ValueError):
            make_schema(cfg)


class ProtocolTests(unittest.TestCase):
    def test_partial_and_concatenated_frames(self):
        codec = OfflineJsonCodec()
        frame = codec.encode({"kind": "observation", "seq": 3})
        self.assertEqual(codec.feed(frame[:5]), [])
        messages = codec.feed(frame[5:] + frame)
        self.assertEqual([m["seq"] for m in messages], [3, 3])

    def test_crc_failure_and_recovery(self):
        codec = OfflineJsonCodec()
        frame = codec.encode({"seq": 3}).replace(b'"seq":3', b'"seq":4')
        with self.assertRaises(ValueError):
            codec.feed(frame)
        self.assertEqual(codec.feed(codec.encode({"seq": 5}))[0]["seq"], 5)

    def test_overlong_or_nonfinite_frame_rejected(self):
        codec = OfflineJsonCodec()
        with self.assertRaises(ValueError):
            codec.feed(b"x" * (MAX_FRAME_BYTES + 1))
        with self.assertRaises(ValueError):
            codec.encode({"obs": [float("nan")]})
        with self.assertRaises(ValueError):
            codec.feed(b'{"body":{"obs":[NaN]},"crc32":"12345678"}\n')


class SessionTests(unittest.TestCase):
    def test_reference_packet_uses_the_same_saved_wheel_limit_as_targets(self):
        policy = FakePolicy([0., 0., 2., 0., 0., -1.2])
        policy.cfg['wheel_action_limit'] = 1.25
        result = PolicySession(policy, clock=Clock()).handle(packet(policy))
        self.assertEqual(result['kind'], 'reference_request')
        self.assertEqual(result['clipped_action'][2::3], [1.25, -1.2])
        self.assertEqual(result['targets']['wheel_speed_ref_rad_s'], [18.75, -18.])
        self.assertFalse(result['motor_enable'])

    def test_invalid_deadlines_are_rejected(self):
        for value in (0., -1., float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                PolicySession(FakePolicy(), inference_deadline_s=value)

    def test_byte_loopback_input_obs_output_targets(self):
        policy, clock, wire = FakePolicy(), Clock(), MemoryTransport()
        codec = OfflineJsonCodec()
        session = PolicySession(policy, clock=clock)
        request = codec.feed(codec.encode(packet(policy)))[0]
        reply = session.handle(request)
        wire.write(codec.encode(reply))
        restored = codec.feed(wire.sent[-1])[0]
        self.assertEqual(restored["kind"], "reference_request")
        self.assertFalse(restored["motor_enable"])
        self.assertTrue(restored["offline_only"])
        self.assertEqual(restored["targets"]["wheel_speed_ref_rad_s"], [0., 0.])

    def test_duplicate_and_clock_restart_latch_fault(self):
        for seq, tick in ((0, 20000), (1, 0)):
            policy = FakePolicy()
            session = PolicySession(policy, clock=Clock())
            session.handle(packet(policy))
            self.assertEqual(session.handle(packet(policy, seq, tick))["kind"], "fallback_request")
            self.assertEqual(session.handle(packet(policy, 2, 40000))["kind"], "fallback_request")

    def test_nan_wrong_schema_or_time_gap(self):
        for mutate in (lambda p: p.update(schema_sha256="wrong"), lambda p: p["obs"].__setitem__(0, float("nan")),
                       lambda p: p.update(seq=True), lambda p: p.update(device_time_us=-1)):
            policy, session = FakePolicy(), None
            session = PolicySession(policy, clock=Clock())
            request = packet(policy)
            mutate(request)
            self.assertEqual(session.handle(request)["kind"], "fallback_request")
        session = PolicySession(FakePolicy(), clock=Clock())
        session.handle(packet(session.policy))
        self.assertEqual(session.handle(packet(session.policy, 1, 101000))["kind"], "fallback_request")

    def test_feedback_timeout_even_before_first_packet(self):
        clock, policy = Clock(), FakePolicy()
        session = PolicySession(policy, clock=clock)
        clock.now = .101
        self.assertEqual(session.check_timeout()["kind"], "fallback_request")
        self.assertEqual(session.handle(packet(policy))["kind"], "fallback_request")

    def test_nonfinite_actor_or_slow_inference_request_fallback(self):
        session = PolicySession(FakePolicy([float("nan")] * 6), clock=Clock())
        self.assertEqual(session.handle(packet(session.policy))["kind"], "fallback_request")
        clock = Clock()
        class SlowPolicy(FakePolicy):
            def __call__(self, obs):
                clock.now += .021
                return super().__call__(obs)
        session = PolicySession(SlowPolicy(), clock=clock)
        self.assertEqual(session.handle(packet(session.policy))["kind"], "fallback_request")


class SerialSafetyTests(unittest.TestCase):
    def test_missing_device_never_opens_port(self):
        calls = []
        with self.assertRaises(ValueError):
            SerialTransport({"port": None, "baudrate": 115200},
                            serial_factory=lambda **kw: calls.append(kw))
        self.assertEqual(calls, [])

    def test_partial_writes_complete_with_explicit_port(self):
        class FakeSerial:
            def __init__(self, **kw):
                self.closed = False
                self.in_waiting = 4
                self.sent = bytearray()
            def open(self):
                self.opened = True
            def reset_input_buffer(self):
                pass
            def read(self, count):
                return b"test"[:count]
            def close(self):
                self.closed = True
            def write(self, data):
                self.sent.extend(data[:2])
                return min(2, len(data))
        transport = SerialTransport({"protocol_confirmed": True, "port_open_reset_behavior_verified": True,
                                     "port": "fake", "baudrate": 115200}, serial_factory=FakeSerial)
        self.assertEqual(transport.read(), b"test")
        self.assertEqual(transport.write(b"action"), 6)
        self.assertEqual(bytes(transport.port.sent), b"action")
        transport.close()
        self.assertTrue(transport.port.closed)


if __name__ == "__main__":
    unittest.main()
