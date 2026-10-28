"""Binary ABI, real PTY transport, and the actual two-file MCU state machine."""
import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import select
import shutil
import struct
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
import zlib

from wheelleg_deploy import wire as w
from wheelleg_deploy.serial_interface import RobotSerialInterface, open_robot_serial

ROOT = Path(__file__).resolve().parents[1]
MCU_SOURCE = ROOT.parent/"series_balance_rtos/control"
if not MCU_SOURCE.exists():
    MCU_SOURCE = ROOT/"mcu_reference"
CONTRACT = "11" * 32


class Clock:
    now = 0.
    def __call__(self):
        return self.now


class Link:
    def __init__(self):
        self.incoming = b""
        self.sent = []
        self.closed = False
    def read(self):
        data, self.incoming = self.incoming, b""
        return data
    def write(self, data):
        self.sent.append(data)
        return len(data)
    def close(self):
        self.closed = True


def status(sequence=1, tick=1000, session=7, contract=CONTRACT, ack=0):
    obs = [0.] * 27
    obs[5], obs[8] = -1., 1.5
    return w.Packet(w.STATUS, session, sequence, tick,
                    w.status_payload(contract, 3, ack, obs, [0.] * 10))


class PythonWireTests(unittest.TestCase):
    def test_fragmentation_noise_corruption_and_frame_lengths(self):
        frames = [
            w.Packet(w.HELLO, 7, 0, 0, w.hello(CONTRACT)).encode(),
            status().encode(),
            w.Packet(w.ACTION, 7, 1, 1, w.action_payload(1, [0.]*6, [0.]*6, [0., 0., .3])).encode(),
            w.Packet(w.STOP, 7, 2, 2, w.STOP_BODY.pack(0)).encode(),
        ]
        self.assertEqual([len(f) for f in frames], [60, 212, 88, 28])
        broken = bytearray(frames[1])
        broken[-7] ^= 1
        codec, decoded = w.BinaryCodec(), []
        stream = b"garbageWLR" + bytes(broken) + b"".join(frames)
        for byte in stream:
            decoded += codec.feed(bytes([byte]))
            self.assertLessEqual(len(codec.pending), w.HEADER.size + w.MAX_PAYLOAD + 4)
        self.assertEqual([p.kind for p in decoded], [1, 2, 3, 4])
        self.assertGreaterEqual(codec.rejected, 1)

    def test_invalid_lengths_versions_and_sessions_resynchronize(self):
        original = status().encode()
        for offset, value in [(4, 9), (5, 9), (6, 255), (8, 0)]:
            bad = bytearray(original)
            bad[offset] = value
            decoded = w.BinaryCodec().feed(bytes(bad) + original)
            self.assertEqual(decoded, [status()])

    def test_integer_wrap_and_invalid_float_or_hash(self):
        self.assertTrue(w.newer32(0, 0xffffffff))
        for value, old in [(1, 1), (0, 1), (0x80000000, 0)]:
            self.assertFalse(w.newer32(value, old))
        for values in [[float("nan")]*6, [float("inf")]*6, [True]*6, [1e40]*6]:
            with self.assertRaises(ValueError):
                w.action_payload(1, values, [0.]*6, [0., 0., .3])
        for digest in [" "*64, "x"*64, "11"*31]:
            with self.assertRaises(ValueError):
                w.hello(digest)
        with patch("wheelleg_deploy.serial_interface.SerialTransport") as transport:
            with self.assertRaises(ValueError):
                open_robot_serial("/dev/not-opened", 2000000, " "*64)
            transport.assert_not_called()

    def test_duplicate_reordered_or_wrong_clock_never_refresh_age(self):
        clock, link = Clock(), Link()
        interface = RobotSerialInterface(link, CONTRACT, session=7, clock=clock)
        link.incoming = status().encode()
        first = interface.receive_observation()
        clock.now = .1
        for packet in [status(), status(0, 2000), status(2, 1000), status(2, 3000, session=9)]:
            link.incoming = packet.encode()
            self.assertIsNone(interface.receive_observation())
            self.assertEqual(interface.last_state, first)
        link.incoming = status(2, 3000).encode()
        self.assertEqual(interface.receive_observation().received_at, .1)
        with self.assertRaises(ValueError):
            interface.send_torques([0.]*6, state_sequence=1,
                                   effective_action=[0.]*6, command=[0., 0., .3])
        interface.send_torques([1., -1., .2, -2., 2., -.3], state_sequence=2,
                               effective_action=[0.]*6, command=[0., 0., .3])
        packet = w.BinaryCodec().feed(link.sent[-1])[0]
        self.assertEqual(w.read_action(packet)[0], 2)
        interface.close()
        self.assertTrue(link.closed)
        self.assertEqual(w.BinaryCodec().feed(link.sent[-1])[0].kind, w.STOP)

    def test_contract_mismatch_fails_before_accepting_state(self):
        link = Link()
        interface = RobotSerialInterface(link, CONTRACT.upper(), session=7)
        link.incoming = status(contract="22"*32).encode()
        with self.assertRaisesRegex(ValueError, "contract"):
            interface.receive_observation()
        self.assertIsNone(interface.last_state)


class CPacket(C.Structure):
    _fields_ = [("kind", C.c_uint8), ("size", C.c_uint16),
                ("session", C.c_uint32), ("sequence", C.c_uint32),
                ("time_us", C.c_uint32), ("payload", C.c_uint8 * 192)]


class CParser(C.Structure):
    _fields_ = [("data", C.c_uint8 * 216), ("used", C.c_uint16), ("rejected", C.c_uint32)]


class CConfig(C.Structure):
    _fields_ = [("contract", C.c_uint8 * 32), ("torque_limit", C.c_float * 6),
                ("action_limit", C.c_float * 6), ("command_min", C.c_float * 3),
                ("command_max", C.c_float * 3), ("obs_action_limit", C.c_float),
                ("timeout", C.c_uint32), ("source_age", C.c_uint32)]


class CCommand(C.Structure):
    _fields_ = [("sequence", C.c_uint32), ("source_sequence", C.c_uint32),
                ("source_time", C.c_uint32), ("received", C.c_uint32),
                ("torque", C.c_float * 6), ("action", C.c_float * 6),
                ("command", C.c_float * 3)]


@unittest.skipUnless(shutil.which("gcc") and (MCU_SOURCE/"rl_host.c").is_file(),
                     "Native C tests require gcc and the MCU source pair")
class McuWireTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        temp = Path(cls.temp.name)
        source = MCU_SOURCE
        wrapper = temp/"inspect.c"
        wrapper.write_text('#include "rl_host.h"\n'
                           'size_t context_size(void){return sizeof(RLHost_Context);}\n'
                           'int fault(const RLHost_Context*c){return c->fault_latched;}\n'
                           'int stopped(const RLHost_Context*c){return c->stop_requested;}\n'
                           'int reason(const RLHost_Context*c){return c->fault_reason;}\n')
        output = temp/"rl_host.so"
        subprocess.run(["gcc", "-std=c99", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC",
                        "-DRL_HOST_PORTABLE_TEST", "-I"+str(source), str(source/"rl_host.c"),
                        str(wrapper), "-lm", "-o", str(output)], check=True, capture_output=True)
        cls.lib = C.CDLL(str(output))
        lib = cls.lib
        lib.context_size.restype = C.c_size_t
        lib.RLHost_Init.argtypes = [C.c_void_p, C.POINTER(CConfig)]
        lib.RLHost_SetPermission.argtypes = [C.c_void_p, C.c_int, C.c_int]
        lib.RLHost_ReceiveBytes.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_uint32]
        lib.RLHost_BuildStatus.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.POINTER(C.c_float),
                                         C.c_uint32, C.c_uint32, C.c_void_p, C.c_size_t]
        lib.RLHost_BuildStatus.restype = C.c_size_t
        lib.RLHost_GetTorques.argtypes = [C.c_void_p, C.c_uint32, C.POINTER(CCommand)]
        lib.RLHost_MarkApplied.argtypes = [C.c_void_p, C.c_uint32, C.c_uint32]
        lib.RLHost_Poll.argtypes = [C.c_void_p, C.c_uint32]
        lib.RLHost_CRC32.argtypes = [C.c_void_p, C.c_size_t]
        lib.RLHost_CRC32.restype = C.c_uint32
        lib.RLHost_Encode.argtypes = [C.POINTER(CPacket), C.c_void_p, C.c_size_t]
        lib.RLHost_Encode.restype = C.c_size_t
        cls.callback_type = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(CPacket))
        lib.RLHost_ParserFeed.argtypes = [C.POINTER(CParser), C.c_void_p, C.c_size_t,
                                        cls.callback_type, C.c_void_p]
        for name in ["fault", "stopped", "reason"]:
            getattr(lib, name).argtypes = [C.c_void_p]

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.cfg = CConfig()
        self.cfg.contract[:] = bytes.fromhex(CONTRACT)
        self.cfg.torque_limit[:] = [35.574, 35.574, 4.80944, 35.574, 35.574, 4.80944]
        self.cfg.action_limit[:] = [1., 1., 1.25, 1., 1., 1.25]
        self.cfg.command_min[:] = [-.8, -.3, .225]
        self.cfg.command_max[:] = [.8, .3, .376]
        self.cfg.obs_action_limit, self.cfg.timeout, self.cfg.source_age = 1., 30000, 30000
        self.ctx = C.create_string_buffer(self.lib.context_size())
        self.assertEqual(self.lib.RLHost_Init(self.ctx, C.byref(self.cfg)), 1)
        self.lib.RLHost_SetPermission(self.ctx, 1, 1)

    def receive(self, packet, now=1200):
        encoded = packet.encode()
        # Split even before the CRC to exercise the actual application parser.
        self.lib.RLHost_ReceiveBytes(self.ctx, encoded[:17], 17, now)
        self.lib.RLHost_ReceiveBytes(self.ctx, encoded[17:], len(encoded)-17, now)

    def publish(self, now=1000):
        obs = (C.c_float * 27)()
        obs[5], obs[8] = -1., 1.5
        raw, output = (C.c_float * 10)(), C.create_string_buffer(216)
        size = self.lib.RLHost_BuildStatus(self.ctx, obs, raw, 3, now, output, 216)
        self.assertEqual(size, 212)
        return w.BinaryCodec().feed(output.raw[:size])[0]

    def connect(self, execute=True, now=1000):
        self.receive(w.Packet(w.HELLO, 7, 0, 0, w.hello(CONTRACT, execute)), now-100)
        return self.publish(now)

    def torque(self, sequence=1, source=0, values=None):
        return w.Packet(w.ACTION, 7, sequence, 100,
                        w.action_payload(source, values or [1., -2., .3, -4., 5., -.6],
                                         [.2, -.3, 1.25, .4, -.5, -1.25], [0., 0., .3]))

    def test_python_and_c_codec_are_byte_identical(self):
        self.assertEqual(self.lib.RLHost_CRC32(b"123456789", 9), 0xcbf43926)
        packets = [status(), self.torque(), w.Packet(w.HELLO, 7, 0, 0, w.hello(CONTRACT, True)),
                   w.Packet(w.STOP, 7, 3, 100, w.STOP_BODY.pack(0))]
        received, parser = [], CParser()
        @self.callback_type
        def callback(_, packet):
            p = packet.contents
            received.append(w.Packet(p.kind, p.session, p.sequence, p.time_us, bytes(p.payload[:p.size])))
        for packet in packets:
            cp, out = CPacket(), C.create_string_buffer(216)
            cp.kind, cp.size, cp.session, cp.sequence, cp.time_us = packet.kind, len(packet.payload), 7, packet.sequence, packet.time_us
            cp.payload[:len(packet.payload)] = packet.payload
            size = self.lib.RLHost_Encode(C.byref(cp), out, 216)
            self.assertEqual(out.raw[:size], packet.encode())
            damaged = bytearray(packet.encode())
            damaged[-1] ^= 1
            data = b"noise" + bytes(damaged) + packet.encode()
            for i in range(0, len(data), 7):
                chunk = data[i:i+7]
                self.lib.RLHost_ParserFeed(C.byref(parser), chunk, len(chunk), callback, None)
        self.assertEqual(received, packets)
        self.assertGreaterEqual(parser.rejected, 4)

    def test_only_actual_application_produces_ack_and_action_history(self):
        self.connect()
        self.receive(self.torque())
        self.assertEqual(self.lib.RLHost_MarkApplied(self.ctx, 1, 1300), 0)
        command = CCommand()
        self.assertEqual(self.lib.RLHost_GetTorques(self.ctx, 1300, C.byref(command)), 1)
        self.assertAlmostEqual(command.torque[0], 1.)
        self.assertEqual(w.read_status(self.publish(1400))[2], 0)
        self.assertEqual(self.lib.RLHost_MarkApplied(self.ctx, 1, 1500), 1)
        _, flags, ack, obs, _ = w.read_status(self.publish(2000))
        self.assertTrue(flags & w.EXECUTION_ACTIVE)
        self.assertEqual(ack, 1)
        self.assertEqual(obs[23], 1.)
        self.assertEqual(obs[26], -1.)

    def test_shadow_never_exposes_a_motor_request(self):
        self.connect(execute=False)
        self.receive(self.torque())
        self.assertEqual(self.lib.RLHost_GetTorques(self.ctx, 1300, C.byref(CCommand())), 0)
        self.assertEqual(self.lib.RLHost_MarkApplied(self.ctx, 1, 1300), 0)
        self.assertFalse(w.read_status(self.publish(1400))[1] & w.EXECUTION_ACTIVE)

    def test_watchdog_latches_and_new_hello_does_not_clear_fault(self):
        self.connect()
        packet = self.torque()
        self.receive(packet)
        self.assertEqual(self.lib.RLHost_GetTorques(self.ctx, 1300, C.byref(CCommand())), 1)
        self.receive(packet, now=30000)  # Repeated frames cannot reset timeout.
        self.lib.RLHost_Poll(self.ctx, 32001)
        self.assertEqual(self.lib.fault(self.ctx), 1)
        self.assertEqual(self.lib.stopped(self.ctx), 1)
        self.receive(w.Packet(w.HELLO, 8, 0, 0, w.hello(CONTRACT, True)), now=33000)
        self.assertEqual(self.lib.fault(self.ctx), 1)
        self.assertEqual(self.lib.RLHost_GetTorques(self.ctx, 33001, C.byref(CCommand())), 0)

    def test_unknown_stale_or_invalid_sensor_source_rejected(self):
        for source, now in [(999, 1200), (0, 32000)]:
            with self.subTest(source=source, now=now):
                self.setUp()
                self.connect()
                self.receive(self.torque(source=source), now=now)
                self.assertEqual(self.lib.reason(self.ctx), 3)
                self.assertEqual(self.lib.RLHost_GetTorques(self.ctx, now, C.byref(CCommand())), 0)

    def test_excess_torque_nonfinite_config_and_command_are_rejected(self):
        for value in [36., float("nan"), float("inf")]:
            self.setUp()
            self.connect()
            payload = w.ACTION_BODY.pack(0, value, *([0.]*11), 0., 0., .3)
            self.receive(w.Packet(w.ACTION, 7, 1, 100, payload))
            self.assertEqual(self.lib.reason(self.ctx), 2)
        self.cfg.torque_limit[0] = float("nan")
        self.assertEqual(self.lib.RLHost_Init(self.ctx, C.byref(self.cfg)), 0)

    def test_stop_and_withdrawn_permission_remove_output(self):
        self.connect()
        self.receive(self.torque())
        self.assertEqual(self.lib.RLHost_GetTorques(self.ctx, 1300, C.byref(CCommand())), 1)
        self.receive(w.Packet(w.STOP, 7, 2, 100, w.STOP_BODY.pack(0)), now=1400)
        self.assertEqual(self.lib.RLHost_GetTorques(self.ctx, 1500, C.byref(CCommand())), 0)
        self.assertEqual(self.lib.stopped(self.ctx), 1)
        self.connect()
        self.receive(self.torque())
        self.lib.RLHost_SetPermission(self.ctx, 0, 1)
        self.assertEqual(self.lib.RLHost_GetTorques(self.ctx, 1500, C.byref(CCommand())), 0)


@unittest.skipUnless(os.name == "posix" and importlib.util.find_spec("serial"),
                     "PTY integration requires Linux/Unix and pyserial")
class PseudoTerminalTests(unittest.TestCase):
    def test_real_pyserial_hello_state_torque_and_stop_roundtrip(self):
        master, slave = os.openpty()
        interface = None
        def read_packet():
            codec, end = w.BinaryCodec(), time.monotonic()+1.
            while time.monotonic() < end:
                if select.select([master], [], [], .05)[0]:
                    packets = codec.feed(os.read(master, 4096))
                    if packets:
                        return packets[0]
            self.fail("Virtual STM32 did not receive a complete packet")
        try:
            interface = open_robot_serial(os.ttyname(slave), 2000000, CONTRACT)
            hello_packet = read_packet()
            self.assertEqual(hello_packet.kind, w.HELLO)
            self.assertEqual(w.HELLO_BODY.unpack(hello_packet.payload)[1], w.SHADOW)
            for n in range(1, 21):
                os.write(master, status(n, n*5000, session=interface.session).encode())
                end, frame = time.monotonic()+1., None
                while frame is None and time.monotonic() < end:
                    frame = interface.receive_observation()
                self.assertIsNotNone(frame)
                interface.send_torques([1., -2., .3, -4., 5., -.6], state_sequence=n,
                                       effective_action=[0.]*6, command=[0., 0., .3])
                reply = read_packet()
                self.assertEqual(reply.kind, w.ACTION)
                self.assertEqual(w.read_action(reply)[0], n)
            interface.close()
            self.assertEqual(read_packet().kind, w.STOP)
        finally:
            if interface is not None and not interface.closed:
                interface.close()
            os.close(master)
            os.close(slave)


if __name__ == "__main__":
    unittest.main()
