"""CLT UART v1: bounded frames shared with control/rl_host.c.

CRC detects corruption, not authentication. Freshness/session checks belong to
the two endpoints. Neither a zero action nor a valid frame enables motors.
"""
from dataclasses import dataclass
import math
import struct
import zlib

MAGIC, VERSION, MAX_PAYLOAD = b"WLRL", 1, 192
HEADER = struct.Struct("<4sBBHIII")
CRC = struct.Struct("<I")
HELLO, STATUS, ACTION, STOP = 1, 2, 3, 4
HELLO_BODY = struct.Struct("<32sI")
STATUS_BODY = struct.Struct("<32sII37f")
ACTION_BODY = struct.Struct("<I6f6f3f")
STOP_BODY = struct.Struct("<I")
SENSOR_VALID, VELOCITY_VALID, EXECUTION_ACTIVE, FAULT = 1, 2, 4, 8
CALIBRATION_VERIFIED, EXECUTION_AVAILABLE = 16, 32
SHADOW, EXECUTE = 0, 1
SIZES = {HELLO: HELLO_BODY.size, STATUS: STATUS_BODY.size,
         ACTION: ACTION_BODY.size, STOP: STOP_BODY.size}


def u32(value, name="value"):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 0xffffffff:
        raise ValueError(name + " must be uint32")
    return value


def newer32(value, previous):
    delta = (u32(value) - u32(previous)) & 0xffffffff
    return 0 < delta < 0x80000000


def vector(values, count):
    if len(values) != count or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                   or not math.isfinite(v) or abs(v) > 3.4028234e38 for v in values):
        raise ValueError("Expected %d finite float32 values" % count)
    return [float(v) for v in values]


def digest_bytes(value):
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Contract requires a SHA256 hexadecimal string")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError("Invalid contract SHA256") from error
    if len(decoded) != 32:
        raise ValueError("Contract requires exactly 32 SHA256 bytes")
    return decoded


@dataclass(frozen=True)
class Packet:
    kind: int
    session: int
    sequence: int
    time_us: int
    payload: bytes

    def encode(self):
        if self.kind not in SIZES or len(self.payload) != SIZES[self.kind]:
            raise ValueError("Packet type/payload length mismatch")
        if not u32(self.session, "session"):
            raise ValueError("Session zero is reserved")
        header = HEADER.pack(MAGIC, VERSION, self.kind, len(self.payload), self.session,
                             u32(self.sequence, "sequence"), u32(self.time_us, "time_us"))
        body = header + self.payload
        return body + CRC.pack(zlib.crc32(body) & 0xffffffff)


class BinaryCodec:
    """Resynchronize after CRC/length/version errors; memory is frame-bounded."""
    def __init__(self):
        self.pending = bytearray()
        self.rejected = 0

    def feed(self, chunk):
        if not isinstance(chunk, bytes):
            raise ValueError("UART chunks must be bytes")
        packets = []
        for value in chunk:
            self.pending.append(value)
            while self.pending:
                if not MAGIC.startswith(self.pending[:min(len(self.pending), 4)]):
                    del self.pending[0]
                    continue
                if len(self.pending) < HEADER.size:
                    break
                _, version, kind, size, session, seq, tick = HEADER.unpack_from(self.pending)
                if version != VERSION or kind not in SIZES or size != SIZES[kind] or not session:
                    self.rejected += 1
                    del self.pending[0]
                    continue
                total = HEADER.size + size + CRC.size
                if len(self.pending) < total:
                    break
                expected, = CRC.unpack_from(self.pending, total - 4)
                if zlib.crc32(self.pending[:total-4]) & 0xffffffff != expected:
                    self.rejected += 1
                    del self.pending[0]
                    continue
                packets.append(Packet(kind, session, seq, tick, bytes(self.pending[HEADER.size:total-4])))
                del self.pending[:total]
        return packets


def hello(contract_sha256, execute=False):
    if not isinstance(execute, bool):
        raise ValueError("execute must be bool")
    return HELLO_BODY.pack(digest_bytes(contract_sha256), int(execute))


def status_payload(contract_sha256, flags, applied_sequence, observation, actuator_state):
    """27 scaled Actor inputs, then unclipped q4/dq4/wheel-speed2 in SI units."""
    return STATUS_BODY.pack(digest_bytes(contract_sha256), u32(flags), u32(applied_sequence),
                             *vector(observation,27), *vector(actuator_state,10))


def read_status(packet):
    if packet.kind != STATUS or len(packet.payload) != STATUS_BODY.size:
        raise ValueError("Expected a status packet")
    contract, flags, applied, *values = STATUS_BODY.unpack(packet.payload)
    return contract.hex(), flags, applied, vector(values[:27],27), vector(values[27:],10)


def action_payload(observation_sequence, torque, effective_action, command):
    """Six output-shaft Nm, six references for echo/history, vx/yaw/height."""
    return ACTION_BODY.pack(u32(observation_sequence), *vector(torque,6),
                            *vector(effective_action,6), *vector(command,3))


def read_action(packet):
    if packet.kind != ACTION or len(packet.payload) != ACTION_BODY.size:
        raise ValueError("Expected an action packet")
    seq, *values = ACTION_BODY.unpack(packet.payload)
    return seq, vector(values[:6],6), vector(values[6:12],6), vector(values[12:],3)
