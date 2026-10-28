"""A bounded, CRC-protected OFFLINE test codec; NOT the agreed MCU protocol.

Replace this codec once the MCU UART framing/endian/fields are provided. CRC is
error detection, not authentication or proof of sensor freshness.
"""

import json
import zlib

MAX_FRAME_BYTES = 4096


def canonical(body):
    return json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class OfflineJsonCodec:
    def __init__(self):
        self.pending = bytearray()

    def encode(self, body):
        frame = canonical({"body": body, "crc32": f"{zlib.crc32(canonical(body)):08x}"}) + b"\n"
        if len(frame) > MAX_FRAME_BYTES:
            raise ValueError("Frame exceeds byte limit")
        return frame

    def feed(self, chunk):
        if not isinstance(chunk, bytes):
            raise ValueError("Transport chunks must be bytes")
        messages = []
        # Bytewise consumption handles both split and concatenated UART frames
        # without growing a persistent buffer beyond one bounded frame.
        for value in chunk:
            self.pending.append(value)
            if len(self.pending) > MAX_FRAME_BYTES:
                self.pending.clear()
                raise ValueError("Overlong/incomplete frame")
            if value != 10:
                continue
            line = bytes(self.pending[:-1])
            self.pending.clear()
            try:
                frame = json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                body = frame["body"]
                if not isinstance(body, dict) or frame["crc32"] != f"{zlib.crc32(canonical(body)):08x}":
                    raise ValueError("CRC or body mismatch")
            except (ValueError, UnicodeError, TypeError, KeyError, RecursionError) as error:
                raise ValueError("Invalid test frame") from error
            messages.append(body)
        return messages
