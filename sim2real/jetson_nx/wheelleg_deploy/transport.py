"""Byte transport; importing never discovers or opens a device."""

import time


class MemoryTransport:
    """Test link; contains no device opening, discovery or OS serial calls."""
    def __init__(self):
        self.sent = []

    def write(self, packet):
        if not isinstance(packet, bytes):
            raise ValueError("Transport expects encoded bytes")
        self.sent.append(packet)
        return len(packet)


class SerialTransport:
    """Explicit raw 8N1 port, bounded reads/writes; no automatic reconnect."""
    def __init__(self, config, *, serial_factory=None):
        port, baud = config.get("port"), config.get("baudrate")
        if not isinstance(port, str) or not port or isinstance(baud, bool) or not isinstance(baud, int) or baud <= 0:
            raise ValueError("Explicit serial port and baudrate required; no auto-discovery")
        if serial_factory is None:
            import serial  # Optional dependency; not imported in offline tools.
            serial_factory = serial.Serial
        self.port = serial_factory(port=None, baudrate=baud, bytesize=8, parity='N', stopbits=1,
                                   timeout=.001, write_timeout=.005, xonxoff=False,
                                   rtscts=False, dsrdtr=False)
        try:
            self.port.dtr = False
            self.port.rts = False
            self.port.port = port
            self.port.open()
            self.port.reset_input_buffer()
        except BaseException:
            self.port.close()
            raise

    def read(self, count=4096):
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 4096:
            raise ValueError("Read size must be 1..4096")
        return self.port.read(min(count, max(1, self.port.in_waiting)))

    def write(self, packet):
        if not isinstance(packet, bytes) or not packet:
            raise ValueError('Expected nonempty encoded bytes')
        deadline, sent = time.monotonic() + .01, 0
        while sent < len(packet):
            size = self.port.write(packet[sent:])
            if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= len(packet)-sent or time.monotonic() > deadline:
                raise TimeoutError('Serial write did not complete within 10 ms')
            sent += size
        return sent

    def close(self):
        self.port.close()
