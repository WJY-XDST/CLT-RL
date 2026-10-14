"""Replaceable byte transport. Physical UART is deliberately not commissioned."""


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
    """Receive-only scaffold. Real output is absent until MCU code is reviewed.

    Merely opening a serial port can toggle DTR/RTS and reset an MCU. Require
    both confirmed framing and reviewed open/reset behaviour even for reads.
    """
    def __init__(self, config, *, serial_factory=None):
        if config.get("protocol_confirmed") is not True or config.get("port_open_reset_behavior_verified") is not True:
            raise ValueError("MCU protocol and serial-port reset behaviour are not confirmed")
        port, baud = config.get("port"), config.get("baudrate")
        if not isinstance(port, str) or not port or isinstance(baud, bool) or not isinstance(baud, int) or baud <= 0:
            raise ValueError("Explicit serial port and baudrate required; no auto-discovery")
        if serial_factory is None:
            import serial  # Optional dependency; not imported in offline tools.
            serial_factory = serial.Serial
        self.port = serial_factory(port=port, baudrate=baud, timeout=.02, write_timeout=.02)

    def read(self, count=4096):
        if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 4096:
            raise ValueError("Read size must be 1..4096")
        return self.port.read(count)

    def write(self, packet):
        raise RuntimeError("Real serial output is not implemented/commissioned; MCU protocol and safety review pending")

    def close(self):
        self.port.close()
