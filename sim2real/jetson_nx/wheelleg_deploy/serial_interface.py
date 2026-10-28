"""Public NX UART API. The user aligns STM32 to PROTOCOL.md.

send_torques uses output-shaft N*m in [LB,LL,LW,RB,RL,RW] order.
No current/CAN/encoder conversions or device discovery occur here.
"""
from dataclasses import dataclass
import secrets
import time

from .transport import SerialTransport
from .wire import (Packet,BinaryCodec,HELLO,STATUS,ACTION,STOP,STOP_BODY,
                   hello,read_status,action_payload,newer32,u32,digest_bytes)


@dataclass(frozen=True)
class ObservationFrame:
    sequence:int
    device_time_us:int
    received_at:float
    flags:int
    last_applied_torque_sequence:int
    observation:list
    actuator_state:list


class RobotSerialInterface:
    def __init__(self,transport,contract_sha256,*,execute=False,session=None,clock=time.monotonic):
        self.transport=transport
        self.contract_sha256=digest_bytes(contract_sha256).hex()
        self.execute=execute
        self.session=u32(session if session is not None else secrets.randbits(32) or 1)
        if not self.session:
            raise ValueError('Session zero is reserved')
        self.clock=clock
        self.codec=BinaryCodec()
        self.tx_sequence=0
        self.last_state=None
        self.closed=False
        self.rejected_states=0
        self.send_hello()

    def _send(self,kind,payload):
        if self.closed:
            raise RuntimeError('Serial interface is closed')
        seq=self.tx_sequence
        packet=Packet(kind,self.session,seq,int(self.clock()*1e6)&0xffffffff,payload)
        self.transport.write(packet.encode())
        self.tx_sequence=(seq+1)&0xffffffff
        return seq

    def send_hello(self):
        return self._send(HELLO,hello(self.contract_sha256,self.execute))

    def receive_observation(self):
        """Read latest valid state or None. Duplicate samples do not refresh age."""
        latest=None
        for packet in self.codec.feed(self.transport.read()):
            if packet.kind!=STATUS or packet.session!=self.session:
                self.rejected_states+=1
                continue
            contract,flags,applied,obs,raw=read_status(packet)
            if contract!=self.contract_sha256:
                self.rejected_states+=1
                raise ValueError('STM32/Actor contract hash mismatch')
            previous=self.last_state
            if previous is not None and (not newer32(packet.sequence,previous.sequence)
                                         or not newer32(packet.time_us,previous.device_time_us)):
                self.rejected_states+=1
                continue
            latest=ObservationFrame(packet.sequence,packet.time_us,self.clock(),flags,applied,obs,raw)
            self.last_state=latest
        return latest

    def send_torques(self,torque_nm,*,state_sequence,effective_action,command):
        """Send final limited Nm, references to echo, and source observation ID."""
        if self.last_state is None or state_sequence!=self.last_state.sequence:
            raise ValueError('Torque must reference the latest accepted state')
        return self._send(ACTION,action_payload(state_sequence,torque_nm,effective_action,command))

    def send_stop(self,reason=0):
        """Explicit stop request; six zero normalized actions would not mean stop."""
        return self._send(STOP,STOP_BODY.pack(u32(reason)))

    def close(self):
        if not self.closed:
            try:
                self.send_stop()
            finally:
                self.closed=True
                self.transport.close()


def open_robot_serial(port,baudrate,contract_sha256,*,execute=False):
    """Open only the explicit device. Constructor/import never discovers ports."""
    hello(contract_sha256,execute)  # Validate before touching an OS device.
    transport=SerialTransport(dict(port=port,baudrate=baudrate))
    try:
        return RobotSerialInterface(transport,contract_sha256,execute=execute)
    except Exception:
        transport.close()
        raise
