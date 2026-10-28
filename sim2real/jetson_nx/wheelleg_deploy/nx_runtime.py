"""NX observation -> Actor -> local PD/gravity -> six final UART torques."""
from collections import OrderedDict
import json
import math
from pathlib import Path
import sys
import time

from .numpy_policy import NumpyPolicy
from .observation import CommandRamp
from .serial_interface import open_robot_serial
from .torque_control import TorqueController
from .wire import (SENSOR_VALID,VELOCITY_VALID,EXECUTION_ACTIVE,FAULT,
                   CALIBRATION_VERIFIED,EXECUTION_AVAILABLE,newer32)


def validate_timing(control_hz,state_timeout_s):
    if isinstance(control_hz,bool) or not isinstance(control_hz,int) or not 50<=control_hz<=1000 or control_hz%50:
        raise ValueError('Control frequency must be an integer multiple of 50 Hz, in 50..1000')
    if isinstance(state_timeout_s,bool) or not isinstance(state_timeout_s,(int,float)) or not math.isfinite(state_timeout_s) or not .005<=state_timeout_s<=.1:
        raise ValueError('State timeout must be 5..100 ms')


def validate_raw_observation(frame,cfg):
    """Reject mismatched scales/order between raw servo state and Actor data."""
    p=frame.actuator_state
    s,c=cfg['obs_scales'],cfg['obs_clip']
    checks=[]
    def clip(v,key):
        return max(-c[key],min(c[key],v))
    for j in range(4):
        checks.append((9+j,clip(p[j]-cfg['leg_joint_position_offset'][j],'theta')*s['dof_pos']))
        checks.append((13+j,clip(p[4+j],'theta_dot')*s['dof_vel']))
    for j in range(2):
        checks.append((19+j,clip(p[8+j],'wheel_vel')*s['dof_vel']))
    if any(abs(frame.observation[i]-expected)>1e-4 for i,expected in checks):
        raise ValueError('obs versus raw actuator-state units/order/scale mismatch')


class NxControlSession:
    """One explicit session; faults latch until a new process/session is started."""
    def __init__(self,policy,interface,*,control_hz=200,state_timeout_s=.03,
                 command_override=None,clock=time.monotonic):
        validate_timing(control_hz,state_timeout_s)
        self.policy,self.interface,self.clock=policy,interface,clock
        self.controller=TorqueController(policy)
        self.period=1./control_hz
        self.policy_period=float(policy.metadata['policy_period_s'])
        if abs(self.policy_period-.02)>1e-8:
            raise ValueError('This runtime requires the selected 50Hz Actor contract')
        self.state_timeout=state_timeout_s
        self.override=command_override
        self.ramp=CommandRamp(policy.cfg)
        self.command=None
        self.started=clock()
        self.next_inference=None
        self.latest=None
        self.fault=None
        self.sent_actions=OrderedDict()
        self.last_ack_change=self.started
        self.last_ack=None
        self.acknowledged_action=[0.]*6
        self.last_step=None

    def _fault(self,reason):
        self.fault=reason
        try:
            self.interface.send_stop(1)
        except (OSError,ValueError,RuntimeError):
            pass
        raise RuntimeError(reason)

    def step(self):
        if self.fault is not None:
            raise RuntimeError('Fault remains latched: '+self.fault)
        start=self.clock()
        try:
            if self.last_step is not None and start-self.last_step>self.state_timeout:
                return self._fault('Host control scheduling gap exceeded the feedback budget')
            self.last_step=start
            frame=self.interface.receive_observation()
            if frame is not None:
                validate_raw_observation(frame,self.policy.cfg)
                if frame.flags&FAULT:
                    return self._fault('MCU reported a fault')
                if frame.flags&(SENSOR_VALID|VELOCITY_VALID)!=(SENSOR_VALID|VELOCITY_VALID):
                    return self._fault('MCU sensors or body-velocity estimate are invalid')
                if self.interface.execute:
                    required=CALIBRATION_VERIFIED|EXECUTION_AVAILABLE
                    if frame.flags&required!=required:
                        return self._fault('MCU has not declared calibrated torque execution available')
                    ack=frame.last_applied_torque_sequence
                    if ack in self.sent_actions:
                        if self.last_ack!=ack:
                            if self.last_ack is not None and not newer32(ack,self.last_ack):
                                return self._fault('MCU torque acknowledgement moved backwards')
                            self.acknowledged_action=list(self.sent_actions[ack])
                            self.last_ack=ack
                            self.last_ack_change=start
                    elif self.sent_actions and (ack!=0 or self.last_ack is not None) and ack!=self.last_ack:
                        return self._fault('MCU acknowledged an unknown torque sequence')
                self.latest=frame
            if self.latest is None:
                if start-self.started>1.:
                    return self._fault('No compatible STM32 observation within 1 second')
                return None
            if start-self.latest.received_at>self.state_timeout:
                return self._fault('Observation freshness timeout')
            if self.interface.execute and self.sent_actions and start-self.last_ack_change>self.state_timeout:
                return self._fault('Torque acknowledgement timeout')
            infer=self.next_inference is None or start+1e-9>=self.next_inference
            obs=list(self.latest.observation)
            if infer:
                s=self.policy.cfg['obs_scales']
                requested=self.override or [obs[6]/s['lin_vel'],obs[7]/s['ang_vel'],obs[8]/s['height_measurements']]
                self.command=self.ramp.step(requested,self.policy_period)
                # Hold one new reference per policy interval, never catch up by
                # generating a burst of inference or ramp steps after a stall.
                if self.next_inference is None:
                    self.next_inference=start+self.policy_period
                else:
                    skipped=max(1,int((start-self.next_inference)/self.policy_period)+1)
                    self.next_inference+=skipped*self.policy_period
            s=self.policy.cfg['obs_scales']
            obs[6:9]=[self.command[0]*s['lin_vel'],self.command[1]*s['ang_vel'],self.command[2]*s['height_measurements']]
            result=self.controller.step(obs,self.latest.actuator_state,infer=infer,
                   previous_action=self.acknowledged_action if self.interface.execute else None)
            compute=self.clock()-start
            if compute>self.period:
                return self._fault('PD/inference missed the configured control computation deadline')
            if self.clock()-self.latest.received_at>self.state_timeout:
                return self._fault('Observation became stale during inference')
            seq=self.interface.send_torques(result['torque_nm'],state_sequence=self.latest.sequence,
                   effective_action=result['effective_action'],command=result['command'])
            if not self.sent_actions:
                self.last_ack_change=self.clock()
            self.sent_actions[seq]=list(result['effective_action'])
            while len(self.sent_actions)>128:
                self.sent_actions.popitem(last=False)
            cycle=self.clock()-start
            if cycle>self.period:
                return self._fault('UART plus control computation missed the configured cycle deadline')
            result.update(torque_sequence=seq,source_state_sequence=self.latest.sequence,
                          device_time_us=self.latest.device_time_us,mode='execute' if self.interface.execute else 'shadow',
                          inference_performed=infer,compute_s=compute,cycle_s=cycle,
                          state_receive_age_s=self.clock()-self.latest.received_at,
                          mcu_execution_active=bool(self.latest.flags&EXECUTION_ACTIVE))
            return result
        except (OSError,ValueError,RuntimeError,OverflowError) as error:
            if self.fault is not None:
                raise
            return self._fault(str(error))


def warmup(policy):
    obs=[0.]*27
    obs[5]=-1.
    obs[8]=sum(policy.cfg['commands']['ranges_height'])/2*policy.cfg['obs_scales']['height_measurements']
    for _ in range(5):
        policy(obs)


def run_serial(bundle,port,*,baudrate=2000000,execute=False,control_hz=200,
               state_timeout_s=.03,log_path=None,duration_s=0.,command_override=None):
    validate_timing(control_hz,state_timeout_s)
    if isinstance(duration_s,bool) or not isinstance(duration_s,(int,float)) or not math.isfinite(duration_s) or duration_s<0:
        raise ValueError('Duration must be finite and nonnegative')
    policy=NumpyPolicy(bundle)
    if command_override is not None:
        CommandRamp(policy.cfg).step(command_override,policy.metadata['policy_period_s'])
    warmup(policy)
    interface,stream=None,None
    try:
        if log_path is not None:
            stream=Path(log_path).open('x',buffering=1,encoding='utf-8')
        interface=open_robot_serial(port,baudrate,policy.metadata['contract_sha256'],execute=execute)
        session=NxControlSession(policy,interface,control_hz=control_hz,state_timeout_s=state_timeout_s,command_override=command_override)
        start=time.monotonic()
        next_tick=start
        count=0
        while duration_s<=0 or time.monotonic()-start<duration_s:
            now=time.monotonic()
            if now<next_tick:
                time.sleep(next_tick-now)
            result=session.step()
            if result is not None:
                count+=1
                if stream and result['inference_performed']:
                    stream.write(json.dumps(dict(host_monotonic_s=time.monotonic(),**result),allow_nan=False)+'\n')
            # A delayed Linux wake-up cannot trigger queued stale torque bursts.
            next_tick+=session.period
            if next_tick<time.monotonic():
                next_tick=time.monotonic()+session.period
        return dict(torque_packets=count,mode='execute' if execute else 'shadow',
                    policy_accepted=False,real_hardware_motion_validated=False)
    finally:
        original_error=sys.exc_info()[0]
        try:
            if interface is not None:
                try:
                    interface.close()
                except (OSError,ValueError,RuntimeError):
                    if original_error is None:
                        raise
        finally:
            if stream:
                stream.close()
