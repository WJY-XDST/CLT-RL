"""Convert calibrated joint targets to leg torque and wheel current.

All feedback is at the joint/wheel output shaft, in rad and rad/s. This module
does not send commands. A hardware supervisor must handle timing, transport,
feedback freshness and the transition to its verified fallback controller.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path


def finite(value, name):
    value=float(value)
    if not math.isfinite(value):raise ValueError(name+' must be finite')
    return value


def positive(value, name):
    value=finite(value,name)
    if value<=0:raise ValueError(name+' must be positive')
    return value


def vector(values, count, name):
    if len(values)!=count:raise ValueError(f'{name} requires {count} values')
    return tuple(finite(v,name) for v in values)


def clip(value, limit):
    return min(limit,max(-limit,value))


@dataclass(frozen=True)
class WheelPIDConfig:
    # Gains produce wheel-end torque, not current: Nm/(rad/s), Nm/rad,
    # and Nm/(rad/s^2). Effective torque_per_amp includes the transmission.
    kp: float
    ki: float
    kd: float
    torque_per_amp: float
    current_limit_a: float
    torque_limit_nm: float
    current_slew_a_s: float
    derivative_filter_s: float
    max_dt_s: float

    def __post_init__(self):
        for name in ('kp','ki','kd','derivative_filter_s'):
            if finite(getattr(self,name),name)<0:raise ValueError(name+' must be nonnegative')
        for name in ('torque_per_amp','current_limit_a','torque_limit_nm','current_slew_a_s','max_dt_s'):
            positive(getattr(self,name),name)


class WheelSpeedCurrentPID:
    """Derivative on feedback, conditional anti-windup, current slew limiting."""
    def __init__(self, config):
        self.cfg=config
        self.reset()

    def reset(self):
        self.integral_torque=0.0
        self.previous_speed=None
        self.filtered_acceleration=0.0
        self.current_a=0.0

    def step(self, target_rad_s, measured_rad_s, dt_s, enabled=True, feedback_age_s=0.0):
        try:
            dt=positive(dt_s,'dt_s');age=finite(feedback_age_s,'feedback_age_s')
            target=finite(target_rad_s,'target_rad_s');speed=finite(measured_rad_s,'measured_rad_s')
            if dt>self.cfg.max_dt_s or age<0 or age>self.cfg.max_dt_s:
                raise ValueError('Control interval or feedback age exceeds configured bound')
        except (ValueError,TypeError):
            self.reset()
            raise
        if not enabled:
            self.reset()
            return {'current_a':0.0,'torque_nm':0.0,'error_rad_s':target-speed,'limited':False}
        c=self.cfg;error=target-speed
        acceleration=0.0 if self.previous_speed is None else (speed-self.previous_speed)/dt
        alpha=dt/(c.derivative_filter_s+dt)
        self.filtered_acceleration+=alpha*(acceleration-self.filtered_acceleration)
        self.previous_speed=speed
        limit=min(c.torque_limit_nm,c.current_limit_a*c.torque_per_amp)
        candidate=clip(self.integral_torque+c.ki*error*dt,limit)
        pd=c.kp*error-c.kd*self.filtered_acceleration

        def apply_limit(torque):
            current=clip(torque,limit)/c.torque_per_amp
            change=clip(current-self.current_a,c.current_slew_a_s*dt)
            return clip(self.current_a+change,c.current_limit_a)

        requested=pd+candidate
        applied=apply_limit(requested)
        # Freeze integration if current/torque/rate limits prevent the error
        # correcting command. Integration in the opposite direction unwinds.
        if (requested-applied*c.torque_per_amp)*error<=0:
            self.integral_torque=candidate
        applied=apply_limit(pd+self.integral_torque)
        requested=pd+self.integral_torque
        self.current_a=applied
        return {'current_a':applied,'torque_nm':applied*c.torque_per_amp,
                'error_rad_s':error,'limited':abs(requested-applied*c.torque_per_amp)>1e-9}


@dataclass(frozen=True)
class LegPDConfig:
    # Logical leg order LB, LL, RB, RL; calibrated raw joint axes.
    kp: tuple
    kd: tuple
    torque_limits_nm: tuple
    joint_lower_rad: tuple
    joint_upper_rad: tuple

    def __post_init__(self):
        for name in ('kp','kd','torque_limits_nm','joint_lower_rad','joint_upper_rad'):
            vector(getattr(self,name),4,name)
        for i in range(4):
            if self.kp[i]<=0 or self.kd[i]<0 or self.torque_limits_nm[i]<=0:
                raise ValueError('Invalid leg gain or torque limit')
            if self.joint_lower_rad[i]>=self.joint_upper_rad[i]:
                raise ValueError('Invalid calibrated joint range')


def leg_joint_pd(config, targets_rad, positions_rad, velocities_rad_s):
    """Limited revolute joints use direct position error, without modulo wrap."""
    targets=vector(targets_rad,4,'targets');positions=vector(positions_rad,4,'positions')
    velocities=vector(velocities_rad_s,4,'velocities')
    for i in range(4):
        if not (config.joint_lower_rad[i]<=targets[i]<=config.joint_upper_rad[i] and
                config.joint_lower_rad[i]<=positions[i]<=config.joint_upper_rad[i]):
            raise ValueError('Leg target or position exceeds calibrated joint limits')
    return tuple(clip(config.kp[i]*(targets[i]-positions[i])-config.kd[i]*velocities[i],
                      config.torque_limits_nm[i]) for i in range(4))


class ActuatorBridge:
    """Joint-space bridge; preserves mirrored wheel axes from the trained model."""
    def __init__(self, leg_config, wheel_configs, wheel_axis_signs, current_polarities):
        if len(wheel_configs)!=2:raise ValueError('Two wheel configurations required')
        self.legs=leg_config
        self.wheels=[WheelSpeedCurrentPID(c) for c in wheel_configs]
        self.axis_signs=vector(wheel_axis_signs,2,'wheel_axis_signs')
        self.current_polarities=vector(current_polarities,2,'current_polarities')
        if any(abs(v)!=1 for v in self.axis_signs+self.current_polarities):
            raise ValueError('Axis/current polarities must be calibrated to +1 or -1')

    def reset(self):
        for wheel in self.wheels:wheel.reset()

    def step(self, leg_targets, leg_positions, leg_velocities, wheel_targets_forward,
             wheel_velocities_raw, dt_s, enabled=True, feedback_age_s=0.0):
        try:
            targets=vector(wheel_targets_forward,2,'wheel_targets_forward')
            speeds=vector(wheel_velocities_raw,2,'wheel_velocities_raw')
            leg_torques=leg_joint_pd(self.legs,leg_targets,leg_positions,leg_velocities)
            wheels=[pid.step(targets[i]*self.axis_signs[i],speeds[i],dt_s,
                             enabled=enabled,feedback_age_s=feedback_age_s)
                    for i,pid in enumerate(self.wheels)]
        except (ValueError,TypeError):
            self.reset()
            raise
        return {'leg_torques_nm':leg_torques if enabled else (0.0,)*4,
                'wheel_currents_a':tuple(w['current_a']*self.current_polarities[i]
                                         for i,w in enumerate(wheels)),
                'wheel_diagnostics':wheels}


def load_bridge_config(path):
    """Template requires explicit hardware calibration; no guessed Nm/A."""
    data=json.loads(Path(path).read_text())
    if data.get('hardware_calibrated') is not True:
        raise ValueError('Complete and verify hardware calibration before loading')
    return ActuatorBridge(LegPDConfig(**data['leg_pd']),
        [WheelPIDConfig(**c) for c in data['wheel_pid']],
        data['wheel_axis_signs'],data['current_polarities'])
