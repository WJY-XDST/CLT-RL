"""NX direct joint/velocity PD plus gravity FF; no IK, VMC feedback or gas FF.

The Actor still requests joint angles and wheel speeds. Final torques are
clamped after PD plus gravity. State feedback must be raw/unclipped SI data.
"""
import math

from .schema import finite_vector,validate_observation
from .targets import clip_action

LEG_COLUMNS=(0,1,3,4)


def gravity_effort(q,pitch,cfg):
    import numpy as np
    g=cfg['five_bar_geometry']
    result=[]
    for side,sign in enumerate((1.,-1.)):
        b,l=g['offsets'][0]+sign*q[2*side],g['offsets'][1]+sign*q[2*side+1]
        c=g['upper'][0]*np.array([math.cos(b),math.sin(b)])
        a=g['upper'][1]*np.array([math.cos(l),math.sin(l)])
        delta=a-c
        distance=float(np.linalg.norm(delta))
        if distance<=1e-5:
            raise ValueError('Five-bar motor geometry is singular')
        along=(g['lower'][0]**2-g['lower'][1]**2+distance**2)/(2*distance)
        h2=g['lower'][0]**2-along**2
        if h2<=1e-10:
            raise ValueError('Five-bar closure is invalid')
        p=c+along*delta/distance+g['branch']*math.sqrt(h2)*np.array([-delta[1],delta[0]])/distance
        u,v=p-c,p-a
        determinant=float(u[0]*v[1]-u[1]*v[0])
        if abs(determinant)<=1e-7:
            raise ValueError('Five-bar Jacobian is singular')
        dc=sign*g['upper'][0]*np.array([-math.sin(b),math.cos(b)])
        da=sign*g['upper'][1]*np.array([-math.sin(l),math.cos(l)])
        rb,rl=float(u@dc),float(v@da)
        cart=g['scale']*np.array([[v[1]*rb,-u[1]*rl],[-v[0]*rb,u[0]*rl]])/determinant
        wheel=g['scale']*p
        length=float(np.linalg.norm(wheel))
        radial=(wheel/length)@cart
        angle=math.atan2(-wheel[1],-wheel[0])
        force=cfg['feedforward_force']
        if cfg['feedforward_mode']=='mcu_pitch_compensated':
            absolute=angle+pitch+cfg['feedforward_angle_bias']
            if absolute>1.1:
                absolute-=2*math.pi
            cosine=math.cos(absolute)
            denominator=math.copysign(max(abs(cosine),1e-6),cosine)
            limit=cfg['feedforward_force_limit']
            force=max(-limit,min(limit,force/denominator))
            if absolute>math.pi/2:
                force=-force
        elif cfg['feedforward_mode']!='constant':
            raise ValueError('Unsupported gravity feedforward contract')
        result.extend((radial*force).tolist())
    return finite_vector(result,4,'gravity effort')


class TorqueController:
    def __init__(self,policy):
        self.policy,self.cfg=policy,policy.cfg
        self.effective_action=[0.]*6
        self.position_target=list(self.cfg['leg_joint_position_offset'])
        self.wheel_target=[0.,0.]

    def step(self,observation,actuator_state,*,infer,previous_action=None):
        cfg=self.cfg
        obs=finite_vector(observation,27,'scaled observation')
        raw=finite_vector(actuator_state,10,'unclipped q4/dq4/wheel speed2')
        q,dq,wheel=raw[:4],raw[4:8],raw[8:10]
        if any(not lo-1e-4<=value<=hi+1e-4 for value,(lo,hi) in zip(q,cfg['leg_joint_position_limits'])):
            raise ValueError('Raw motor angle is outside the selected joint limits')
        limit=cfg['obs_clip']['action']
        history=self.effective_action if previous_action is None else finite_vector(previous_action,6,'acknowledged action')
        obs[21:27]=[max(-limit,min(limit,a)) for a in history]
        validate_observation(obs,self.policy.schema,cfg)
        if infer:
            action=clip_action(self.policy(obs),cfg)
            for j,column in enumerate(LEG_COLUMNS):
                center,scale=cfg['leg_joint_position_offset'][j],cfg['leg_joint_position_scale'][j]
                lo,hi=cfg['leg_joint_position_limits'][j]
                lo=max(lo+cfg['leg_joint_target_limit_margin'],center-scale)
                hi=min(hi-cfg['leg_joint_target_limit_margin'],center+scale)
                if not lo<=center<=hi:
                    raise ValueError('Invalid joint target bounds')
                self.position_target[j]=max(lo,min(hi,center+scale*action[column]))
                action[column]=(self.position_target[j]-center)/scale
            self.effective_action=action
            self.wheel_target=[action[j]*cfg['action_scale_vel'] for j in (2,5)]
        pitch=math.asin(max(-1.,min(1.,obs[3])))
        raw_ff=gravity_effort(q,pitch,cfg) if cfg['leg_joint_gravity_feedforward'] else [0.]*4
        # Match the trained external-effort channel before imposing the final
        # physical actuator limit on PD + feedforward together.
        ff=[max(-cfg['leg_effort_limit'],min(cfg['leg_effort_limit'],v)) for v in raw_ff]
        torque=[0.]*6
        before=[]
        for j,column in enumerate(LEG_COLUMNS):
            value=cfg['leg_joint_stiffness']*(self.position_target[j]-q[j])-cfg['leg_joint_damping']*dq[j]+ff[j]
            before.append(value)
            torque[column]=max(-cfg['leg_effort_limit'],min(cfg['leg_effort_limit'],value))
        for j,column in enumerate((2,5)):
            value=cfg['wheel_damping']*(self.wheel_target[j]-wheel[j])
            torque[column]=max(-cfg['wheel_effort_limit'],min(cfg['wheel_effort_limit'],value))
        command=[obs[6]/cfg['obs_scales']['lin_vel'],obs[7]/cfg['obs_scales']['ang_vel'],obs[8]/cfg['obs_scales']['height_measurements']]
        return dict(torque_nm=finite_vector(torque,6,'total motor torque'),
                    effective_action=list(self.effective_action),position_target_rad=list(self.position_target),
                    wheel_target_rad_s=list(self.wheel_target),gravity_effort_nm=ff,
                    gravity_unclipped_effort_nm=raw_ff,
                    leg_total_before_clamp_nm=before,observation=obs,command=command,
                    gas_spring_feedforward=False)
