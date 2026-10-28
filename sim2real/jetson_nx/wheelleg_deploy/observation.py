"""Physical telemetry -> the exact trained 27D observation, scaled once."""
from .schema import finite_vector, validate_observation


def build_observation(physical, command, cfg, schema):
    # Physical24: gyro3, gravity3, q4, dq4, heading velocity2, wheels2, applied action6.
    p = finite_vector(physical,24,"physical telemetry")
    command = finite_vector(command,3,"command vx/yaw/height")
    s,c = cfg['obs_scales'],cfg['obs_clip']
    def block(values, key, scale=1.):
        bound=float(c[key])
        return [min(bound,max(-bound,v))*scale for v in values]
    result = block(p[:3],'ang_vel',s['ang_vel']) + block(p[3:6],'projected_gravity')
    result += [command[0]*s['lin_vel'],command[1]*s['ang_vel'],command[2]*s['height_measurements']]
    result += block([q-o for q,o in zip(p[6:10],cfg['leg_joint_position_offset'])],'theta',s['dof_pos'])
    result += block(p[10:14],'theta_dot',s['dof_vel'])
    result += block(p[14:16],'lin_vel',s['lin_vel'])
    result += block(p[16:18],'wheel_vel',s['dof_vel'])
    result += block(p[18:24],'action')
    return validate_observation(result,schema,cfg)


class CommandRamp:
    """Advance only on policy ticks, not on reads, logging or status packets."""
    def __init__(self,cfg):
        self.cfg=cfg
        self.applied_vx=0.

    def step(self,target,dt):
        vx,yaw,height=finite_vector(target,3,'requested command')
        cmd=self.cfg['commands']
        ranges=[cmd['ranges_lin_vel_x'],cmd['ranges_reverse_lin_vel_x']]
        lower=min(0.,*[v[0] for v in ranges]); upper=max(0.,*[v[1] for v in ranges])
        # Match schema validation: float32 UART representation can turn the
        # exact 0.8 boundary into 0.8000000119. Admit representation error only,
        # then restore the original range; do not expand the command envelope.
        checked=[]
        for value,(lo,hi) in zip((vx,yaw,height),((lower,upper),cmd['ranges_ang_vel_yaw'],cmd['ranges_height'])):
            if not lo-1e-6<=value<=hi+1e-6:
                raise ValueError('Command outside the selected policy range')
            checked.append(max(lo,min(hi,value)))
        vx,yaw,height=checked
        if not 0<dt<=.1:
            raise ValueError('Invalid policy interval')
        maximum=cmd['linear_acceleration_limit']*dt
        self.applied_vx+=max(-maximum,min(maximum,vx-self.applied_vx))
        return [self.applied_vx,yaw,height]
