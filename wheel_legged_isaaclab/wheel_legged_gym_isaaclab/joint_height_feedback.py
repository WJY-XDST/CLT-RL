"""Forward geometry for diagnosing a possible height training loss.

This staged module does not run in production. It never replaces policy
actions, solves inverse kinematics or computes control efforts.
"""
import copy
import math
from pathlib import Path

import torch
import yaml

from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state
from wheel_legged_gym_isaaclab.height_commands import SAFE_LEG_LENGTH_RANGE_M, LEG_STOP_MARGIN_M
from wheel_legged_gym_isaaclab.leg_control import joint_position_targets
from wheel_legged_gym_isaaclab.robot_model import MODEL
from wheel_legged_gym_isaaclab.tracking_goal_feedback import load_tracking_goal_criteria


def validate_coefficient(value):
    if (isinstance(value,bool) or not isinstance(value,(int,float))
            or not math.isfinite(value) or not 0.<=value<=.25):
        raise ValueError('Height feedback coefficient must be finite and within [0, .25]')
    return float(value)


def resolve_coefficient(checkpoint=None,requested=None):
    if requested is not None:
        return validate_coefficient(requested)
    if checkpoint is None:
        return 0.
    path=Path(checkpoint).parent/'params/agent.yaml'
    if not path.is_file():
        return 0.
    algorithm=yaml.safe_load(path.read_text())['algorithm']
    value=validate_coefficient(algorithm.get('joint_height_feedback_coef',0.))
    if value and algorithm.get('class_name')!='JointMeanGuardPPO':
        raise ValueError('Height feedback requires the qualified PPO variant')
    return value


def model_from_config(cfg, joint_limits):
    if cfg['leg_control_mode'] != 'implicit_joint_position' or cfg['observation_space'] != 27:
        raise ValueError('Require the unchanged direct joint-angle policy contract')
    return dict(geometry=copy.deepcopy(cfg['five_bar_geometry']),
        offset=list(cfg['leg_joint_position_offset']), scale=list(cfg['leg_joint_position_scale']),
        joint_limits=joint_limits, margin=cfg['leg_joint_target_limit_margin'],
        wheel_radius=cfg['wheel_radius'], hip_z=MODEL['hip_z'], wheel_track_width=MODEL['wheel_track_width'],
        height_scale=cfg['obs_scales']['height_measurements'], gyro_scale=cfg['obs_scales']['ang_vel'],
        relative_tolerance=load_tracking_goal_criteria()[0]['relative_error'], training_fraction=.5,
        safe_leg_length_range=list(SAFE_LEG_LENGTH_RANGE_M), stop_margin=LEG_STOP_MARGIN_M)


def predict_support_heights(means, observations, model):
    if means.ndim != 2 or means.shape[1] != 6 or observations.shape != (means.shape[0],27):
        raise ValueError('Expected six raw means and the original 27 observations')
    limits=means.new_tensor(model['joint_limits']).expand(means.shape[0],-1,-1)
    target,_=joint_position_targets(means[:,(0,1,3,4)],model['offset'],model['scale'],
                                   limits,model['margin'],validate=False)
    length,angle,_,valid=five_bar_state(target[:,::2],target[:,1::2],**model['geometry'])
    gravity=observations[:,3:6].detach()
    x=-length*angle.sin()
    y=length.new_tensor([model['wheel_track_width']/2,-model['wheel_track_width']/2])
    z=model['hip_z']-length*angle.cos()
    height=model['wheel_radius']+gravity[:,0,None]*x+gravity[:,1,None]*y+gravity[:,2,None]*z
    return height,length,valid


def height_proxy_loss(means, observations, model):
    height,length,valid=predict_support_heights(means,observations,model)
    command=observations[:,8].detach()/model['height_scale']
    tolerance=model['relative_tolerance']*command.clamp_min(.001)
    height_excess=((height-command[:,None]).abs()/tolerance[:,None]-model['training_fraction']).clamp_min(0.)
    low,high=model['safe_leg_length_range']
    stop_excess=((low-length).clamp_min(0.)+(length-high).clamp_min(0.))/model['stop_margin']
    reliable=((observations[:,5].detach() < -math.cos(math.radians(20.)))
              &(torch.linalg.vector_norm(observations[:,:3].detach(),dim=-1)/model['gyro_scale'] < 2.)
              &(command>0.))
    def huber(values):
        return torch.where(values<1.,.5*values.square(),values-.5)
    return (huber(height_excess)*reliable[:,None]*valid+huber(stop_excess)*valid).mean()
