import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import train_joint_pd_reward_block as block
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M


@pytest.fixture
def config():
    return dict(leg_control_mode='implicit_joint_position',joint_reset_profile='joint_target_nominal_v1',
        commands=dict(ranges_height=list(SAFE_BASE_HEIGHT_RANGE_M),full_height_range_from_start=True,
            standing_only_steps=0,yaw_start_steps=0,reverse_ramp_start_steps=0,motion_ramp_steps=48000,
            ranges_lin_vel_x=[.2,.8],ranges_ang_vel_yaw=[-.3,.3]),
        rewards=dict(base_height_error_sq=-50.,base_height_penalty_clip=30.,orientation=-400.,
            tracking_lin_vel=2.,tracking_ang_vel=1.,tracking_goal_component_violation=-.1),
        robot=dict(spawn=dict(usd_path='calibrated.usda')),leg_joint_drive_stiffness=300.,leg_joint_drive_damping=3.)


def test_only_strengthens_height_cost_and_preserves_source(config):
    old=copy.deepcopy(config)
    revised=block.revised_config(config,'base_height_error_sq',-500.)
    assert config==old
    assert revised['rewards']['base_height_error_sq']==-500.
    revised['rewards']['base_height_error_sq']=-50.
    assert revised==old


@pytest.mark.parametrize(('key','value'),[
    ('base_height_error_sq',-50.),('base_height_error_sq',-5.),('base_height_error_sq',-1001.),
    ('base_height_error_sq',float('nan')),('base_height_error_sq',float('inf')),
    ('base_height_error_sq',True),('orientation',-2000.),('tracking_lin_vel',20.),
])
def test_rejects_weakening_unbounded_or_unrelated_revision(config,key,value):
    with pytest.raises(ValueError,match='bounded'):
        block.revised_config(config,key,value)


@pytest.mark.parametrize(('group','key','value'),[
    (None,'leg_control_mode','explicit_vmc'),(None,'joint_reset_profile','cad_zero_v0'),
    ('commands','ranges_height',[.29,.31]),('commands','full_height_range_from_start',False),
    ('commands','standing_only_steps',100000),('commands','yaw_start_steps',100000),
    ('commands','reverse_ramp_start_steps',100000),('commands','motion_ramp_steps',96000),
])
def test_reward_revision_cannot_freeze_motion_or_narrow_height(config,group,key,value):
    (config if group is None else config[group])[key]=value
    with pytest.raises(ValueError,match='Preserve normal'):
        block.revised_config(config,'base_height_error_sq',-500.)


def test_cannot_flatten_height_feedback_with_a_small_clip(config):
    config['rewards']['base_height_penalty_clip']=5.
    with pytest.raises(ValueError,match='flat'):
        block.revised_config(config,'base_height_error_sq',-500.)


def samples(scale=-500.):
    values=np.zeros(3,dtype=[('height_m',float),('height_cmd_m',float),('reward_base_height_error_sq',float)])
    values['height_m']=[.3,.3,1.]
    values['height_cmd_m']=[.3,.225,.3]
    values['reward_base_height_error_sq']=np.maximum(scale*(values['height_m']-values['height_cmd_m'])**2,-30.)*.02
    return values


def test_independent_height_reward_formula_includes_coefficient_dt_and_clip(config):
    config['rewards']['base_height_error_sq']=-500.
    assert block.quadratic_height_discrepancy(samples(),config,.02)==0.
    assert block.quadratic_height_discrepancy(samples(-50.),config,.02)>.01


def test_nonfinite_actual_reward_is_not_hidden(config):
    values=samples()
    values['reward_base_height_error_sq'][0]=float('nan')
    with pytest.raises(ValueError,match='Nonfinite'):
        block.quadratic_height_discrepancy(values,config,.02)
