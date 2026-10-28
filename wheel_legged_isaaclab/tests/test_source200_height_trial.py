import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import train_joint_pd_source200_height_trial as trial
from joint_pd_continuation import HEIGHT_DEFAULTS
from wheel_legged_gym_isaaclab.joint_reset import nominal_reset_config
from wheel_legged_gym_isaaclab.robot_model import MODEL, MAPPED_FIVE_BAR


@pytest.fixture
def configs():
    source = dict(leg_control_mode='implicit_joint_position', observation_space=27,
        leg_joint_position_offset=[-.0707452938,-.0336936327,.0707452938,.0336936327],
        leg_joint_position_scale=[.3]*4, wheel_radius=.06, five_bar_geometry=copy.deepcopy(MAPPED_FIVE_BAR),
        robot=dict(spawn=dict(usd_path='historical.usd'),
                   init_state=dict(pos=[0.,0.,MODEL['nominal_root_height']],joint_pos={'.*':0.})),
        commands=dict(HEIGHT_DEFAULTS,ranges_height=[.28,.32],standing_only_steps=48000,yaw_start_steps=48000,
            reverse_ramp_start_steps=48000,motion_ramp_steps=48000),
        rewards=dict(base_height_error_sq=0.,orientation=-20.,tracking_lin_vel=2.,tracking_ang_vel=1.))
    current = nominal_reset_config(source)
    current['robot']['spawn']['usd_path'] = 'calibrated.usda'
    current['commands'].update(ranges_height=list(trial.SAFE_BASE_HEIGHT_RANGE_M),
        full_height_range_from_start=True,standing_only_steps=0,yaw_start_steps=0,reverse_ramp_start_steps=0)
    current['rewards'].update(base_height_error_sq=-500.,orientation=-400.,tracking_goal_component_violation=-.1)
    return source, current


def test_owner_fork_preserves_action_map_and_source_while_adopting_current_contract(configs):
    source, current = configs
    before = copy.deepcopy(source)
    result = trial.branch_config(source, current)
    assert source == before
    assert result['rewards'] == current['rewards']
    assert result['commands'] == current['commands']
    assert result['robot'] == current['robot']
    assert result['leg_joint_position_offset'] == source['leg_joint_position_offset']
    assert trial.UPDATES == 100


@pytest.mark.parametrize(('group','key','value'), [
    ('commands','ranges_height',[.28,.32]),('commands','full_height_range_from_start',False),
    ('commands','standing_only_steps',48000),('commands','yaw_start_steps',48000),
    ('commands','reverse_ramp_start_steps',48000),('commands','motion_ramp_steps',96000),
    ('rewards','base_height_error_sq',-50.),('rewards','orientation',-20.),
    ('rewards','tracking_lin_vel',0.),('rewards','tracking_ang_vel',0.),
    ('rewards','tracking_goal_component_violation',0.),
])
def test_cannot_return_to_narrow_height_static_only_or_unreinforced_training(configs,group,key,value):
    source, current = configs
    current[group][key] = value
    with pytest.raises(ValueError):
        trial.branch_config(source, current)


def test_cannot_silently_change_joint_action_offsets(configs):
    source, current = configs
    current['leg_joint_position_offset'][0] += .01
    with pytest.raises(ValueError):
        trial.branch_config(source, current)


def reset_record():
    return dict(completed_ppo_updates=201,initial_common_step_counter=201*48,motion_progress=.201,
        height_bounds_m=list(trial.SAFE_BASE_HEIGHT_RANGE_M),
        sampled_command_height_min_m=trial.SAFE_BASE_HEIGHT_RANGE_M[0],
        sampled_command_height_max_m=trial.SAFE_BASE_HEIGHT_RANGE_M[1])


def test_real_Adam_clock_not_filename_and_full_height_on_first_reset():
    trial.audit_first_reset(reset_record(),201)


@pytest.mark.parametrize(('key','value'),[
    ('completed_ppo_updates',200),('initial_common_step_counter',0),('motion_progress',0.),
    ('height_bounds_m',[.28,.32]),('sampled_command_height_min_m',.28),('sampled_command_height_max_m',.32)])
def test_rejects_reset_clock_restart_or_height_warmup(key,value):
    start = reset_record(); start[key] = value
    with pytest.raises(ValueError):
        trial.audit_first_reset(start,201)
