import copy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'mine_model/scripts'))
import train_joint_target_length_trial as trial


@pytest.fixture
def config():
    return dict(leg_control_mode='implicit_joint_position',joint_reset_profile='joint_target_nominal_v1',
        commands=dict(ranges_height=list(trial.SAFE_BASE_HEIGHT_RANGE_M),full_height_range_from_start=True,
            standing_only_steps=0,yaw_start_steps=0,reverse_ramp_start_steps=0,motion_ramp_steps=48000),
        rewards=dict(base_height_error_sq=-500.,orientation=-400.,tracking_lin_vel=2.,tracking_ang_vel=1.,
            base_height_penalty_clip=30.))


def test_only_adds_the_explicit_decision_cost_not_old_rewards_or_commands(config):
    old = copy.deepcopy(config)
    result = trial.revised_config(config)
    assert config == old
    assert result['rewards'].pop(trial.KEY) == -500.
    assert result == old
    assert trial.UPDATES == 100


@pytest.mark.parametrize(('group','key','value'),[
    ('commands','ranges_height',[.28,.32]),('commands','full_height_range_from_start',False),
    ('commands','standing_only_steps',48000),('commands','yaw_start_steps',48000),
    ('commands','reverse_ramp_start_steps',48000),('commands','motion_ramp_steps',96000),
    ('rewards','base_height_error_sq',0.),('rewards','orientation',-20.),
    ('rewards','tracking_lin_vel',0.),('rewards','tracking_ang_vel',0.),
    ('rewards',trial.KEY,-500.),
])
def test_cannot_repeat_started_reward_or_narrow_height_or_suppress_motion(config,group,key,value):
    config[group][key] = value
    with pytest.raises(ValueError):
        trial.revised_config(config)


def test_reward_cannot_become_flat_across_requested_height_span(config):
    config['rewards']['base_height_penalty_clip'] = 5.
    with pytest.raises(ValueError,match='flatten'):
        trial.revised_config(config)


def test_reward_snapshot_guard_rejects_actuator_execution_changes(tmp_path):
    path = tmp_path/'wheel_legged_vmc_flat_env.py'
    original = 'class WheelLeggedVMCFlatEnv:\n    def _apply_action(self):\n        return 35.574\n'
    path.write_text(original.replace('35.574','100.'))
    snapshot = {str(path):dict(sha256=hashlib.sha256(original.encode()).hexdigest(),source_text=original)}
    with pytest.raises(ValueError,match='boundaries'):
        trial.verify_reward_only_revision(snapshot)


def test_reward_snapshot_guard_rejects_nonzero_legacy_default():
    with pytest.raises(ValueError,match='default'):
        trial.normalized_revision_tree('class RewardsCfg:\n    '+trial.KEY+' = -500.\n',
                                       'wheel_legged_vmc_flat_env_cfg.py')


def test_reward_snapshot_cannot_hide_pre_revision_text_corruption(tmp_path):
    path = tmp_path/'reward_signals.py';path.write_text('x=1\n')
    with pytest.raises(ValueError,match='corrupt'):
        trial.verify_reward_only_revision({str(path):dict(sha256='wrong',source_text='x=1\n')})
