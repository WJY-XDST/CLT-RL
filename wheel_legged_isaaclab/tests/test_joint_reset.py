import copy
from types import SimpleNamespace

import pytest
import torch

from wheel_legged_gym_isaaclab.joint_reset import (
    CAD_ZERO_RESET, NOMINAL_JOINT_RESET, DEFAULT_JOINT_OFFSETS,
    closed_chain_reset_pose, configure_reset_pose, require_matching_reset_profile,
    nominal_reset_config,
)
from wheel_legged_gym_isaaclab.robot_model import MAPPED_FIVE_BAR, MODEL
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state


def test_nominal_reset_actual_active_joints_equal_zero_action_targets():
    pose = closed_chain_reset_pose(DEFAULT_JOINT_OFFSETS, MAPPED_FIVE_BAR)
    assert pose['root_height_m'] == pytest.approx(.3, abs=1.e-10)
    assert len(pose['joint_positions']) == 14
    names = ('LB_joint', 'LL_joint', 'RB_joint', 'RL_joint')
    assert [pose['joint_positions'][name] for name in names] == list(DEFAULT_JOINT_OFFSETS)
    assert all(abs(leg['angle_rad']) < 1.e-9 for leg in pose['legs'])
    assert all(leg['length_m'] == pytest.approx(.26465354469) for leg in pose['legs'])


def test_zero_motor_reset_reproduces_CAD_height_and_zero_passive_coordinates():
    pose = closed_chain_reset_pose([0.]*4, MAPPED_FIVE_BAR)
    assert all(abs(value) < 1.e-12 for value in pose['joint_positions'].values())
    assert pose['root_height_m'] == pytest.approx(MODEL['nominal_root_height'], abs=1.e-12)


def test_closed_chain_reset_includes_mirrored_passive_and_closure_angles():
    pose = closed_chain_reset_pose(DEFAULT_JOINT_OFFSETS, MAPPED_FIVE_BAR)
    q = pose['joint_positions']
    for suffix in ('B_joint', 'L_joint', 'L1_joint', 'L2_joint', 'L3_joint', 'L3_LS_closure'):
        right = suffix.replace('_LS_', '_RS_')
        assert q['L'+suffix] == pytest.approx(-q['R'+right], abs=1.e-10)
    assert abs(q['LL1_joint']) > .001
    assert q['LL3_joint'] == pytest.approx(-q['LL3_LS_closure'])
    assert 'LS_joint' not in q and 'RS_joint' not in q


def test_passive_link_orientations_match_independent_tensor_forward_kinematics():
    pose = closed_chain_reset_pose(DEFAULT_JOINT_OFFSETS, MAPPED_FIVE_BAR)
    q = torch.tensor([DEFAULT_JOINT_OFFSETS], dtype=torch.float64)
    length, angle, _, valid = five_bar_state(q[:, ::2], q[:, 1::2], **MAPPED_FIVE_BAR)
    assert valid.all()
    for i, leg in enumerate(pose['legs']):
        assert leg['length_m'] == pytest.approx(float(length[0, i]), abs=1.e-12)
        assert leg['angle_rad'] == pytest.approx(float(angle[0, i]), abs=1.e-12)


@pytest.fixture
def cfg():
    return SimpleNamespace(leg_control_mode='implicit_joint_position', joint_reset_profile=NOMINAL_JOINT_RESET,
        leg_joint_position_offset=DEFAULT_JOINT_OFFSETS, five_bar_geometry=copy.deepcopy(MAPPED_FIVE_BAR),
        wheel_radius=.06, robot=SimpleNamespace(init_state=SimpleNamespace(
            pos=(0., 0., MODEL['nominal_root_height']), joint_pos={'.*': 0.})))


def test_nominal_configuration_updates_complete_physical_reset_template(cfg):
    configure_reset_pose(cfg)
    assert cfg.robot.init_state.pos[2] == pytest.approx(.3)
    assert cfg.robot.init_state.joint_pos['LB_joint'] == DEFAULT_JOINT_OFFSETS[0]
    assert len(cfg.robot.init_state.joint_pos) == 14
    assert '.*' not in cfg.robot.init_state.joint_pos


def test_historical_replay_retains_CAD_reset(cfg):
    configure_reset_pose(cfg)
    cfg.joint_reset_profile = CAD_ZERO_RESET
    configure_reset_pose(cfg)
    assert cfg.robot.init_state.pos[2] == MODEL['nominal_root_height']
    assert cfg.robot.init_state.joint_pos == {'.*': 0.}


def test_hanging_fixture_keeps_its_explicit_root_height(cfg):
    cfg.robot.init_state.pos = (0., 0., .9)
    configure_reset_pose(cfg)
    assert cfg.robot.init_state.pos[2] == .9


def test_old_checkpoint_cannot_silently_change_reset_profile():
    old = dict(leg_control_mode='implicit_joint_position')
    new = dict(old, joint_reset_profile=NOMINAL_JOINT_RESET)
    with pytest.raises(ValueError, match='allow_reset_pose_change'):
        require_matching_reset_profile(old, new)
    assert require_matching_reset_profile(old, new, True)['changed']
    with pytest.raises(ValueError): require_matching_reset_profile(new, old, True)


def test_nonfinite_or_asymmetric_reset_is_rejected():
    with pytest.raises(ValueError): closed_chain_reset_pose([float('nan')]*4, MAPPED_FIVE_BAR)
    with pytest.raises(ValueError): closed_chain_reset_pose([.1, 0., 0., 0.], MAPPED_FIVE_BAR)


def test_declared_migration_changes_only_reset_profile_and_complete_initial_pose():
    original=dict(leg_control_mode='implicit_joint_position',
        leg_joint_position_offset=list(DEFAULT_JOINT_OFFSETS),five_bar_geometry=copy.deepcopy(MAPPED_FIVE_BAR),
        wheel_radius=.06,rewards={'base_height_error_sq':-50.},commands={'ranges_height':[.22534645531,.37534645531]},
        robot={'init_state':{'pos':[0.,0.,MODEL['nominal_root_height']], 'joint_pos':{'.*':0.},
                            'joint_vel':{'.*':0.}},'spawn':{'usd_path':'unchanged.usda'}})
    before=copy.deepcopy(original)
    current=nominal_reset_config(original)
    assert original==before
    assert current['joint_reset_profile']==NOMINAL_JOINT_RESET
    assert current['robot']['init_state']['pos'][2]==pytest.approx(.3)
    assert len(current['robot']['init_state']['joint_pos'])==14
    current.pop('joint_reset_profile')
    current['robot']['init_state']=copy.deepcopy(original['robot']['init_state'])
    assert current==original
