import copy
import math

import pytest

import diagnose_plain_joint_pd_height as diagnostic


def test_plain_ppo_needs_no_optional_auxiliary_model():
    agent = dict(algorithm=dict(class_name='PPO'), policy=dict(activation='elu'))
    before = copy.deepcopy(agent)
    diagnostic.require_plain_ppo(agent)
    assert agent == before and 'joint_height_feedback_model' not in agent['algorithm']


@pytest.mark.parametrize('field', ['class_name', 'joint_mean_guard_coef', 'joint_height_feedback_coef'])
def test_active_auxiliary_training_is_not_silently_treated_as_plain_ppo(field):
    agent = dict(algorithm=dict(class_name='PPO'), policy=dict(activation='elu'))
    agent['algorithm'][field] = 'JointMeanGuardPPO' if field == 'class_name' else .1
    with pytest.raises(ValueError, match='without auxiliary'):
        diagnostic.require_plain_ppo(agent)


def test_geometry_has_no_training_profile_or_invented_hard_limits():
    cfg = dict(leg_control_mode='implicit_joint_position', observation_space=27,
        five_bar_geometry=dict(upper=[.21,.21]), leg_joint_position_offset=[0.]*4,
        leg_joint_position_scale=[.3]*4, leg_joint_target_limit_margin=.01,
        wheel_radius=.06, wheel_track_width=.4199, obs_scales=dict(height_measurements=5.))
    limits = [[-1., 2.], [-2., 1.], [-3., .5], [-.5, 3.]]
    before = copy.deepcopy(cfg)
    model = diagnostic.geometry(cfg, limits)
    assert cfg == before and model['joint_limits'] == limits
    assert 'command_counterfactuals' not in model and 'relative_tolerance' not in model


def test_usd_limits_use_authored_degrees_in_requested_motor_order(tmp_path):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdPhysics
    path = tmp_path/'limits.usda'
    stage = Usd.Stage.CreateNew(str(path))
    for name, limit in [('LL_joint', 30.), ('LB_joint', 90.)]:
        joint = UsdPhysics.RevoluteJoint.Define(stage, '/'+name)
        joint.CreateLowerLimitAttr(-limit)
        joint.CreateUpperLimitAttr(limit)
    stage.GetRootLayer().Save()
    values = diagnostic.usd_joint_limits(path, ['LB_joint', 'LL_joint'])
    assert values[0] == pytest.approx([-math.pi/2, math.pi/2])
    assert values[1] == pytest.approx([-math.pi/6, math.pi/6])
    with pytest.raises(ValueError, match='authored hard limit'):
        diagnostic.usd_joint_limits(path, ['missing_joint'])
