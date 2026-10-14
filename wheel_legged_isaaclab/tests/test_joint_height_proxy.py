import json
from pathlib import Path

import numpy as np
import pytest
import torch

from wheel_legged_gym_isaaclab.joint_height_feedback import model_from_config, predict_support_heights, height_proxy_loss
from wheel_legged_gym_isaaclab.robot_model import MODEL

ROOT=Path('/home/aaa/studyRL/src/CLT-RL')
REPLAY=ROOT/'mine_model/results/joint_pd_optimization_20261015_004022_041465_mass_recovery_cumulative/early_4830_replay_verified'


@pytest.fixture
def model():
    cfg=json.loads((REPLAY/'resolved_env.json').read_text())
    return model_from_config(cfg,[[-1.56999993,1.56999993]]*4)


def observations(command=.3):
    value=torch.zeros(1,27,dtype=torch.float64)
    value[:,5]=-1.
    value[:,8]=command*5.
    return value


def test_nominal_forward_pose_matches_height_without_an_ik_solution(model):
    means=torch.zeros(1,6,dtype=torch.float64)
    height,length,valid=predict_support_heights(means,observations(),model)
    torch.testing.assert_close(height,torch.full((1,2),.3,dtype=torch.float64),atol=1.e-8,rtol=0.)
    assert valid.all() and height_proxy_loss(means,observations(),model)==0.
    assert float(length.mean())==pytest.approx(.3-model['wheel_radius']+MODEL['hip_z'])


@pytest.mark.parametrize('command',[.22534645531,.37534645531])
def test_small_direct_joint_gradient_step_reduces_geometric_height_error(model,command):
    means=torch.zeros(1,6,dtype=torch.float64,requires_grad=True)
    obs=observations(command)
    loss=height_proxy_loss(means,obs,model)
    loss.backward()
    assert torch.isfinite(means.grad).all()
    torch.testing.assert_close(means.grad[:,(2,5)],torch.zeros(1,2,dtype=torch.float64),rtol=0.,atol=0.)
    original=(predict_support_heights(means,obs,model)[0]-command).abs().mean()
    candidate=(predict_support_heights(means.detach()-.001*means.grad,obs,model)[0]-command).abs().mean()
    assert candidate<original


def test_unreliable_fast_rotation_does_not_supply_height_proxy_gradient(model):
    obs=observations(.37534645531)
    obs[:,0]=3.*model['gyro_scale']
    assert height_proxy_loss(torch.zeros(1,6,dtype=torch.float64),obs,model)==0.


def test_actual_frozen_policy_samples_get_finite_leg_only_direct_gradients(model):
    with np.load(REPLAY/'policy_inputs.npz') as saved:
        obs=torch.tensor(saved['observations'][-150:].reshape(-1,27),dtype=torch.float64)
        means=torch.tensor(saved['means'][-150:].reshape(-1,6),dtype=torch.float64,requires_grad=True)
    loss=height_proxy_loss(means,obs,model)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(means.grad).all()
    assert means.grad[:,(0,1,3,4)].abs().sum()>0.
    assert means.grad[:,(2,5)].abs().max()==0.


def test_existing_gate_and_mechanical_margin_define_the_proxy(model):
    assert model['relative_tolerance']==.1
    assert model['training_fraction']==.5
    assert model['safe_leg_length_range']==[.19,.34]
    assert model['stop_margin']==.01
