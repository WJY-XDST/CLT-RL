import numpy as np
import pytest

from continue_joint_pd_stationary100_v2 import verify_rollout_motion


def test_actual_observer_schema_is_read_without_a_generic_commands_alias(tmp_path):
    path=tmp_path/'rollout.npz'
    np.savez(path,native_commands=np.zeros((48,32,3)))
    verify_rollout_motion(path,0.)


def test_height_commands_remain_variable_during_stationary_training(tmp_path):
    path=tmp_path/'rollout.npz';commands=np.zeros((48,32,3));commands[...,2]=.37534645531
    np.savez(path,native_commands=commands)
    verify_rollout_motion(path,0.)


@pytest.mark.parametrize('index',[0,1])
def test_nonzero_motion_in_stationary_data_is_not_silently_accepted(tmp_path,index):
    path=tmp_path/'rollout.npz';commands=np.zeros((48,32,3));commands[20,7,index]=.001
    np.savez(path,native_commands=commands)
    with pytest.raises(ValueError,match='nonzero'):verify_rollout_motion(path,0.)


def test_motion_phase_preserves_nonzero_commands(tmp_path):
    path=tmp_path/'rollout.npz';commands=np.ones((48,32,3))*.05
    np.savez(path,native_commands=commands)
    verify_rollout_motion(path,.125)
