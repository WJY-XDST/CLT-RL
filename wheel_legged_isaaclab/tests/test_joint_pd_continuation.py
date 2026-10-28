"""Cumulative progress and physical-contract safeguards, without a simulator."""
import copy
import hashlib

import pytest
import torch

from joint_pd_continuation import (continuation_command, optimizer_step, stagnant,
                                   validate_continuation_config, qualified_wheel_mass_asset)


def test_resume_keeps_adam_and_policy_and_uses_loaded_iteration(tmp_path):
    checkpoint=tmp_path/'model_2438_interrupted.pt'
    before=dict(iter=2438,model_state_dict={'actor.6.weight':torch.ones(6,64)},
                optimizer_state_dict={'state':{0:{'step':torch.tensor(28900.),'exp_avg':torch.ones(6,64)}},
                                      'param_groups':[{'lr':.0000759375}]})
    torch.save(before,checkpoint)
    argv,final,loaded=continuation_command(checkpoint,{'rewards':{'orientation':-60.}},'resume',100,
                                         {'ranges_height':[.22534645531,.37534645531]},tmp_path/'STOP')
    assert final==2537
    assert argv[argv.index('--target_iteration')+1]=='2538'
    assert '--reset_optimizer' not in argv and '--reset_exploration_std' not in argv
    assert optimizer_step(loaded)==28900
    assert loaded['optimizer_state_dict']['param_groups'][0]['lr']==.0000759375


def test_reset_pose_migration_is_explicit_and_never_resets_policy_or_Adam(tmp_path):
    checkpoint=tmp_path/'model_349.pt'
    saved=dict(iter=349,model_state_dict={'actor':torch.ones(1)},
               optimizer_state_dict={'state':{0:{'step':torch.tensor(7000.)}},'param_groups':[{'lr':.00002}]})
    torch.save(saved,checkpoint)
    cfg=dict(leg_control_mode='implicit_joint_position',rewards={})
    original_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='allow_reset_pose_change'):
        continuation_command(checkpoint,cfg,'reset',5,{},tmp_path/'STOP',
                             joint_reset_profile='joint_target_nominal_v1')
    command,_,loaded=continuation_command(checkpoint,cfg,'reset',5,{},tmp_path/'STOP',
        joint_reset_profile='joint_target_nominal_v1',allow_reset_pose_change=True)
    assert '--allow_reset_pose_change' in command
    assert 'env.joint_reset_profile="joint_target_nominal_v1"' in command
    assert optimizer_step(loaded)==7000 and loaded['optimizer_state_dict']['param_groups'][0]['lr']==.00002
    assert '--reset_optimizer' not in command and '--reset_exploration_std' not in command
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==original_hash
    legacy,_,_=continuation_command(checkpoint,cfg,'legacy',5,{},tmp_path/'STOP')
    assert 'env.joint_reset_profile="cad_zero_v0"' in legacy
    assert '--allow_reset_pose_change' not in legacy


def test_only_the_declared_height_command_change_is_permitted():
    previous={'commands':{'ranges_height':[.28,.32],'ranges_ang_vel_yaw':[-.3,.3]},
              'rewards':{'base_height_relative_deadband':.05},'leg_joint_stiffness':300.}
    commands=dict(previous['commands'],ranges_height=[.22534645531,.37534645531],
                  height_expansion_from=[.28,.32],height_expansion_start_steps=117024,
                  height_expansion_ramp_steps=48000)
    current=dict(previous,commands=commands)
    validate_continuation_config(previous,current,commands)
    modified=copy.deepcopy(current); modified['leg_joint_stiffness']=400.
    with pytest.raises(ValueError,match='leg_joint_stiffness'):
        validate_continuation_config(previous,modified,commands)


def test_mass_calibrated_resume_retains_weights_adam_and_all_other_settings(tmp_path):
    checkpoint = tmp_path/'model_3237.pt'
    saved = dict(iter=3237, model_state_dict={'actor.6.weight':torch.ones(6,64)},
                 optimizer_state_dict={'state':{0:{'step':torch.tensor(45060.)}},
                                       'param_groups':[{'lr':.00003375}]})
    torch.save(saved, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    previous = {'robot':{'spawn':{'usd_path':'/tmp/original_cad.usd'}},
                'commands':{}, 'rewards':{'orientation':-60.}, 'leg_joint_stiffness':300.}
    commands = dict(height_expansion_from=None, height_expansion_start_steps=0,
                    height_expansion_ramp_steps=48000)
    calibrated = '/tmp/owner_confirmed_mass.usda'
    argv, final, before = continuation_command(checkpoint, previous, 'mass', 100, commands,
        tmp_path/'STOP', minimum_wheel_std=.03, robot_asset=calibrated)
    assert '--allow_physics_contract_change' in argv
    assert 'env.robot.spawn.usd_path="/tmp/owner_confirmed_mass.usda"' in argv
    assert '--reset_optimizer' not in argv and '--reset_exploration_std' not in argv
    assert final == 3336 and optimizer_step(before) == 45060
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    current = copy.deepcopy(previous)
    current['commands'] = commands
    current['robot']['spawn']['usd_path'] = calibrated
    validate_continuation_config(previous, current, commands, robot_asset=calibrated)
    with pytest.raises(ValueError, match='robot'):
        validate_continuation_config(previous, current, commands)
    current['leg_joint_stiffness'] = 400.
    with pytest.raises(ValueError, match='leg_joint_stiffness'):
        validate_continuation_config(previous, current, commands, robot_asset=calibrated)


def test_calibration_admission_uses_verified_owner_profile_only(tmp_path):
    from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST
    asset, proof = qualified_wheel_mass_asset(ASSETS / MANIFEST['wheel_mass_calibration'])
    assert asset == str((ASSETS / MANIFEST['training_usd']).resolve())
    assert proof['wheel_masses_kg'] == {'LW_link': .665, 'RW_link': .665}
    assert proof['inertia_measured'] is False
    assert len(proof['asset_chain_sha256']) >= 8
    with pytest.raises(ValueError, match='owner-confirmed'):
        qualified_wheel_mass_asset(tmp_path/'unqualified.json')


def test_stagnation_uses_three_independent_milestones_and_respects_progress():
    assert not stagnant([4.,4.])
    assert stagnant([4.,4.01,4.02])
    assert not stagnant([4.,3.6,3.3])
    assert not stagnant([.9,.9,.9])


def test_prepared_wheel_reward_carries_the_source_and_optimizer(tmp_path):
    checkpoint=tmp_path/'model_2544_interrupted.pt'
    torch.save(dict(iter=2544,model_state_dict={'actor.6.weight':torch.ones(6,64)},
                    optimizer_state_dict={'state':{0:{'step':torch.tensor(31040.)}},
                                          'param_groups':[{'lr':.00003375}]}),checkpoint)
    cfg={'rewards':{'orientation':-60.,'yaw_rate_error_sq':-10.,'base_height_relative_deadband':.05}}
    argv,final,loaded=continuation_command(checkpoint,cfg,'feedback',100,{},tmp_path/'STOP',
                                         reward_overrides={'rewards.wheel_action_saturation':-2.})
    assert 'env.rewards.wheel_action_saturation=-2.0' in argv
    assert 'env.rewards.orientation=-60.0' in argv
    assert 'env.rewards.yaw_rate_error_sq=-10.0' in argv
    assert final==2643 and optimizer_step(loaded)==31040
    assert '--reset_optimizer' not in argv and '--reset_exploration_std' not in argv


def test_only_prepared_reward_is_changed_during_continuation():
    previous={'commands':{},'rewards':{'orientation':-60.,'base_height_relative_deadband':.05}}
    commands={'height_expansion_from':None,'height_expansion_start_steps':0,'height_expansion_ramp_steps':48000}
    current={'commands':commands,'rewards':dict(previous['rewards'],wheel_action_saturation=-2.,
                                               wheel_action_saturation_penalty_clip=40.)}
    change={'rewards.wheel_action_saturation':-2.}
    validate_continuation_config(previous,current,commands,change)
    current['rewards']['orientation']=-80.
    with pytest.raises(ValueError,match='rewards'):
        validate_continuation_config(previous,current,commands,change)


def test_selective_wheel_floor_uses_checkpoint_without_reset_or_mutation(tmp_path):
    checkpoint=tmp_path/'model_2841.pt'
    saved=dict(iter=2841,model_state_dict={'actor.6.weight':torch.ones(6,64),
                                        'std':torch.tensor([.015,.012,.011,.012,.01,.014])},
               optimizer_state_dict={'state':{0:{'step':torch.tensor(37060.),'exp_avg':torch.ones(6,64)}},
                                     'param_groups':[{'lr':.00003375}]})
    torch.save(saved,checkpoint);digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    argv,final,loaded=continuation_command(checkpoint,{'rewards':{'wheel_action_saturation':-2.}},
        'wheel_exploration',100,{'height_expansion_start_steps':117024},tmp_path/'STOP',minimum_wheel_std=.03)
    assert final==2940 and optimizer_step(loaded)==37060
    assert argv[argv.index('--min_wheel_exploration_std')+1]=='0.03'
    assert '--reset_optimizer' not in argv and '--reset_exploration_std' not in argv
    assert '--wheel_exploration_std' not in argv
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digest
    torch.testing.assert_close(loaded['model_state_dict']['std'],saved['model_state_dict']['std'],rtol=0,atol=0)
    assert 'env.commands.height_expansion_start_steps=117024' in argv


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),0.,.009,.051])
def test_invalid_wheel_floor_cannot_launch_a_child(tmp_path,bad):
    with pytest.raises(ValueError,match='wheel exploration floor'):
        continuation_command(tmp_path/'nonexistent.pt',{'rewards':{}},'invalid',100,{},tmp_path/'STOP',
                             minimum_wheel_std=bad)
