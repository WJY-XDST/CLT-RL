"""Configuration def/use and active-mode checks without booting Isaac Sim."""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from env_source import ENV_DIRECTORY, environment_tree
from wheel_legged_gym_isaaclab.leg_control import (
    DIRECT_JOINT_POSITION_MODE, LEG_CONTROL_MODES, validate_joint_position_map,
)
from wheel_legged_gym_isaaclab.config_overrides import RETIRED_CONFIG_FIELDS, RETIRED_SAMPLING_FIELDS
from wheel_legged_gym_isaaclab.tracking_goal_feedback import validate_tracking_goal_reward_config

ROOT = Path(__file__).resolve().parents[2]
CFG = ENV_DIRECTORY / 'wheel_legged_vmc_flat_env_cfg.py'


def declarations():
    result = {}
    for cls in (n for n in ast.parse(CFG.read_text()).body if isinstance(n, ast.ClassDef)):
        fields = {}
        for n in cls.body:
            targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, ast.AnnAssign) else []
            for target in targets:
                if isinstance(target, ast.Name):
                    fields[target.id] = n.value
        result[cls.name] = fields
    return result


def test_retired_declarations_cannot_reappear():
    groups = dict(obs_scales='ObsScalesCfg', commands='CommandsCfg', rewards='RewardsCfg')
    fields = declarations()
    for key in RETIRED_CONFIG_FIELDS:
        group, field = key.split('.')
        assert field not in fields[groups[group]]
    # A reward named dof_acc is active in VMC; only its unused obs scale retired.
    assert 'dof_acc' in fields['RewardsCfg']
    assert not RETIRED_SAMPLING_FIELDS.intersection(fields['CommandsCfg'])


def test_remaining_group_fields_have_code_references_not_just_documentation():
    files = [p for base in ('wheel_legged_isaaclab', 'mine_model/scripts', 'sim2real/wheelleg_deploy')
             for p in (ROOT / base).rglob('*.py')
             if not any(s in p.parts for s in ('tests', '__pycache__', 'learning', 'checkpoints'))
             and not p.name.startswith('test_')]
    trees = [ast.parse(p.read_text()) for p in files]
    names = {n.attr for t in trees for n in ast.walk(t)
             if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)}
    names |= {n.value for t in trees for n in ast.walk(t)
              if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    for group, fields in declarations().items():
        if group != 'WheelLeggedVMCFlatEnvCfg':
            assert not set(fields) - names, (group, set(fields) - names)


def test_sdk_owned_fields_are_retained_and_read_by_lab():
    source = (ROOT / 'IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py').read_text()
    fields = declarations()['WheelLeggedVMCFlatEnvCfg']
    for name in ('episode_length_s', 'state_space'):
        assert name in fields
        assert name in source


def validation_fixture():
    def literals(cls):
        result = {}
        for name, value in declarations()[cls].items():
            try:
                result[name] = ast.literal_eval(value)
            except (ValueError, TypeError):
                pass
        return result
    commands = literals('CommandsCfg')
    commands.update(grouped_training=True, fixed_standing_groups=True, heading_command=False,
                    ranges_ang_vel_yaw=(-.3, .3), ranges_height=(.28, .32),
                    grouped_low_speed_fraction=1./3.)
    rewards = literals('RewardsCfg')
    rewards.update(base_height_target=.30, leg_length_target_height_offset=-.03534645531,
                   leg_length_target_cap=.32)
    cfg = literals('WheelLeggedVMCFlatEnvCfg')
    cfg.update(leg_model='mine_five_bar', leg_control_mode=DIRECT_JOINT_POSITION_MODE,
               leg_joint_stiffness=300., leg_joint_damping=3.,
               leg_joint_position_offset=(-.0707452938, -.0336936327, .0707452938, .0336936327),
               leg_joint_position_scale=(.30,)*4, action_scale_l0=.09, l0_offset=.27,
               l0_ref_min=.18, l0_ref_max=.36, forward_support_min_leg_length=.30,
               height_feedback_gain=1., height_feedback_max_adjustment=.03, height_reference_blend=.75,
               wheel_radius=.06, wheel_track_width=.4199, action_scale_vel=15.,
               commands=SimpleNamespace(**commands), rewards=SimpleNamespace(**rewards))
    method = next(n for n in ast.walk(environment_tree())
                  if isinstance(n, ast.FunctionDef) and n.name == '_validate_configuration')
    scope = dict(math=math, DIRECT_JOINT_POSITION_MODE=DIRECT_JOINT_POSITION_MODE,
                 LEG_CONTROL_MODES=LEG_CONTROL_MODES, validate_joint_position_map=validate_joint_position_map,
                 validate_tracking_goal_reward_config=validate_tracking_goal_reward_config)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual_validator>', 'exec'), scope)
    return SimpleNamespace(cfg=SimpleNamespace(**cfg), _action_dim=6), scope['_validate_configuration']


def test_direct_pd_does_not_validate_unused_vmc_target_maps():
    env, validate = validation_fixture()
    validate(env)
    env.cfg.height_reference_blend = 9.
    env.cfg.l0_offset = 1.
    validate(env)
    env.cfg.leg_control_mode = 'explicit_vmc'
    with pytest.raises(ValueError, match='Leg-length action map'):
        validate(env)


def test_only_grouped_training_and_fixed_speed_replay_are_supported():
    env, validate = validation_fixture()
    validate(env)
    env.cfg.commands.grouped_training = False
    with pytest.raises(ValueError, match='non-grouped sampler is retired'):
        validate(env)
    env.cfg.commands.ranges_lin_vel_x = (0., 0.)
    validate(env)


def test_active_joint_and_vmc_maps_still_fail_closed():
    env, validate = validation_fixture()
    invalid = deepcopy(env)
    invalid.cfg.leg_joint_position_scale = (0.,)*4
    with pytest.raises(ValueError):
        validate(invalid)
    env.cfg.leg_control_mode = 'explicit_vmc'
    validate(env)
    env.cfg.rewards.leg_length_target_cap = 1.
    with pytest.raises(ValueError, match='leg_length_target_cap'):
        validate(env)


def test_training_has_no_partial_head_cli_or_freeze_hook():
    source = (ROOT / 'wheel_legged_isaaclab/scripts/rsl_rl/train.py').read_text()
    assert 'enable_wheel_head_finetuning' not in source
    assert 'wheel_head_only_finetune' not in source
    assert 'Full-policy training unexpectedly has frozen parameters' in source


def test_retired_tools_are_absent_and_current_diagnostic_entries_exist():
    for relative in (
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/precision_finetune.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/checkpoint_initialization.py',
        'wheel_legged_isaaclab/scripts/rsl_rl/optimize_training.py',
        'wheel_legged_isaaclab/scripts/rsl_rl/optimize_yaw.py',
        'wheel_legged_isaaclab/scripts/live_viewer.py',
    ):
        assert not (ROOT / relative).exists()
    for relative in ('mine_model/scripts/joint_pd_optimization.py',
                     'mine_model/scripts/optimize_vmc_feedback.py',
                     'wheel_legged_isaaclab/scripts/check_direct_joint_pd.py',
                     'wheel_legged_isaaclab/scripts/check_mcu_vmc_support.py'):
        assert (ROOT / relative).is_file()
