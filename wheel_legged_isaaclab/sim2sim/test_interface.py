"""Shared parity fixtures for the imported closed-chain vehicle only."""
import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

from core import PROJECT, FIVE_BAR, LEG_CONTROL, ROBOT, load_config
import torch


def vehicle_fixture():
    checkpoint = os.environ.get('MINE_SIM2SIM_CHECKPOINT')
    if not checkpoint:
        runs = sorted((PROJECT.parent / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat').glob('*/model_*.pt'), key=lambda p: p.stat().st_mtime_ns)
        if not runs:
            raise FileNotFoundError('Set MINE_SIM2SIM_CHECKPOINT to a supported direct-joint or VMC checkpoint')
        checkpoint = runs[-1]
    checkpoint = Path(checkpoint)
    folder = checkpoint.parent / 'params' if (checkpoint.parent / 'params').is_dir() else checkpoint.parent
    cfg = load_config(folder / 'env.yaml')
    mjcf = Path(os.environ.get('MINE_SIM2SIM_MJCF', PROJECT.parent / 'mine_model/exports/mujoco_protected_stops_v2_20261004/scene_protected_contact.xml'))
    return checkpoint, cfg, mjcf


def namespace(value):
    return SimpleNamespace(**{key: namespace(item) for key, item in value.items()}) if isinstance(value, dict) else value


def reference_class():
    # Extract simulator-independent methods from the real source, avoiding Kit startup.
    source = PROJECT / "wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py"
    original = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    methods = {"_get_wheel_vel_forward_positive", "_apply_action", "_update_forward_kinematics",
               "_vmc", "_sanitize_observation", "_get_observations",
               "_get_heading_frame_horizontal_velocity", "_modify_leg_references",
               "_get_leg_force_bias", "_get_gravity_feedforward", "_get_leg_observation",
               "_apply_direct_joint_position_action", "_wheel_effort_from_reference"}
    components = (source, *sorted(source.parent.glob('env_*.py')))
    implementations = [node for path in components
                       for cls in ast.parse(path.read_text()).body if isinstance(cls, ast.ClassDef)
                       for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    names = [node.name for node in implementations]
    if set(names) != methods or len(names) != len(methods):
        raise AssertionError('Missing or duplicate production parity methods')
    original.body = implementations
    original.bases, original.decorator_list = [], []

    def quat_apply(q, v):
        xyz = q[:, 1:]
        cross = 2 * torch.cross(xyz, v, dim=-1)
        return v + q[:, :1] * cross + torch.cross(xyz, cross, dim=-1)

    scope = {"torch": torch, "quat_apply": quat_apply,
             "five_bar_state": FIVE_BAR.five_bar_state, "five_bar_torques": FIVE_BAR.five_bar_torques,
             "joint_position_targets": LEG_CONTROL.joint_position_targets,
             "LEG_ACTION_COLUMNS": LEG_CONTROL.LEG_ACTION_COLUMNS,
             "DIRECT_JOINT_POSITION_MODE": LEG_CONTROL.DIRECT_JOINT_POSITION_MODE,
             "gravity_feedforward": ROBOT._hardware.gravity_feedforward}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])), str(source), "exec"), scope)
    return scope[original.name]
