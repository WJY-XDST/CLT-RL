"""Imported CAD geometry and versioned firmware execution parameters.

The CAD main loop is smaller than the wheel-end equivalent five-bar. Runtime
kinematics uses the larger equivalent links, so the scale is applied once only.
This module intentionally has no simulator dependency or example-robot defaults.
"""
import copy
import json
import math
from pathlib import Path
import importlib.util

# sim2sim and the offline deployment exporter also load this file directly,
# without a Python package context or an Isaac installation.
_hardware_spec = importlib.util.spec_from_file_location(
    "mine_mcu_hardware_contract", Path(__file__).with_name("mcu_hardware.py"))
_hardware = importlib.util.module_from_spec(_hardware_spec)
_hardware_spec.loader.exec_module(_hardware)
HARDWARE_PROFILE = _hardware.HARDWARE_PROFILE
LEG_EFFORT_LIMIT = _hardware.LEG_EFFORT_LIMIT
WHEEL_EFFORT_LIMIT = _hardware.WHEEL_EFFORT_LIMIT


ASSETS = Path(__file__).resolve().parents[1] / "assets/robots/mine"
MANIFEST = json.loads((ASSETS / "manifest.json").read_text())
MODEL = json.loads((ASSETS / MANIFEST["metadata"]).read_text())


def mapped_five_bar_geometry(geometry):
    """Equivalent wheel-end links in metres; offsets stay at CAD motor zero."""
    result = copy.deepcopy(geometry)
    scale = float(result["scale"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Five-bar mapping scale must be positive and finite")
    for name in ("upper", "lower"):
        if len(result[name]) != 2:
            raise ValueError(f"Five-bar {name} must contain two link lengths")
        result[name] = [float(value) * scale for value in result[name]]
        if any(not math.isfinite(value) or value <= 0 for value in result[name]):
            raise ValueError("Mapped link lengths must be positive and finite")
    if len(result["offsets"]) != 2 or not all(math.isfinite(v) for v in result["offsets"]):
        raise ValueError("Five-bar requires two finite CAD zero offsets")
    if result["branch"] != 1:
        raise ValueError("This vehicle's CAD five-bar assembly requires branch 1")
    result["scale"] = 1.0
    return result


MAPPED_FIVE_BAR = mapped_five_bar_geometry(MODEL["five_bar"])


def validate_robot_contract(cfg):
    """Fail closed instead of interpreting another vehicle as this one."""
    if cfg.get("leg_model") != "mine_five_bar":
        raise ValueError("Only the imported mapped closed-five-bar vehicle is supported")
    for key in ("leg_joint_names", "wheel_joint_names"):
        if list(cfg[key]) != MODEL[key]:
            raise ValueError(f"{key} differs from the imported vehicle metadata")
    for key in ("wheel_radius", "wheel_track_width"):
        if not math.isclose(float(cfg[key]), float(MODEL[key]), rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"{key} differs from the imported vehicle metadata")
    profile = cfg.get("hardware_profile")
    if profile != HARDWARE_PROFILE:
        raise ValueError(f"Checkpoint must use this vehicle's MCU hardware profile: {profile}")
    # 这里只有当前新车的下位机标定，不再回退到 CAD 导入时的旧力矩默认值。
    limits = {"leg_effort_limit": LEG_EFFORT_LIMIT, "wheel_effort_limit": WHEEL_EFFORT_LIMIT}
    for key in ("leg_effort_limit", "wheel_effort_limit"):
        if not math.isclose(float(cfg[key]), float(limits[key]), rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"{key} differs from the selected hardware profile")
    mapped = mapped_five_bar_geometry(cfg["five_bar_geometry"])
    for key in ("upper", "lower", "offsets"):
        if any(not math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-9)
               for a, b in zip(mapped[key], MAPPED_FIVE_BAR[key])):
            raise ValueError("Checkpoint five-bar geometry differs from the imported vehicle")
    return mapped


def validate_checkpoint_robot(checkpoint):
    """Reject unmatched weights even when actor dimensions happen to match."""
    import yaml
    class Loader(yaml.SafeLoader):
        pass
    Loader.add_constructor('tag:yaml.org,2002:python/tuple', Loader.construct_sequence)
    checkpoint = Path(checkpoint).expanduser().resolve()
    folder = checkpoint.parent / 'params' if (checkpoint.parent / 'params').is_dir() else checkpoint.parent
    path = folder / 'env.yaml'
    if not path.is_file():
        raise ValueError(f'Checkpoint requires its paired vehicle configuration: {path}')
    cfg = yaml.load(path.read_text(), Loader=Loader)
    from wheel_legged_gym_isaaclab.config_overrides import strip_retired_config_fields
    cfg = strip_retired_config_fields(cfg)
    validate_robot_contract(cfg)
    return cfg
