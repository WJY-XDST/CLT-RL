"""Exercise the production replay-configuration block without starting Kit."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wheel_legged_gym_isaaclab.config_overrides import apply_config_overrides
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.physics_contract import (
    checkpoint_force_iteration_setting, checkpoint_robot_asset_setting,
)


def configured_cases(saved, overrides=None):
    """Execute the actual initializer statements, not a duplicated case builder."""
    path = Path(__file__).resolve().parents[1]/'scripts/evaluate_mine_policy.py'
    main = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'main')
    def assigns(node, name):
        return isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets)
    start = next(i for i,n in enumerate(main.body) if assigns(n, 'saved_rewards'))
    end = next(i for i,n in enumerate(main.body) if assigns(n, 'cases'))
    body = ast.Module(body=main.body[start:end+1], type_ignores=[])
    def factory():
        return SimpleNamespace(
            commands=SimpleNamespace(ranges_height=(.225, .375)),
            rewards=SimpleNamespace(base_height_target=.3),
            robot=SimpleNamespace(spawn=SimpleNamespace(usd_path='/default.usd'),
                                  actuators={'wheels': SimpleNamespace(damping=.5)}),
            wheel_control_mode='implicit_velocity', wheel_damping=.5,
            enable_external_forces_every_iteration=True)
    namespace = dict(saved_config=copy.deepcopy(saved),
        args=SimpleNamespace(overrides_json=None if overrides is None else
            SimpleNamespace(read_text=lambda: json.dumps(overrides))),
        WheelLeggedVMCFlatEnvCfg=factory, json=json,
        apply_config_overrides=apply_config_overrides,
        checkpoint_robot_asset_setting=checkpoint_robot_asset_setting,
        checkpoint_force_iteration_setting=checkpoint_force_iteration_setting)
    exec(compile(body, str(path), 'exec'), namespace)
    return namespace['cfg'], namespace['cases']


@pytest.fixture
def historical():
    return dict(commands=dict(ranges_height=[.28, .32]),
                rewards=dict(base_height_target=.3),
                robot=dict(spawn=dict(usd_path='/historical.usd')),
                enable_external_forces_every_iteration=True)


def test_unmodified_historical_replay_keeps_its_original_cases(historical):
    cfg,cases = configured_cases(historical)
    assert cfg.robot.spawn.usd_path == '/historical.usd'
    assert cases[-2][-1] == .28 and cases[-1][-1] == .32
    assert len(cases) == 9 and cases[0] == ('stand', 0., 0., .3)


def test_explicit_full_height_override_changes_actual_cases_not_only_cfg(historical):
    cfg,cases = configured_cases(historical, {
        'commands.ranges_height': list(SAFE_BASE_HEIGHT_RANGE_M),
        'robot.spawn.usd_path': '/calibrated.usda'})
    assert cfg.robot.spawn.usd_path == '/calibrated.usda'
    assert tuple(c[-1] for c in cases[-2:]) == SAFE_BASE_HEIGHT_RANGE_M
    assert historical['commands']['ranges_height'] == [.28, .32]
    assert historical['robot']['spawn']['usd_path'] == '/historical.usd'


def test_nominal_height_override_is_reflected_in_nominal_cases(historical):
    _,cases = configured_cases(historical, {'rewards.base_height_target': .31})
    assert all(c[-1] == .31 for c in cases[:7])


def test_mass_only_override_does_not_expand_historical_height(historical):
    _,cases = configured_cases(historical, {'robot.spawn.usd_path': '/calibrated.usda'})
    assert tuple(c[-1] for c in cases[-2:]) == (.28, .32)


def test_override_preserves_canonical_motion_targets(historical):
    _,cases = configured_cases(historical, {'commands.ranges_height': list(SAFE_BASE_HEIGHT_RANGE_M)})
    assert [(c[0], c[1], c[2]) for c in cases] == [
        ('stand',0.,0.),('forward_0p3',.3,0.),('forward_0p6',.6,0.),
        ('reverse_0p3',-.3,0.),('reverse_0p6',-.6,0.),
        ('spin_left',0.,.3),('spin_right',0.,-.3),
        ('height_low',0.,0.),('height_high',0.,0.)]


def test_unknown_override_is_not_silently_accepted(historical):
    with pytest.raises(ValueError, match='Unknown override'):
        configured_cases(historical, {'commands.unknown': 1})


def test_wheel_damping_override_keeps_actuator_in_sync(historical):
    cfg,_ = configured_cases(historical, {'wheel_damping': .4})
    assert cfg.wheel_damping == cfg.robot.actuators['wheels'].damping == .4
