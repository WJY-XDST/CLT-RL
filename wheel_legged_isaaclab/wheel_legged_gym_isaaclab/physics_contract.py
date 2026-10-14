"""Version solver settings and the paired rigid-body asset without the SDK."""

from pathlib import Path

EXTERNAL_FORCE_FIELD = 'enable_external_forces_every_iteration'
ROBOT_ASSET_FIELD = 'robot.spawn.usd_path'


def checkpoint_robot_asset_setting(saved_config, overrides=None):
    """Replay historical CAD masses unless a different asset is explicitly requested."""
    value = (overrides or {}).get(ROBOT_ASSET_FIELD,
                                saved_config['robot']['spawn']['usd_path'])
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Checkpoint requires a nonempty robot.spawn.usd_path')
    return value


def require_matching_robot_asset_setting(saved_config, current, allow_change=False):
    """Measured mass changes belong to a distinct physics experiment."""
    previous = checkpoint_robot_asset_setting(saved_config)
    if not isinstance(current, str) or not current.strip():
        raise ValueError('Current robot.spawn.usd_path must be nonempty')
    if (Path(previous).expanduser().resolve() != Path(current).expanduser().resolve()
            and not allow_change):
        raise ValueError('Robot asset physics contract changed; explicitly allow the '
                         'physics contract change for a new calibrated-mass experiment')


def checkpoint_force_iteration_setting(saved_config, overrides=None):
    """Old weights used PhysX's False default; never silently replay them as True."""
    value = (overrides or {}).get(EXTERNAL_FORCE_FIELD,
                                saved_config.get(EXTERNAL_FORCE_FIELD, False))
    if not isinstance(value, bool):
        raise ValueError(f'{EXTERNAL_FORCE_FIELD} must be a boolean')
    return value


def require_matching_force_iteration_setting(saved_config, current, allow_change=False):
    previous = checkpoint_force_iteration_setting(saved_config)
    current = checkpoint_force_iteration_setting({EXTERNAL_FORCE_FIELD: current})
    if previous != current and not allow_change:
        raise ValueError('External-effort physics contract changed; start fresh or explicitly '
                         'allow the physics contract change for a new experiment')


def apply_force_iteration_setting(stage, physics_prim_path, enabled):
    """Author before physics starts, and verify the actual USD scene setting.

    Isaac Lab 2.3.1 does not expose this PhysX flag in SimulationCfg. TGS normally
    applies external efforts once per frame; the flag distributes their effect
    over internal solver iterations. It does not multiply commanded motor torque.
    """
    from pxr import PhysxSchema

    enabled = checkpoint_force_iteration_setting({EXTERNAL_FORCE_FIELD: enabled})
    # stage 是 LAB SimulationContext 提供的 USD 场景；这里找的是物理场景节点，
    # 不是机器人。GetPrimAtPath 只查询节点，所以必须检查是否真实存在。
    scene = stage.GetPrimAtPath(physics_prim_path)
    if not scene.IsValid():
        raise ValueError(f'Physics scene not found: {physics_prim_path}')
    # PhysxSceneAPI 是 USD 的 PhysX 扩展 schema，Apply 将所需属性附到该节点。
    # LAB 2.3.1 的 SimulationCfg 没暴露这个开关，故在物理启动前通过 USD 设置。
    api = PhysxSchema.PhysxSceneAPI.Apply(scene)
    api.CreateEnableExternalForcesEveryIterationAttr(enabled)
    # Create...Attr 写值，Get...Attr().Get() 读回验证；不会额外执行一次仿真。
    # True 改变 TGS 内部外力作用方式，不乘大力矩、不修改重力、不提高控制频率。
    actual = api.GetEnableExternalForcesEveryIterationAttr().Get()
    if actual != enabled:
        raise RuntimeError('Failed to author external-effort physics contract')
    return actual
