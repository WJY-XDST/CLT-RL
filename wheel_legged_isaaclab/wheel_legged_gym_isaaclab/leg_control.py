"""Explicit leg-execution contracts, independent of the simulator SDK."""
import math


DIRECT_JOINT_POSITION_MODE = 'implicit_joint_position'
LEG_CONTROL_MODES = ('explicit_vmc', DIRECT_JOINT_POSITION_MODE)
LEG_ACTION_COLUMNS = (0, 1, 3, 4)  # LB, LL, RB, RL: two active hips per leg


def validate_joint_position_map(offset, scale, margin):
    if len(offset) != 4 or len(scale) != 4:
        raise ValueError('Direct joint position actions require four offsets and scales in LB/LL/RB/RL order')
    if not all(math.isfinite(v) for v in offset):
        raise ValueError('Joint position offsets must be finite')
    if not all(math.isfinite(v) and v > 0 for v in scale):
        raise ValueError('Joint position scales must be finite and positive')
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('Joint limit margin must be finite and nonnegative')


def joint_position_targets(actions, offset, scale, limits, margin=0.0, *, validate=True):
    """Four normalized policy coordinates -> raw CAD motor angles [rad].

    No virtual-leg reference, IK, height feedback or explicit joint PD
    participates in this position map. Gravity feedforward is a separate
    effort channel, never an angle offset. Limits are the articulation's
    hard position limits, intersected with the configured action envelope.
    """
    # 纯张量函数，不调用 LAB/PhysX：调用方传入 Articulation.data.joint_pos_limits。
    # actions 为 [N, 4]，limits 为 [N, 4, 2]；四列顺序是 LB、LL、RB、RL。
    import torch
    if actions.ndim != 2 or actions.shape[1] != 4:
        raise ValueError('Expected a batch of four active-joint actions')
    if validate:
        validate_joint_position_map(offset, scale, margin)
    if limits.shape != (*actions.shape, 2):
        raise ValueError('Joint limits must have shape [num_envs, 4, 2]')
    if validate and (not torch.isfinite(limits).all() or not ((limits[..., 1] - limits[..., 0]) > 2 * margin).all()):
        raise ValueError('Joint limits must be finite and wider than twice the safety margin')
    center, amplitude = actions.new_tensor(offset), actions.new_tensor(scale)
    # 允许区间是「nominal±动作尺度」与「USD 关节角限位减 margin」的交集。
    # 全部用 rad；它不是 CAD 限位块接触包络，也不是虚拟腿长的硬限制。
    lower = torch.maximum(limits[..., 0] + margin, center - amplitude)
    upper = torch.minimum(limits[..., 1] - margin, center + amplitude)
    if validate and (not (lower < upper).all() or not ((center >= lower) & (center <= upper)).all()):
        raise ValueError('Joint action envelope and safe articulation limits must contain the reference pose')
    normalized = torch.nan_to_num(actions, nan=0., posinf=1., neginf=-1.).clamp(-1., 1.)
    target = torch.maximum(torch.minimum(center + normalized * amplitude, upper), lower)
    # 第二项将最终角度反算为有效归一化动作，供 last_actions 和奖励使用。
    # 这里只生成参考；LAB set_joint_position_target() 和隐式 PD 在环境中执行。
    return target, (target - center) / amplitude


def require_matching_policy_contract(saved_cfg, target_cfg):
    """Equal tensor dimensions do not make virtual-leg and joint-angle policies compatible."""
    old = saved_cfg.get('leg_control_mode', 'explicit_vmc')
    new = target_cfg.get('leg_control_mode', 'explicit_vmc')
    require_leg_control_mode(old)
    require_leg_control_mode(new)
    if old != new:
        raise ValueError('Checkpoint action/observation contract differs; start a fresh policy for the new leg control mode')
    if new == DIRECT_JOINT_POSITION_MODE:
        for name in ('leg_joint_position_offset', 'leg_joint_position_scale', 'leg_joint_target_limit_margin',
                     'leg_joint_gravity_feedforward', 'observation_space', 'action_space'):
            a, b = saved_cfg.get(name), target_cfg.get(name)
            if name in ('leg_joint_position_offset', 'leg_joint_position_scale'):
                a, b = list(a) if a is not None else a, list(b) if b is not None else b
            if a != b:
                raise ValueError(f'Checkpoint direct-joint contract differs at {name}')


def require_leg_control_mode(actual, required=None):
    if actual == 'implicit_joint_reference':
        raise ValueError('The legacy IK leg control mode has been removed; its checkpoints cannot be replayed or resumed. Start a fresh policy with a supported mode')
    if actual not in LEG_CONTROL_MODES:
        raise ValueError(f'Unknown leg control mode: {actual}')
    if required is not None and actual != required:
        raise ValueError(f'Required leg control mode {required}, but resolved configuration is {actual}')


def leg_drive_gains(mode, stiffness, damping):
    """VMC uses external efforts, never an additional joint-position PD."""
    require_leg_control_mode(mode)
    if mode == 'explicit_vmc':
        # 环境已通过 Jᵀ[F,T] 计算反馈力矩，LAB 关节驱动不能再叠加一套位置 PD。
        return 0.0, 0.0
    if not math.isfinite(stiffness) or stiffness <= 0 or not math.isfinite(damping) or damping < 0:
        raise ValueError('Direct joint position control requires positive stiffness and nonnegative damping')
    return float(stiffness), float(damping)


def resolve_training_leg_mode(overrides, requested=None, saved_mode=None):
    """Pin each experiment to one execution chain; never reuse incompatible assessments."""
    if requested is not None:
        require_leg_control_mode(requested)
    if saved_mode is not None:
        require_leg_control_mode(saved_mode)
        if requested is not None and requested != saved_mode:
            raise ValueError('Changing the leg execution chain requires a new --output directory')
    result = dict(overrides)
    mode = requested or result.get('leg_control_mode', 'explicit_vmc')
    require_leg_control_mode(mode)
    result['leg_control_mode'] = mode
    return result
