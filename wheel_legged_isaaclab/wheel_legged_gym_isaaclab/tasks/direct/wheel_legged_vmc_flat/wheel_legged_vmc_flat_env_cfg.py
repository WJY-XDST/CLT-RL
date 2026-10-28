# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause

"""用户新车的唯一平地环境配置；直接从此文件阅读和修改训练参数。"""

from __future__ import annotations

from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, MODEL, MAPPED_FIVE_BAR, GRAVITY_FORCE
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.joint_reset import (
    DEFAULT_JOINT_OFFSETS, NOMINAL_JOINT_RESET, closed_chain_reset_pose,
)
from wheel_legged_gym_isaaclab.mcu_hardware import (
    HARDWARE_PROFILE, LEG_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT, WHEEL_RADIUS,
    GRAVITY_FORCE_LIMIT, GRAVITY_ANGLE_BIAS,
)

_DIRECT_NOMINAL_RESET = closed_chain_reset_pose(DEFAULT_JOINT_OFFSETS, MAPPED_FIVE_BAR)


@configclass
class ObsScalesCfg:
    """Observation scales (mirrors `normalization.obs_scales` in the original project)."""

    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05
    height_measurements = 5.0
    l0 = 5.0
    l0_dot = 0.25


@configclass
class ObsClipCfg:
    """Physical bounds applied before observation scaling.

    These limits are deliberately wider than the normal operating envelope.
    They are numerical guards, not task constraints.
    """

    lin_vel = 5.0  # [m/s]
    ang_vel = 20.0  # [rad/s]
    projected_gravity = 1.0
    theta = 3.1415926536  # [rad]
    theta_dot = 50.0  # [rad/s]
    leg_length = 0.5  # [m]
    leg_length_dot = 5.0  # [m/s]
    wheel_vel = 100.0  # [rad/s]
    action = 1.0


@configclass
class CommandsCfg:
    """Command configuration (mirrors `commands` in the original project)."""

    resampling_time = 5.0  # [s]
    linear_acceleration_limit = 0.8  # [m/s^2], ramps abrupt speed changes
    # 训练转向指令默认不用航向保持；固定航向回放可开启。
    heading_command = False
    heading_kp = 1.5  # [1/s], heading error -> target yaw rate
    heading_rate_limit = 0.5  # [rad/s], maximum corrective yaw rate
    ranges_lin_vel_x = (0.2, 0.8)  # [m/s], forward training commands
    ranges_reverse_lin_vel_x = (-0.8, -0.2)  # [m/s], reverse training commands
    speed_boundary_fraction = 0.3  # main-range samples explicitly at +/- maximum speed
    height_boundary_fraction = 0.4  # half at each height endpoint; remaining samples uniform
    # Bridge the gap between standing and the positive/negative main ranges.
    ranges_transition_lin_vel_x = (-0.2, 0.2)  # [m/s]
    ranges_ang_vel_yaw = (-0.3, 0.3)
    yaw_min_abs_rate = 0.15  # [rad/s], before curriculum scaling
    yaw_boundary_fraction = 0.2
    yaw_start_steps = 48000
    yaw_ramp_steps = 48000
    grouped_training = True
    # 速度阶段固定保留前 25% 环境练习站立；静止阶段全部环境为零速度。
    fixed_standing_groups = True
    grouped_low_speed_fraction = 1.0 / 3.0  # within each moving group, remaining samples cover main range
    # CAD 限位块腿长 0.19～0.34 m，含至少 10 mm 余量；机体直立时换算。
    ranges_height: tuple[float, float] = SAFE_BASE_HEIGHT_RANGE_M
    # 显式解耦：完整高度可从首轮开启，不必跳过站立/速度/转向课程。
    full_height_range_from_start = True
    # 静止姿态和全高度复验通过后，由每 100 次更新的审查逐档放开。
    # 仅缩放训练的线速度/转向命令，不缩放高度，不改变固定命令验收。
    training_motion_scale: float = 0.0
    height_expansion_from: tuple[float, float] | None = None
    height_expansion_start_steps = 0
    height_expansion_ramp_steps = 48000
    standing_only_steps = 48000
    motion_ramp_steps = 48000
    # Explicit zero-speed population retained after motion is introduced.
    standing_env_fraction = 0.25
    # 前进与后退同时引入；后退幅度按自己的课程时钟增长。
    reverse_ramp_start_steps = 48000
    reverse_ramp_steps = 48000


@configclass
class RewardsCfg:
    """Reward scales for flat-terrain velocity, height, posture, and standing control."""

    profile = 'joint_pd_balance_v1'
    tracking_lin_vel = 2.0
    tracking_lin_vel_enhance = 0.0
    # 窄核项目前关闭；速度误差以线速度跟踪和平方误差为主。
    tracking_lin_vel_precise = 0.0
    lin_vel_error_sq = -2.0
    lin_vel_error_lateral_multiplier = 1.0
    standing_velocity = 0.0
    base_height_error_sq = 0.0
    joint_target_leg_length_error_sq = 0.0  # opt-in FK decision cost; never changes joint targets
    tracking_ang_vel = 1.0
    yaw_rate_error_sq = -1.0
    spin_center_velocity = 0.0
    base_height = 2.0
    nominal_state = 0.0
    standing_leg_angle = 0.0
    standing_wheel_tracking = 0.0
    lin_vel_z = 0.0
    ang_vel_xy = 0.0
    orientation = -20.0
    dof_vel = 0.0
    dof_acc = 0.0
    torques = 0.0
    action_rate = -0.02
    action_smooth = 0.0
    # VMC 长度动作软饱和；当前直接关节模式不使用这三项。
    leg_length_action_saturation = 0.0
    leg_length_action_difference = 0.0
    leg_length_target_underreach = 0.0
    joint_position_action_saturation = -0.5
    # Opt-in diagnosed feedback for discarded wheel outputs; legacy reward is unchanged at zero.
    wheel_action_saturation = 0.0
    # Opt-in pointwise feedback for the worst error relative to the existing gate.
    tracking_goal_violation = 0.0
    tracking_goal_penalty_clip = 40.0
    # Restore feedback on simultaneous failures hidden by the worst-error term.
    tracking_goal_component_violation = 0.0
    tracking_goal_component_penalty_clip = 200.0
    tracking_task_quality_gate_floor = 1.0  # 1 disables; an opt-in floor never removes motion learning
    collision = -1.0
    dof_pos_limits = -1.0
    # One-time cost for unsafe termination, not a per-second reward scale.
    termination = -20.0

    clip_single_reward = 5.0
    nominal_state_penalty_clip = 3.0  # keep the 6-9 degree asymmetry penalty below its cap
    orientation_penalty_clip = 30.0
    base_height_penalty_clip = 30.0
    standing_leg_angle_penalty_clip = 3.0
    lin_vel_penalty_clip = 8.0
    yaw_rate_penalty_clip = 4.0
    standing_wheel_penalty_clip = 1.0
    action_rate_penalty_clip = 1.0
    action_smooth_penalty_clip = 3.0
    wheel_action_saturation_penalty_clip = 40.0
    action_smooth_wheel_multiplier = 5.0
    tracking_sigma = 0.25
    tracking_sigma_ang = 0.25  # separate yaw width so turning can be trained without changing linear tracking
    tracking_sigma_enhance = 0.025
    tracking_sigma_precise = 0.01
    # 小角度时投影重力 y 主要反映 roll，乘数单独调整通用姿态代价。
    orientation_roll_multiplier = 2.0
    base_height_target: float = 0.30
    base_height_sigma = 0.0004  # [m^2], experimental height-tracking width
    # 0 保持历史 reward；诊断后的续训显式使用 .05，容差区小于最终 10% 验收界限。
    base_height_relative_deadband = 0.0
    leg_length_action_soft_limit = 0.85
    joint_position_action_soft_limit = 0.85
    leg_length_target_height_offset: float = MODEL['hip_z'] - MODEL['wheel_radius']
    leg_length_target_cap: float = 0.32


@configclass
class WheelLeggedVMCFlatEnvCfg(DirectRLEnvCfg):
    """唯一环境配置：新车数据、执行器、观测、奖励、命令与物理设置。

    默认直接关节目标角加隐式 PD；explicit_vmc 模式保留在同一环境中。
    不继承旧车配置，不在 mine_env_cfg 或 reward profile 中二次覆写。
    """

    # -- env --
    episode_length_s = 20.0
    decimation = 4
    action_space = 6
    observation_space = 27
    state_space = 0
    leg_model: str = 'mine_five_bar'
    leg_joint_names: list[str] = MODEL['leg_joint_names']
    wheel_joint_names: list[str] = MODEL['wheel_joint_names']
    leg_body_names: list[str] = [s + n + '_link' for s in ('L', 'R') for n in ('B', 'S', 'L', 'L1', 'L2', 'L3')]
    penalised_body_pattern: str = '(base_link|[LR](B|S|L[123]?)_link)'
    five_bar_geometry: dict = MAPPED_FIVE_BAR
    leg_effort_limit: float = LEG_EFFORT_LIMIT
    wheel_effort_limit: float = WHEEL_EFFORT_LIMIT
    # Root linear [m/s] and angular [rad/s] velocity perturbations at reset.
    reset_velocity_initial = 0.05
    reset_velocity_final = 0.5
    # A nearly inverted body retains the original delayed failure condition.
    fail_to_terminal_time_s = 1.0
    # Base-link contact is unsafe even when the robot is not fully inverted.
    base_contact_terminal_time_s = 0.15  # [s]
    base_contact_force_threshold = 10.0  # [N]
    min_root_height: float = 0.22
    low_height_terminal_time_s = 0.15
    # Sustained non-wheel leg-link contact is treated as a failure.
    leg_contact_terminal_time_s = 0.2
    leg_contact_force_threshold = 5.0  # [N]
    collision_force_threshold = 1.0  # [N]
    max_body_pitch = 0.5235987756  # [rad], 30 degrees
    body_pitch_terminal_time_s = 0.25
    # 不要求左右腿等长；只对持续过大的机身横滚角终止。
    max_body_roll = 0.3490658504  # [rad], 20 degrees
    body_roll_terminal_time_s = 0.2

    # Keep the interactive Isaac Sim viewport centered on env_0's robot.
    viewer = ViewerCfg(
        eye=(2.0, 2.0, 1.2), lookat=(0.0, 0.0, -0.08),
        origin_type="asset_root", env_index=0, asset_name="robot",
    )

    # -- virtual-coordinate controller --
    action_scale_theta = 0.2  # [rad] per action unit
    theta0_ref_min = -0.2  # [rad], hard VMC target bound
    theta0_ref_max = 0.2  # [rad], hard VMC target bound
    action_scale_l0: float = 0.09
    l0_offset: float = 0.27
    l0_ref_min: float = 0.18
    l0_ref_max: float = 0.36
    leg_length_height_offset: float = MODEL['hip_z'] - MODEL['wheel_radius']
    forward_support_speed_threshold = 0.6  # [m/s]
    forward_support_min_leg_length: float = 0.3
    forward_support_height_margin: float = leg_length_height_offset
    height_feedback_gain: float = 1.0
    height_feedback_max_adjustment: float = 0.03
    height_reference_blend: float = 0.75
    wheel_radius: float = WHEEL_RADIUS
    wheel_track_width: float = MODEL['wheel_track_width']
    action_scale_vel: float = 15.0
    # Reference clamp is separate from scale and the motor's rated speed.
    # Historical checkpoints omitted this field and used a unit clamp.
    wheel_action_limit: float = 1.0
    # 使用当前质量校准的整车重力的一半；新增 1.3 kg 后为 13.64*9.81/2=66.9042 N。
    # 下位机旧版的 12.34 kg、每腿 60.5277 N 仅作为历史参考保留在 mcu_hardware.py。
    feedforward_force: float = GRAVITY_FORCE
    hardware_profile: str | None = HARDWARE_PROFILE
    feedforward_mode = 'mcu_pitch_compensated'
    feedforward_angle_bias = GRAVITY_ANGLE_BIAS
    feedforward_force_limit = GRAVITY_FORCE_LIMIT
    kp_theta: float = 50.0
    kd_theta: float = 3.0
    kp_l0: float = 900.0
    kd_l0: float = 90.0
    wheel_damping: float = 0.5
    wheel_control_mode: str = 'implicit_velocity'
    leg_control_mode: str = 'implicit_joint_position'
    leg_joint_stiffness: float = 300.0
    leg_joint_damping: float = 3.0
    # Direct-joint mode: normalized columns 0/1/3/4 map to LB/LL/RB/RL.
    leg_joint_position_offset: tuple[float, float, float, float] = DEFAULT_JOINT_OFFSETS
    # Zero action and the actual reset pose agree, including passive loop joints.
    # Historical checkpoints explicitly retain cad_zero_v0 during replay.
    joint_reset_profile: str = NOMINAL_JOINT_RESET
    leg_joint_position_scale: tuple[float, float, float, float] = (0.3, 0.3, 0.3, 0.3)
    leg_joint_target_limit_margin = 0.01  # [rad] inside articulation hard stops
    # Direct motor-position mode: optional J^T [gravity force, 0] only.
    leg_joint_gravity_feedforward = True

    # -- robot --
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # 整车实测 12.34 kg；每侧整腿含轮 1.935 kg、不含机身上的髋电机。
            # 两轮各 0.665 kg；碳纤维腿部质量、质心和惯量按 CAD 体积估计。
            # 独立质量层保留闭链、几何与限位；各连杆参数尚非单独实测。
            # 两轮主惯量取右轮 CAD 值，并按左右坐标镜像；尚非实测惯量。
            # 旧权重回放使用其保存的配套资产。
            usd_path=str(ASSETS / MANIFEST.get("training_usd", MANIFEST["usd"].replace(
                'mine_closed_chain.usd', 'mine_closed_chain_training_protected_stops_v2.usd'))),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=128,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., _DIRECT_NOMINAL_RESET['root_height_m']),
            joint_pos=dict(_DIRECT_NOMINAL_RESET['joint_positions']), joint_vel={".*": 0.},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=MODEL["leg_joint_names"],
                stiffness=leg_joint_stiffness, damping=leg_joint_damping, effort_limit_sim=LEG_EFFORT_LIMIT,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=MODEL["wheel_joint_names"],
                stiffness=0., damping=wheel_damping, effort_limit_sim=WHEEL_EFFORT_LIMIT,
            ),
            "passive": ImplicitActuatorCfg(
                joint_names_expr=["[LR]L[123]_joint", "[LR]L3_[LR]S_closure"],
                stiffness=0., damping=0., effort_limit_sim=0.,
            ),
        },
    )

    # -- contact sensor (used for termination and collision penalty) --
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )

    # -- scene --
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False,
    )

    # -- simulation --
    enable_external_forces_every_iteration: bool = True
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
    )

    # -- terrain (flat plane, matches `mesh_type = "plane"`) --
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # -- sub-configs --
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    obs_clip: ObsClipCfg = ObsClipCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 只增大详细闭链碰撞的 GPU 缓冲容量，不改变接触力或物理参数。
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
