# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause
"""新车平地环境：cfg 配置，env 执行，train.py 负责 PPO。

所有 LAB 生命周期回调都在这个类中。仅闭链运动学、硬件标定及命令
抽样等数学计算放在独立功能文件；不再存在旧车环境或 Mine 适配子类。
"""
from __future__ import annotations

from collections.abc import Sequence
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, wrap_to_pi

from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state, five_bar_torques
from wheel_legged_gym_isaaclab.leg_control import (
    DIRECT_JOINT_POSITION_MODE, LEG_CONTROL_MODES, LEG_ACTION_COLUMNS,
    joint_position_targets, leg_drive_gains, validate_joint_position_map,
)
from wheel_legged_gym_isaaclab.mcu_hardware import gravity_feedforward, synchronize_actuator_limits
from wheel_legged_gym_isaaclab.physics_contract import apply_force_iteration_setting
from wheel_legged_gym_isaaclab.wheel_control import apply_wheel_action_limit, validated_wheel_action_limit
from wheel_legged_gym_isaaclab.robot_model import validate_robot_contract
from wheel_legged_gym_isaaclab.reward_signals import (
    action_second_difference_penalty, horizontal_velocity_error_squared, merge_step_diagnostics,
    height_tracking_kernel, wheel_action_overshoot_penalty,
    joint_target_length_error_squared, validate_joint_target_length_reward,
    guard_tracking_bonuses, validate_task_quality_gate,
)
from wheel_legged_gym_isaaclab.tracking_goal_feedback import (
    load_tracking_goal_criteria, tracking_goal_violation_penalty,
    tracking_goal_component_violation_penalty,
    validate_tracking_goal_reward_config,
)
from wheel_legged_gym_isaaclab.yaw_training import (
    TASK_NAMES, grouped_task_ids,
    spin_center_velocity_penalty,
)
from .wheel_legged_vmc_flat_env_cfg import WheelLeggedVMCFlatEnvCfg
from .env_commands import resample_commands
from wheel_legged_gym_isaaclab.joint_reset import configure_reset_pose

REWARD_SCALE_FIELDS = {
    'track_lin_vel': 'tracking_lin_vel',
    'track_lin_vel_enhance': 'tracking_lin_vel_enhance',
    'track_lin_vel_precise': 'tracking_lin_vel_precise',
    'track_ang_vel': 'tracking_ang_vel',
}

REWARD_NAMES = (
    'track_lin_vel', 'track_lin_vel_enhance', 'track_lin_vel_precise',
    'lin_vel_error_sq', 'standing_velocity', 'base_height_error_sq', 'joint_target_leg_length_error_sq',
    'track_ang_vel', 'yaw_rate_error_sq', 'spin_center_velocity',
    'base_height', 'nominal_state', 'standing_leg_angle',
    'standing_wheel_tracking', 'lin_vel_z', 'ang_vel_xy',
    'orientation', 'dof_vel', 'dof_acc',
    'torques', 'action_rate', 'action_smooth',
    'leg_length_action_saturation', 'leg_length_action_difference', 'leg_length_target_underreach',
    'joint_position_action_saturation', 'wheel_action_saturation', 'collision', 'dof_pos_limits',
    'termination',
)


class WheelLeggedVMCFlatEnv(DirectRLEnv):
    """一个新车环境；关节目标角 PD 和显式 VMC 由 cfg.leg_control_mode 选择。

    LAB step 顺序：动作预处理 → 物理子步 → 终止 → 奖励 → reset → 观测。
    位置/速度/力矩目标只写入缓存，scene.write_data_to_sim() 才送到 PhysX。
    """
    cfg: WheelLeggedVMCFlatEnvCfg

    # ---- 初始化与场景 ----
    def __init__(
        self, cfg: WheelLeggedVMCFlatEnvCfg, render_mode: str | None = None, **kwargs
    ):
        validated_wheel_action_limit(cfg.wheel_action_limit)
        cfg.five_bar_geometry = validate_robot_contract(cfg.to_dict())
        configure_reset_pose(cfg)  # 完整闭链姿态；历史回放保留原 reset profile。
        # LAB 基类先创建 self.sim / self.scene，再回调本类的 _setup_scene()。
        super().__init__(cfg, render_mode, **kwargs)

        # single_action_space 描述「单个机器人」的动作空间；张量实际是 [N, 6]。
        self._action_dim = int(self.single_action_space.shape[0])
        self._validate_configuration()
        # LAB find_joints() 返回「仿真关节下标、匹配名称」，而不是电机对象。
        leg_joint_names = list(self.cfg.leg_joint_names)
        wheel_joint_names = list(self.cfg.wheel_joint_names)
        torque_joint_names = [
            leg_joint_names[0], leg_joint_names[1], wheel_joint_names[0],
            leg_joint_names[2], leg_joint_names[3], wheel_joint_names[1],
        ]
        self._leg_joint_ids, resolved_legs = self._robot.find_joints(
            leg_joint_names, preserve_order=True
        )
        self._wheel_joint_ids, resolved_wheels = self._robot.find_joints(
            wheel_joint_names, preserve_order=True
        )
        self._torque_joint_ids, resolved_torques = self._robot.find_joints(
            torque_joint_names, preserve_order=True
        )
        if (
            resolved_legs != leg_joint_names
            or resolved_wheels != wheel_joint_names
            or resolved_torques != torque_joint_names
            or self._action_dim != len(torque_joint_names)
        ):
            raise ValueError(
                "VMC joint/action layout does not match the robot articulation: "
                f"legs={resolved_legs}, wheels={resolved_wheels}, "
                f"torques={resolved_torques}, actions={self._action_dim}"
            )
        print(
            "[INFO] VMC joint mapping: "
            f"articulation={self._robot.joint_names}, "
            f"legs={self._leg_joint_ids}, wheels={self._wheel_joint_ids}, "
            f"torque_targets={self._torque_joint_ids}"
        )

        # ContactSensor.find_bodies() 返回传感器力张量的 body 下标；它不是 joint ID。
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        # 名称可用正则：这里只选择机身/腿部，不把正常轮子触地纳入惩罚。
        self._penalised_contact_ids, _ = self._contact_sensor.find_bodies(
            self.cfg.penalised_body_pattern
        )
        leg_body_names = list(self.cfg.leg_body_names)
        self._leg_contact_ids, resolved_leg_bodies = self._contact_sensor.find_bodies(
            leg_body_names, preserve_order=True
        )
        if resolved_leg_bodies != leg_body_names:
            raise ValueError(
                "Leg contact layout does not match the robot: "
                f"expected={leg_body_names}, resolved={resolved_leg_bodies}"
            )

        # Torque limits come from the selected model configuration.
        self._torque_limits = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        # 力矩列顺序为 [LB, LL, 左轮, RB, RL, 右轮]。
        self._torque_limits[:, (0, 1, 3, 4)] = self.cfg.leg_effort_limit
        self._torque_limits[:, (2, 5)] = self.cfg.wheel_effort_limit
        self._leg_gravity_effort_applied = torch.zeros(self.num_envs, 4, device=self.device)

        # raw 是未执行限幅的策略输出，用于饱和惩罚；actions 是有效执行动作。
        self._raw_actions = torch.zeros_like(self._torque_limits)
        self._actions = torch.zeros_like(self._raw_actions)
        self._last_actions = torch.zeros(
            self.num_envs, self._action_dim, 2, device=self.device
        )

        # 命令缓存：[前向速度 m/s, yaw rate rad/s, 机身高度 m, 可选航向 rad]。
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._target_lin_vel_x = torch.zeros(self.num_envs, device=self.device)
        self._last_grouped_command_phase = -1
        self._command_ranges = torch.tensor(
            [
                list(self.cfg.commands.ranges_lin_vel_x),
                list(self.cfg.commands.ranges_ang_vel_yaw),
                list(self.cfg.commands.ranges_height),
            ],
            device=self.device,
        )
        self._dedicated_standing_count = (
            int(self.num_envs * self.cfg.commands.standing_env_fraction)
            if self.cfg.commands.grouped_training and self.cfg.commands.fixed_standing_groups else None
        )
        if self.cfg.commands.grouped_training:
            groups = grouped_task_ids(torch.arange(self.num_envs),
                                      dedicated_standing_count=self._dedicated_standing_count)
            counts = torch.bincount(groups, minlength=len(TASK_NAMES))
            print(f"[INFO] Command task groups: {dict(zip(TASK_NAMES, counts.tolist()))}", flush=True)
            if self._dedicated_standing_count is not None:
                print(f"[INFO] Dedicated standing environments: {self._dedicated_standing_count}; "
                      "speed/yaw remain zero across command phases and resets.", flush=True)
        # obs scale for commands: [lin_vel, ang_vel, height]
        self._commands_scale = torch.tensor(
            [
                self.cfg.obs_scales.lin_vel,
                self.cfg.obs_scales.ang_vel,
                self.cfg.obs_scales.height_measurements,
            ],
            device=self.device,
        )

        # 每个环境两条腿，虚拟状态均为 [N, 2]；Jacobian 为 [N, 腿, 虚拟坐标, 主动关节]。
        for name in ('_theta0', '_theta0_dot', '_L0', '_L0_dot'):
            setattr(self, name, torch.zeros(self.num_envs, 2, device=self.device))
        self._five_bar_jacobian = torch.zeros(self.num_envs, 2, 2, 2, device=self.device)
        self._five_bar_valid = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._l0_ref_applied = torch.zeros(self.num_envs, 2, device=self.device)
        self._theta0_ref_applied = torch.zeros_like(self._l0_ref_applied)
        self._joint_position_ref_applied = torch.zeros(self.num_envs, 4, device=self.device)
        if self.cfg.leg_control_mode == DIRECT_JOINT_POSITION_MODE:
            # Validate once, not via GPU synchronizations on every substep.
            joint_position_targets(
                self._joint_position_ref_applied, self.cfg.leg_joint_position_offset,
                self.cfg.leg_joint_position_scale, self._robot.data.joint_pos_limits[:, self._leg_joint_ids],
                self.cfg.leg_joint_target_limit_margin,
            )

        # termination bookkeeping
        self._fail_buf = torch.zeros(self.num_envs, device=self.device)
        self._base_contact_buf = torch.zeros(self.num_envs, device=self.device)
        self._pitch_fail_buf = torch.zeros(self.num_envs, device=self.device)
        self._roll_fail_buf = torch.zeros(self.num_envs, device=self.device)
        self._leg_contact_buf = torch.zeros(self.num_envs, device=self.device)
        self._low_height_buf = torch.zeros(self.num_envs, device=self.device)
        self._termination_reasons: dict[str, torch.Tensor] = {}

        # episode reward sums for logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in REWARD_NAMES
        }
        self._tracking_goal_criteria = None
        self._tracking_goal_evidence = None
        if (self.cfg.rewards.tracking_goal_violation != 0.
                or self.cfg.rewards.tracking_goal_component_violation != 0.):
            self._tracking_goal_criteria, self._tracking_goal_evidence = load_tracking_goal_criteria()
            for key in ('tracking_goal_violation', 'tracking_goal_component_violation'):
                if getattr(self.cfg.rewards, key) != 0.:
                    self._episode_sums[key] = torch.zeros(
                        self.num_envs, dtype=torch.float, device=self.device)
        self._last_reward_terms: dict[str, torch.Tensor] = {}
        self._tracking_diagnostics: dict[str, float] = {}
        self._command_resample_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # LAB 的 _robot.data 是带时间戳的状态缓存，scene.update() 推进后按需刷新。
        self._base_height = self._robot.data.root_pos_w[:, 2]

    def _setup_scene(self):
        # LAB 在基类初始化期间调用此钩子；这里构建场景，不执行策略推理。
        self._external_forces_every_iteration = apply_force_iteration_setting(
            self.sim.stage, self.cfg.sim.physics_prim_path,
            self.cfg.enable_external_forces_every_iteration,
        )
        print('[INFO] PhysX external forces every iteration: '
              f'{self._external_forces_every_iteration}')
        # 将顶层力矩上限同步到嵌套 actuator 配置，必须先于 Articulation 初始化。
        synchronize_actuator_limits(self.cfg)
        leg_stiffness, leg_damping = leg_drive_gains(
            self.cfg.leg_control_mode, self.cfg.leg_joint_stiffness, self.cfg.leg_joint_damping
        )
        # 显式 VMC 把关节位置 PD 设为 0/0；关节角模式则启用配置的 PD。
        self.cfg.robot.actuators['legs'].stiffness = leg_stiffness
        self.cfg.robot.actuators['legs'].damping = leg_damping
        self._robot = Articulation(self.cfg.robot)
        from pxr import UsdPhysics
        for name in self.cfg.leg_joint_names:
            # 下面是 USD 底层 API，不是读取机器人当前角度；Prim 表示场景中的节点。
            prim=self.sim.stage.GetPrimAtPath(f'/World/envs/env_0/Robot/joints/{name}')
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                raise ValueError(f'Leg drive joint not found: {name}')
            drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
            # Apply 给该关节附加角驱动 schema；Create*Attr 写入驱动配置属性。
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(leg_stiffness)
            drive.CreateDampingAttr(leg_damping)
            drive.CreateMaxForceAttr(self.cfg.leg_effort_limit)
            if self.cfg.leg_control_mode == DIRECT_JOINT_POSITION_MODE:
                # 普通 maxForce 仅限制驱动 PD；PhysX envelope 同时约束 PD 和前馈。
                from omni.physx.bindings._physx import PERF_ENV_API
                if not prim.ApplyAPI(PERF_ENV_API, 'angular'):
                    raise RuntimeError('Cannot enforce combined joint PD + feedforward effort limit')
                if not prim.HasAPI(PERF_ENV_API, 'angular'):
                    raise RuntimeError('Combined joint effort envelope was not applied')
        print(f'[INFO] Leg execution: {self.cfg.leg_control_mode}; '
              f'joint-drive stiffness={leg_stiffness}, damping={leg_damping}.')
        if self.cfg.wheel_control_mode == "implicit_velocity":
            from pxr import UsdPhysics
            # 原 CAD 没有导入器驱动；只为两轮创建所需的速度阻尼驱动。
            for name in self.cfg.wheel_joint_names:
                path = f"/World/envs/env_0/Robot/joints/{name}"
                prim = self.sim.stage.GetPrimAtPath(path)
                if not prim.IsA(UsdPhysics.RevoluteJoint):
                    raise ValueError(f"Wheel drive joint not found: {path}")
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(0.0)
                drive.CreateDampingAttr(self.cfg.wheel_damping)
                drive.CreateMaxForceAttr(self.cfg.wheel_effort_limit)
                drive.CreateTargetVelocityAttr(0.0)
        # 注册到 LAB scene，后续 scene.write_data_to_sim()/update()/reset()
        self.scene.articulations["robot"] = self._robot
        from wheel_legged_gym_isaaclab.mine_collision_filters import relocate_collision_groups
        relocate_collision_groups(self.sim.stage, self.scene.env_prim_paths)
        # LAB 按配置的 prim_path 建立接触传感器；读取 .data 才取得力数据。
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # LAB 的 class_type 是配置携带的构造类；调用它创建地形及各 env 的原点。
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone 将 env_0 扩展成各 env。False 表示继承源 USD，而不是完整复制；
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            # CPU 路径显式过滤不同 env 的互撞；地形 prim 是所有 env 共用的地面。
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # DomeLightCfg 是配置，.func(path, cfg) 才实际生成灯光；不参与控制和 reward。
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ---- 动作与控制 ----

    def _pre_physics_step(self, actions: torch.Tensor):
        # Episode events belong to this step only; a step without resets must
        # not repeat the previous step's ended episodes in RSL's rollout log.
        self.extras['log'] = {}
        # Preserve raw Gaussian outputs; leg targets keep their unit envelope.
        # A separately saved wheel clamp supplies transient reference headroom.
        self._raw_actions = torch.nan_to_num(
            actions, nan=0.0, posinf=1.0, neginf=-1.0
        )
        self._actions = torch.clamp(self._raw_actions, -1.0, 1.0)
        self._actions = apply_wheel_action_limit(
            self._actions, self._raw_actions, self.cfg.wheel_action_limit,
        )

    def _apply_action(self):
        """Apply this policy's configured actuator contract.

        Called once per physics substep (200 Hz). Mirrors the original
        `leg_post_physics_step` + `_compute_torques` inside the decimation loop.
        """
        self._update_forward_kinematics()

        if self.cfg.leg_control_mode == DIRECT_JOINT_POSITION_MODE:
            self._apply_direct_joint_position_action()
            return  # Direct motor references; gravity was handled above.

        # reference targets from actions
        theta0_ref = self._actions[:, (0, 3)] * self.cfg.action_scale_theta
        # Keep the virtual-leg angle target inside its physical safety range
        theta0_ref = torch.clamp(
            theta0_ref,
            min=self.cfg.theta0_ref_min,
            max=self.cfg.theta0_ref_max,
        )
        # 每个子步从 raw 重建长度参考，避免对已修正的 _actions 重复叠加高度修正。
        l0_ref = (
            torch.clamp(self._raw_actions[:, (1, 4)], -1.0, 1.0)
            * self.cfg.action_scale_l0
            + self.cfg.l0_offset
        )
        nominal_l0_ref = self._commands[:, 2:3] + self.cfg.leg_length_height_offset
        l0_ref = (
            (1.0 - self.cfg.height_reference_blend) * l0_ref
            + self.cfg.height_reference_blend * nominal_l0_ref
        )
        height_error = self._commands[:, 2] - self._robot.data.root_pos_w[:, 2]
        height_correction = torch.clamp(
            height_error * self.cfg.height_feedback_gain,
            min=-self.cfg.height_feedback_max_adjustment,
            max=self.cfg.height_feedback_max_adjustment,
        )
        l0_ref = l0_ref + height_correction[:, None]
        # 高度混合和外环修正后再限制腿长，动作限幅不能替代这一层参考限幅。
        l0_ref = torch.clamp(
            l0_ref, min=self.cfg.l0_ref_min, max=self.cfg.l0_ref_max
        )
        forward_support = (
            self._commands[:, 0] >= self.cfg.forward_support_speed_threshold
        )
        support_floor = torch.minimum(
            torch.full_like(height_error, self.cfg.forward_support_min_leg_length),
            self._commands[:, 2] + self.cfg.forward_support_height_margin,
        )
        l0_ref = torch.where(
            forward_support[:, None],
            torch.maximum(l0_ref, support_floor[:, None]),
            l0_ref,
        )
        # 固定控制器和导轨诊断可覆写这两个钩子；普通平地控制不额外改参考。
        l0_ref, theta0_ref = self._modify_leg_references(l0_ref, theta0_ref)
        self._l0_ref_applied = l0_ref
        self._theta0_ref_applied = theta0_ref
        self._actions[:, (1, 4)] = (
            l0_ref - self.cfg.l0_offset
        ) / self.cfg.action_scale_l0
        wheel_vel_ref = self._actions[:, (2, 5)] * self.cfg.action_scale_vel

        # VMC leg control (PD on virtual coordinates theta0 / L0)
        torque_leg = (
            self.cfg.kp_theta * (theta0_ref - self._theta0)
            - self.cfg.kd_theta * self._theta0_dot
        )
        force_leg = (
            self.cfg.kp_l0 * (l0_ref - self._L0) - self.cfg.kd_l0 * self._L0_dot
        )
        torque_wheel = self._wheel_effort_from_reference(wheel_vel_ref)

        # map virtual forces/torques to actual joint torques (closed chain)
        force_bias = self._get_leg_force_bias()
        T1, T2 = self._vmc(force_leg + self._get_gravity_feedforward() + force_bias, torque_leg)

        torques = torch.stack(
            (
                T1[:, 0], T2[:, 0], torque_wheel[:, 0],
                T1[:, 1], T2[:, 1],
                # 写入边界将前进为正的右轮力矩转换回镜像关节坐标。
                -torque_wheel[:, 1],
            ),
            dim=1,
        )
        torques = torch.clip(torques, -self._torque_limits, self._torque_limits)
        self._robot.set_joint_effort_target(torques, joint_ids=self._torque_joint_ids)

    def _apply_direct_joint_position_action(self):
        """Four active motor positions through PhysX joint PD, plus unchanged wheels."""
        target, applied = joint_position_targets(
            self._raw_actions[:, LEG_ACTION_COLUMNS], self.cfg.leg_joint_position_offset,
            self.cfg.leg_joint_position_scale, self._robot.data.joint_pos_limits[:, self._leg_joint_ids],
            self.cfg.leg_joint_target_limit_margin, validate=False,
        )
        self._actions[:, LEG_ACTION_COLUMNS] = applied
        self._joint_position_ref_applied = target
        self._robot.set_joint_position_target(target, joint_ids=self._leg_joint_ids)
        self._robot.set_joint_velocity_target(torch.zeros_like(target), joint_ids=self._leg_joint_ids)
        # 直接关节模式的显式 effort 只含重力前馈，不能叠加另一套 VMC 反馈。
        torques = torch.zeros_like(self._actions)
        if self.cfg.leg_joint_gravity_feedforward:
            t1, t2 = five_bar_torques(
                self._five_bar_jacobian, self._get_gravity_feedforward(), torch.zeros_like(self._theta0),
            )
            gravity_effort = torch.stack((t1[:, 0], t2[:, 0], t1[:, 1], t2[:, 1]), dim=-1)
            self._leg_gravity_effort_applied = gravity_effort.clamp(-self.cfg.leg_effort_limit, self.cfg.leg_effort_limit)
        else:
            self._leg_gravity_effort_applied.zero_()
        torques[:, LEG_ACTION_COLUMNS] = self._leg_gravity_effort_applied
        wheel_ref = self._actions[:, (2, 5)] * self.cfg.action_scale_vel
        wheel_effort = self._wheel_effort_from_reference(wheel_ref)
        if self.cfg.wheel_control_mode != 'implicit_velocity':
            torques[:, (2, 5)] = wheel_effort * wheel_effort.new_tensor([1., -1.])
        self._robot.set_joint_effort_target(
            torques.clamp(-self._torque_limits, self._torque_limits), joint_ids=self._torque_joint_ids,
        )
        # FK of motor targets is telemetry only, never a control reference map.
        length, angle, _, _ = five_bar_state(target[:, ::2], target[:, 1::2], **self.cfg.five_bar_geometry)
        self._l0_ref_applied = length
        self._theta0_ref_applied = angle

    def _wheel_effort_from_reference(self, reference):
        """Shared forward-positive wheel controller for both leg modes."""
        if self.cfg.wheel_control_mode == 'implicit_velocity':
            self._robot.set_joint_velocity_target(
                reference * reference.new_tensor([1., -1.]), joint_ids=self._wheel_joint_ids,
            )
            return torch.zeros_like(reference)
        return self.cfg.wheel_damping * (reference - self._get_wheel_vel_forward_positive())

    def _modify_leg_references(self, length, angle):
        return length, angle

    def _get_leg_force_bias(self):
        return torch.zeros_like(self._L0)

    def _get_gravity_feedforward(self):
        pitch = torch.asin(self._robot.data.projected_gravity_b[:, 0].clamp(-1., 1.))[:, None]
        return gravity_feedforward(
            self._theta0, pitch, mode=self.cfg.feedforward_mode,
            force=self.cfg.feedforward_force, angle_bias=self.cfg.feedforward_angle_bias,
            force_limit=self.cfg.feedforward_force_limit,
        )

    def _update_forward_kinematics(self, env_ids: torch.Tensor | None = None):
        """Mapped large five-bar state, Jacobian and analytic virtual velocity."""
        dof_pos = self._robot.data.joint_pos
        dof_vel = self._robot.data.joint_vel
        if env_ids is not None:
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
        q_b = dof_pos[:, self._leg_joint_ids[::2]]
        q_l = dof_pos[:, self._leg_joint_ids[1::2]]
        length, angle, jacobian, valid = five_bar_state(q_b, q_l, **self.cfg.five_bar_geometry)
        velocity = torch.stack((dof_vel[:, self._leg_joint_ids[::2]],
                                dof_vel[:, self._leg_joint_ids[1::2]]), dim=-1)
        virtual_velocity = (jacobian @ velocity[..., None]).squeeze(-1)
        selection = slice(None) if env_ids is None else env_ids
        self._L0[selection] = length
        self._theta0[selection] = angle
        self._L0_dot[selection] = virtual_velocity[..., 0]
        self._theta0_dot[selection] = virtual_velocity[..., 1]
        self._five_bar_jacobian[selection] = jacobian
        self._five_bar_valid[selection] = valid

    def _vmc(self, F: torch.Tensor, T: torch.Tensor):
        """J^T [F,T] in raw motor coordinates, including left/right axis signs."""
        return five_bar_torques(self._five_bar_jacobian, F, T)

    # ---- 观测与命令 ----

    def _get_observations(self) -> dict:
        # command resampling (once every resampling_time seconds)
        self._update_commands()
        max_command_step = self.cfg.commands.linear_acceleration_limit * self.step_dt
        self._commands[:, 0] += torch.clamp(
            self._target_lin_vel_x - self._commands[:, 0],
            min=-max_command_step,
            max=max_command_step,
        )
        # heading command -> yaw-rate command
        if self.cfg.commands.heading_command:
            forward_axis = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device
            ).expand(self.num_envs, -1)
            forward = quat_apply(
                self._robot.data.root_quat_w, forward_axis
            )
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self._commands[:, 1] = torch.clip(
                self.cfg.commands.heading_kp * wrap_to_pi(self._commands[:, 3] - heading),
                -self.cfg.commands.heading_rate_limit,
                self.cfg.commands.heading_rate_limit,
            )

        # base height (flat ground)
        self._base_height = self._robot.data.root_pos_w[:, 2]

        # store previous actions for reward computation
        self._last_actions[:, :, 1] = self._last_actions[:, :, 0]
        self._last_actions[:, :, 0] = self._actions[:]

        heading_velocity = self._get_heading_frame_horizontal_velocity()
        wheel_velocity_forward = self._get_wheel_vel_forward_positive()
        command_obs = torch.nan_to_num(
            self._commands[:, :3], nan=0.0, posinf=0.0, neginf=0.0
        )
        obs = torch.cat(
            (
                self._sanitize_observation(
                    self._robot.data.root_ang_vel_b, self.cfg.obs_clip.ang_vel
                ) * self.cfg.obs_scales.ang_vel,
                self._sanitize_observation(
                    self._robot.data.projected_gravity_b,
                    self.cfg.obs_clip.projected_gravity,
                ),
                command_obs * self._commands_scale,
                self._get_leg_observation(),
                # 轮角连续累积而不输入 obs；保留实际机身水平速度和两轮转速。
                self._sanitize_observation(
                    heading_velocity, self.cfg.obs_clip.lin_vel
                ) * self.cfg.obs_scales.lin_vel,
                self._sanitize_observation(
                    wheel_velocity_forward,
                    self.cfg.obs_clip.wheel_vel,
                ) * self.cfg.obs_scales.dof_vel,
                self._sanitize_observation(
                    self._actions, self.cfg.obs_clip.action
                ),
            ),
            dim=-1,
        )
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        # reset 会替换 extras['log']，在返回观测前补回本控制步的诊断统计。
        if self._tracking_diagnostics:
            self.extras['log'] = merge_step_diagnostics(
                self.extras.get('log', {}), self._tracking_diagnostics,
            )
        return {"policy": obs}

    def _get_leg_observation(self):
        if self.cfg.leg_control_mode == DIRECT_JOINT_POSITION_MODE:
            position = self._robot.data.joint_pos[:, self._leg_joint_ids]
            offset = position.new_tensor(self.cfg.leg_joint_position_offset)
            return torch.cat((
                self._sanitize_observation(position - offset, self.cfg.obs_clip.theta) * self.cfg.obs_scales.dof_pos,
                self._sanitize_observation(
                    self._robot.data.joint_vel[:, self._leg_joint_ids], self.cfg.obs_clip.theta_dot,
                ) * self.cfg.obs_scales.dof_vel,
            ), dim=-1)
        return torch.cat((
            self._sanitize_observation(self._theta0, self.cfg.obs_clip.theta) * self.cfg.obs_scales.dof_pos,
            self._sanitize_observation(self._theta0_dot, self.cfg.obs_clip.theta_dot) * self.cfg.obs_scales.dof_vel,
            self._sanitize_observation(self._L0, self.cfg.obs_clip.leg_length) * self.cfg.obs_scales.l0,
            self._sanitize_observation(self._L0_dot, self.cfg.obs_clip.leg_length_dot) * self.cfg.obs_scales.l0_dot,
        ), dim=-1)

    def _get_wheel_vel_forward_positive(self) -> torch.Tensor:
        """Return wheel speeds with positive values meaning forward motion.

        The mirrored URDF hip frames make the raw right-wheel joint axis point
        opposite to the raw left-wheel joint axis. Canonicalizing the right
        coordinate here gives the policy one consistent wheel convention.
        """
        wheel_vel_raw = self._robot.data.joint_vel[:, self._wheel_joint_ids]
        return torch.stack(
            (wheel_vel_raw[:, 0], -wheel_vel_raw[:, 1]), dim=1
        )

    def _get_heading_frame_horizontal_velocity(self) -> torch.Tensor:
        """Return horizontal velocity in the yaw-aligned heading frame.

        Unlike ``root_lin_vel_b[:, :2]``, this projection is independent of
        body roll and pitch. The policy therefore cannot change its measured
        forward speed merely by leaning the chassis.
        """
        forward_axis_b = torch.zeros(self.num_envs, 3, device=self.device)
        forward_axis_b[:, 0] = 1.0
        forward_axis_w = quat_apply(self._robot.data.root_quat_w, forward_axis_b)
        heading_forward = forward_axis_w[:, :2]
        heading_forward = heading_forward / torch.clamp(
            torch.linalg.vector_norm(heading_forward, dim=1, keepdim=True),
            min=1.0e-6,
        )
        heading_left = torch.stack(
            (-heading_forward[:, 1], heading_forward[:, 0]), dim=1
        )
        velocity_xy_w = self._robot.data.root_lin_vel_w[:, :2]
        return torch.stack(
            (
                torch.sum(velocity_xy_w * heading_forward, dim=1),
                torch.sum(velocity_xy_w * heading_left, dim=1),
            ),
            dim=1,
        )

    @staticmethod
    def _sanitize_observation(value: torch.Tensor, limit: float) -> torch.Tensor:
        """Replace non-finite values and clip one physical observation block."""
        return torch.clamp(
            torch.nan_to_num(value, nan=0.0, posinf=limit, neginf=-limit),
            min=-limit,
            max=limit,
        )

    def _update_commands(self):
        """Resample commands for environments that reached the resampling time."""
        if self.cfg.commands.grouped_training and self.cfg.commands.ranges_lin_vel_x[0] != self.cfg.commands.ranges_lin_vel_x[1]:
            phase = self.common_step_counter // max(1, int(self.cfg.commands.resampling_time / self.step_dt))
            if phase != self._last_grouped_command_phase:
                self._last_grouped_command_phase = phase
                # 固定站立组保持零速，其余任务组按全局阶段轮换，并一起重采样。
                self._resample_commands_for(torch.arange(self.num_envs, device=self.device))
            return
        env_ids = (
            self.episode_length_buf
            % int(self.cfg.commands.resampling_time / self.step_dt)
            == 0
        ).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return
        self._resample_commands_for(env_ids)

    def _resample_commands_for(self, env_ids):
        # 长抽样流程在功能文件中；仍只修改当前环境的同一份命令缓存。
        resample_commands(self, env_ids)
        self._command_resample_step[env_ids] = self.common_step_counter

    # ---- 奖励 ----

    def _get_rewards(self) -> torch.Tensor:
        self._update_forward_kinematics()
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        proj_gravity = self._robot.data.projected_gravity_b
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        joint_acc = self._robot.data.joint_acc
        torque = self._robot.data.applied_torque
        contact_forces = self._contact_sensor.data.net_forces_w
        heading_velocity = self._get_heading_frame_horizontal_velocity()
        wheel_velocity = self._get_wheel_vel_forward_positive()

        # 使用航向对齐的水平速度，避免机身倾斜改变“实际前进速度”的定义。
        lin_vel_error = torch.square(
            self._commands[:, 0] - heading_velocity[:, 0]
        )
        lin_vel_penalty = horizontal_velocity_error_squared(
            heading_velocity, self._commands[:, 0], self.cfg.rewards.lin_vel_error_lateral_multiplier,
        )
        r_track_lin = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        r_track_lin_enhance = (
            torch.exp(
                -lin_vel_error / self.cfg.rewards.tracking_sigma_enhance
            )
            - 1.0
        )
        r_track_lin_precise = torch.exp(
            -lin_vel_error / self.cfg.rewards.tracking_sigma_precise
        )
        standing = (torch.abs(self._commands[:, 0]) < 0.01) & (
            torch.abs(self._target_lin_vel_x) < 0.01
        )
        r_standing_velocity = torch.sum(torch.square(heading_velocity), dim=1) * standing
        r_standing_leg_angle = torch.sum(self._theta0.square(), dim=1) * standing
        wheel_reference = self._actions[:, (2, 5)] * self.cfg.action_scale_vel
        r_standing_wheel_tracking = torch.sum(
            (wheel_velocity - wheel_reference).square(), dim=1
        ) * (standing & (self._commands[:, 1].abs() < 0.01))
        ang_vel_error = torch.square(self._commands[:, 1] - root_ang_vel_b[:, 2])
        r_track_ang = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma_ang)
        r_spin_center = spin_center_velocity_penalty(
            wheel_velocity, self.cfg.wheel_radius,
            self._commands[:, 0], self._target_lin_vel_x, self._commands[:, 1],
        )
        base_height_error = torch.square(
            self._robot.data.root_pos_w[:, 2] - self._commands[:, 2]
        )
        if self.cfg.rewards.base_height < 0:
            r_base_height = torch.sqrt(base_height_error)
        else:
            r_base_height = height_tracking_kernel(
                self._robot.data.root_pos_w[:, 2], self._commands[:, 2],
                self.cfg.rewards.base_height_sigma, self.cfg.rewards.base_height_relative_deadband,
            )
        theta_difference = torch.square(self._theta0[:, 0] - self._theta0[:, 1])
        if self.cfg.rewards.nominal_state < 0:
            r_nominal_state = theta_difference
        else:
            r_nominal_state = torch.exp(-theta_difference / 0.1)
        r_lin_vel_z = torch.square(root_lin_vel_b[:, 2])
        r_ang_vel_xy = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
        # 小角度时投影重力 x 主要反映 pitch，y 主要反映 roll。
        r_orientation = torch.square(proj_gravity[:, 0]) + (
            self.cfg.rewards.orientation_roll_multiplier
            * torch.square(proj_gravity[:, 1])
        )
        r_dof_vel = torch.sum(torch.square(joint_vel[:, self._leg_joint_ids]), dim=1)
        r_dof_acc = torch.sum(torch.square(joint_acc), dim=1)
        r_torques = torch.sum(torch.square(torque), dim=1)
        r_action_rate = torch.sum(torch.square(self._last_actions[:, :, 0] - self._actions), dim=1)
        r_action_smooth = action_second_difference_penalty(
            self._actions, self._last_actions[:, :, 0], self._last_actions[:, :, 1],
            self.cfg.rewards.action_smooth_wheel_multiplier,
        )
        r_wheel_action_saturation = wheel_action_overshoot_penalty(self._raw_actions)
        # 软饱和读取执行裁剪前的动作，使超出 ±1 的输出仍能被区分和惩罚。
        zero = torch.zeros_like(self._raw_actions[:, 0])
        if self.cfg.leg_control_mode == DIRECT_JOINT_POSITION_MODE:
            # 直接模式的 1/4 列是关节角动作，不能再套用虚拟腿长度动作惩罚。
            r_leg_length_action_saturation = r_leg_length_action_difference = r_leg_length_target_underreach = zero
            raw_joint_actions = self._raw_actions[:, LEG_ACTION_COLUMNS].clamp(-10., 10.)
            r_joint_position_action_saturation = torch.relu(
                raw_joint_actions.abs() - self.cfg.rewards.joint_position_action_soft_limit,
            ).square().sum(dim=1)
        else:
            raw_length_actions = self._raw_actions[:, (1, 4)].clamp(-10.0, 10.0)
            r_leg_length_action_saturation = torch.relu(
                raw_length_actions.abs() - self.cfg.rewards.leg_length_action_soft_limit,
            ).square().sum(dim=1)
            r_leg_length_action_difference = (self._actions[:, 1] - self._actions[:, 4]).square()
            leg_length_refs = (
                self._raw_actions[:, (1, 4)].clamp(-1.0, 1.0) * self.cfg.action_scale_l0 + self.cfg.l0_offset
            )
            minimum_leg_length_ref = (
                self._commands[:, 2:3] + self.cfg.rewards.leg_length_target_height_offset
            ).clamp(max=self.cfg.rewards.leg_length_target_cap)
            r_leg_length_target_underreach = torch.relu(
                minimum_leg_length_ref - leg_length_refs,
            ).square().mean(dim=1)
            r_joint_position_action_saturation = zero
        contact_force_norm = torch.norm(
            contact_forces[:, self._penalised_contact_ids, :], dim=-1
        )
        r_collision = torch.sum(
            1.0 * (contact_force_norm > self.cfg.collision_force_threshold), dim=1
        )
        dof_pos_limits = self._robot.data.joint_pos_limits
        leg_pos = joint_pos[:, self._leg_joint_ids]
        leg_pos_limits = dof_pos_limits[:, self._leg_joint_ids]
        out_of_limits = -(leg_pos - leg_pos_limits[:, :, 0]).clip(max=0.0)
        out_of_limits += (leg_pos - leg_pos_limits[:, :, 1]).clip(min=0.0)
        r_dof_pos_limits = torch.sum(out_of_limits, dim=1)
        # Keep each signal and its config scale in one auditable table.
        signals = {
            "track_lin_vel": r_track_lin,
            "track_lin_vel_enhance": r_track_lin_enhance,
            "track_lin_vel_precise": r_track_lin_precise,
            "lin_vel_error_sq": lin_vel_penalty,
            "standing_velocity": r_standing_velocity,
            "base_height_error_sq": base_height_error,
            "joint_target_leg_length_error_sq": joint_target_length_error_squared(
                self._l0_ref_applied, self._commands[:, 2], self.cfg.rewards.leg_length_target_height_offset,
            ) if self.cfg.leg_control_mode == DIRECT_JOINT_POSITION_MODE and self.cfg.rewards.joint_target_leg_length_error_sq else zero,
            "track_ang_vel": r_track_ang,
            "yaw_rate_error_sq": ang_vel_error,
            "spin_center_velocity": r_spin_center,
            "base_height": r_base_height,
            "nominal_state": r_nominal_state,
            "lin_vel_z": r_lin_vel_z,
            "ang_vel_xy": r_ang_vel_xy,
            "orientation": r_orientation,
            "dof_vel": r_dof_vel,
            "dof_acc": r_dof_acc,
            "torques": r_torques,
            "action_rate": r_action_rate,
            "action_smooth": r_action_smooth,
            "leg_length_action_saturation": r_leg_length_action_saturation,
            "leg_length_action_difference": r_leg_length_action_difference,
            "leg_length_target_underreach": r_leg_length_target_underreach,
            "joint_position_action_saturation": r_joint_position_action_saturation,
            "wheel_action_saturation": r_wheel_action_saturation,
            "collision": r_collision,
            "dof_pos_limits": r_dof_pos_limits,
        }
        signals = guard_tracking_bonuses(signals, self._robot.data.root_pos_w[:, 2], self._commands[:, 2],
                                        proj_gravity, self._tracking_goal_criteria, self.cfg.rewards.tracking_task_quality_gate_floor)
        rewards = {
            key: value * getattr(self.cfg.rewards, REWARD_SCALE_FIELDS.get(key, key)) * self.step_dt
            for key, value in signals.items()
            if key != 'joint_target_leg_length_error_sq' or self.cfg.rewards.joint_target_leg_length_error_sq
        }

        # 各项负奖励单独设上限，避免姿态等平方代价过早进入通用裁剪平坦区。
        reward_bound = self.cfg.rewards.clip_single_reward * self.step_dt
        penalty_bounds = {
            "nominal_state": self.cfg.rewards.nominal_state_penalty_clip * self.step_dt,
            "orientation": self.cfg.rewards.orientation_penalty_clip * self.step_dt,
            "base_height_error_sq": self.cfg.rewards.base_height_penalty_clip * self.step_dt,
            "joint_target_leg_length_error_sq": self.cfg.rewards.base_height_penalty_clip * self.step_dt,
            "standing_leg_angle": self.cfg.rewards.standing_leg_angle_penalty_clip * self.step_dt,
            "lin_vel_error_sq": self.cfg.rewards.lin_vel_penalty_clip * self.step_dt,
            "yaw_rate_error_sq": self.cfg.rewards.yaw_rate_penalty_clip * self.step_dt,
            "standing_wheel_tracking": self.cfg.rewards.standing_wheel_penalty_clip * self.step_dt,
            "action_rate": self.cfg.rewards.action_rate_penalty_clip * self.step_dt,
            "action_smooth": self.cfg.rewards.action_smooth_penalty_clip * self.step_dt,
            "wheel_action_saturation": self.cfg.rewards.wheel_action_saturation_penalty_clip * self.step_dt,
        }
        rewards["standing_leg_angle"] = r_standing_leg_angle * self.cfg.rewards.standing_leg_angle * self.step_dt
        rewards["standing_wheel_tracking"] = r_standing_wheel_tracking * self.cfg.rewards.standing_wheel_tracking * self.step_dt
        if (
            not self._tracking_diagnostics or self.common_step_counter % 100 == 0
        ):
            # 记录实际应用命令与速度；诊断统计不参与动作执行或奖励求和。
            diagnostics={}
            opposite = (self._commands[:, 0] * heading_velocity[:, 0] < 0) & (heading_velocity[:, 0].abs() > .05)
            for name,mask in (
                ('standing',standing & (self._commands[:,1].abs()<.01)),
                ('forward',self._commands[:,0]>.05),
                ('reverse',self._commands[:,0]<-.05),
                ('spin',standing & (self._commands[:,1].abs()>.05)),
            ):
                count=mask.sum().clamp_min(1)
                diagnostics[f'Tracking/{name}_fraction']=mask.float().mean().item()
                for label,value in (
                    ('command_m_s',self._commands[:,0]),
                    ('speed_m_s',heading_velocity[:,0]),
                    ('speed_mae_m_s',(self._commands[:,0]-heading_velocity[:,0]).abs()),
                    ('yaw_mae_rad_s',(self._commands[:,1]-root_ang_vel_b[:,2]).abs()),
                ):
                    diagnostics[f'Tracking/{name}_{label}']=((value*mask).sum()/count).item()
                diagnostics[f'Tracking/{name}_opposite_direction_fraction'] = ((opposite * mask).sum() / count).item()
            diagnostics['Tracking/wheel_reference_delta_rms_rad_s']=torch.sqrt(
                (self._actions[:,(2,5)]-self._last_actions[:,(2,5),0]).square().mean()
            ).item()*self.cfg.action_scale_vel
            diagnostics['Tracking/wheel_reference_second_difference_rms_rad_s'] = torch.sqrt(
                (self._actions[:, (2, 5)] - 2 * self._last_actions[:, (2, 5), 0]
                 + self._last_actions[:, (2, 5), 1]).square().mean()
            ).item() * self.cfg.action_scale_vel
            for key in ('lin_vel_error_sq','yaw_rate_error_sq','standing_wheel_tracking','action_rate','action_smooth','orientation'):
                bound=penalty_bounds[key]
                diagnostics[f'Reward_Clipping/{key}']=(rewards[key]<=-bound).float().mean().item()
            self._tracking_diagnostics = diagnostics
        rewards = {
            key: torch.nan_to_num(
                torch.clamp(
                    value,
                    min=-penalty_bounds.get(key, reward_bound),
                    max=reward_bound,
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            for key, value in rewards.items()
        }
        # LAB 先计算 reset_terminated；不安全终止只扣一次，超时不扣此项。
        rewards["termination"] = self.reset_terminated.float() * self.cfg.rewards.termination
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        if getattr(self.cfg.rewards, 'tracking_goal_violation', 0.) != 0.:
            cost = tracking_goal_violation_penalty(
                self._commands[:, :3], heading_velocity[:, 0], root_ang_vel_b[:, 2],
                self._robot.data.root_pos_w[:, 2], proj_gravity,
                self._L0, self._l0_ref_applied, self._theta0, self._theta0_ref_applied,
                wheel_velocity, wheel_reference, self._tracking_goal_criteria,
            )
            term = cost * self.cfg.rewards.tracking_goal_violation * self.step_dt
            term = torch.nan_to_num(term.clamp(min=-self.cfg.rewards.tracking_goal_penalty_clip * self.step_dt),
                                    nan=0., posinf=0., neginf=0.)
            rewards['tracking_goal_violation'] = term
            reward = reward + term
        if getattr(self.cfg.rewards, 'tracking_goal_component_violation', 0.) != 0.:
            cost = tracking_goal_component_violation_penalty(
                self._commands[:, :3], heading_velocity[:, 0], root_ang_vel_b[:, 2],
                self._robot.data.root_pos_w[:, 2], proj_gravity,
                self._L0, self._l0_ref_applied, self._theta0, self._theta0_ref_applied,
                wheel_velocity, wheel_reference, self._tracking_goal_criteria,
            )
            term = cost * self.cfg.rewards.tracking_goal_component_violation * self.step_dt
            term = torch.nan_to_num(term.clamp(
                min=-self.cfg.rewards.tracking_goal_component_penalty_clip * self.step_dt),
                nan=0., posinf=0., neginf=0.)
            rewards['tracking_goal_component_violation'] = term
            reward = reward + term
        self._last_reward_terms = {
            key: value.detach() for key, value in rewards.items()
        }
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # ---- 终止与配置检查 ----

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        contact_forces = self._contact_sensor.data.net_forces_w
        gravity = self._robot.data.projected_gravity_b
        base_contact = torch.norm(contact_forces[:, self._base_id, :], dim=-1).gt(
            self.cfg.base_contact_force_threshold,
        ).any(dim=1)
        self._base_contact_buf, base_died = self._advance_failure_counter(
            self._base_contact_buf, base_contact, self.cfg.base_contact_terminal_time_s,
        )
        self._fail_buf, fallen_died = self._advance_failure_counter(
            self._fail_buf, gravity[:, 2] > -0.1, self.cfg.fail_to_terminal_time_s,
        )
        body_pitch = torch.asin(gravity[:, 0].clamp(-1.0, 1.0)).abs()
        self._pitch_fail_buf, pitch_died = self._advance_failure_counter(
            self._pitch_fail_buf, body_pitch > self.cfg.max_body_pitch,
            self.cfg.body_pitch_terminal_time_s,
        )
        body_roll = torch.asin(gravity[:, 1].clamp(-1.0, 1.0)).abs()
        self._roll_fail_buf, roll_died = self._advance_failure_counter(
            self._roll_fail_buf, body_roll > self.cfg.max_body_roll,
            self.cfg.body_roll_terminal_time_s, inclusive=True,
        )
        # Only wheels may maintain ground contact. Brief impacts are tolerated.
        leg_contact = torch.norm(contact_forces[:, self._leg_contact_ids, :], dim=-1).gt(
            self.cfg.leg_contact_force_threshold,
        ).any(dim=1)
        self._leg_contact_buf, leg_died = self._advance_failure_counter(
            self._leg_contact_buf, leg_contact, self.cfg.leg_contact_terminal_time_s,
        )
        # Height also catches a collapsed chassis with intermittent force readings.
        self._low_height_buf, low_height_died = self._advance_failure_counter(
            self._low_height_buf, self._robot.data.root_pos_w[:, 2] < self.cfg.min_root_height,
            self.cfg.low_height_terminal_time_s, inclusive=True, minimum_steps=1.0,
        )
        numerical_failure = ~(
            torch.isfinite(self._robot.data.root_pos_w).all(dim=1)
            & torch.isfinite(self._robot.data.root_quat_w).all(dim=1)
            & torch.isfinite(self._robot.data.root_lin_vel_w).all(dim=1)
            & torch.isfinite(self._robot.data.root_ang_vel_w).all(dim=1)
            & torch.isfinite(self._robot.data.joint_pos).all(dim=1)
            & torch.isfinite(self._robot.data.joint_vel).all(dim=1)
            & torch.isfinite(contact_forces).all(dim=(1, 2))
        )
        self._termination_reasons = {
            "base_contact": base_died,
            "fallen": fallen_died,
            "pitch": pitch_died,
            "roll": roll_died,
            "leg_contact": leg_died,
            "numerical": numerical_failure,
            "kinematic_singularity": ~self._five_bar_valid.all(dim=1),
            "low_height": low_height_died,
        }
        died = torch.stack(tuple(self._termination_reasons.values())).any(dim=0)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out

    def _advance_failure_counter(self, counter, unsafe, seconds, *, inclusive=False, minimum_steps=None):
        """Count continuous unsafe steps; a safe step clears only this timer."""
        counter = torch.where(unsafe, counter + 1.0, 0.0)
        threshold = seconds / self.step_dt
        if minimum_steps is not None:
            threshold = max(minimum_steps, threshold)
        died = counter >= threshold if inclusive else counter > threshold
        return counter, died

    def _validate_configuration(self):
        """Fail early when coupled command/action settings become inconsistent."""
        if self._action_dim != 6:
            raise ValueError(f"WheelLeggedVMC expects 6 actions, got {self._action_dim}.")
        if self.cfg.leg_control_mode not in LEG_CONTROL_MODES:
            raise ValueError('Unknown leg control mode')
        if self.cfg.leg_control_mode == DIRECT_JOINT_POSITION_MODE:
            if self.cfg.leg_joint_stiffness<=0 or self.cfg.leg_joint_damping<0:
                raise ValueError('Implicit leg references require the CAD five-bar and positive joint stiffness')
            validate_joint_position_map(
                self.cfg.leg_joint_position_offset, self.cfg.leg_joint_position_scale,
                self.cfg.leg_joint_target_limit_margin,
            )
            if not 0 < self.cfg.rewards.joint_position_action_soft_limit <= 1:
                raise ValueError('Joint action soft limit must be in (0, 1]')
        if not 0.0 <= self.cfg.reset_velocity_initial <= self.cfg.reset_velocity_final:
            raise ValueError("Reset velocity magnitudes must satisfy 0 <= initial <= final.")
        if self.cfg.rewards.termination > 0.0:
            raise ValueError("Unsafe termination must not receive a positive reward.")
        if (not math.isfinite(self.cfg.rewards.base_height_relative_deadband)
                or not 0 <= self.cfg.rewards.base_height_relative_deadband < .1):
            raise ValueError('Height reward deadband must be smaller than ten percent')
        wheel_scale = self.cfg.rewards.wheel_action_saturation
        wheel_cap = self.cfg.rewards.wheel_action_saturation_penalty_clip
        if not math.isfinite(wheel_scale) or wheel_scale > 0:
            raise ValueError('Wheel action saturation reward must be finite and nonpositive')
        if not math.isfinite(wheel_cap) or wheel_cap <= 0 or wheel_cap < 17.*abs(wheel_scale):
            raise ValueError('Wheel action saturation cap must retain feedback throughout the bounded raw-action range')
        validate_joint_target_length_reward(self.cfg.leg_control_mode, self.cfg.rewards.joint_target_leg_length_error_sq)
        validate_task_quality_gate(self.cfg.rewards.tracking_task_quality_gate_floor, self.cfg.rewards.tracking_goal_component_violation or self.cfg.rewards.tracking_goal_violation)
        validate_tracking_goal_reward_config(self.cfg.rewards, self.cfg.leg_control_mode)
        if not math.isfinite(self.cfg.rewards.action_smooth_wheel_multiplier) or self.cfg.rewards.action_smooth_wheel_multiplier < 0:
            raise ValueError("Wheel smoothness multiplier must be finite and nonnegative.")
        if not math.isfinite(self.cfg.rewards.action_smooth_penalty_clip) or self.cfg.rewards.action_smooth_penalty_clip <= 0:
            raise ValueError("Action smoothness cap must be finite and positive.")

        standing_fraction = float(self.cfg.commands.standing_env_fraction)
        yaw_low, yaw_high = self.cfg.commands.ranges_ang_vel_yaw
        fixed_speed = self.cfg.commands.ranges_lin_vel_x[0] == self.cfg.commands.ranges_lin_vel_x[1]
        if not fixed_speed and (yaw_low != 0.0 or yaw_high != 0.0):
            if self.cfg.commands.heading_command:
                raise ValueError("Yaw-rate training requires heading_command=False; heading hold would overwrite turn commands.")
            if not (yaw_low < 0 < yaw_high and 0 <= self.cfg.commands.yaw_min_abs_rate <= min(-yaw_low, yaw_high)):
                raise ValueError("Yaw training needs both turn directions and a minimum within its range.")
        if self.cfg.commands.yaw_ramp_steps <= 0 or self.cfg.commands.yaw_start_steps < 0:
            raise ValueError("Yaw curriculum requires a nonnegative start and positive ramp duration.")
        if not 0 <= self.cfg.commands.grouped_low_speed_fraction <= 1:
            raise ValueError("grouped_low_speed_fraction must be in [0, 1].")
        if self.cfg.commands.grouped_training and not fixed_speed:
            if not isinstance(self.cfg.commands.fixed_standing_groups, bool):
                raise ValueError("fixed_standing_groups must be boolean")
            if yaw_low == yaw_high == 0.0:
                raise ValueError("Grouped turning training requires nonzero yaw ranges.")
            if not (self.cfg.commands.ranges_transition_lin_vel_x[0] < 0 < self.cfg.commands.ranges_transition_lin_vel_x[1]):
                raise ValueError("Grouped low-speed practice requires both forward and reverse transition ranges.")
        if not 0 <= self.cfg.commands.yaw_boundary_fraction <= 1:
            raise ValueError("Yaw boundary fraction must be in [0, 1].")
        if self.cfg.rewards.tracking_sigma_ang <= 0 or self.cfg.rewards.spin_center_velocity > 0:
            raise ValueError("Yaw tracking width must be positive and spin drift penalty nonpositive.")
        if not 0.0 <= standing_fraction < 1.0:
            raise ValueError(
                "standing_env_fraction must satisfy 0 <= value < 1, "
                f"got {standing_fraction}."
            )
        if not fixed_speed and not self.cfg.commands.grouped_training:
            raise ValueError("Random training requires grouped_training=True; the non-grouped sampler is retired.")
        if self.cfg.commands.reverse_ramp_steps <= 0:
            raise ValueError("reverse_ramp_steps must be greater than zero.")
        if self.cfg.commands.linear_acceleration_limit <= 0.0:
            raise ValueError("linear_acceleration_limit must be greater than zero.")
        if not 0.0 <= self.cfg.commands.speed_boundary_fraction <= 1.0:
            raise ValueError("speed_boundary_fraction must be between zero and one.")
        if not 0.0 <= self.cfg.commands.height_boundary_fraction <= 1.0:
            raise ValueError("height_boundary_fraction must be between zero and one.")

        transition_min, transition_max = self.cfg.commands.ranges_transition_lin_vel_x
        main_min, main_max = self.cfg.commands.ranges_lin_vel_x
        reverse_min, reverse_max = self.cfg.commands.ranges_reverse_lin_vel_x
        if not fixed_speed and not (
            reverse_min
            <= reverse_max
            <= transition_min
            <= 0.0
            <= transition_max
            <= main_min
            <= main_max
        ):
            raise ValueError(
                "Linear-velocity ranges must satisfy "
                "reverse_min <= reverse_max <= transition_min <= 0 <= transition_max <= main_min <= main_max, "
                f"got transition={self.cfg.commands.ranges_transition_lin_vel_x}, "
                f"forward={self.cfg.commands.ranges_lin_vel_x}, "
                f"reverse={self.cfg.commands.ranges_reverse_lin_vel_x}."
            )

        height_min, height_max = self.cfg.commands.ranges_height
        if height_min > height_max:
            raise ValueError(
                "Height command range must satisfy min <= max, "
                f"got {self.cfg.commands.ranges_height}."
            )
        if not fixed_speed and not (
            height_min <= self.cfg.rewards.base_height_target <= height_max
        ):
            raise ValueError(
                "base_height_target must lie inside the training height range, "
                f"got target={self.cfg.rewards.base_height_target}, "
                f"range={self.cfg.commands.ranges_height}."
            )
        if self.cfg.leg_control_mode != DIRECT_JOINT_POSITION_MODE:
            # 仅 VMC 模式检查虚拟腿参考映射；直接关节角模式不使用这些参数。
            mapped_min = self.cfg.l0_offset - self.cfg.action_scale_l0
            mapped_max = self.cfg.l0_offset + self.cfg.action_scale_l0
            if (abs(mapped_min - self.cfg.l0_ref_min) > 1.e-6
                    or abs(mapped_max - self.cfg.l0_ref_max) > 1.e-6):
                raise ValueError('Leg-length action map must cover the physical range exactly.')
            if not fixed_speed and height_max + self.cfg.rewards.leg_length_target_height_offset > self.cfg.l0_ref_max:
                raise ValueError('The minimum leg-length reference must fit inside the VMC range.')
            for name, value in (
                ('leg_length_target_cap', self.cfg.rewards.leg_length_target_cap),
                ('forward_support_min_leg_length', self.cfg.forward_support_min_leg_length),
            ):
                if not self.cfg.l0_ref_min <= value <= self.cfg.l0_ref_max:
                    raise ValueError(f'{name} must be inside the VMC range.')
            if self.cfg.height_feedback_gain < 0 or self.cfg.height_feedback_max_adjustment < 0:
                raise ValueError('Height feedback gain and adjustment limit must be non-negative.')
            if not 0 <= self.cfg.height_reference_blend <= 1:
                raise ValueError('height_reference_blend must be between zero and one.')
            if not fixed_speed and not main_min <= self.cfg.forward_support_speed_threshold <= main_max:
                raise ValueError('forward_support_speed_threshold must be in the forward command range.')

        max_command_speed = float(max(
            abs(value) for value in (
                transition_min, transition_max, main_min, main_max, reverse_min, reverse_max
            )
        ))
        max_wheel_speed = self.cfg.wheel_radius * self.cfg.action_scale_vel * self.cfg.wheel_action_limit
        if max_command_speed > max_wheel_speed + 1.0e-6:
            raise ValueError(
                "Linear command magnitude exceeds the ideal wheel-speed action range: "
                f"command_magnitude={max_command_speed:.3f} m/s, "
                f"reachable={max_wheel_speed:.3f} m/s."
            )

        for name in (
            'base_height_sigma', 'nominal_state_penalty_clip', 'orientation_penalty_clip',
            'tracking_sigma', 'tracking_sigma_enhance', 'tracking_sigma_precise',
            'orientation_roll_multiplier',
        ):
            value = getattr(self.cfg.rewards, name)
            if value <= 0.0:
                raise ValueError(f'{name} must be greater than zero, got {value}.')
        for name in (
            'lin_vel_penalty_clip', 'yaw_rate_penalty_clip',
            'standing_wheel_penalty_clip', 'action_rate_penalty_clip',
        ):
            value = getattr(self.cfg.rewards, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f'{name} must be finite and positive.')
        if self.cfg.wheel_radius <= 0.0 or self.cfg.wheel_track_width <= 0.0:
            raise ValueError(
                "wheel_radius and wheel_track_width must both be greater than zero, "
                f"got radius={self.cfg.wheel_radius}, track={self.cfg.wheel_track_width}."
            )

    # ---- 回合重置 ----

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # env_ids 选择「要重置的机器人」，不是关节下标；未给出时重置全部实例。
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        terminated_command_x = self._commands[env_ids, 0].clone()
        terminated_target_x = self._target_lin_vel_x[env_ids].clone()
        terminated_command_height = self._commands[env_ids, 2].clone()
        # Grouped commands rotate on the global clock, independently of each
        # episode's randomized age. A reset also resamples just its own IDs.
        steps_since_resample = self.common_step_counter - self._command_resample_step[env_ids]
        # Articulation.reset() 清理执行器等缓存，并不会把刚体瞬移回初始姿态。
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # default_* 已按 reset profile 统一主动/被动闭链姿态；clone 不污染其他 env。
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        # root_state 为 [位置 xyz, 四元数 wxyz, 世界线速度 xyz, 世界角速度 xyz] 共 13 维。
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        # 默认位置是相对各 env 原点的；加上 env_origins 才是世界坐标位置。
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # 训练重置扰动随课程放大；固定命令回放使用配置的最终扰动幅度。
        fixed_command = float(self._command_ranges[0, 0].item()) == float(
            self._command_ranges[0, 1].item()
        )
        progress = 1.0 if fixed_command else min(max(
            (self.common_step_counter - self.cfg.commands.standing_only_steps)
            / max(self.cfg.commands.motion_ramp_steps, 1), 0.0
        ), 1.0)
        reset_velocity = self.cfg.reset_velocity_initial + progress * (
            self.cfg.reset_velocity_final - self.cfg.reset_velocity_initial
        )
        default_root_state[:, 7:13] = (
            2.0 * torch.rand(len(env_ids), 6, device=self.device) - 1.0
        ) * reset_velocity
        # LAB write_*_to_sim() 直接写入物理状态和相关缓存；不是 PD 的目标参考。
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # 第三个参数 joint_ids=None 表示所有关节；仅作用于 env_ids 选中的实例。
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._update_forward_kinematics(env_ids)

        # reset buffers
        self._raw_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._joint_position_ref_applied[env_ids] = 0.0
        self._l0_ref_applied[env_ids] = 0.0
        self._theta0_ref_applied[env_ids] = 0.0
        self._leg_gravity_effort_applied[env_ids] = 0.0
        if self.cfg.leg_control_mode == DIRECT_JOINT_POSITION_MODE:
            # 状态复位不等于目标清零：还要清理旧参考和前馈，避免新回合继承饱和动作。
            self._joint_position_ref_applied[env_ids] = self._joint_position_ref_applied.new_tensor(
                self.cfg.leg_joint_position_offset,
            )
            # LAB set_joint_*_target() 写目标缓存，下一次 write_data_to_sim() 才应用。
            self._robot.set_joint_position_target(
                self._joint_position_ref_applied[env_ids], joint_ids=self._leg_joint_ids, env_ids=env_ids,
            )
            self._robot.set_joint_velocity_target(
                torch.zeros((len(env_ids), 4), device=self.device), joint_ids=self._leg_joint_ids, env_ids=env_ids,
            )
            self._robot.set_joint_effort_target(
                torch.zeros((len(env_ids), 6), device=self.device), joint_ids=self._torque_joint_ids, env_ids=env_ids,
            )
        self._fail_buf[env_ids] = 0.0
        self._base_contact_buf[env_ids] = 0.0
        self._pitch_fail_buf[env_ids] = 0.0
        self._roll_fail_buf[env_ids] = 0.0
        self._leg_contact_buf[env_ids] = 0.0
        self._low_height_buf[env_ids] = 0.0
        self._commands[env_ids] = 0.0
        if self.cfg.commands.heading_command:
            forward_axis = torch.tensor(
                [1.0, 0.0, 0.0], device=self.device
            ).expand(len(env_ids), -1)
            # LAB quat_apply(q, v) 用 wxyz 四元数把机体系前向向量旋转到世界系。
            forward = quat_apply(default_root_state[:, 3:7], forward_axis)
            self._commands[env_ids, 3] = torch.atan2(forward[:, 1], forward[:, 0])
        self._resample_commands_for(env_ids)

        # LAB extras["log"] 会经 wrapper 交给 RSL-RL 记录；只是统计，不直接改奖励。
        extras = dict()
        for key in self._episode_sums.keys():
            extras["Episode_Reward/" + key] = self._episode_sums[key][env_ids].clone() / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/unsafe_contact_or_orientation"] = self.reset_terminated[env_ids].float()
        extras["Episode_Termination/time_out"] = self.reset_time_outs[env_ids].float()
        for reason, flags in self._termination_reasons.items():
            extras[f"Episode_Termination/{reason}"] = flags[env_ids].float()
        command_groups = {
            "standing": torch.abs(terminated_command_x) < 0.05,
            "reverse": terminated_command_x < -0.2,
            "reverse_fast": terminated_command_x < -0.5,
            "reverse_slow": (terminated_command_x >= -0.5) & (terminated_command_x < -0.2),
            "forward": terminated_command_x > 0.2,
            "forward_fast": terminated_command_x >= 0.6,
            "forward_slow": (terminated_command_x > 0.2) & (terminated_command_x < 0.6),
            "high_height": terminated_command_height >= (
                self.cfg.commands.ranges_height[0]
                + .75 * (self.cfg.commands.ranges_height[1] - self.cfg.commands.ranges_height[0])
            ),
            "target_reverse_fast": terminated_target_x < -0.5,
            "target_forward_fast": terminated_target_x >= 0.6,
        }
        for group, mask in command_groups.items():
            if torch.any(mask):
                extras[f"Episode_Termination/{group}_unsafe"] = self.reset_terminated[env_ids][mask].float()
        unsafe_mask = self.reset_terminated[env_ids]
        if torch.any(unsafe_mask):
            extras["Episode_Termination/unsafe_within_1s_of_command"] = (
                steps_since_resample[unsafe_mask] < int(1.0 / self.step_dt)
            ).float()
        self.extras["log"].update(extras)
