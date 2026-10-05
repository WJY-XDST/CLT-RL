# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause
#
# DirectRLEnv for the wheel-legged robot using Virtual Model Control (VMC).
# Faithfully migrated from `wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py`
# and `wheel_legged_gym/envs/base/legged_robot.py` (Isaac Gym Preview 4) to Isaac Lab.

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, wrap_to_pi

from wheel_legged_gym_isaaclab.vmc import leg_coordinates, virtual_leg_torques
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state, five_bar_torques, five_bar_inverse
from wheel_legged_gym_isaaclab.yaw_training import (
    TASK_NAMES, sample_grouped_motion, sample_yaw_rates, spin_center_velocity_penalty,
)

from .wheel_legged_vmc_flat_env_cfg import WheelLeggedVMCFlatEnvCfg


class WheelLeggedVMCFlatEnv(DirectRLEnv):
    """Wheel-legged robot environment with VMC control on flat terrain."""

    cfg: WheelLeggedVMCFlatEnvCfg

    def __init__(
        self, cfg: WheelLeggedVMCFlatEnvCfg, render_mode: str | None = None, **kwargs
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # DirectRLEnv exposes Gymnasium spaces in Isaac Lab 2.3 rather than
        # the deprecated ``num_actions`` attribute.
        self._action_dim = int(self.single_action_space.shape[0])
        self._validate_configuration()

        # Isaac Sim's articulation order can differ from the URDF/action order.
        # Keep each lookup in the requested logical order and map torque columns
        # explicitly when sending them to the articulation.
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

        # body indices for contact handling
        self._base_id, _ = self._contact_sensor.find_bodies("base_link")
        # penalised contacts: left/right leg links + base
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
        # Action/torque order is [left B, left L, left wheel,
        # right B, right L, right wheel] for the five-bar robot.
        self._torque_limits[:, (0, 1, 3, 4)] = self.cfg.leg_effort_limit
        self._torque_limits[:, (2, 5)] = self.cfg.wheel_effort_limit

        # Keep both the Gaussian policy output and the physically applied
        # action. The latter is always clamped to the safe normalized range.
        self._raw_actions = torch.zeros(
            self.num_envs, self._action_dim, device=self.device
        )
        self._actions = torch.zeros(
            self.num_envs, self._action_dim, device=self.device
        )
        self._last_actions = torch.zeros(
            self.num_envs, self._action_dim, 2, device=self.device
        )

        # commands: [lin_vel_x, ang_vel_yaw (or heading-error), height, heading]
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
        if self.cfg.commands.grouped_training:
            counts = torch.bincount(torch.arange(self.num_envs) % len(TASK_NAMES), minlength=len(TASK_NAMES))
            print(f"[INFO] Command task groups: {dict(zip(TASK_NAMES, counts.tolist()))}", flush=True)
        # obs scale for commands: [lin_vel, ang_vel, height]
        self._commands_scale = torch.tensor(
            [
                self.cfg.obs_scales.lin_vel,
                self.cfg.obs_scales.ang_vel,
                self.cfg.obs_scales.height_measurements,
            ],
            device=self.device,
        )

        # VMC forward-kinematics state
        self.pi = torch.acos(torch.zeros(1, device=self.device)) * 2
        self._theta1 = torch.zeros(self.num_envs, 2, device=self.device)
        self._theta2 = torch.zeros(self.num_envs, 2, device=self.device)
        self._theta0 = torch.zeros(self.num_envs, 2, device=self.device)
        self._theta0_dot = torch.zeros(self.num_envs, 2, device=self.device)
        self._L0 = torch.zeros(self.num_envs, 2, device=self.device)
        self._L0_dot = torch.zeros(self.num_envs, 2, device=self.device)
        self._five_bar_jacobian = torch.zeros(self.num_envs, 2, 2, 2, device=self.device)
        self._five_bar_valid = torch.ones(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._l0_ref_applied = torch.zeros(self.num_envs, 2, device=self.device)

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
            for key in [
                "track_lin_vel",
                "track_lin_vel_enhance",
                "track_lin_vel_precise",
                "lin_vel_error_sq",
                "standing_velocity",
                "base_height_error_sq",
                "track_ang_vel",
                "yaw_rate_error_sq",
                "spin_center_velocity",
                "base_height",
                "nominal_state",
                "standing_leg_angle",
                "standing_wheel_tracking",
                "lin_vel_z",
                "ang_vel_xy",
                "orientation",
                "dof_vel",
                "dof_acc",
                "torques",
                "action_rate",
                "action_smooth",
                "leg_length_action_saturation",
                "leg_length_action_difference",
                "leg_length_target_underreach",
                "collision",
                "dof_pos_limits",
                "termination",
            ]
        }
        self._last_reward_terms: dict[str, torch.Tensor] = {}

        # base height from terrain (flat ground at z=0)
        self._base_height = self._robot.data.root_pos_w[:, 2]

    def _validate_configuration(self):
        """Fail early when coupled command/action settings become inconsistent."""
        if self._action_dim != 6:
            raise ValueError(f"WheelLeggedVMC expects 6 actions, got {self._action_dim}.")
        if self.cfg.leg_control_mode not in ('explicit_vmc','implicit_joint_reference'):
            raise ValueError('Unknown leg control mode')
        if self.cfg.leg_control_mode=='implicit_joint_reference':
            if self.cfg.leg_model!='mine_five_bar' or self.cfg.leg_joint_stiffness<=0 or self.cfg.leg_joint_damping<0:
                raise ValueError('Implicit leg references require the CAD five-bar and positive joint stiffness')
        if not 0.0 <= self.cfg.reset_velocity_initial <= self.cfg.reset_velocity_final:
            raise ValueError("Reset velocity magnitudes must satisfy 0 <= initial <= final.")
        if self.cfg.rewards.termination > 0.0:
            raise ValueError("Unsafe termination must not receive a positive reward.")

        mapped_l0_min = self.cfg.l0_offset - self.cfg.action_scale_l0
        mapped_l0_max = self.cfg.l0_offset + self.cfg.action_scale_l0
        if (
            abs(mapped_l0_min - self.cfg.l0_ref_min) > 1.0e-6
            or abs(mapped_l0_max - self.cfg.l0_ref_max) > 1.0e-6
        ):
            raise ValueError(
                "Leg-length action map must cover the physical range exactly: "
                f"mapped=[{mapped_l0_min:.6f}, {mapped_l0_max:.6f}], "
                f"limits=[{self.cfg.l0_ref_min:.6f}, {self.cfg.l0_ref_max:.6f}]."
            )

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
            if yaw_low == yaw_high == 0.0:
                raise ValueError("Grouped turning training requires nonzero yaw ranges.")
            if not (self.cfg.commands.ranges_transition_lin_vel_x[0] < 0 < self.cfg.commands.ranges_transition_lin_vel_x[1]):
                raise ValueError("Grouped low-speed practice requires both forward and reverse transition ranges.")
        if not (0 <= self.cfg.commands.yaw_env_fraction <= 1 and 0 <= self.cfg.commands.yaw_boundary_fraction <= 1):
            raise ValueError("Yaw population and boundary fractions must be in [0, 1].")
        if self.cfg.rewards.tracking_sigma_ang <= 0 or self.cfg.rewards.spin_center_velocity > 0:
            raise ValueError("Yaw tracking width must be positive and spin drift penalty nonpositive.")
        transition_fraction = float(self.cfg.commands.transition_env_fraction)
        reverse_fraction = float(self.cfg.commands.reverse_env_fraction)
        if not 0.0 <= standing_fraction < 1.0:
            raise ValueError(
                "standing_env_fraction must satisfy 0 <= value < 1, "
                f"got {standing_fraction}."
            )
        if transition_fraction < 0.0 or standing_fraction + transition_fraction > 1.0:
            raise ValueError(
                "Command population fractions must satisfy "
                "standing_env_fraction + transition_env_fraction <= 1, "
                f"got {standing_fraction + transition_fraction}."
            )
        if (
            reverse_fraction < 0.0
            or standing_fraction + transition_fraction + reverse_fraction > 1.0
        ):
            raise ValueError(
                "Command fractions must satisfy standing + transition + reverse <= 1, "
                f"got {standing_fraction + transition_fraction + reverse_fraction}."
            )
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
        fixed_evaluation_command = main_min == main_max
        if not fixed_evaluation_command and not (
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
        if not fixed_evaluation_command and not (
            height_min <= self.cfg.rewards.base_height_target <= height_max
        ):
            raise ValueError(
                "base_height_target must lie inside the training height range, "
                f"got target={self.cfg.rewards.base_height_target}, "
                f"range={self.cfg.commands.ranges_height}."
            )
        if (
            (self.cfg.leg_model == "legacy_serial" and self.cfg.rewards.leg_length_target_height_offset < 0.0)
            or (
                not fixed_evaluation_command
                and height_max + self.cfg.rewards.leg_length_target_height_offset
                > self.cfg.l0_ref_max
            )
        ):
            raise ValueError(
                "The minimum leg-length reference must fit inside the VMC range."
            )
        if not (
            self.cfg.l0_ref_min
            <= self.cfg.rewards.leg_length_target_cap
            <= self.cfg.l0_ref_max
        ):
            raise ValueError("leg_length_target_cap must be inside the VMC range.")
        if not (
            self.cfg.l0_ref_min
            <= self.cfg.forward_support_min_leg_length
            <= self.cfg.l0_ref_max
        ):
            raise ValueError("forward_support_min_leg_length must be inside the VMC range.")
        if self.cfg.leg_model == "legacy_serial" and self.cfg.forward_support_height_margin < 0.0:
            raise ValueError("forward_support_height_margin must be non-negative.")
        if self.cfg.height_feedback_gain < 0.0 or self.cfg.height_feedback_max_adjustment < 0.0:
            raise ValueError("Height feedback gain and adjustment limit must be non-negative.")
        if not 0.0 <= self.cfg.height_reference_blend <= 1.0:
            raise ValueError("height_reference_blend must be between zero and one.")
        if self.cfg.leg_model == "legacy_serial" and self.cfg.leg_length_height_offset < 0.0:
            raise ValueError("leg_length_height_offset must be non-negative.")
        if not fixed_evaluation_command and not (
            main_min <= self.cfg.forward_support_speed_threshold <= main_max
        ):
            raise ValueError("forward_support_speed_threshold must be in the forward command range.")

        max_command_speed = float(
            max(
                abs(transition_min),
                abs(transition_max),
                abs(main_min),
                abs(main_max),
                abs(reverse_min),
                abs(reverse_max),
            )
        )
        max_wheel_speed = self.cfg.wheel_radius * self.cfg.action_scale_vel
        if max_command_speed > max_wheel_speed + 1.0e-6:
            raise ValueError(
                "Linear command magnitude exceeds the ideal wheel-speed action range: "
                f"command_magnitude={max_command_speed:.3f} m/s, "
                f"reachable={max_wheel_speed:.3f} m/s."
            )

        if self.cfg.rewards.base_height_sigma <= 0.0:
            raise ValueError("base_height_sigma must be greater than zero.")
        if self.cfg.rewards.nominal_state_penalty_clip <= 0.0:
            raise ValueError("nominal_state_penalty_clip must be greater than zero.")
        if self.cfg.rewards.orientation_penalty_clip <= 0.0:
            raise ValueError("orientation_penalty_clip must be greater than zero.")
        for name in ('lin_vel_penalty_clip','yaw_rate_penalty_clip','standing_wheel_penalty_clip','action_rate_penalty_clip'):
            if not math.isfinite(getattr(self.cfg.rewards,name)) or getattr(self.cfg.rewards,name) <= 0.:
                raise ValueError(f'{name} must be finite and positive.')
        if self.cfg.rewards.tracking_sigma <= 0.0:
            raise ValueError(
                "tracking_sigma must be greater than zero, "
                f"got {self.cfg.rewards.tracking_sigma}."
            )
        if self.cfg.rewards.tracking_sigma_enhance <= 0.0:
            raise ValueError("tracking_sigma_enhance must be greater than zero.")
        if self.cfg.rewards.tracking_sigma_precise <= 0.0:
            raise ValueError(
                "tracking_sigma_precise must be greater than zero, "
                f"got {self.cfg.rewards.tracking_sigma_precise}."
            )
        if self.cfg.rewards.orientation_roll_multiplier <= 0.0:
            raise ValueError(
                "orientation_roll_multiplier must be greater than zero, "
                f"got {self.cfg.rewards.orientation_roll_multiplier}."
            )
        if self.cfg.wheel_radius <= 0.0 or self.cfg.wheel_track_width <= 0.0:
            raise ValueError(
                "wheel_radius and wheel_track_width must both be greater than zero, "
                f"got radius={self.cfg.wheel_radius}, track={self.cfg.wheel_track_width}."
            )

    def _setup_scene(self):
        if self.cfg.leg_control_mode=='implicit_joint_reference':
            self.cfg.robot.actuators['legs'].stiffness=self.cfg.leg_joint_stiffness
            self.cfg.robot.actuators['legs'].damping=self.cfg.leg_joint_damping
        self._robot = Articulation(self.cfg.robot)
        if self.cfg.leg_control_mode=='implicit_joint_reference':
            from pxr import UsdPhysics
            for name in self.cfg.leg_joint_names:
                prim=self.sim.stage.GetPrimAtPath(f'/World/envs/env_0/Robot/joints/{name}')
                if not prim.IsA(UsdPhysics.RevoluteJoint):
                    raise ValueError(f'Leg drive joint not found: {name}')
                drive=UsdPhysics.DriveAPI.Apply(prim,'angular')
                drive.CreateTypeAttr('force')
                drive.CreateStiffnessAttr(self.cfg.leg_joint_stiffness)
                drive.CreateDampingAttr(self.cfg.leg_joint_damping)
                drive.CreateMaxForceAttr(self.cfg.leg_effort_limit)
        if self.cfg.wheel_control_mode == "implicit_velocity":
            from pxr import UsdPhysics
            # The CAD export deliberately removes importer drives. Author only
            # the two requested wheel drives before cloning/starting physics.
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
        self.scene.articulations["robot"] = self._robot
        from wheel_legged_gym_isaaclab.mine_collision_filters import relocate_collision_groups
        relocate_collision_groups(self.sim.stage, self.scene.env_prim_paths)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Step / control
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        # PPO uses an unbounded Gaussian action distribution.  Clamp at the
        # environment boundary as a final safety guard so virtual leg-length,
        # leg-angle, and wheel-speed references always stay in their designed
        # normalized range.
        self._raw_actions = torch.nan_to_num(
            actions, nan=0.0, posinf=1.0, neginf=-1.0
        )
        self._actions = torch.clamp(self._raw_actions, -1.0, 1.0)

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

    def _apply_action(self):
        """Compute VMC joint torques and apply them as effort targets.

        Called once per physics substep (200 Hz). Mirrors the original
        `leg_post_physics_step` + `_compute_torques` inside the decimation loop.
        """
        self._update_forward_kinematics()

        wheel_vel = self._get_wheel_vel_forward_positive()

        # reference targets from actions
        theta0_ref = (
            torch.cat(
                (
                    self._actions[:, 0].unsqueeze(1),
                    self._actions[:, 3].unsqueeze(1),
                ),
                dim=1,
            )
            * self.cfg.action_scale_theta
        )
        # Keep the virtual-leg angle target inside its physical safety range
        # even if action scaling is changed independently in the future.
        theta0_ref = torch.clamp(
            theta0_ref,
            min=self.cfg.theta0_ref_min,
            max=self.cfg.theta0_ref_max,
        )
        # Rebuild from the policy output each substep: _actions records the
        # applied reference, so using it here would accumulate the correction.
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
        # The normalized action clamp is not sufficient by itself because
        # scale/offset values may change. Enforce the physical reference range
        # independently at the VMC boundary.
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
        self._l0_ref_applied = l0_ref
        self._actions[:, (1, 4)] = (
            l0_ref - self.cfg.l0_offset
        ) / self.cfg.action_scale_l0
        wheel_vel_ref = (
            torch.cat(
                (
                    self._actions[:, 2].unsqueeze(1),
                    self._actions[:, 5].unsqueeze(1),
                ),
                dim=1,
            )
            * self.cfg.action_scale_vel
        )

        # VMC leg control (PD on virtual coordinates theta0 / L0)
        torque_leg = (
            self.cfg.kp_theta * (theta0_ref - self._theta0)
            - self.cfg.kd_theta * self._theta0_dot
        )
        force_leg = (
            self.cfg.kp_l0 * (l0_ref - self._L0) - self.cfg.kd_l0 * self._L0_dot
        )
        # wheel velocity damping control
        if self.cfg.wheel_control_mode == "implicit_velocity":
            # The velocity targets use raw mirrored joint coordinates. PhysX
            # applies the actuator's damping and effort limit implicitly.
            raw_wheel_ref = wheel_vel_ref * wheel_vel_ref.new_tensor([1.0, -1.0])
            self._robot.set_joint_velocity_target(raw_wheel_ref, joint_ids=self._wheel_joint_ids)
            torque_wheel = torch.zeros_like(wheel_vel_ref)
        else:
            torque_wheel = self.cfg.wheel_damping * (
                wheel_vel_ref - wheel_vel
            )

        # map virtual forces/torques to actual joint torques (closed chain)
        T1, T2 = self._vmc(force_leg + self.cfg.feedforward_force, torque_leg)

        right_leg_sign = 1.0 if self.cfg.leg_model == "mine_five_bar" else -1.0
        torques = torch.cat(
            (
                T1[:, 0].unsqueeze(1),
                T2[:, 0].unsqueeze(1),
                torque_wheel[:, 0].unsqueeze(1),
                right_leg_sign*T1[:, 1].unsqueeze(1),
                right_leg_sign*T2[:, 1].unsqueeze(1),
                # Convert the canonical forward-positive right-wheel torque
                # back to its mirrored raw joint coordinate.
                -torque_wheel[:, 1].unsqueeze(1),
            ),
            dim=1,
        )
        torques = torch.clip(torques, -self._torque_limits, self._torque_limits)
        if self.cfg.leg_control_mode=='implicit_joint_reference':
            # External loop constraints and joint drive forces are solved together
            # by PhysX. This is a different actuator contract from explicit VMC;
            # the same virtual-leg action references and raw CAD axes are retained.
            b,l,_=five_bar_inverse(l0_ref,theta0_ref,**self.cfg.five_bar_geometry)
            target=torch.stack((b[:,0],l[:,0],b[:,1],l[:,1]),-1)
            self._robot.set_joint_position_target(target,joint_ids=self._leg_joint_ids)
            torques[:,(0,1,3,4)]=0.
        self._robot.set_joint_effort_target(torques, joint_ids=self._torque_joint_ids)

    def _update_forward_kinematics(self, env_ids: torch.Tensor | None = None):
        """Update VMC virtual-coordinate states from current joint states."""
        dof_pos = self._robot.data.joint_pos
        dof_vel = self._robot.data.joint_vel
        if env_ids is not None:
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]

        if self.cfg.leg_model == "mine_five_bar":
            q_b = dof_pos[:, self._leg_joint_ids[::2]]
            q_l = dof_pos[:, self._leg_joint_ids[1::2]]
            length,angle,jacobian,valid = five_bar_state(q_b,q_l,**self.cfg.five_bar_geometry)
            velocity = torch.stack((dof_vel[:, self._leg_joint_ids[::2]],
                                    dof_vel[:, self._leg_joint_ids[1::2]]),dim=-1)
            virtual_velocity = (jacobian@velocity[...,None]).squeeze(-1)
            selection = slice(None) if env_ids is None else env_ids
            self._theta1[selection] = q_b
            self._theta2[selection] = q_l
            self._L0[selection] = length
            self._theta0[selection] = angle
            self._L0_dot[selection] = virtual_velocity[...,0]
            self._theta0_dot[selection] = virtual_velocity[...,1]
            self._five_bar_jacobian[selection] = jacobian
            self._five_bar_valid[selection] = valid
            return

        theta1 = torch.cat(
            (
                dof_pos[:, self._leg_joint_ids[0]].unsqueeze(1),
                -dof_pos[:, self._leg_joint_ids[2]].unsqueeze(1),
            ),
            dim=1,
        )
        theta2 = torch.cat(
            (
                (dof_pos[:, self._leg_joint_ids[1]] + self.pi / 2).unsqueeze(1),
                (-dof_pos[:, self._leg_joint_ids[3]] + self.pi / 2).unsqueeze(1),
            ),
            dim=1,
        )
        theta1_dot = torch.cat(
            (
                dof_vel[:, self._leg_joint_ids[0]].unsqueeze(1),
                -dof_vel[:, self._leg_joint_ids[2]].unsqueeze(1),
            ),
            dim=1,
        )
        theta2_dot = torch.cat(
            (
                dof_vel[:, self._leg_joint_ids[1]].unsqueeze(1),
                -dof_vel[:, self._leg_joint_ids[3]].unsqueeze(1),
            ),
            dim=1,
        )

        L0, theta0 = self._forward_kinematics(theta1, theta2)
        # finite-difference approximation of virtual velocities (dt=0.001 as original)
        dt = 0.001
        L0_temp, theta0_temp = self._forward_kinematics(
            theta1 + theta1_dot * dt, theta2 + theta2_dot * dt
        )
        L0_dot = (L0_temp - L0) / dt
        theta0_dot = (theta0_temp - theta0) / dt

        if env_ids is None:
            self._theta1 = theta1
            self._theta2 = theta2
            self._theta0 = theta0
            self._theta0_dot = theta0_dot
            self._L0 = L0
            self._L0_dot = L0_dot
        else:
            self._theta1[env_ids] = theta1
            self._theta2[env_ids] = theta2
            self._theta0[env_ids] = theta0
            self._theta0_dot[env_ids] = theta0_dot
            self._L0[env_ids] = L0
            self._L0_dot[env_ids] = L0_dot

    def _forward_kinematics(self, theta1: torch.Tensor, theta2: torch.Tensor):
        return leg_coordinates(
            theta1, theta2, l1=self.cfg.l1, l2=self.cfg.l2, offset=self.cfg.offset
        )

    def _vmc(self, F: torch.Tensor, T: torch.Tensor):
        """Map virtual force F and torque T to joint torques T1, T2 (per leg side)."""
        if self.cfg.leg_model == "mine_five_bar":
            return five_bar_torques(self._five_bar_jacobian,F,T)
        return virtual_leg_torques(
            self._theta1, self._theta2, self._L0, self._theta0, F, T,
            l1=self.cfg.l1, l2=self.cfg.l2,
            legacy_angular_mapping=self.cfg.vmc_legacy_angular_mapping,
        )

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
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
                self._sanitize_observation(
                    self._theta0, self.cfg.obs_clip.theta
                ) * self.cfg.obs_scales.dof_pos,
                self._sanitize_observation(
                    self._theta0_dot, self.cfg.obs_clip.theta_dot
                ) * self.cfg.obs_scales.dof_vel,
                self._sanitize_observation(
                    self._L0, self.cfg.obs_clip.leg_length
                ) * self.cfg.obs_scales.l0,
                self._sanitize_observation(
                    self._L0_dot, self.cfg.obs_clip.leg_length_dot
                ) * self.cfg.obs_scales.l0_dot,
                # Wheel joint angles are continuous and unbounded, so they are
                # not a stationary policy input. Preserve the 27-D interface
                # while exposing yaw-aligned horizontal body velocity.
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
        return {"policy": obs}

    def _update_commands(self):
        """Resample commands for environments that reached the resampling time."""
        if self.cfg.commands.grouped_training and self.cfg.commands.ranges_lin_vel_x[0] != self.cfg.commands.ranges_lin_vel_x[1]:
            phase = self.common_step_counter // max(1, int(self.cfg.commands.resampling_time / self.step_dt))
            if phase != self._last_grouped_command_phase:
                self._last_grouped_command_phase = phase
                # Rotate all groups together, retaining exact population counts while
                # teaching starts/stops, reversals and transitions into/out of turns.
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

    # ------------------------------------------------------------------
    # Rewards (ported from legged_robot.py reward functions)
    # ------------------------------------------------------------------
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

        # Keep the source formula, but measure forward speed in the yaw-aligned
        # horizontal frame so pitch and roll do not redefine forward velocity.
        lin_vel_error = torch.square(
            self._commands[:, 0] - heading_velocity[:, 0]
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
            (self._get_wheel_vel_forward_positive() - wheel_reference).square(), dim=1
        ) * (standing & (self._commands[:, 1].abs() < 0.01))
        ang_vel_error = torch.square(self._commands[:, 1] - root_ang_vel_b[:, 2])
        r_track_ang = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma_ang)
        r_spin_center = spin_center_velocity_penalty(
            self._get_wheel_vel_forward_positive(), self.cfg.wheel_radius,
            self._commands[:, 0], self._target_lin_vel_x, self._commands[:, 1],
        )

        base_height_error = torch.square(
            self._robot.data.root_pos_w[:, 2] - self._commands[:, 2]
        )
        if self.cfg.rewards.base_height < 0:
            r_base_height = torch.sqrt(base_height_error)
        else:
            r_base_height = torch.exp(
                -base_height_error / self.cfg.rewards.base_height_sigma
            )

        theta_difference = torch.square(self._theta0[:, 0] - self._theta0[:, 1])
        if self.cfg.rewards.nominal_state < 0:
            r_nominal_state = theta_difference
        else:
            r_nominal_state = torch.exp(-theta_difference / 0.1)

        r_lin_vel_z = torch.square(root_lin_vel_b[:, 2])
        r_ang_vel_xy = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
        # For projected gravity in the body frame, x is primarily pitch and y
        # is primarily roll. Penalize roll more strongly while still allowing
        # unequal leg lengths when terrain later requires them.
        r_orientation = torch.square(proj_gravity[:, 0]) + (
            self.cfg.rewards.orientation_roll_multiplier
            * torch.square(proj_gravity[:, 1])
        )
        r_dof_vel = torch.sum(
            torch.square(joint_vel[:, self._leg_joint_ids]), dim=1
        )
        r_dof_acc = torch.sum(torch.square(joint_acc), dim=1)
        r_torques = torch.sum(torch.square(torque), dim=1)
        r_action_rate = torch.sum(
            torch.square(self._last_actions[:, :, 0] - self._actions), dim=1
        )
        r_action_smooth = torch.sum(
            torch.square(
                self._actions[:, :2]
                - 2 * self._last_actions[:, :2, 0]
                + self._last_actions[:, :2, 1]
            ),
            dim=1,
        ) + torch.sum(
            torch.square(
                self._actions[:, 3:5]
                - 2 * self._last_actions[:, 3:5, 0]
                + self._last_actions[:, 3:5, 1]
            ),
            dim=1,
        )
        # Use the pre-clipped policy output so actions above +/-1 remain
        # distinguishable to PPO. The wide clamp is only a numerical guard.
        raw_leg_length_actions = torch.clamp(
            self._raw_actions[:, (1, 4)], -10.0, 10.0
        )
        leg_length_action_excess = torch.relu(
            torch.abs(raw_leg_length_actions)
            - self.cfg.rewards.leg_length_action_soft_limit
        )
        r_leg_length_action_saturation = torch.sum(
            torch.square(leg_length_action_excess), dim=1
        )
        r_leg_length_action_difference = torch.square(
            self._actions[:, 1] - self._actions[:, 4]
        )
        leg_length_refs = (
            torch.clamp(self._raw_actions[:, (1, 4)], -1.0, 1.0)
            * self.cfg.action_scale_l0
            + self.cfg.l0_offset
        )
        minimum_leg_length_ref = (
            self._commands[:, 2:3]
            + self.cfg.rewards.leg_length_target_height_offset
        )
        minimum_leg_length_ref = torch.clamp(
            minimum_leg_length_ref,
            max=self.cfg.rewards.leg_length_target_cap,
        )
        r_leg_length_target_underreach = torch.mean(
            torch.square(torch.relu(minimum_leg_length_ref - leg_length_refs)),
            dim=1,
        )

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

        rewards = {
            "track_lin_vel": r_track_lin
            * self.cfg.rewards.tracking_lin_vel
            * self.step_dt,
            "track_lin_vel_enhance": r_track_lin_enhance
            * self.cfg.rewards.tracking_lin_vel_enhance
            * self.step_dt,
            "track_lin_vel_precise": r_track_lin_precise
            * self.cfg.rewards.tracking_lin_vel_precise
            * self.step_dt,
            "lin_vel_error_sq": lin_vel_error
            * self.cfg.rewards.lin_vel_error_sq
            * self.step_dt,
            "standing_velocity": r_standing_velocity
            * self.cfg.rewards.standing_velocity
            * self.step_dt,
            "base_height_error_sq": base_height_error
            * self.cfg.rewards.base_height_error_sq
            * self.step_dt,
            "track_ang_vel": r_track_ang
            * self.cfg.rewards.tracking_ang_vel
            * self.step_dt,
            "yaw_rate_error_sq": ang_vel_error
            * self.cfg.rewards.yaw_rate_error_sq
            * self.step_dt,
            "spin_center_velocity": r_spin_center * self.cfg.rewards.spin_center_velocity * self.step_dt,
            "base_height": r_base_height
            * self.cfg.rewards.base_height
            * self.step_dt,
            "nominal_state": r_nominal_state
            * self.cfg.rewards.nominal_state
            * self.step_dt,
            "lin_vel_z": r_lin_vel_z * self.cfg.rewards.lin_vel_z * self.step_dt,
            "ang_vel_xy": r_ang_vel_xy * self.cfg.rewards.ang_vel_xy * self.step_dt,
            "orientation": r_orientation
            * self.cfg.rewards.orientation
            * self.step_dt,
            "dof_vel": r_dof_vel * self.cfg.rewards.dof_vel * self.step_dt,
            "dof_acc": r_dof_acc * self.cfg.rewards.dof_acc * self.step_dt,
            "torques": r_torques * self.cfg.rewards.torques * self.step_dt,
            "action_rate": r_action_rate
            * self.cfg.rewards.action_rate
            * self.step_dt,
            "action_smooth": r_action_smooth
            * self.cfg.rewards.action_smooth
            * self.step_dt,
            "leg_length_action_saturation": r_leg_length_action_saturation
            * self.cfg.rewards.leg_length_action_saturation
            * self.step_dt,
            "leg_length_action_difference": r_leg_length_action_difference
            * self.cfg.rewards.leg_length_action_difference
            * self.step_dt,
            "leg_length_target_underreach": r_leg_length_target_underreach
            * self.cfg.rewards.leg_length_target_underreach
            * self.step_dt,
            "collision": r_collision * self.cfg.rewards.collision * self.step_dt,
            "dof_pos_limits": r_dof_pos_limits
            * self.cfg.rewards.dof_pos_limits
            * self.step_dt,
        }

        # Keep wider negative bounds for symmetry and attitude penalties so
        # their increased weights retain useful differences at observed errors.
        reward_bound = self.cfg.rewards.clip_single_reward * self.step_dt
        penalty_bounds = {
            "nominal_state": self.cfg.rewards.nominal_state_penalty_clip * self.step_dt,
            "orientation": self.cfg.rewards.orientation_penalty_clip * self.step_dt,
            "base_height_error_sq": self.cfg.rewards.base_height_penalty_clip * self.step_dt,
            "standing_leg_angle": self.cfg.rewards.standing_leg_angle_penalty_clip * self.step_dt,
            "lin_vel_error_sq": self.cfg.rewards.lin_vel_penalty_clip * self.step_dt,
            "yaw_rate_error_sq": self.cfg.rewards.yaw_rate_penalty_clip * self.step_dt,
            "standing_wheel_tracking": self.cfg.rewards.standing_wheel_penalty_clip * self.step_dt,
            "action_rate": self.cfg.rewards.action_rate_penalty_clip * self.step_dt,
        }
        rewards["standing_leg_angle"] = r_standing_leg_angle * self.cfg.rewards.standing_leg_angle * self.step_dt
        rewards["standing_wheel_tracking"] = r_standing_wheel_tracking * self.cfg.rewards.standing_wheel_tracking * self.step_dt
        if self.cfg.leg_model == 'mine_five_bar' and self.common_step_counter % 100 == 0:
            # Record actual training commands and measured velocity, rather
            # than infer tracking from the sum of clipped reward terms.
            diagnostics={}
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
            diagnostics['Tracking/wheel_reference_delta_rms_rad_s']=torch.sqrt(
                (self._actions[:,(2,5)]-self._last_actions[:,(2,5),0]).square().mean()
            ).item()*self.cfg.action_scale_vel
            for key in ('lin_vel_error_sq','yaw_rate_error_sq','standing_wheel_tracking','action_rate'):
                bound=penalty_bounds[key]
                diagnostics[f'Reward_Clipping/{key}']=(rewards[key]<=-bound).float().mean().item()
            self.extras['log'].update(diagnostics)
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
        # DirectRLEnv sets reset_terminated before calling _get_rewards. A
        # terminal event has a fixed cost regardless of the control timestep;
        # applying step_dt or the per-step cap would erase this learning signal.
        rewards["termination"] = self.reset_terminated.float() * self.cfg.rewards.termination
        self._last_reward_terms = {
            key: value.detach() for key, value in rewards.items()
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        contact_forces = self._contact_sensor.data.net_forces_w

        # Terminate base-link contact independently and much sooner than the
        # general fallen-posture condition. Intermittent contacts reset the
        # timer so a single impact does not immediately end the episode.
        base_force = torch.norm(contact_forces[:, self._base_id, :], dim=-1)
        base_contact = torch.any(
            base_force > self.cfg.base_contact_force_threshold, dim=1
        )
        self._base_contact_buf = torch.where(
            base_contact, self._base_contact_buf + 1.0, 0.0
        )
        base_contact_died = self._base_contact_buf > (
            self.cfg.base_contact_terminal_time_s / self.step_dt
        )

        # Keep the original delayed termination for a nearly inverted body.
        fallen = self._robot.data.projected_gravity_b[:, 2] > -0.1
        self._fail_buf = torch.where(fallen, self._fail_buf + 1.0, 0.0)
        fallen_died = self._fail_buf > (
            self.cfg.fail_to_terminal_time_s / self.step_dt
        )

        # Terminate sustained excessive pitch before the policy can exploit a
        # strongly leaned body as a normal forward-driving posture.
        body_pitch = torch.abs(
            torch.asin(
                torch.clamp(
                    self._robot.data.projected_gravity_b[:, 0], -1.0, 1.0
                )
            )
        )
        excessive_pitch = body_pitch > self.cfg.max_body_pitch
        self._pitch_fail_buf = torch.where(
            excessive_pitch, self._pitch_fail_buf + 1.0, 0.0
        )
        pitch_died = self._pitch_fail_buf > (
            self.cfg.body_pitch_terminal_time_s / self.step_dt
        )

        # Treat sustained lateral tilt as unsafe without constraining how the
        # two legs achieve a level body. The timer tolerates brief roll while
        # crossing an edge or height discontinuity.
        body_roll = torch.abs(
            torch.asin(
                torch.clamp(
                    self._robot.data.projected_gravity_b[:, 1], -1.0, 1.0
                )
            )
        )
        excessive_roll = body_roll > self.cfg.max_body_roll
        self._roll_fail_buf = torch.where(
            excessive_roll, self._roll_fail_buf + 1.0, 0.0
        )
        roll_died = self._roll_fail_buf >= (
            self.cfg.body_roll_terminal_time_s / self.step_dt
        )

        # Wheels are allowed to touch the ground; sustained contact by any leg
        # link indicates a collapsed or dragging posture.
        leg_force = torch.norm(
            contact_forces[:, self._leg_contact_ids, :], dim=-1
        )
        leg_contact = torch.any(
            leg_force > self.cfg.leg_contact_force_threshold, dim=1
        )
        self._leg_contact_buf = torch.where(
            leg_contact, self._leg_contact_buf + 1.0, 0.0
        )
        leg_contact_died = self._leg_contact_buf > (
            self.cfg.leg_contact_terminal_time_s / self.step_dt
        )
        # Net contact forces can intermittently drop below the threshold while
        # a collapsed chassis remains on the floor. Height is an independent guard.
        low_height = self._robot.data.root_pos_w[:, 2] < self.cfg.min_root_height
        self._low_height_buf = torch.where(low_height, self._low_height_buf + 1.0, 0.0)
        low_height_died = self._low_height_buf >= max(
            1.0, self.cfg.low_height_terminal_time_s / self.step_dt
        )
        # Reset numerical failures immediately before invalid physics can
        # contaminate the remaining vectorized environments or PPO rollout.
        numerical_failure = ~(
            torch.isfinite(self._robot.data.root_pos_w).all(dim=1)
            & torch.isfinite(self._robot.data.root_quat_w).all(dim=1)
            & torch.isfinite(self._robot.data.root_lin_vel_w).all(dim=1)
            & torch.isfinite(self._robot.data.root_ang_vel_w).all(dim=1)
            & torch.isfinite(self._robot.data.joint_pos).all(dim=1)
            & torch.isfinite(self._robot.data.joint_vel).all(dim=1)
            & torch.isfinite(contact_forces).all(dim=(1, 2))
        )
        singularity = ~self._five_bar_valid.all(dim=1)
        died = (
            base_contact_died
            | fallen_died
            | pitch_died
            | roll_died
            | leg_contact_died
            | numerical_failure
            | singularity
            | low_height_died
        )
        self._termination_reasons = {
            "base_contact": base_contact_died,
            "fallen": fallen_died,
            "pitch": pitch_died,
            "roll": roll_died,
            "leg_contact": leg_contact_died,
            "numerical": numerical_failure,
            "kinematic_singularity": singularity,
            "low_height": low_height_died,
        }

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        terminated_command_x = self._commands[env_ids, 0].clone()
        terminated_target_x = self._target_lin_vel_x[env_ids].clone()
        terminated_command_height = self._commands[env_ids, 2].clone()
        resample_steps = max(int(self.cfg.commands.resampling_time / self.step_dt), 1)
        steps_since_resample = self.episode_length_buf[env_ids].clone() % resample_steps
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # reset joint state to default
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        # reset root state
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # Start fresh training with gentle perturbations. Fixed-command replay
        # always tests the final disturbance range, not the easier first stage.
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
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._update_forward_kinematics(env_ids)

        # reset buffers
        self._raw_actions[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
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
            forward = quat_apply(default_root_state[:, 3:7], forward_axis)
            self._commands[env_ids, 3] = torch.atan2(forward[:, 1], forward[:, 0])
        self._resample_commands_for(env_ids)

        # episode logs
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/unsafe_contact_or_orientation"] = torch.mean(
            self.reset_terminated[env_ids].float()
        ).item()
        extras["Episode_Termination/time_out"] = torch.mean(
            self.reset_time_outs[env_ids].float()
        ).item()
        for reason, flags in self._termination_reasons.items():
            extras[f"Episode_Termination/{reason}"] = torch.mean(
                flags[env_ids].float()
            ).item()
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
                extras[f"Episode_Termination/{group}_unsafe"] = torch.mean(
                    self.reset_terminated[env_ids][mask].float()
                ).item()
        unsafe_mask = self.reset_terminated[env_ids]
        if torch.any(unsafe_mask):
            extras["Episode_Termination/unsafe_within_1s_of_command"] = torch.mean(
                (steps_since_resample[unsafe_mask] < int(1.0 / self.step_dt)).float()
            ).item()
        self.extras["log"].update(extras)

    def _resample_commands_for(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        previous_speed = self._commands[env_ids, 0].clone()

        lin_min = float(self._command_ranges[0, 0].item())
        lin_max = float(self._command_ranges[0, 1].item())

        # A collapsed linear-velocity range is used by play.py for deterministic
        # evaluation. Bypass the training curriculum so --fixed_command 0.5 ...
        # really evaluates 0.5 m/s immediately.
        if lin_min == lin_max:
            rand = torch.rand(env_ids.numel(), 3, device=self.device)
            self._commands[env_ids, :3] = (
                self._command_ranges[:, 1] - self._command_ranges[:, 0]
            )[None, :] * rand + self._command_ranges[:, 0][None, :]
        else:
            step = float(self.common_step_counter)
            standing_steps = float(self.cfg.commands.standing_only_steps)
            ramp_steps = max(float(self.cfg.commands.motion_ramp_steps), 1.0)
            progress = min(max((step - standing_steps) / ramp_steps, 0.0), 1.0)
            reverse_progress = min(
                max(
                    (step - self.cfg.commands.reverse_ramp_start_steps)
                    / self.cfg.commands.reverse_ramp_steps,
                    0.0,
                ),
                1.0,
            )
            moving_fraction = (
                1.0 - self.cfg.commands.standing_env_fraction
            ) * progress

            # Every resample starts as an explicit standing command. A growing
            # subset is then replaced with motion commands. Height is
            # sampled independently for every environment, including standing
            # samples, so height control is not coupled to forward motion.
            self._commands[env_ids, :3] = 0.0
            height_min, height_max = self.cfg.commands.ranges_height
            height_center = self.cfg.rewards.base_height_target
            progressive_height_min = height_center + (
                height_min - height_center
            ) * progress
            progressive_height_max = height_center + (
                height_max - height_center
            ) * progress
            self._commands[env_ids, 2] = progressive_height_min + (
                progressive_height_max - progressive_height_min
            ) * torch.rand(env_ids.numel(), device=self.device)
            height_sample = torch.rand(env_ids.numel(), device=self.device)
            half_boundary_fraction = self.cfg.commands.height_boundary_fraction / 2.0
            low_height_ids = env_ids[height_sample < half_boundary_fraction]
            high_height_ids = env_ids[
                (height_sample >= half_boundary_fraction)
                & (height_sample < 2.0 * half_boundary_fraction)
            ]
            self._commands[low_height_ids, 2] = progressive_height_min
            self._commands[high_height_ids, 2] = progressive_height_max
            if self.cfg.commands.grouped_training:
                yaw_progress = (step - self.cfg.commands.yaw_start_steps) / self.cfg.commands.yaw_ramp_steps
                speed, yaw = sample_grouped_motion(
                    env_ids, self.cfg.commands, progress, reverse_progress, yaw_progress,
                    phase=self.common_step_counter // max(1, int(self.cfg.commands.resampling_time / self.step_dt)),
                )
                self._commands[env_ids, 0] = speed
                self._commands[env_ids, 1] = yaw
            else:
                moving_mask = (
                    torch.rand(env_ids.numel(), device=self.device) < moving_fraction
                )
                moving_ids = env_ids[moving_mask]
                if moving_ids.numel() > 0:
                    # At full progress the population is split into explicit
                    # standing, bidirectional low-speed transition, and the main
                    # forward/reverse ranges.
                    non_standing_fraction = 1.0 - self.cfg.commands.standing_env_fraction
                    transition_share = (
                        self.cfg.commands.transition_env_fraction / non_standing_fraction
                    )
                    transition_mask = (
                        torch.rand(moving_ids.numel(), device=self.device) < transition_share
                    )
                    transition_ids = moving_ids[transition_mask]
                    main_ids = moving_ids[~transition_mask]

                    if transition_ids.numel() > 0:
                        transition_min, transition_max = (
                            self.cfg.commands.ranges_transition_lin_vel_x
                        )
                        progressive_transition_min = transition_min * progress
                        progressive_transition_max = (
                            transition_max * progress
                        )
                        self._commands[transition_ids, 0] = progressive_transition_min + (
                            progressive_transition_max - progressive_transition_min
                        ) * torch.rand(transition_ids.numel(), device=self.device)

                    if main_ids.numel() > 0:
                        main_fraction = (
                            non_standing_fraction
                            - self.cfg.commands.transition_env_fraction
                        )
                        reverse_share = (
                            self.cfg.commands.reverse_env_fraction
                            / max(main_fraction, 1.0e-6)
                            * reverse_progress
                        )
                        reverse_mask = (
                            torch.rand(main_ids.numel(), device=self.device) < reverse_share
                        )
                        reverse_ids = main_ids[reverse_mask]
                        forward_ids = main_ids[~reverse_mask]
                        progressive_vmax = lin_min + (lin_max - lin_min) * progress
                        if forward_ids.numel() > 0:
                            self._commands[forward_ids, 0] = lin_min + (
                                progressive_vmax - lin_min
                            ) * torch.rand(forward_ids.numel(), device=self.device)
                            boundary = (
                                torch.rand(forward_ids.numel(), device=self.device)
                                < self.cfg.commands.speed_boundary_fraction
                            )
                            self._commands[forward_ids[boundary], 0] = progressive_vmax
                        if reverse_ids.numel() > 0:
                            reverse_min, reverse_max = (
                                self.cfg.commands.ranges_reverse_lin_vel_x
                            )
                            progressive_reverse_min = reverse_max + (
                                reverse_min - reverse_max
                            ) * reverse_progress
                            self._commands[reverse_ids, 0] = progressive_reverse_min + (
                                reverse_max - progressive_reverse_min
                            ) * torch.rand(reverse_ids.numel(), device=self.device)
                            boundary = (
                                torch.rand(reverse_ids.numel(), device=self.device)
                                < self.cfg.commands.speed_boundary_fraction
                            )
                            self._commands[reverse_ids[boundary], 0] = progressive_reverse_min
                    yaw_min, yaw_max = self.cfg.commands.ranges_ang_vel_yaw
                    self._commands[moving_ids, 1] = yaw_min + (yaw_max - yaw_min) * torch.rand(
                        moving_ids.numel(), device=self.device
                    )
                yaw_min, yaw_max = self.cfg.commands.ranges_ang_vel_yaw
                if yaw_min != 0.0 or yaw_max != 0.0:
                    # Apply to ALL resampled environments, including zero-speed spins.
                    # Preserve zero-yaw samples for straight motion and stationary balance.
                    yaw_progress = (step - self.cfg.commands.yaw_start_steps) / self.cfg.commands.yaw_ramp_steps
                    self._commands[env_ids, 1] = sample_yaw_rates(
                        env_ids.numel(), device=self.device, low=yaw_min, high=yaw_max,
                        fraction=self.cfg.commands.yaw_env_fraction, minimum=self.cfg.commands.yaw_min_abs_rate,
                        boundary_fraction=self.cfg.commands.yaw_boundary_fraction,
                        progress=min(progress, yaw_progress),
                    )
        self._target_lin_vel_x[env_ids] = self._commands[env_ids, 0]
        self._commands[env_ids, 0] = previous_speed
