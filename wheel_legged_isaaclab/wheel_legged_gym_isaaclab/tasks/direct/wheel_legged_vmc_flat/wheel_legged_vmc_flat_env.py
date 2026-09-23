# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause
#
# DirectRLEnv for the wheel-legged robot using Virtual Model Control (VMC).
# Faithfully migrated from `wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py`
# and `wheel_legged_gym/envs/base/legged_robot.py` (Isaac Gym Preview 4) to Isaac Lab.

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, wrap_to_pi

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

        # Isaac Sim's articulation order can differ from the URDF/action order.
        # Keep each lookup in the requested logical order and map torque columns
        # explicitly when sending them to the articulation.
        leg_joint_names = ["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"]
        wheel_joint_names = ["l_wheel_Joint", "r_wheel_Joint"]
        torque_joint_names = [
            "lf0_Joint", "lf1_Joint", "l_wheel_Joint",
            "rf0_Joint", "rf1_Joint", "r_wheel_Joint",
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
            "(lf|rf|base).*"
        )
        leg_body_names = ["lf0_Link", "lf1_Link", "rf0_Link", "rf1_Link"]
        self._leg_contact_ids, resolved_leg_bodies = self._contact_sensor.find_bodies(
            leg_body_names, preserve_order=True
        )
        if resolved_leg_bodies != leg_body_names:
            raise ValueError(
                "Leg contact layout does not match the robot: "
                f"expected={leg_body_names}, resolved={resolved_leg_bodies}"
            )

        # torque limits (from URDF: legs 30 N*m, wheels 5 N*m)
        self._torque_limits = torch.zeros(self.num_envs, self._action_dim, device=self.device)
        # Action/torque order is [left hip, left knee, left wheel,
        # right hip, right knee, right wheel].
        self._torque_limits[:, (0, 1, 3, 4)] = 30.0
        self._torque_limits[:, (2, 5)] = 5.0

        # actions / previous actions
        self._actions = torch.zeros(
            self.num_envs, self._action_dim, device=self.device
        )
        self._last_actions = torch.zeros(
            self.num_envs, self._action_dim, 2, device=self.device
        )

        # commands: [lin_vel_x, ang_vel_yaw (or heading-error), height, heading]
        self._commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._command_ranges = torch.tensor(
            [
                list(self.cfg.commands.ranges_lin_vel_x),
                list(self.cfg.commands.ranges_ang_vel_yaw),
                list(self.cfg.commands.ranges_height),
            ],
            device=self.device,
        )
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

        # termination bookkeeping
        self._fail_buf = torch.zeros(self.num_envs, device=self.device)
        self._base_contact_buf = torch.zeros(self.num_envs, device=self.device)
        self._pitch_fail_buf = torch.zeros(self.num_envs, device=self.device)
        self._leg_contact_buf = torch.zeros(self.num_envs, device=self.device)

        # episode reward sums for logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel",
                "track_lin_vel_enhance",
                "lin_vel_error_sq",
                "track_ang_vel",
                "yaw_rate_error",
                "base_height",
                "nominal_state",
                "lin_vel_z",
                "ang_vel_xy",
                "orientation",
                "dof_vel",
                "dof_acc",
                "torques",
                "action_rate",
                "action_smooth",
                "collision",
                "dof_pos_limits",
                "leg_length_below_min",
                "base_height_below_target",
                "termination",
            ]
        }

        # base height from terrain (flat ground at z=0)
        self._base_height = self._robot.data.root_pos_w[:, 2]

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
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
        self._actions = torch.clamp(actions, -1.0, 1.0)

    def _apply_action(self):
        """Compute VMC joint torques and apply them as effort targets.

        Called once per physics substep (200 Hz). Mirrors the original
        `leg_post_physics_step` + `_compute_torques` inside the decimation loop.
        """
        self._update_forward_kinematics()

        dof_pos = self._robot.data.joint_pos
        dof_vel = self._robot.data.joint_vel

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
        l0_ref = (
            torch.cat(
                (
                    self._actions[:, 1].unsqueeze(1),
                    self._actions[:, 4].unsqueeze(1),
                ),
                dim=1,
            )
            * self.cfg.action_scale_l0
            + self.cfg.l0_offset
        )
        # The normalized action clamp is not sufficient by itself because
        # scale/offset values may change. Enforce the physical reference range
        # independently at the VMC boundary.
        l0_ref = torch.clamp(
            l0_ref, min=self.cfg.l0_ref_min, max=self.cfg.l0_ref_max
        )
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
        torque_wheel = self.cfg.wheel_damping * (
            wheel_vel_ref - dof_vel[:, self._wheel_joint_ids]
        )

        # map virtual forces/torques to actual joint torques (closed chain)
        T1, T2 = self._vmc(force_leg + self.cfg.feedforward_force, torque_leg)

        torques = torch.cat(
            (
                T1[:, 0].unsqueeze(1),
                T2[:, 0].unsqueeze(1),
                torque_wheel[:, 0].unsqueeze(1),
                -T1[:, 1].unsqueeze(1),
                -T2[:, 1].unsqueeze(1),
                torque_wheel[:, 1].unsqueeze(1),
            ),
            dim=1,
        )
        torques = torch.clip(torques, -self._torque_limits, self._torque_limits)
        self._robot.set_joint_effort_target(torques, joint_ids=self._torque_joint_ids)

    def _update_forward_kinematics(self):
        """Update VMC virtual-coordinate states from current joint states."""
        dof_pos = self._robot.data.joint_pos
        dof_vel = self._robot.data.joint_vel

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

        self._theta1 = theta1
        self._theta2 = theta2
        self._theta0 = theta0
        self._theta0_dot = theta0_dot
        self._L0 = L0
        self._L0_dot = L0_dot

    def _forward_kinematics(self, theta1: torch.Tensor, theta2: torch.Tensor):
        end_x = (
            self.cfg.offset
            + self.cfg.l1 * torch.cos(theta1)
            + self.cfg.l2 * torch.cos(theta1 + theta2)
        )
        end_y = self.cfg.l1 * torch.sin(theta1) + self.cfg.l2 * torch.sin(
            theta1 + theta2
        )
        L0 = torch.sqrt(end_x**2 + end_y**2)
        theta0 = torch.arctan2(end_y, end_x) - self.pi / 2
        return L0, theta0

    def _vmc(self, F: torch.Tensor, T: torch.Tensor):
        """Map virtual force F and torque T to joint torques T1, T2 (per leg side)."""
        theta0 = self._theta0 + self.pi / 2
        t11 = self.cfg.l1 * torch.sin(theta0 - self._theta1) - self.cfg.l2 * torch.sin(
            self._theta1 + self._theta2 - theta0
        )
        t12 = (
            self.cfg.l1 * torch.cos(theta0 - self._theta1)
            - self.cfg.l2 * torch.cos(self._theta1 + self._theta2 - theta0)
        ) / self._L0
        t21 = -self.cfg.l2 * torch.sin(self._theta1 + self._theta2 - theta0)
        t22 = (
            -self.cfg.l2 * torch.cos(self._theta1 + self._theta2 - theta0)
        ) / self._L0
        T1 = t11 * F - t12 * T
        T2 = t21 * F - t22 * T
        return T1, T2

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # command resampling (once every resampling_time seconds)
        self._update_commands()
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
                1.5 * wrap_to_pi(self._commands[:, 3] - heading), -5, 5
            )

        # base height (flat ground)
        self._base_height = self._robot.data.root_pos_w[:, 2]

        # store previous actions for reward computation
        self._last_actions[:, :, 1] = self._last_actions[:, :, 0]
        self._last_actions[:, :, 0] = self._actions[:]

        obs = torch.cat(
            (
                self._robot.data.root_ang_vel_b * self.cfg.obs_scales.ang_vel,
                self._robot.data.projected_gravity_b,
                self._commands[:, :3] * self._commands_scale,
                self._theta0 * self.cfg.obs_scales.dof_pos,
                self._theta0_dot * self.cfg.obs_scales.dof_vel,
                self._L0 * self.cfg.obs_scales.l0,
                self._L0_dot * self.cfg.obs_scales.l0_dot,
                self._robot.data.joint_pos[:, self._wheel_joint_ids]
                * self.cfg.obs_scales.dof_pos,
                self._robot.data.joint_vel[:, self._wheel_joint_ids]
                * self.cfg.obs_scales.dof_vel,
                self._actions,
            ),
            dim=-1,
        )
        return {"policy": obs}

    def _update_commands(self):
        """Resample commands for environments that reached the resampling time."""
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
        root_lin_vel_b = self._robot.data.root_lin_vel_b
        root_ang_vel_b = self._robot.data.root_ang_vel_b
        proj_gravity = self._robot.data.projected_gravity_b
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel
        joint_acc = self._robot.data.joint_acc
        torque = self._robot.data.applied_torque
        default_joint_pos = self._robot.data.default_joint_pos
        contact_forces = self._contact_sensor.data.net_forces_w

        # velocity tracking (exponential reward)
        lin_vel_error = torch.square(self._commands[:, 0] - root_lin_vel_b[:, 0])
        r_track_lin = torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)
        r_track_lin_enh = (
            torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma / 10) - 1
        )
        ang_vel_error = torch.square(self._commands[:, 1] - root_ang_vel_b[:, 2])
        r_track_ang = torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
        # The exponential tracking reward is nearly zero for a large yaw
        # error, which by itself provides little incentive to recover from a
        # persistent spin.  Keep the same error as an explicit dense cost.
        r_yaw_rate_error = ang_vel_error

        # base height
        base_height_error = torch.square(
            self._base_height - self._commands[:, 2]
        )
        if self.cfg.rewards.base_height >= 0:
            r_base_height = torch.exp(-base_height_error / 0.001)
        else:
            r_base_height = torch.sqrt(base_height_error)

        # nominal state (left/right theta0 symmetry)
        theta_diff = torch.square(self._theta0[:, 0] - self._theta0[:, 1])
        if self.cfg.rewards.nominal_state < 0:
            r_nominal = theta_diff
        else:
            r_nominal = torch.exp(-theta_diff / 0.1)

        # motion penalties
        r_lin_vel_z = torch.square(root_lin_vel_b[:, 2])
        r_ang_vel_xy = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
        r_orientation = torch.sum(torch.square(proj_gravity[:, :2]), dim=1)
        r_dof_vel = torch.sum(torch.square(joint_vel[:, self._leg_joint_ids]), dim=1)
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

        # collision penalty on leg links + base
        contact_force_norm = torch.norm(
            contact_forces[:, self._penalised_contact_ids, :], dim=-1
        )
        r_collision = torch.sum(
            1.0 * (contact_force_norm > self.cfg.collision_force_threshold), dim=1
        )

        # joint position limit penalty (legs)
        dof_pos_limits = self._robot.data.joint_pos_limits  # (num_envs, num_dof, 2)
        leg_pos = joint_pos[:, self._leg_joint_ids]
        leg_pos_limits = dof_pos_limits[:, self._leg_joint_ids]
        out_of_limits = -(leg_pos - leg_pos_limits[:, :, 0]).clip(max=0.0)
        out_of_limits += (leg_pos - leg_pos_limits[:, :, 1]).clip(min=0.0)
        r_dof_pos_limits = torch.sum(out_of_limits, dim=1)

        # Safety costs are one-sided: valid/longer legs and heights above the
        # command are not penalized by these terms.
        leg_length_deficit = torch.clamp(
            self.cfg.l0_ref_min - self._L0, min=0.0
        )
        r_leg_length_below_min = torch.sum(
            torch.square(leg_length_deficit), dim=1
        )
        base_height_deficit = torch.clamp(
            self._commands[:, 2] - self._base_height, min=0.0
        )
        r_base_height_below_target = torch.square(base_height_deficit)

        # termination penalty
        r_termination = (
            self.reset_terminated * ~self.reset_time_outs
        )

        rewards = {
            "track_lin_vel": r_track_lin * self.cfg.rewards.tracking_lin_vel * self.step_dt,
            "track_lin_vel_enhance": r_track_lin_enh
            * self.cfg.rewards.tracking_lin_vel_enhance
            * self.step_dt,
            "lin_vel_error_sq": lin_vel_error
            * self.cfg.rewards.lin_vel_error_sq
            * self.step_dt,
            "track_ang_vel": r_track_ang * self.cfg.rewards.tracking_ang_vel * self.step_dt,
            "yaw_rate_error": r_yaw_rate_error
            * self.cfg.rewards.yaw_rate_error
            * self.step_dt,
            "base_height": r_base_height * self.cfg.rewards.base_height * self.step_dt,
            "nominal_state": r_nominal * self.cfg.rewards.nominal_state * self.step_dt,
            "lin_vel_z": r_lin_vel_z * self.cfg.rewards.lin_vel_z * self.step_dt,
            "ang_vel_xy": r_ang_vel_xy * self.cfg.rewards.ang_vel_xy * self.step_dt,
            "orientation": r_orientation * self.cfg.rewards.orientation * self.step_dt,
            "dof_vel": r_dof_vel * self.cfg.rewards.dof_vel * self.step_dt,
            "dof_acc": r_dof_acc * self.cfg.rewards.dof_acc * self.step_dt,
            "torques": r_torques * self.cfg.rewards.torques * self.step_dt,
            "action_rate": r_action_rate * self.cfg.rewards.action_rate * self.step_dt,
            "action_smooth": r_action_smooth
            * self.cfg.rewards.action_smooth
            * self.step_dt,
            "collision": r_collision * self.cfg.rewards.collision * self.step_dt,
            "dof_pos_limits": r_dof_pos_limits
            * self.cfg.rewards.dof_pos_limits
            * self.step_dt,
            "leg_length_below_min": r_leg_length_below_min
            * self.cfg.rewards.leg_length_below_min
            * self.step_dt,
            "base_height_below_target": r_base_height_below_target
            * self.cfg.rewards.base_height_below_target
            * self.step_dt,
            "termination": r_termination * self.cfg.rewards.termination,
        }
        # Keep the usual per-step clip for dense rewards.  Preserve the
        # increased linear-tracking weight instead of clipping its positive
        # reward back to one step_dt; allow a wider negative yaw-error range.
        # The one-off terminal penalty remains exempt.
        for key, value in rewards.items():
            if key == "track_lin_vel":
                rewards[key] = torch.clamp(
                    value,
                    min=-self.step_dt,
                    max=self.cfg.rewards.tracking_lin_vel * self.step_dt,
                )
            elif key == "yaw_rate_error":
                rewards[key] = torch.clamp(
                    value,
                    min=-self.cfg.rewards.yaw_rate_error_clip_multiplier * self.step_dt,
                    max=self.step_dt,
                )
            elif key == "orientation":
                rewards[key] = torch.clamp(
                    value,
                    min=-self.cfg.rewards.orientation_clip_multiplier * self.step_dt,
                    max=self.step_dt,
                )
            elif key == "collision":
                rewards[key] = torch.clamp(
                    value,
                    min=-self.cfg.rewards.collision_clip_multiplier * self.step_dt,
                    max=self.step_dt,
                )
            elif key in ("leg_length_below_min", "base_height_below_target"):
                rewards[key] = torch.clamp(
                    value,
                    min=-self.cfg.rewards.safety_penalty_clip_multiplier * self.step_dt,
                    max=self.step_dt,
                )
            elif key != "termination":
                rewards[key] = torch.clamp(value, min=-self.step_dt, max=self.step_dt)
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # logging
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
        died = base_contact_died | fallen_died | pitch_died | leg_contact_died

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return died, time_out

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # reset joint state to default
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        # reset root state
        default_root_state = self._robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # randomize base velocities
        default_root_state[:, 7:13] = (
            torch.rand(len(env_ids), 6, device=self.device) - 0.5
        )  # [-0.5, 0.5]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset buffers
        self._actions[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._fail_buf[env_ids] = 0.0
        self._base_contact_buf[env_ids] = 0.0
        self._pitch_fail_buf[env_ids] = 0.0
        self._leg_contact_buf[env_ids] = 0.0
        self._commands[env_ids] = 0.0
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
        extras["Episode_Termination/unsafe_contact_or_orientation"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        self.extras["log"].update(extras)

    def _resample_commands_for(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

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
            moving_fraction = (
                1.0 - self.cfg.commands.standing_env_fraction
            ) * progress

            # Every resample starts as an explicit standing command.  A growing
            # subset is then replaced with forward-motion commands.
            self._commands[env_ids, :3] = 0.0
            self._commands[env_ids, 2] = self.cfg.rewards.base_height_target
            moving_mask = (
                torch.rand(env_ids.numel(), device=self.device) < moving_fraction
            )
            moving_ids = env_ids[moving_mask]
            if moving_ids.numel() > 0:
                progressive_vmax = lin_min + (lin_max - lin_min) * progress
                self._commands[moving_ids, 0] = lin_min + (
                    progressive_vmax - lin_min
                ) * torch.rand(moving_ids.numel(), device=self.device)
                yaw_min, yaw_max = self.cfg.commands.ranges_ang_vel_yaw
                self._commands[moving_ids, 1] = yaw_min + (yaw_max - yaw_min) * torch.rand(
                    moving_ids.numel(), device=self.device
                )
                height_min, height_max = self.cfg.commands.ranges_height
                self._commands[moving_ids, 2] = height_min + (
                    height_max - height_min
                ) * torch.rand(moving_ids.numel(), device=self.device)

        if self.cfg.commands.heading_command:
            self._commands[env_ids, 3] = (
                self.cfg.commands.ranges_heading[0]
                + (self.cfg.commands.ranges_heading[1] - self.cfg.commands.ranges_heading[0])
                * torch.rand(env_ids.numel(), device=self.device)
            )
