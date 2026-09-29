# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause
#
# Environment configuration for the wheel-legged robot (VMC control) on flat terrain.
# Migrated from Wheel-Legged-Gym (Isaac Gym Preview 4) to Isaac Lab 2.3.x / Isaac Sim 5.1.

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from wheel_legged_gym_isaaclab import WHEEL_LEGGED_GYM_ISAACLAB_ROOT_DIR


@configclass
class ObsScalesCfg:
    """Observation scales (mirrors `normalization.obs_scales` in the original project)."""

    lin_vel = 2.0
    ang_vel = 0.25
    dof_pos = 1.0
    dof_vel = 0.05
    dof_acc = 0.0025
    height_measurements = 5.0
    torque = 0.05
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

    num_commands = 3
    resampling_time = 5.0  # [s]
    # Hold the initial world-frame heading throughout each episode.
    heading_command = True
    heading_kp = 1.5  # [1/s], heading error -> target yaw rate
    heading_rate_limit = 0.5  # [rad/s], maximum corrective yaw rate
    ranges_lin_vel_x = (0.3, 0.8)  # [m/s], forward training commands
    ranges_reverse_lin_vel_x = (-0.8, -0.3)  # [m/s], reverse training commands
    # Bridge the gap between standing and the positive/negative main ranges.
    ranges_transition_lin_vel_x = (-0.2, 0.2)  # [m/s]
    ranges_ang_vel_yaw = (0.0, 0.0)  # [rad/s]
    # Introduce height control around the nominal 0.18 m stance.  The sampler
    # expands progressively from the midpoint to this complete interval using
    # the same curriculum progress as the velocity commands.
    ranges_height = (0.16, 0.20)  # [m]
    # Environment control steps. Start introducing motion immediately and
    # reach the full command mixture after 500 PPO iterations (48 steps each).
    standing_only_steps = 0
    motion_ramp_steps = 24_000
    # Explicit zero-speed population retained after motion is introduced.
    standing_env_fraction = 0.25
    # At full curriculum progress, this fraction receives commands from the
    # bidirectional low-speed transition range. Main-range samples are split
    # between forward and reverse as the reverse curriculum progresses.
    transition_env_fraction = 0.25
    # Resume from the 2000-iteration forward policy, then introduce reverse
    # commands gradually over another 1000 iterations (48 steps per iteration).
    reverse_env_fraction = 0.25
    reverse_ramp_start_steps = 96_000
    reverse_ramp_steps = 48_000


@configclass
class RewardsCfg:
    """Reward scales restored from the original Wheel-Legged-Gym task.

    Isaac-Lab-specific safety termination remains outside the reward set,
    while the learning objective follows the source project.
    """

    tracking_lin_vel = 1.0
    tracking_lin_vel_enhance = 1.0
    # A narrower tracking kernel and an explicit squared-error term preserve
    # useful gradients both near and far from the commanded forward speed.
    tracking_lin_vel_precise = 1.0
    lin_vel_error_sq = -1.0
    base_height_error_sq = -40.0
    tracking_ang_vel = 1.0
    yaw_rate_error_sq = -5.0
    base_height = 1.0
    nominal_state = -2.0
    lin_vel_z = -2.0
    ang_vel_xy = -0.05
    orientation = -15.0
    dof_vel = -5e-5
    dof_acc = -2.5e-7
    torques = -0.0001
    action_rate = -0.01
    action_smooth = -0.01
    # Softly discourage virtual-leg length commands from remaining close to
    # their normalized action limits. This still permits unequal leg lengths.
    leg_length_action_saturation = -5.0
    leg_length_action_difference = -0.5
    leg_length_target_underreach = -200.0
    collision = -1.0
    dof_pos_limits = -1.0

    # parameters
    clip_single_reward = 1.0
    tracking_sigma = 0.25
    tracking_sigma_enhance = 0.025
    tracking_sigma_precise = 0.04
    # Projected-gravity y is dominated by body roll; weight it more strongly
    # without constraining the two virtual-leg lengths to be identical.
    orientation_roll_multiplier = 4.0
    base_height_target = 0.18
    base_height_sigma = 0.0004  # [m^2], experimental height-tracking width
    max_contact_force = 100.0
    leg_length_action_soft_limit = 0.85
    leg_length_target_height_offset = 0.04  # [m], minimum reference above commanded root height


@configclass
class WheelLeggedVMCFlatEnvCfg(DirectRLEnvCfg):
    """Configuration for the wheel-legged robot VMC task on flat terrain."""

    # -- env --
    episode_length_s = 20.0
    decimation = 4
    action_space = 6
    observation_space = 27
    state_space = 0
    # A nearly inverted body retains the original delayed failure condition.
    fail_to_terminal_time_s = 1.0
    # Base-link contact is unsafe even when the robot is not fully inverted.
    base_contact_terminal_time_s = 0.15  # [s]
    base_contact_force_threshold = 10.0  # [N]
    # Sustained non-wheel leg-link contact is treated as a failure.
    leg_contact_terminal_time_s = 0.2
    leg_contact_force_threshold = 5.0  # [N]
    collision_force_threshold = 1.0  # [N]
    max_body_pitch = 0.5235987756  # [rad], 30 degrees
    body_pitch_terminal_time_s = 0.25
    # Unequal leg lengths remain allowed for uneven terrain. Terminate only
    # when their resulting body roll stays unsafe for a meaningful duration.
    max_body_roll = 0.3490658504  # [rad], 20 degrees
    body_roll_terminal_time_s = 0.2

    # Keep the interactive Isaac Sim viewport centered on env_0's robot.
    # Without this, the generic world camera is far from this small robot and
    # a successful replay can look like an empty scene.
    viewer = ViewerCfg(
        eye=(2.0, 2.0, 1.2),
        lookat=(0.0, 0.0, 0.18),
        origin_type="asset_root",
        env_index=0,
        asset_name="robot",
    )

    # -- VMC control gains (from `wheel_legged_vmc_config.py`) --
    action_scale_theta = 0.2  # [rad] per action unit
    theta0_ref_min = -0.2  # [rad], hard VMC target bound
    theta0_ref_max = 0.2  # [rad], hard VMC target bound
    # Map the complete normalized interval [-1, 1] onto [0.12, 0.26] m.
    # The extra upper travel improves high-stance reach without the roll
    # increase observed when testing a 0.27 m reference ceiling.
    action_scale_l0 = 0.07  # [m] per action unit
    l0_offset = 0.19  # [m], midpoint of the virtual-leg target range
    l0_ref_min = 0.12  # [m], hard lower bound for the VMC reference
    l0_ref_max = 0.26  # [m], hard upper bound for the VMC reference
    # A 0.0675 m wheel needs 11.85 rad/s for the maximum 0.8 m/s command.
    # Keep some control margin instead of making the fastest command
    # unreachable at action=1.
    wheel_radius = 0.0675  # [m]
    wheel_track_width = 0.341  # [m], wheel-center separation from the URDF
    action_scale_vel = 15.0  # [rad/s] per action unit
    # Approximate per-leg support force for the 12.28 kg assembly. Start at
    # 50 N because the wheel masses are supported directly at ground contact;
    # refine this value with the zero-action static diagnostic.
    feedforward_force = 50.0  # [N]
    kp_theta = 50.0  # [N*m/rad]
    kd_theta = 3.0  # [N*m*s/rad]
    kp_l0 = 900.0  # [N/m]
    kd_l0 = 20.0  # [N*s/m]
    wheel_damping = 0.5  # [N*m*s/rad]

    # leg kinematic parameters (from `wheel_legged_config.py`)
    l1 = 0.15  # [m]
    l2 = 0.25  # [m]
    offset = 0.054  # [m]

    # -- robot --
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=f"{WHEEL_LEGGED_GYM_ISAACLAB_ROOT_DIR}/assets/robots/wl/urdf/wl.urdf",
            usd_dir=f"{WHEEL_LEGGED_GYM_ISAACLAB_ROOT_DIR}/assets/robots/wl/usd",
            usd_file_name="wl.usd",
            make_instanceable=True,
            fix_base=False,
            activate_contact_sensors=True,
            # VMC supplies effort targets in the environment, so no importer
            # position drive may apply a competing torque.
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0
                )
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),
            joint_pos={
                "lf0_Joint": 0.5,
                "lf1_Joint": 0.35,
                "l_wheel_Joint": 0.0,
                "rf0_Joint": -0.5,
                "rf1_Joint": -0.35,
                "r_wheel_Joint": 0.0,
            },
        ),
        actuators={
            # All joint torques are computed by the VMC controller in the env,
            # so the implicit PD gains are set to zero.
            "legs": ImplicitActuatorCfg(
                joint_names_expr=["lf0_Joint", "lf1_Joint", "rf0_Joint", "rf1_Joint"],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=30.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=["l_wheel_Joint", "r_wheel_Joint"],
                stiffness=0.0,
                damping=0.0,
                effort_limit_sim=5.0,
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
        num_envs=12288, env_spacing=4.0, replicate_physics=True
    )

    # -- simulation --
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
