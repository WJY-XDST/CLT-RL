"""User CAD closed-chain robot, expressed in X-forward/Y-left/Z-up base axes."""
import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from wheel_legged_gym_isaaclab import WHEEL_LEGGED_GYM_ISAACLAB_ROOT_DIR
from .wheel_legged_vmc_flat_env_cfg import WheelLeggedVMCFlatEnvCfg

ASSETS = Path(WHEEL_LEGGED_GYM_ISAACLAB_ROOT_DIR) / "assets/robots/mine"
MANIFEST = json.loads((ASSETS / "manifest.json").read_text())
MODEL = json.loads((ASSETS / MANIFEST["metadata"]).read_text())


@configclass
class MineWheelLeggedVMCFlatEnvCfg(WheelLeggedVMCFlatEnvCfg):
    leg_model = "mine_five_bar"
    leg_joint_names = MODEL["leg_joint_names"]
    wheel_joint_names = MODEL["wheel_joint_names"]
    leg_body_names = [s+n+"_link" for s in ("L", "R") for n in ("B", "S", "L", "L1", "L2", "L3")]
    penalised_body_pattern = "(base_link|[LR](B|S|L[123]?)_link)"
    five_bar_geometry = MODEL["five_bar"]
    leg_effort_limit = MODEL["leg_effort_limit"]
    wheel_effort_limit = MODEL["wheel_effort_limit"]
    wheel_radius = MODEL["wheel_radius"]
    wheel_track_width = MODEL["wheel_track_width"]
    feedforward_force = MODEL["supported_mass_kg"] * 9.81 / 2
    # Solve wheel velocity damping inside PhysX: the CAD axial inertia is too
    # small for an explicit 0.5 Nm*s/rad controller at a 5 ms physics step.
    wheel_control_mode = "implicit_velocity"
    wheel_damping = 0.5
    action_scale_l0 = 0.09
    l0_offset = 0.27
    l0_ref_min = 0.18
    l0_ref_max = 0.36
    min_root_height = 0.22
    leg_length_height_offset = MODEL["hip_z"] - MODEL["wheel_radius"]
    forward_support_height_margin = leg_length_height_offset
    forward_support_min_leg_length = 0.30
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSETS / MANIFEST["usd"]),
            activate_contact_sensors=True,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=64,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0., 0., MODEL["nominal_root_height"]),
            joint_pos={".*": 0.}, joint_vel={".*": 0.},
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=MODEL["leg_joint_names"],
                stiffness=0., damping=0., effort_limit_sim=MODEL["leg_effort_limit"],
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=MODEL["wheel_joint_names"],
                stiffness=0., damping=0.5, effort_limit_sim=MODEL["wheel_effort_limit"],
            ),
            "passive": ImplicitActuatorCfg(
                joint_names_expr=["[LR]S_joint", "[LR]L[123]_joint"],
                stiffness=0., damping=0., effort_limit_sim=0.,
            ),
        },
    )

    def __post_init__(self):
        super().__post_init__()
        # Detailed CAD collision geometry is more expensive than the legacy asset.
        self.scene.num_envs = 256
        self.robot.actuators["wheels"].damping = self.wheel_damping
        # With 256 environments the old 200-iteration startup contains only
        # 1/48 of the original 12288-environment sample count. Keep the first
        # 1000 iterations for balance; subsequent rounds ramp all motion tasks.
        self.commands.standing_only_steps = 48_000
        self.commands.motion_ramp_steps = 48_000
        self.commands.reverse_ramp_start_steps = 48_000
        self.commands.reverse_ramp_steps = 48_000
        self.commands.yaw_start_steps = 48_000
        self.commands.yaw_ramp_steps = 48_000
        self.commands.ranges_ang_vel_yaw = (-0.3, 0.3)
        self.commands.heading_command = False
        self.commands.grouped_training = True
        self.commands.ranges_height = (0.28, 0.32)
        self.rewards.base_height_target = 0.30
        # Preserve a height-error learning signal throughout a full collapse.
        # The legacy 1 reward/s cap saturated already at 31.6 mm error.
        self.rewards.base_height_penalty_clip = 30.0
        self.rewards.standing_leg_angle = -5.0
        self.rewards.leg_length_target_height_offset = self.leg_length_height_offset
        self.rewards.leg_length_target_cap = 0.32
        # asset_root coordinates are offsets from the moving robot root.
        self.viewer.lookat = (0., 0., -0.08)
