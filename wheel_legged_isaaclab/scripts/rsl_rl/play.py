# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL (external project: wheel_legged_gym_isaaclab)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import csv
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
from heading_feedback import HeadingFeedback
from keyboard_control import (
    KeyboardConfig, IsaacKeyboardController, apply_keyboard_command, supports_keyboard_task,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument('--overrides_json',type=Path,help='Restore the accepted model control settings before replay')
parser.add_argument(
    "--camera_mode", choices=("orbit", "free", "follow"), default="orbit",
    help="Interactive camera: orbit follows position with mouse navigation; free stops tracking; follow locks the angle.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the published pre-trained checkpoint instead of a local run checkpoint.",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Load an explicit checkpoint path, including a checkpoint stored in this repository.",
)
parser.add_argument(
    "--fixed_command",
    type=float,
    nargs=3,
    metavar=("LIN_VEL_X", "YAW_RATE", "HEIGHT"),
    default=None,
    help=(
        "Evaluate with a constant command instead of randomized commands. "
        "Provide: forward velocity [m/s], yaw rate [rad/s], base height [m]."
    ),
)
parser.add_argument(
    "--velocity_cycle",
    type=float,
    nargs="+",
    default=None,
    metavar="LIN_VEL_X",
    help=(
        "Cycle through forward-velocity commands during one replay. Requires "
        "--fixed_command to provide the fixed yaw-rate and height values."
    ),
)
parser.add_argument(
    "--yaw_cycle", type=float, nargs="+", default=None, metavar="YAW_RATE",
    help="Cycle yaw-rate commands [rad/s] alongside velocity/height; zero holds the heading reached at phase entry.",
)
parser.add_argument("--heading_feedback", action="store_true",
                    help="Single-robot navigation test: track the integral of requested yaw rate with bounded PI heading feedback.")
parser.add_argument("--constant_yaw_rate", action="store_true",
                    help="Apply the requested yaw rate directly, including zero, as in fixed-command acceptance.")
parser.add_argument(
    "--height_cycle",
    type=float,
    nargs="+",
    default=None,
    metavar="HEIGHT",
    help="Cycle body-height commands using --velocity_phase_duration; requires --fixed_command.",
)
parser.add_argument(
    "--velocity_phase_duration",
    type=float,
    default=5.0,
    metavar="SECONDS",
    help="Duration of each --velocity_cycle phase in seconds (default: 5.0).",
)
parser.add_argument(
    "--fixed_leg_length",
    type=float,
    default=None,
    metavar="LENGTH",
    help=(
        "For WheelLeggedVMC replay only, override both virtual-leg target lengths "
        "while retaining the checkpoint's training configuration. Units: m."
    ),
)
parser.add_argument(
    "--print_obs",
    action="store_true",
    default=False,
    help="Print the first environment's 27-dimensional policy observation during replay.",
)
parser.add_argument(
    "--print_obs_interval",
    type=int,
    default=50,
    metavar="STEPS",
    help="Number of control steps between observation prints (default: 50, or 1 second at 50 Hz).",
)
parser.add_argument(
    "--zero_actions",
    action="store_true",
    default=False,
    help="Ignore the policy output and apply zero normalized actions for VMC symmetry diagnostics.",
)
parser.add_argument(
    "--fixed_wheel_action",
    type=float,
    nargs=2,
    metavar=("LEFT", "RIGHT"),
    default=None,
    help=(
        "Override only the policy's normalized left/right wheel actions for an open-loop drive test. "
        "Positive values command forward wheel motion; each value must be in [-1, 1]."
    ),
)
parser.add_argument(
    "--fixed_wheel_action_start",
    type=int,
    default=0,
    metavar="STEPS",
    help="Wait this many control steps before applying --fixed_wheel_action (default: 0).",
)
parser.add_argument(
    "--print_vmc",
    action="store_true",
    default=False,
    help="Print raw left/right VMC states and articulation-order applied torques.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    metavar="STEPS",
    help="Stop replay automatically after this many control steps.",
)
parser.add_argument(
    "--trace_csv",
    type=str,
    default=None,
    metavar="PATH",
    help="Write env state, actions, torques, and per-term rewards to a CSV file.",
)
parser.add_argument(
    "--trace_interval",
    type=int,
    default=1,
    metavar="STEPS",
    help="Control-step interval between trace rows (default: 1).",
)
parser.add_argument(
    "--trace_env_id",
    type=int,
    default=0,
    metavar="ENV_ID",
    help="Environment index recorded by --trace_csv (default: 0).",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--live_stats", action="store_true", default=False,
    help="Show simulation progress, speed, height, and leg states in a GUI panel.",
)
parser.add_argument("--startup_focus", action="store_true",
                    help="Pin the first five seconds of individual leg angles and show settling metrics above live charts.")
keyboard_group = parser.add_mutually_exclusive_group()
keyboard_group.add_argument('--keyboard', dest='keyboard', action='store_true',
                            help='Enable WASD/QE keyboard commands in a GUI replay')
keyboard_group.add_argument('--no_keyboard', dest='keyboard', action='store_false',
                            help='Disable automatic keyboard control for reproducible evaluation')
parser.set_defaults(keyboard=None)
parser.add_argument('--keyboard_speed', type=float, default=0.8, help='Held W/S speed [m/s]')
parser.add_argument('--keyboard_yaw_rate', type=float, default=0.3, help='Held A/D yaw rate [rad/s]')
parser.add_argument('--keyboard_height_rate', type=float, default=0.02, help='Held Q/E height change [m/s]')
parser.add_argument('--keyboard_height_step', type=float, default=0.005, help='Q/E initial press height step [m]')
parser.add_argument('--keyboard_height_slew_rate', type=float, default=0.02,
                    help='Maximum ramped policy height-command rate [m/s], including after key release')
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# LAB 将 --headless、--device、--enable_cameras 等应用参数加入同一个 parser。
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# LAB AppLauncher 创建 Kit/Isaac Sim 应用；.app 才是应用句柄。
# 顺序不能反过来：需要仿真扩展的 isaaclab.envs、omni 等模块在启动后再导入。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import math
import os
import time
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
import wheel_legged_gym_isaaclab.tasks  # noqa: F401  (registers WheelLeggedVMC-Flat-v0)


# LAB 装饰器从 Gym 注册信息取得环境/PPO 配置，合并 Hydra 的 env.xxx=... 覆盖项，
# 再把配置对象传给 main；不是策略网络的一层，也不启动 PPO 更新。
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    if args_cli.print_obs_interval <= 0:
        raise ValueError("--print_obs_interval must be greater than zero.")
    if args_cli.trace_interval <= 0:
        raise ValueError("--trace_interval must be greater than zero.")
    if args_cli.fixed_wheel_action_start < 0:
        raise ValueError("--fixed_wheel_action_start must be non-negative.")
    if args_cli.velocity_phase_duration <= 0.0:
        raise ValueError("--velocity_phase_duration must be greater than zero.")
    if any(cycle is not None for cycle in (args_cli.velocity_cycle, args_cli.height_cycle, args_cli.yaw_cycle)) and args_cli.fixed_command is None:
        raise ValueError("Command cycles require --fixed_command for the remaining command values.")
    if args_cli.height_cycle is not None and any(height <= 0.0 for height in args_cli.height_cycle):
        raise ValueError("--height_cycle values must be positive.")
    if args_cli.fixed_wheel_action is not None:
        if any(abs(value) > 1.0 for value in args_cli.fixed_wheel_action):
            raise ValueError("--fixed_wheel_action values must both be in [-1, 1].")
        print(
            "[INFO] Open-loop wheel override enabled: "
            f"left={args_cli.fixed_wheel_action[0]:.3f}, "
            f"right={args_cli.fixed_wheel_action[1]:.3f} (forward-positive normalized actions)."
        )
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.overrides_json:
        from wheel_legged_gym_isaaclab.config_overrides import apply_config_overrides
        apply_config_overrides(env_cfg,json.loads(args_cli.overrides_json.read_text()))
        if env_cfg.wheel_control_mode=='implicit_velocity':
            env_cfg.robot.actuators['wheels'].damping=env_cfg.wheel_damping
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    keyboard_enabled = args_cli.keyboard
    if keyboard_enabled is None:
        keyboard_enabled = (not args_cli.headless and not args_cli.video
                            and supports_keyboard_task(task_name)
                            and not args_cli.startup_focus and not args_cli.zero_actions
                            and args_cli.fixed_leg_length is None and args_cli.fixed_wheel_action is None)
    if keyboard_enabled and (args_cli.headless or args_cli.video):
        raise ValueError('--keyboard requires a GUI replay without --video')
    if keyboard_enabled and (args_cli.fixed_leg_length is not None or args_cli.fixed_wheel_action is not None or args_cli.zero_actions):
        raise ValueError('Keyboard commands cannot be combined with open-loop action overrides')
    if keyboard_enabled and args_cli.startup_focus:
        raise ValueError('Keyboard commands cannot be combined with fixed startup capture; use --no_keyboard')
    keyboard_config = None
    keyboard_starts_active = keyboard_enabled and args_cli.fixed_command is None
    if keyboard_enabled:
        if hasattr(env_cfg, 'jump'):
            # Interactive jumps require SPACE, never a periodic auto-trigger.
            env_cfg.jump.automatic = False
        if not supports_keyboard_task(task_name):
            raise ValueError('Keyboard commands support the registered wheel-legged VMC and joint-PD flat tasks only')
        if args_cli.num_envs is None:
            env_cfg.scene.num_envs = 1
        height_min, height_max = env_cfg.commands.ranges_height
        initial_height = args_cli.fixed_command[2] if args_cli.fixed_command else env_cfg.rewards.base_height_target
        keyboard_config = KeyboardConfig(args_cli.keyboard_speed, args_cli.keyboard_yaw_rate,
                                         args_cli.keyboard_height_rate, height_min, height_max, initial_height,
                                         height_step=args_cli.keyboard_height_step,
                                         height_slew_rate=args_cli.keyboard_height_slew_rate)
        if args_cli.heading_feedback and keyboard_config.yaw_rate > 0.5:
            raise ValueError('Heading feedback requires --keyboard_yaw_rate <= 0.5 rad/s; '
                             'use direct keyboard yaw-rate mode for larger commands')
        if keyboard_starts_active:
            args_cli.fixed_command = [0.0, 0.0, initial_height]
    if args_cli.heading_feedback and (env_cfg.scene.num_envs != 1 or args_cli.fixed_command is None):
        raise ValueError("--heading_feedback requires --num_envs 1 and --fixed_command")
    if args_cli.startup_focus and (
        not args_cli.live_stats or args_cli.headless or env_cfg.scene.num_envs != 1
        or args_cli.fixed_command is None or any(args_cli.fixed_command[:2])
        or any(cycle is not None for cycle in (args_cli.velocity_cycle, args_cli.height_cycle, args_cli.yaw_cycle))
    ):
        raise ValueError("--startup_focus requires a single GUI robot, --live_stats and a constant standing command")
    if args_cli.heading_feedback and any(abs(v) > .5 for v in (args_cli.yaw_cycle or [args_cli.fixed_command[1]])):
        raise ValueError("Heading feedback test supports requested yaw rates within +/-0.5 rad/s")

    # Zero yaw holds the initial world-frame heading. Nonzero yaw turns at a
    # constant commanded rate.
    if args_cli.fixed_command is not None:
        lin_vel_x, yaw_rate, height = args_cli.fixed_command
        env_cfg.commands.heading_command = (yaw_rate == 0.0 and not keyboard_starts_active and not args_cli.heading_feedback
                                           and not args_cli.constant_yaw_rate)
        env_cfg.commands.ranges_lin_vel_x = (lin_vel_x, lin_vel_x)
        env_cfg.commands.ranges_ang_vel_yaw = (yaw_rate, yaw_rate)
        env_cfg.commands.ranges_height = (height, height)
        print(
            "[INFO] Using fixed evaluation command: "
            f"lin_vel_x={lin_vel_x:.3f} m/s, yaw_rate={yaw_rate:.3f} rad/s, height={height:.3f} m, "
            f"heading_hold={env_cfg.commands.heading_command}"
        )

    # Do not alter ``l0_offset`` for a single replay experiment: it is part of
    # the training task definition.  Instead convert a requested target length
    # to the normalized VMC leg-length action and override only action indices
    # 1 and 4 (left/right l0) after policy inference.
    fixed_leg_action = None
    if args_cli.fixed_leg_length is not None:
        if env_cfg.leg_control_mode == 'implicit_joint_position':
            raise ValueError('--fixed_leg_length is incompatible with a direct-joint policy; use height commands, not length-action overrides')
        required_leg_cfg = ("l0_offset", "action_scale_l0", "l0_ref_min", "l0_ref_max")
        if not all(hasattr(env_cfg, name) for name in required_leg_cfg):
            raise ValueError("--fixed_leg_length is supported only by WheelLeggedVMC tasks.")
        if not env_cfg.l0_ref_min <= args_cli.fixed_leg_length <= env_cfg.l0_ref_max:
            raise ValueError(
                f"Requested leg length {args_cli.fixed_leg_length:.3f} m is outside the physical range "
                f"[{env_cfg.l0_ref_min:.3f}, {env_cfg.l0_ref_max:.3f}] m."
            )
        fixed_leg_action = (args_cli.fixed_leg_length - env_cfg.l0_offset) / env_cfg.action_scale_l0
        if not -1.0 <= fixed_leg_action <= 1.0:
            raise ValueError(
                "The requested physical leg length cannot be represented by the normalized action map; "
                "check l0_offset/action_scale_l0 against l0_ref_min/l0_ref_max."
            )
        # A diagnostic fixed target must bypass the outer height correction
        # and the speed-dependent support floor.
        env_cfg.height_feedback_gain = 0.0
        env_cfg.height_reference_blend = 0.0
        env_cfg.forward_support_min_leg_length = env_cfg.l0_ref_min
        print(
            "[INFO] Fixing both virtual-leg targets at "
            f"{args_cli.fixed_leg_length:.3f} m (normalized action {fixed_leg_action:.3f})."
        )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "IsaacLab",
            "logs",
            "rsl_rl",
            agent_cfg.experiment_name,
        )
    )
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        raise ValueError("This vehicle requires locally trained weights and their paired env.yaml; published example-vehicle weights are not supported.")
    if args_cli.checkpoint_path is not None:
        resume_path = str(Path(args_cli.checkpoint_path).expanduser().resolve())
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = (
        os.path.dirname(resume_path)
        if args_cli.checkpoint_path is None
        else os.path.join(log_root_path, "external_checkpoint_play")
    )
    if args_cli.checkpoint_path is not None:
        os.makedirs(log_dir, exist_ok=True)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    from wheel_legged_gym_isaaclab.physics_contract import (
        EXTERNAL_FORCE_FIELD, ROBOT_ASSET_FIELD, checkpoint_force_iteration_setting,
        checkpoint_robot_asset_setting,
    )
    saved_config = validate_checkpoint_robot(resume_path)
    from wheel_legged_gym_isaaclab.leg_control import require_matching_policy_contract
    require_matching_policy_contract(saved_config, env_cfg.to_dict())
    physics_overrides = json.loads(args_cli.overrides_json.read_text()) if args_cli.overrides_json else {}
    from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
    if any(arg.startswith('env.wheel_action_limit=') for arg in hydra_args):
        physics_overrides['wheel_action_limit'] = env_cfg.wheel_action_limit
    env_cfg.wheel_action_limit = checkpoint_wheel_action_limit(saved_config, physics_overrides)
    if not any(arg.startswith('env.action_scale_vel=') for arg in hydra_args):
        env_cfg.action_scale_vel = saved_config.get('action_scale_vel', env_cfg.action_scale_vel)
    if keyboard_enabled and keyboard_config.speed > (
            env_cfg.wheel_radius * env_cfg.action_scale_vel * env_cfg.wheel_action_limit):
        raise ValueError('Keyboard speed exceeds the saved wheel action range')
    if any(arg.startswith(f'env.{ROBOT_ASSET_FIELD}=') for arg in hydra_args):
        physics_overrides[ROBOT_ASSET_FIELD] = env_cfg.robot.spawn.usd_path
    env_cfg.robot.spawn.usd_path = checkpoint_robot_asset_setting(saved_config, physics_overrides)
    if any(arg.startswith(f'env.{EXTERNAL_FORCE_FIELD}=') for arg in hydra_args):
        physics_overrides[EXTERNAL_FORCE_FIELD] = env_cfg.enable_external_forces_every_iteration
    env_cfg.enable_external_forces_every_iteration = checkpoint_force_iteration_setting(
        saved_config, physics_overrides,
    )
    # Gym 根据注册入口创建 LAB 环境；cfg 是已解析的配置，scene.num_envs 决定批量。
    # 此调用会触发 DirectRLEnv 初始化、_setup_scene() 和物理句柄创建。
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # .unwrapped 取出最内层 LAB 环境，越过 Gym/视频/RSL-RL 包装器。
    # 下面是通用多智能体兼容分支；当前轮腿任务本身是单智能体。
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # LAB wrapper 将 dict obs 包为 TensorDict，并合并 terminated/truncated 为 done。
    # 返回接口从 Gym 的 5 项变为 RSL-RL 的 4 项；不改变物理控制器。
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # OnPolicyRunner 属于 RSL-RL，不是 LAB：此处借它构建 actor/critic 并加载权重。
    # 不调用 runner.learn()，所以回放没有 PPO 参数更新。
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # get_inference_policy() 切到 eval 模式，返回 actor.act_inference 的可调用句柄。
    # policy(obs) 得到动作均值 [N, 6]，不进行训练时的高斯采样。
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    try:
        policy_nn = runner.alg.policy  # version 2.3 onwards
    except AttributeError:
        policy_nn = runner.alg.actor_critic  # version 2.2 and below

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # LAB 导出 actor 和可选观测归一化器，不导出环境、reward、PD 或安全控制器。
    # 导出的网络仍需要配套 obs/action 单位与顺序，不能单靠文件就部署到实车。
    export_model_dir = os.path.join(log_dir, "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # LAB step_dt = sim.dt × decimation，是策略周期；不是物理子步长。
    # 当前为 0.020 s；键盘斜坡、阶段时长和统计都按这个周期计算。
    dt = env.unwrapped.step_dt
    keyboard_controller = IsaacKeyboardController(
        keyboard_config, keyboard_starts_active,
        jump_handler=getattr(env.unwrapped, 'request_jump', None),
    ) if keyboard_enabled else None

    velocity_phase_steps = None
    last_velocity_phase = None
    if any(cycle is not None for cycle in (args_cli.velocity_cycle, args_cli.height_cycle, args_cli.yaw_cycle)):
        velocity_phase_steps = max(1, round(args_cli.velocity_phase_duration / dt))
    if args_cli.velocity_cycle is not None:
        reachable_speed = (env.unwrapped.cfg.wheel_radius * env.unwrapped.cfg.action_scale_vel
                           * env.unwrapped.cfg.wheel_action_limit)
        if any(abs(value) > reachable_speed for value in args_cli.velocity_cycle):
            raise ValueError(
                "A --velocity_cycle command exceeds the ideal wheel-speed action range "
                f"of +/-{reachable_speed:.3f} m/s."
            )
        print(
            "[INFO] Velocity-cycle evaluation enabled: "
            f"sequence={args_cli.velocity_cycle}, "
            f"phase_duration={args_cli.velocity_phase_duration:.3f} s "
            f"({velocity_phase_steps} control steps)."
        )

    def apply_velocity_cycle_command(observation, step: int):
        """Synchronize the active test speed with the environment and policy input."""
        nonlocal last_velocity_phase
        if keyboard_controller is not None and keyboard_controller.state.active:
            return
        if velocity_phase_steps is None:
            return
        phase = step // velocity_phase_steps
        base_env = env.unwrapped
        command_speed = args_cli.fixed_command[0]
        if args_cli.velocity_cycle is not None:
            command_speed = args_cli.velocity_cycle[phase % len(args_cli.velocity_cycle)]
            base_env._command_ranges[0, :] = command_speed
            base_env._target_lin_vel_x[:] = command_speed
        if args_cli.height_cycle is not None:
            command_height = args_cli.height_cycle[phase % len(args_cli.height_cycle)]
            base_env._command_ranges[2, :] = command_height
            base_env._commands[:, 2] = command_height
        if args_cli.yaw_cycle is not None:
            command_yaw = args_cli.yaw_cycle[phase % len(args_cli.yaw_cycle)]
            base_env._command_ranges[1, :] = command_yaw
            base_env.cfg.commands.heading_command = (command_yaw == 0.0 and not args_cli.heading_feedback
                                                     and not args_cli.constant_yaw_rate)
            if command_yaw != 0.0 or args_cli.heading_feedback or args_cli.constant_yaw_rate:
                base_env._commands[:, 1] = command_yaw
            elif phase != last_velocity_phase and not args_cli.heading_feedback:
                # Hold the heading reached after a turn, instead of returning to the initial heading.
                q = base_env._robot.data.root_quat_w
                base_env._commands[:, 3] = torch.atan2(
                    2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                    1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2),
                )
                base_env._commands[:, 1] = 0.0
        # The policy receives the ramped command already applied by the environment.
        command_obs = base_env._commands[:, :3] * base_env._commands_scale
        policy_obs = (
            observation["policy"]
            if hasattr(observation, "keys") and "policy" in observation.keys()
            else observation
        )
        policy_obs[:, 6:9] = command_obs
        if phase != last_velocity_phase:
            print(
                f"[COMMAND phase={phase} step={step}] "
                f"lin_vel_x={command_speed:.3f} m/s, "
                f"yaw_rate={base_env._commands[0, 1].item():.3f} rad/s, "
                f"height={base_env._commands[0, 2].item():.3f} m",
                flush=True,
            )
            last_velocity_phase = phase

    trace_file = None
    trace_writer = None
    if args_cli.trace_csv is not None:
        if not 0 <= args_cli.trace_env_id < env.unwrapped.num_envs:
            raise ValueError(
                f"--trace_env_id must be in [0, {env.unwrapped.num_envs - 1}], "
                f"got {args_cli.trace_env_id}."
            )
        trace_path = Path(args_cli.trace_csv).expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_file = trace_path.open("w", newline="", encoding="utf-8")
        print(f"[INFO] Writing replay diagnostics to: {trace_path}")

    camera_controller = getattr(env.unwrapped, "viewport_camera_controller", None)
    camera_mode = "free"
    camera_window = None
    camera_status = None
    camera_subscription = None
    previous_robot_position = None

    def robot_camera_position():
        return env.unwrapped.scene[env_cfg.viewer.asset_name or "robot"].data.root_pos_w[
            env_cfg.viewer.env_index
        ].detach().cpu().tolist()

    def set_camera_mode(mode):
        nonlocal camera_mode, previous_robot_position
        if camera_controller is None:
            return
        camera_mode = mode
        previous_robot_position = robot_camera_position()
        if mode == "follow":
            camera_controller.update_view_to_asset_root(env_cfg.viewer.asset_name or "robot")
        else:
            # Keep the current view, but stop the per-frame asset tracking callback
            # from overwriting mouse orbit/pan/zoom edits.
            camera_controller.cfg.origin_type = "world"
        if camera_status is not None:
            camera_status.text = {
                "orbit": "Mode: Follow + orbit (angle and zoom remain adjustable)",
                "free": "Mode: Free view (camera stays in world space)",
                "follow": "Mode: Fixed follow (angle locked)",
            }[mode]

    def center_camera():
        if camera_controller is None:
            return
        mode = camera_mode
        camera_controller.update_view_to_asset_root(env_cfg.viewer.asset_name or "robot")
        set_camera_mode(mode)

    def update_orbit_camera(event):
        nonlocal previous_robot_position
        if camera_mode != "orbit":
            return
        position = robot_camera_position()
        delta = Gf.Vec3d(*(now - old for now, old in zip(position, previous_robot_position)))
        previous_robot_position = position
        viewport = get_active_viewport()
        if viewport is not None:
            state = ViewportCameraState(viewport=viewport)
            # Translate the *current* mouse-edited camera and its orbit pivot together.
            # rotate=False preserves orientation, zoom distance and local center of interest.
            state.set_position_world(state.position_world + delta, False)

    if camera_controller is not None and not args_cli.headless and not args_cli.video:
        import omni.ui as ui
        import omni.kit.app
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf

        # Place the camera at the familiar robot view once, then release it for navigation.
        center_camera()
        camera_window = ui.Window("Camera controls", width=480, height=125, position_x=340, position_y=60)
        with camera_window.frame:
            with ui.VStack(spacing=5):
                camera_status = ui.Label("", height=22)
                with ui.HStack(height=30, spacing=5):
                    ui.Button("Follow + orbit", clicked_fn=lambda: set_camera_mode("orbit"))
                    ui.Button("Free view", clicked_fn=lambda: set_camera_mode("free"))
                    ui.Button("Fixed follow", clicked_fn=lambda: set_camera_mode("follow"))
                    ui.Button("Center robot", clicked_fn=center_camera)
                ui.Label("Over scene: Alt + left drag = orbit; wheel = zoom.", height=22)
        set_camera_mode(args_cli.camera_mode)
        camera_subscription = omni.kit.app.get_app_interface().get_post_update_event_stream().create_subscription_to_pop(
            update_orbit_camera
        )
        print(f"[INFO] Interactive camera mode: {args_cli.camera_mode}.", flush=True)

    live_window = None
    live_labels = {}
    live_plots = {}
    live_history = {}
    live_chart_titles = {}
    live_error_labels = {}
    live_error_plots = {}
    next_live_update = 0.0
    live_wall_start = time.monotonic()
    startup_capture = None
    startup_labels, startup_plots = {}, {}
    startup_trial = 1
    if args_cli.startup_focus:
        from startup_metrics import StartupCapture
        startup_capture = StartupCapture(dt)
    if args_cli.live_stats and not args_cli.headless:
        import omni.ui as ui
        import omni.appwindow
        from collections import deque

        live_chart_specs = {
            "yaw": ("Yaw rate (rad/s)", -0.5, 0.5, ("cmd_yaw", "yaw_rate_body"), ("Command", "Measured")),
            "yaw_angle": ("Yaw turn since reset (deg)", -90.0, 90.0,
                          ("yaw_turn_reference_deg", "yaw_turn_measured_deg"),
                          (("Target turn" if args_cli.heading_feedback else "Requested rate integral"), "Measured turn")),
            "speed": ("Forward speed (m/s)", -1.0, 1.0, ("cmd_x_target", "vel_x_heading", "cmd_x"),
                      ("Target", "Measured", "Ramped cmd")),
            "height": ("Body height (mm)", 140.0, 220.0, ("height_target_mm", "height_measured_mm"),
                       ("Target", "Measured")),
            "body": ("Body angle (deg)", -5.0, 5.0, ("zero", "pitch_deg", "roll_deg"),
                     ("Upright target", "Pitch", "Roll")),
            "symmetry": ("Left - right leg angle (deg)", -3.0, 3.0, ("zero", "leg_difference_deg"),
                         ("Target", "Measured")),
        }
        # Use the loaded robot's height range, not the legacy 0.14--0.22 m
        # chart base which compresses the own-model height adjustments.
        chart_height_min, chart_height_max = (
            (keyboard_config.height_min, keyboard_config.height_max) if keyboard_config is not None
            else env.unwrapped.cfg.commands.ranges_height)
        live_chart_specs["height"] = (
            "Body height (mm)", chart_height_min * 1000 - 10, chart_height_max * 1000 + 10,
            ("height_target_mm", "height_measured_mm"), ("Target", "Measured"))
        if keyboard_config is not None:
            live_chart_specs["height"] = (
                "Body height (mm)", chart_height_min * 1000 - 10, chart_height_max * 1000 + 10,
                ("height_goal_mm", "height_target_mm", "height_measured_mm"),
                ("Keyboard goal", "Ramped cmd", "Measured"))
        yaw_chart_limit = max(0.5, keyboard_config.yaw_rate) if keyboard_config is not None else 0.5
        live_chart_specs["yaw"] = (
            "Yaw rate (rad/s)", -yaw_chart_limit, yaw_chart_limit,
            ("cmd_yaw", "yaw_rate_body"), ("Command", "Measured"))
        if args_cli.heading_feedback:
            live_chart_specs["yaw"] = ("Yaw rate (rad/s)", -0.5, 0.5,
                                       ("yaw_rate_requested", "yaw_rate_body", "cmd_yaw"),
                                       ("Requested", "Measured", "Corrected cmd"))
        colors = (0xFF43C8FF, 0xFFFFAD54, 0xFF8DD781)
        app_window = omni.appwindow.get_default_app_window()
        live_window = ui.Window("Wheel-legged commands and feedback", width=760,
                               height=min(1200, app_window.get_height() - 80),
                               position_x=max(0, app_window.get_width() - 784), position_y=40)
        with live_window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=4):
                    if startup_capture is not None:
                        ui.Label("STARTUP LEG ANGLES | first 5 seconds stay visible", height=24)
                        startup_labels["status"] = ui.Label("Collecting at control frequency...", height=22)
                        ui.Label("Settling band: final 2-second mean +/-0.5 deg (not a policy target)", height=20)
                        for side, color in zip(("left", "right"), colors):
                            startup_labels[side] = ui.Label(f"{side.title()}: collecting...", height=24)
                            with ui.ZStack(height=100):
                                ui.Rectangle(style={"background_color": 0xFF242424})
                                for key, line_color in (("measured", color), ("low", 0xFF999999), ("high", 0xFF999999)):
                                    startup_plots[(side, key)] = ui.Plot(
                                        ui.Type.LINE, -12., 12., 0., 0.,
                                        style={"color": line_color, "background_color": 0x0})
                        ui.Label("X: seconds since startup | Y: leg angle in degrees | gray: settling band", height=20)
                    for field in ("progress", "speed", "yaw", "heading", "height", "body", "angles", "left", "right", "status"):
                        live_labels[field] = ui.Label("Waiting for simulation...", height=20)
                    live_labels["plot_time"] = ui.Label("Time window: waiting for samples", height=20)
                    ui.Label("Right plots: error in pink; limits in gray. Orange number = outside band.", height=20)
                    ui.Label("Yaw turn reference integrates rate commands; it is not an angle command.", height=20)
                    for name, (title, low, high, keys, legends) in live_chart_specs.items():
                        live_chart_titles[name] = ui.Label(title, height=20)
                        with ui.HStack(height=18):
                            for legend, color in zip(legends, colors):
                                ui.Label(legend, style={"color": color})
                        with ui.HStack(height=100, spacing=10):
                            with ui.ZStack():
                                ui.Rectangle(style={"background_color": 0xFF242424})
                                for key, color in zip(keys, colors):
                                    plot = ui.Plot(ui.Type.LINE, low, high, 0.0, 0.0,
                                                   style={"color": color, "background_color": 0x0})
                                    live_plots[(name, key)] = plot
                                    live_history.setdefault(key, deque(maxlen=200))
                            with ui.VStack(width=235, spacing=2):
                                for field in ("value", "band", "peak"):
                                    live_error_labels[(name, field)] = ui.Label("Waiting...", height=20)
                                with ui.ZStack(height=38):
                                    ui.Rectangle(style={"background_color": 0xFF191919})
                                    for suffix, color in (("zero", 0xFF555555), ("low", 0xFF999999),
                                                          ("high", 0xFF999999), ("value", 0xFF8982FF)):
                                        key = f"error_{name}_{suffix}"
                                        live_history[key] = deque(maxlen=200)
                                        live_error_plots[(name, suffix)] = ui.Plot(
                                            ui.Type.LINE, -1.0, 1.0, 0.0, 0.0,
                                            style={"color": color, "background_color": 0x0})
        print("[INFO] Live command/feedback panel enabled: six rolling charts including yaw angle, 10 Hz simulation samples.", flush=True)

    # wrapper 构造时已 reset。这里计算当前观测，不再重置或推进物理时间；
    # Direct 环境仍会调用 _get_observations()，其中包含命令/动作历史的更新。
    obs = env.get_observations()
    track_yaw = trace_file is not None or bool(live_labels) or args_cli.heading_feedback
    heading_feedback = HeadingFeedback() if args_cli.heading_feedback else None
    yaw_turn_reference = 0.0
    yaw_turn_measured = 0.0

    def measured_yaw_degrees():
        q = env.unwrapped._robot.data.root_quat_w[args_cli.trace_env_id]
        return math.degrees(torch.atan2(
            2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
        ).item())

    previous_yaw = measured_yaw_degrees() if track_yaw else 0.0
    timestep = 0
    # simulation_app 是 Kit 应用句柄；is_running() 判断应用是否仍运行。
    while simulation_app.is_running():
        start_time = time.time()
        # PyTorch inference_mode 禁用自动求导和训练图，不会冻结 PhysX 中的机器人。
        with torch.inference_mode():
            apply_velocity_cycle_command(obs, timestep)
            keyboard_command = keyboard_controller.advance(dt) if keyboard_controller is not None else None
            apply_keyboard_command(env.unwrapped, obs, keyboard_command)
            requested_yaw_rate = (args_cli.yaw_cycle[(timestep // velocity_phase_steps) % len(args_cli.yaw_cycle)]
                                  if args_cli.yaw_cycle is not None else
                                  (args_cli.fixed_command[1] if args_cli.fixed_command is not None else
                                   env.unwrapped._commands[args_cli.trace_env_id, 1].item()))
            if keyboard_command is not None:
                requested_yaw_rate = keyboard_command[1]
            if heading_feedback is not None:
                corrected_yaw_rate = heading_feedback.step(requested_yaw_rate, math.radians(yaw_turn_measured), dt)
                env.unwrapped._commands[:, 1] = corrected_yaw_rate
                policy_obs = obs["policy"] if hasattr(obs, "keys") and "policy" in obs.keys() else obs
                policy_obs[:, 7] = corrected_yaw_rate * env.unwrapped._commands_scale[1]
            if track_yaw:
                # Capture the command actually presented to the policy for this step.
                yaw_rate_applied = env.unwrapped._commands[args_cli.trace_env_id, 1].item()
            # 推理输入为 LAB wrapper 返回的观测组；动作如何变成力矩由环境决定。
            actions = policy(obs)
            if args_cli.zero_actions:
                actions = torch.zeros_like(actions)
            if fixed_leg_action is not None:
                actions = actions.clone()
                actions[:, (1, 4)] = fixed_leg_action
            if (
                args_cli.fixed_wheel_action is not None
                and timestep >= args_cli.fixed_wheel_action_start
            ):
                actions = actions.clone()
                actions[:, 2] = args_cli.fixed_wheel_action[0]
                actions[:, 5] = args_cli.fixed_wheel_action[1]
            # wrapper.step() 内部走 LAB：动作预处理 → 4 个物理子步 → 终止/奖励
            # → 必要的自动 reset → 新观测。这里忽略 reward/done/extras，并非没计算。
            obs, _, _, _ = env.step(actions)
            if track_yaw:
                yaw_degrees = measured_yaw_degrees()
                did_reset = bool(env.unwrapped.reset_terminated[args_cli.trace_env_id]) or bool(
                    env.unwrapped.reset_time_outs[args_cli.trace_env_id]
                )
                if did_reset:
                    yaw_turn_reference = yaw_turn_measured = 0.0
                    if heading_feedback is not None:
                        heading_feedback.reset()
                    for history in live_history.values():
                        history.clear()
                    if startup_capture is not None:
                        startup_capture = StartupCapture(dt)
                        startup_trial += 1
                        for plot in startup_plots.values():
                            plot.set_data(0.0, 0.0)
                else:
                    if heading_feedback is not None:
                        yaw_turn_reference = math.degrees(heading_feedback.reference)
                    else:
                        yaw_turn_reference += math.degrees(requested_yaw_rate * dt)
                    # Unwrap +/-180-degree crossings before accumulating the measured turn.
                    yaw_turn_measured += (yaw_degrees - previous_yaw + 180.0) % 360.0 - 180.0
                previous_yaw = yaw_degrees
            if startup_capture is not None and not did_reset:
                angles = env.unwrapped._theta0[args_cli.trace_env_id].detach().cpu().tolist()
                was_complete = startup_capture.complete
                startup_capture.add(*angles)
                if startup_capture.complete and not was_complete:
                    print(f"[STARTUP trial={startup_trial}] {json.dumps(startup_capture.report())}", flush=True)
            apply_velocity_cycle_command(obs, timestep)
            write_trace = trace_file is not None and timestep % args_cli.trace_interval == 0
            update_live = bool(live_labels) and time.monotonic() >= next_live_update
            sample_live = bool(live_labels) and timestep % max(1, round(0.1 / dt)) == 0
            if write_trace or update_live or sample_live:
                base_env = env.unwrapped
                env_id = args_cli.trace_env_id
                # LAB .data 提供状态张量：_w 是世界系、_b 是机体系；不能混用速度分量。
                # 当前显示的前向/侧向速度另投影到水平航向系，不随 pitch/roll 倾斜。
                heading_velocity = base_env._get_heading_frame_horizontal_velocity()[env_id]
                gravity = base_env._robot.data.projected_gravity_b[env_id]
                wheel_velocity = base_env._get_wheel_vel_forward_positive()[env_id]
                # 隐式执行器的 applied_torque 是 LAB 估计量，不是原生 PD 实测反馈力矩。
                logical_torque = base_env._robot.data.applied_torque[
                    env_id, base_env._torque_joint_ids
                ]
                row = {
                    "step": timestep,
                    "sim_time_s": timestep * dt,
                    "env_id": env_id,
                    # LAB 在 step 返回前已自动 reset；done 行的状态可能属于新回合。
                    # 因此终止标志必须一起保存，不能把重置后的状态算作旧回合稳态。
                    "terminated": int(base_env.reset_terminated[env_id].item()),
                    "time_out": int(base_env.reset_time_outs[env_id].item()),
                    "episode_step": int(base_env.episode_length_buf[env_id].item()),
                    "cmd_x": base_env._commands[env_id, 0].item(),
                    "cmd_x_target": base_env._target_lin_vel_x[env_id].item(),
                    "cmd_yaw": yaw_rate_applied if heading_feedback is not None else base_env._commands[env_id, 1].item(),
                    "yaw_world_deg": yaw_degrees,
                    "yaw_rate_applied": yaw_rate_applied,
                    "yaw_turn_reference_deg": yaw_turn_reference,
                    "yaw_turn_measured_deg": yaw_turn_measured,
                    "yaw_turn_error_deg": yaw_turn_measured - yaw_turn_reference,
                    "heading_hold": int(base_env.cfg.commands.heading_command),
                    "heading_feedback": int(heading_feedback is not None),
                    "keyboard_active": int(keyboard_command is not None),
                    "keyboard_jump_requests": keyboard_controller.state.jump_requests if keyboard_controller else 0,
                    "yaw_rate_requested": requested_yaw_rate,
                    "height_cmd": base_env._commands[env_id, 2].item(),
                    "keyboard_height_target": (keyboard_controller.state.target_height
                                               if keyboard_command is not None
                                               else base_env._commands[env_id, 2].item()),
                    "vel_x_heading": heading_velocity[0].item(),
                    "vel_y_heading": heading_velocity[1].item(),
                    "vel_z_body": base_env._robot.data.root_lin_vel_b[env_id, 2].item(),
                    "yaw_rate_body": base_env._robot.data.root_ang_vel_b[env_id, 2].item(),
                    "base_height": base_env._base_height[env_id].item(),
                    "gravity_x": gravity[0].item(),
                    "gravity_y": gravity[1].item(),
                    "gravity_z": gravity[2].item(),
                    "pitch_est_rad": torch.asin(torch.clamp(gravity[0], -1.0, 1.0)).item(),
                    "roll_est_rad": torch.asin(torch.clamp(gravity[1], -1.0, 1.0)).item(),
                    "theta_left": base_env._theta0[env_id, 0].item(),
                    "theta_right": base_env._theta0[env_id, 1].item(),
                    "length_left": base_env._L0[env_id, 0].item(),
                    "length_right": base_env._L0[env_id, 1].item(),
                    "target_length_left": base_env._l0_ref_applied[env_id, 0].item(),
                    "target_length_right": base_env._l0_ref_applied[env_id, 1].item(),
                    "wheel_vel_left": wheel_velocity[0].item(),
                    "wheel_vel_right": wheel_velocity[1].item(),
                }
                row.update(
                    {f"action_{index}": value.item() for index, value in enumerate(actions[env_id])}
                )
                row.update(
                    {f"torque_{index}": value.item() for index, value in enumerate(logical_torque)}
                )
                reward_terms = getattr(base_env, "_last_reward_terms", {})
                row.update(
                    {
                        f"reward_{key}": value[env_id].item()
                        for key, value in reward_terms.items()
                    }
                )
                row["reward_total"] = sum(
                    row[key] for key in row if key.startswith("reward_")
                )
                if write_trace:
                    if trace_writer is None:
                        trace_writer = csv.DictWriter(trace_file, fieldnames=list(row))
                        trace_writer.writeheader()
                    trace_writer.writerow(row)
                    trace_file.flush()
                if live_labels:
                    error_values = {
                        "speed": (row["vel_x_heading"] - row["cmd_x_target"],
                                  abs(row["cmd_x_target"]) * 0.05 if abs(row["cmd_x_target"]) > 1e-6 else 0.02),
                        "height": (1000 * (row["base_height"] - row["height_cmd"]), 50 * row["height_cmd"]),
                        "yaw": (row["yaw_rate_body"] - row["cmd_yaw"], 0.0),
                        "yaw_angle": (row["yaw_turn_error_deg"], 0.0),
                        "body": (57.29578 * max(abs(row["pitch_est_rad"]), abs(row["roll_est_rad"])), 3.0),
                        "symmetry": (57.29578 * (row["theta_left"] - row["theta_right"]), 3.0),
                    }
                if sample_live:
                    chart_row = dict(row, zero=0.0,
                                     height_goal_mm=1000 * row["keyboard_height_target"],
                                     height_target_mm=1000 * row["height_cmd"],
                                     height_measured_mm=1000 * row["base_height"],
                                     pitch_deg=57.29578 * row["pitch_est_rad"],
                                     roll_deg=57.29578 * row["roll_est_rad"],
                                     leg_difference_deg=57.29578 * (row["theta_left"] - row["theta_right"]))
                    for name, (value, band) in error_values.items():
                        chart_row.update({f"error_{name}_value": value, f"error_{name}_high": band,
                                          f"error_{name}_low": -band, f"error_{name}_zero": 0.0})
                    for key, history in live_history.items():
                        history.append(chart_row[key])
                if update_live:
                    if startup_capture is not None:
                        duration = len(startup_capture.samples) * dt
                        startup_labels["status"].text = (
                            f"Trial {startup_trial} | {'FROZEN' if startup_capture.complete else 'COLLECTING'} | "
                            f"0 to {duration:.2f} s | Kp {base_env.cfg.kp_theta:g}, Kd {base_env.cfg.kd_theta:g}")
                        report = startup_capture.report()
                        for index, side in enumerate(("left", "right")):
                            values = [sample[index] for sample in startup_capture.samples]
                            span = max(2., max(map(abs, values), default=0.) * 1.15)
                            for key in ("measured", "low", "high"):
                                plot = startup_plots[(side, key)]
                                plot.scale_min, plot.scale_max = -span, span
                            if len(values) >= 2:
                                startup_plots[(side, "measured")].set_data(*values)
                            if report is not None:
                                result = report[side]
                                settling = (f"{result['settling_s']:.2f} s" if result['settling_s'] is not None else ">5 s")
                                startup_labels[side].text = (
                                    f"{side.title()}: settle {settling} | peak {result['peak_abs_deg']:.2f} deg | "
                                    f"mean {result['steady_deg']:+.2f} deg | 2s travel {result['variation_first_2s_deg']:.2f} deg")
                                for key, offset in (("low", -.5), ("high", .5)):
                                    startup_plots[(side, key)].set_data(*([result["steady_deg"] + offset] * len(values)))
                            else:
                                startup_labels[side].text = f"{side.title()}: collecting | peak {max(map(abs, values), default=0.):.2f} deg"
                    elapsed = time.monotonic() - live_wall_start
                    live_labels["progress"].text = (
                        f"Sim {(timestep + 1) * dt:.1f} s | step {timestep + 1} | "
                        f"{(timestep + 1) / max(elapsed, 1e-6):.1f} steps/s"
                    )
                    live_labels["speed"].text = (
                        f"Vx measured {row['vel_x_heading']:+.3f} | target {row['cmd_x_target']:+.2f} | "
                        f"ramped cmd {row['cmd_x']:+.3f} m/s"
                    )
                    live_labels["yaw"].text = (
                        f"Yaw rate measured {row['yaw_rate_body']:+.3f} | cmd {row['cmd_yaw']:+.3f} rad/s | "
                        + (f"requested {row['yaw_rate_requested']:+.3f}" if heading_feedback is not None else
                           f"Vy {row['vel_y_heading']:+.3f} m/s")
                    )
                    live_labels["heading"].text = (
                        f"Yaw angle {row['yaw_world_deg']:+.1f} deg | "
                        + ("Tracking target heading (PI)" if heading_feedback is not None else
                           ("Holding reached heading" if row["heading_hold"] else "Turning (yaw-rate command)"))
                    )
                    live_labels["height"].text = (
                        f"Body height: {1000 * row['base_height']:.1f} / "
                        f"target {1000 * row['height_cmd']:.0f} mm"
                    )
                    live_labels["body"].text = (
                        f"Pitch {57.29578 * row['pitch_est_rad']:+.2f} | "
                        f"Roll {57.29578 * row['roll_est_rad']:+.2f} deg | target 0"
                    )
                    live_labels["angles"].text = (
                        f"Leg angle L {57.29578 * row['theta_left']:+.2f} | "
                        f"R {57.29578 * row['theta_right']:+.2f} | "
                        f"diff {57.29578 * abs(row['theta_left'] - row['theta_right']):.2f} deg | target diff 0"
                    )
                    for side in ("left", "right"):
                        live_labels[side].text = (
                            f"{side.title()} leg: {1000 * row[f'length_{side}']:.1f} / "
                            f"target {1000 * row[f'target_length_{side}']:.1f} mm"
                        )
                    live_labels["status"].text = (
                        f"Errors: Vx {row['vel_x_heading'] - row['cmd_x_target']:+.3f} m/s | "
                        f"height {1000 * (row['base_height'] - row['height_cmd']):+.1f} mm"
                    )
                    count = len(live_history["zero"])
                    sample_step = max(1, round(0.1 / dt))
                    last_sample_time = (timestep // sample_step) * sample_step * dt
                    live_labels["plot_time"].text = (
                        f"Charts (10 Hz): simulation {max(0.0, last_sample_time - (count - 1) * sample_step * dt):.1f}"
                        f" to {last_sample_time:.1f} s, oldest at left / newest at right"
                    )
                    for name, (title, low, high, keys, _) in live_chart_specs.items():
                        values = [v for key in keys for v in live_history[key]]
                        if values:
                            low, high = min(low, min(values)), max(high, max(values))
                            margin = (high - low) * 0.03
                            live_chart_titles[name].text = f"{title} | range {low:.2f} to {high:.2f}"
                            for key in keys:
                                plot = live_plots[(name, key)]
                                plot.scale_min, plot.scale_max = low - margin, high + margin
                                plot.set_data(*live_history[key])
                        error_samples = live_history[f"error_{name}_value"]
                        if error_samples:
                            error, band = error_values[name]
                            peak = max(abs(v) for v in error_samples)
                            unit = {"speed": "m/s", "height": "mm", "yaw": "rad/s", "yaw_angle": "deg",
                                    "body": "deg", "symmetry": "deg"}[name]
                            decimals = 3 if name in ("speed", "yaw") else 2
                            live_error_labels[(name, "value")].text = f"Error {error:+.{decimals}f} {unit}"
                            if name == "speed" and abs(row["cmd_x_target"]) < 1e-6:
                                drift = (row["vel_x_heading"] ** 2 + row["vel_y_heading"] ** 2) ** 0.5
                                detail = f"Drift {drift:.3f} / limit 0.020"
                                outside = drift > .02
                            elif name in ("speed", "height"):
                                percent = abs(error) / band * 5.0
                                detail = f"{percent:.2f}% / limit 5%"
                                outside = abs(error) > band
                            elif name == "yaw":
                                detail = (f"{100 * abs(error / row['cmd_yaw']):.1f}% of command"
                                          if not row["heading_hold"] and abs(row["cmd_yaw"]) > 1e-6
                                          else "Heading hold: absolute error")
                                outside = False
                            elif name == "yaw_angle":
                                detail, outside = "Measured - rate integral", False
                            else:
                                detail, outside = f"|error| {abs(error):.2f} / limit 3 deg", abs(error) > band
                            live_error_labels[(name, "band")].text = detail
                            live_error_labels[(name, "value")].style = {
                                "color": 0xFFE0E0E0 if name in ("yaw", "yaw_angle") else (0xFF55A5FF if outside else 0xFF8DD781)
                            }
                            live_error_labels[(name, "peak")].text = f"Sampled peak {peak:.{decimals}f} {unit}"
                            span = max(peak, max(live_history[f"error_{name}_high"]), .01) * 1.15
                            for suffix in ("zero", "low", "high", "value"):
                                plot = live_error_plots[(name, suffix)]
                                plot.scale_min, plot.scale_max = -span, span
                                plot.set_data(*live_history[f"error_{name}_{suffix}"])
                    next_live_update = time.monotonic() + 0.1
            if args_cli.print_obs and timestep % args_cli.print_obs_interval == 0:
                policy_obs = obs["policy"] if hasattr(obs, "keys") and "policy" in obs.keys() else obs
                obs_values = policy_obs[0].detach().cpu().tolist()
                print(
                    f"[OBS step={timestep}] policy="
                    f"{[round(value, 5) for value in obs_values]}",
                    flush=True,
                )
                if len(obs_values) == 27:
                    leg_fields = (
                        f"active_joint_position_error={obs_values[9:13]}, active_joint_velocity={obs_values[13:17]}, "
                        if env_cfg.leg_control_mode == 'implicit_joint_position' else
                        f"theta0={obs_values[9:11]}, theta0_dot={obs_values[11:13]}, "
                        f"leg_length={obs_values[13:15]}, leg_length_dot={obs_values[15:17]}, "
                    )
                    print(
                        "[OBS fields] "
                        f"ang_vel={obs_values[0:3]}, gravity={obs_values[3:6]}, "
                        f"command={obs_values[6:9]}, {leg_fields}lin_vel_xy={obs_values[17:19]}, "
                        f"wheel_vel={obs_values[19:21]}, last_action={obs_values[21:27]}",
                        flush=True,
                    )
            if args_cli.print_vmc and timestep % args_cli.print_obs_interval == 0:
                base_env = env.unwrapped
                theta0 = base_env._theta0[0].detach().cpu().tolist()
                leg_length = base_env._L0[0].detach().cpu().tolist()
                applied_torque = base_env._robot.data.applied_torque[0].detach().cpu().tolist()
                print(
                    "[VMC symmetry] "
                    f"theta0={[round(value, 6) for value in theta0]}, "
                    f"theta_diff={theta0[0] - theta0[1]:.6f}, "
                    f"L0={[round(value, 6) for value in leg_length]}, "
                    f"L0_diff={leg_length[0] - leg_length[1]:.6f}, "
                    f"joint_names={base_env._robot.joint_names}, "
                    f"applied_torque={[round(value, 5) for value in applied_torque]}",
                    flush=True,
                )
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close diagnostics and simulator
    if keyboard_controller is not None:
        keyboard_controller.close()
    if trace_file is not None:
        trace_file.close()
    if live_window is not None:
        live_window.destroy()
    if camera_subscription is not None:
        camera_subscription.unsubscribe()
    if camera_window is not None:
        camera_window.destroy()
    # 关闭 LAB 环境，释放场景、传感器和回调；外层 finally 再关闭 Kit 应用。
    env.close()


if __name__ == "__main__":
    # run the main function
    try:
        main()
    finally:
        # Release the simulator even if inference, input setup or tracing fails.
        simulation_app.close()
