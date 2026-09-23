# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL (external project: wheel_legged_gym_isaaclab)."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
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
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
import wheel_legged_gym_isaaclab.tasks  # noqa: F401  (registers WheelLeggedVMC-Flat-v0)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    if args_cli.print_obs_interval <= 0:
        raise ValueError("--print_obs_interval must be greater than zero.")
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Evaluation should be repeatable.  Disable the heading-to-yaw controller
    # and collapse command ranges to a supplied constant command when asked.
    if args_cli.fixed_command is not None:
        lin_vel_x, yaw_rate, height = args_cli.fixed_command
        env_cfg.commands.heading_command = False
        env_cfg.commands.ranges_lin_vel_x = (lin_vel_x, lin_vel_x)
        env_cfg.commands.ranges_ang_vel_yaw = (yaw_rate, yaw_rate)
        env_cfg.commands.ranges_height = (height, height)
        print(
            "[INFO] Using fixed evaluation command: "
            f"lin_vel_x={lin_vel_x:.3f} m/s, yaw_rate={yaw_rate:.3f} rad/s, height={height:.3f} m"
        )

    # Do not alter ``l0_offset`` for a single replay experiment: it is part of
    # the training task definition.  Instead convert a requested target length
    # to the normalized VMC leg-length action and override only action indices
    # 1 and 4 (left/right l0) after policy inference.
    fixed_leg_action = None
    if args_cli.fixed_leg_length is not None:
        if not hasattr(env_cfg, "l0_offset") or not hasattr(env_cfg, "action_scale_l0"):
            raise ValueError("--fixed_leg_length is supported only by WheelLeggedVMC tasks.")
        fixed_leg_action = (args_cli.fixed_leg_length - env_cfg.l0_offset) / env_cfg.action_scale_l0
        if not -1.0 <= fixed_leg_action <= 1.0:
            min_length = env_cfg.l0_offset - env_cfg.action_scale_l0
            max_length = env_cfg.l0_offset + env_cfg.action_scale_l0
            raise ValueError(
                f"Requested leg length {args_cli.fixed_leg_length:.3f} m is outside the policy action range "
                f"[{min_length:.3f}, {max_length:.3f}] m."
            )
        print(
            "[INFO] Fixing both virtual-leg targets at "
            f"{args_cli.fixed_leg_length:.3f} m (normalized action {fixed_leg_action:.3f})."
        )

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
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

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            if args_cli.zero_actions:
                actions = torch.zeros_like(actions)
            if fixed_leg_action is not None:
                actions = actions.clone()
                actions[:, (1, 4)] = fixed_leg_action
            # env stepping
            obs, _, _, _ = env.step(actions)
            if args_cli.print_obs and timestep % args_cli.print_obs_interval == 0:
                policy_obs = obs["policy"] if hasattr(obs, "keys") and "policy" in obs.keys() else obs
                obs_values = policy_obs[0].detach().cpu().tolist()
                print(
                    f"[OBS step={timestep}] policy="
                    f"{[round(value, 5) for value in obs_values]}",
                    flush=True,
                )
                if len(obs_values) == 27:
                    print(
                        "[OBS fields] "
                        f"ang_vel={obs_values[0:3]}, gravity={obs_values[3:6]}, "
                        f"command={obs_values[6:9]}, theta0={obs_values[9:11]}, "
                        f"theta0_dot={obs_values[11:13]}, leg_length={obs_values[13:15]}, "
                        f"leg_length_dot={obs_values[15:17]}, wheel_pos={obs_values[17:19]}, "
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

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
