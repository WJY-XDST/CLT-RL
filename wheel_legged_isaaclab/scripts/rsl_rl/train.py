# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL (external project: wheel_legged_gym_isaaclab)."""

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
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument('--stop_file', type=str, default=None,
                    help='Save a complete checkpoint and exit after an update when this marker exists.')
parser.add_argument(
    "--target_iteration", type=int, default=None,
    help="Stop at a total PPO iteration target, including iterations loaded from a checkpoint.",
)
parser.add_argument("--reset_exploration_std", type=float, default=None,
                    help="Reset policy exploration after loading a failed pilot checkpoint.")
parser.add_argument("--wheel_exploration_std", type=float, default=None,
                    help="Set wheel-action exploration only, preserving stable leg exploration and optimizer state.")
parser.add_argument("--min_exploration_std", type=float, default=1.e-4,
                    help="Positive floor for all learned exploration std entries, enforced after optimizer steps.")
parser.add_argument('--min_wheel_exploration_std', type=float, default=None,
                    help='Optional persistent floor for wheel entries 2/5 in direct joint PD; retain leg noise and Adam.')
parser.add_argument('--joint_height_feedback_coef', type=float, default=None, help='Optional forward-height PPO training loss; no action override')
parser.add_argument('--joint_mean_guard_coef', type=float, default=None,
                    help='Optional PPO loss for raw leg means outside [-1, 1]; inherit saved coefficient on resume.')
parser.add_argument('--require_leg_control_mode', choices=('explicit_vmc','implicit_joint_position'), default=None,
                    help='Fail if Hydra overrides select a different leg execution chain.')
parser.add_argument('--allow_physics_contract_change', action='store_true',
                    help='Explicitly allow a different robot asset or external-effort solver flag for a new experiment.')
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default=None,
    help="Resume from an explicit checkpoint path, including a checkpoint stored in this repository.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
import wheel_legged_gym_isaaclab.tasks  # noqa: F401  (registers WheelLeggedVMC-Flat-v0)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.checkpoint_path is not None:
        agent_cfg.resume = True
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    from wheel_legged_gym_isaaclab.leg_control import require_leg_control_mode
    require_leg_control_mode(env_cfg.leg_control_mode, args_cli.require_leg_control_mode)
    print(f'[INFO]: Resolved leg control mode: {env_cfg.leg_control_mode}')

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
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume:
        if args_cli.checkpoint_path is not None:
            resume_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint_path))
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
        source_config = validate_checkpoint_robot(resume_path)
        from wheel_legged_gym_isaaclab.leg_control import require_matching_policy_contract
        require_matching_policy_contract(source_config, env_cfg.to_dict())
        source_mode = source_config.get('leg_control_mode')
        require_leg_control_mode(source_mode)
        require_leg_control_mode(env_cfg.leg_control_mode, source_mode)
        from wheel_legged_gym_isaaclab.physics_contract import (
            require_matching_force_iteration_setting, require_matching_robot_asset_setting,
        )
        require_matching_robot_asset_setting(
            source_config, env_cfg.robot.spawn.usd_path,
            allow_change=args_cli.allow_physics_contract_change,
        )
        require_matching_force_iteration_setting(
            source_config, env_cfg.enable_external_forces_every_iteration,
            allow_change=args_cli.allow_physics_contract_change,
        )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    from wheel_legged_gym_isaaclab.joint_mean_guard import (
        configured_runner_dict, register_runner_algorithm, resolve_coefficient,
    )
    joint_mean_guard_coef = resolve_coefficient(
        resume_path if agent_cfg.resume else None, args_cli.joint_mean_guard_coef,
    )
    from wheel_legged_gym_isaaclab.joint_height_feedback import resolve_coefficient as resolve_height, model_from_config
    joint_height_feedback_coef=resolve_height(resume_path if agent_cfg.resume else None,args_cli.joint_height_feedback_coef)
    height_model=(model_from_config(env_cfg.to_dict(),env.unwrapped._robot.data.joint_pos_limits[0,env.unwrapped._leg_joint_ids].cpu().tolist())
                  if joint_height_feedback_coef else None)
    if (joint_mean_guard_coef or joint_height_feedback_coef) and env_cfg.leg_control_mode != 'implicit_joint_position':
        raise ValueError('The joint mean guard is only defined for direct joint-position PD.')
    if joint_mean_guard_coef or joint_height_feedback_coef:
        register_runner_algorithm()
    runner_config = configured_runner_dict(agent_cfg.to_dict(), joint_mean_guard_coef,joint_height_feedback_coef,height_model)
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_config, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume:
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer)
        if args_cli.reset_optimizer:
            print("[INFO]: Loaded policy weights with a fresh optimizer state.")
        else:
            from wheel_legged_gym_isaaclab.training_state import restore_adaptive_learning_rate
            restored_lr = restore_adaptive_learning_rate(runner.alg)
            agent_cfg.algorithm.learning_rate = restored_lr
            print(f"[INFO]: Restored PPO adaptive learning rate from optimizer: {restored_lr}")
        # The environment curriculum is driven by control steps, whereas the
        # checkpoint stores PPO iterations. Keep both clocks aligned so a
        # resumed run does not silently return to the standing-only phase.
        if hasattr(env.unwrapped, "common_step_counter"):
            curriculum_step = int(runner.current_learning_iteration) * int(
                agent_cfg.num_steps_per_env
            )
            env.unwrapped.common_step_counter = curriculum_step
            print(
                "[INFO]: Restored environment curriculum step to "
                f"{curriculum_step} from iteration {runner.current_learning_iteration}."
            )

    # dump the configuration into log-directory
    if args_cli.reset_exploration_std is not None:
        if args_cli.reset_exploration_std <= 0:
            raise ValueError("Exploration std must be positive.")
        policy = runner.alg.policy
        with torch.no_grad():
            if hasattr(policy, "std"):
                policy.std.fill_(args_cli.reset_exploration_std)
            elif hasattr(policy, "log_std"):
                policy.log_std.fill_(float(torch.log(torch.tensor(args_cli.reset_exploration_std))))
            else:
                raise ValueError("Unsupported policy exploration parameter.")
        print(f"[INFO]: Reset exploration std to {args_cli.reset_exploration_std}")
    if args_cli.wheel_exploration_std is not None:
        from wheel_legged_gym_isaaclab.exploration import set_wheel_exploration_std
        set_wheel_exploration_std(runner.alg.policy, args_cli.wheel_exploration_std)
        print(f"[INFO]: Set wheel-only exploration std to {args_cli.wheel_exploration_std}; leg exploration retained.")
    from wheel_legged_gym_isaaclab.exploration import enforce_positive_exploration
    if args_cli.min_wheel_exploration_std is not None and env_cfg.leg_control_mode != 'implicit_joint_position':
        raise ValueError('A selective wheel exploration floor requires direct joint-position PD.')
    exploration_guard = enforce_positive_exploration(
        runner.alg.policy, runner.alg.optimizer, args_cli.min_exploration_std,
        minimum_wheel_std=args_cli.min_wheel_exploration_std)
    print(f'[INFO]: Learned exploration std floor: {args_cli.min_exploration_std}; all entries remain trainable.')
    if args_cli.min_wheel_exploration_std is not None:
        print(f'[INFO]: Wheel-only exploration floor: {args_cli.min_wheel_exploration_std}; saved leg noise retained.')
    # Checkpoints store weights, not gradient hooks. A rebuilt policy trains
    # every leg/wheel output and the shared features; partial-head mode retired.
    frozen = [name for name, parameter in runner.alg.policy.named_parameters()
              if not parameter.requires_grad]
    if frozen:
        raise ValueError(f'Full-policy training unexpectedly has frozen parameters: {frozen}')
    dump_yaml(os.path.join(log_dir, 'params', 'precision_mode.yaml'), {
        'mode': 'full_policy', 'source_checkpoint': resume_path if agent_cfg.resume else None,
        'trainable_actor_rows': list(range(env.num_actions)),
        'trainable_std_entries': list(range(env.num_actions)),
        'shared_actor_features_frozen': False, 'leg_output_rows_frozen': [],
        'critic_trainable': True, 'inference_contract_unchanged': True,
        'exploration_std_floor': args_cli.min_exploration_std,
        'wheel_exploration_std_floor': args_cli.min_wheel_exploration_std})
    print('[INFO]: Full-policy training: actor features, all leg/wheel outputs, exploration and critic train.')
    if args_cli.target_iteration is not None:
        remaining = args_cli.target_iteration - runner.current_learning_iteration
        if remaining <= 0:
            raise ValueError("Target iteration must exceed the loaded checkpoint iteration.")
        agent_cfg.max_iterations = remaining
        print(f"[INFO]: Target total iteration {args_cli.target_iteration}; training {remaining} remaining iterations.")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"),
              configured_runner_dict(agent_cfg.to_dict(), joint_mean_guard_coef,joint_height_feedback_coef,height_model))
    from wheel_legged_gym_isaaclab.mcu_hardware import MCU_LIMIT_EVIDENCE
    dump_yaml(os.path.join(log_dir, 'params', 'control_contract.yaml'), {
        'hardware_profile': env_cfg.hardware_profile,
        'hardware_limit_authority': MCU_LIMIT_EVIDENCE,
        'joint_action_scale_is_hardware_limit': False,
        'wheel_action_scale_is_hardware_limit': False,
        'leg_control_mode': env_cfg.leg_control_mode,
        'leg_action_order': (['LB_joint_position', 'LL_joint_position', 'LW_speed', 'RB_joint_position', 'RL_joint_position', 'RW_speed']
                             if env_cfg.leg_control_mode == 'implicit_joint_position' else
                             ['left_virtual_angle', 'left_virtual_length', 'LW_speed', 'right_virtual_angle', 'right_virtual_length', 'RW_speed']),
        'leg_observation_contract': ('active_joint_position_relative_to_fixed_offset4_and_velocity4' if env_cfg.leg_control_mode == 'implicit_joint_position'
                                     else 'virtual_angle2_velocity2_length2_velocity2'),
        'leg_joint_position_offset': list(env_cfg.leg_joint_position_offset),
        'leg_joint_position_scale': list(env_cfg.leg_joint_position_scale),
        'leg_joint_target_limit_margin': env_cfg.leg_joint_target_limit_margin,
        'leg_joint_gravity_feedforward': env_cfg.leg_joint_gravity_feedforward,
        'joint_pd_total_effort_limit': ('PhysxDrivePerformanceEnvelopeAPI:angular'
                                       if env_cfg.leg_control_mode == 'implicit_joint_position' else None),
        'enable_external_forces_every_iteration': env.unwrapped._external_forces_every_iteration,
        'leg_effort_path': ('virtual_coordinate_pd_jacobian_transpose' if
                            env_cfg.leg_control_mode == 'explicit_vmc' else
                            'direct_joint_position_implicit_pd'),
        'leg_gravity_feedforward_applied': (env_cfg.leg_control_mode == 'explicit_vmc' or
                                          (env_cfg.leg_control_mode == 'implicit_joint_position' and env_cfg.leg_joint_gravity_feedforward)),
        'leg_joint_drive_stiffness': env_cfg.robot.actuators['legs'].stiffness,
        'leg_joint_drive_damping': env_cfg.robot.actuators['legs'].damping,
        'kp_theta': env_cfg.kp_theta, 'kd_theta': env_cfg.kd_theta,
        'kp_l0': env_cfg.kp_l0, 'kd_l0': env_cfg.kd_l0,
        'feedforward_force': env_cfg.feedforward_force,
        'feedforward_mode': env_cfg.feedforward_mode,
        'feedforward_angle_bias': env_cfg.feedforward_angle_bias,
        'feedforward_force_limit': env_cfg.feedforward_force_limit,
        'leg_effort_limit': env_cfg.leg_effort_limit,
        'wheel_effort_limit': env_cfg.wheel_effort_limit,
        'wheel_control_mode': env_cfg.wheel_control_mode,
        'wheel_damping': env_cfg.wheel_damping,
        'policy_dt': env_cfg.sim.dt * env_cfg.decimation, 'physics_dt': env_cfg.sim.dt,
        'vmc_dt': env_cfg.sim.dt,
        'solver_position_iterations': env_cfg.robot.spawn.articulation_props.solver_position_iteration_count,
        'solver_velocity_iterations': env_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count,
        'fresh_weights': not agent_cfg.resume,
        'hardware_controller_calibrated': False,
    })

    # 用户／服务正常停止时，等一次完整更新结束再保存，避免信号打断原生物理线程。
    from wheel_legged_gym_isaaclab.training_state import (
        CheckpointStopRequested, install_checkpoint_stop, install_episode_log_aggregation,
    )
    install_episode_log_aggregation(runner)
    install_checkpoint_stop(runner, args_cli.stop_file, log_dir)
    # run training
    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except CheckpointStopRequested as error:
        print(f'[INFO]: Training stopped at a saved update boundary: {error}', flush=True)
    except KeyboardInterrupt:
        path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}_interrupted.pt")
        runner.save(path)
        print(f"[INFO]: Interrupted training checkpoint saved to {path}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
