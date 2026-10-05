# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause

"""PPO agent configurations for the wheel-legged VMC flat task."""

from __future__ import annotations

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class WheelLeggedVMCFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 2000
    save_interval = 100
    experiment_name = "wheel_legged_vmc_flat"
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    # Preserve the Gaussian policy output until it reaches the environment so
    # reward terms can penalize pre-clipped saturation. The environment still
    # clamps every action to [-1, 1] before applying it to the VMC.
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        # 0.3 corresponds to roughly 0.30 m/s wheel-surface noise at startup;
        # 0.2 retains exploration without overwhelming the standing controller.
        init_noise_std=0.2,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MineWheelLeggedVMCFlatPPORunnerCfg(WheelLeggedVMCFlatPPORunnerCfg):
    experiment_name = "mine_wheel_legged_vmc_flat"
    max_iterations = 1000

    def __post_init__(self):
        # The 256-environment collapse pilot lost nearly all exploration
        # (std 0.3 -> 0.04) before learning a standing episode. Keep entropy
        # pressure specific to the new-model experiment.
        self.algorithm.entropy_coef = 0.01
