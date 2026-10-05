# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause

"""Registration of the wheel-legged VMC flat task."""

import gymnasium as gym

from . import agents

gym.register(
    id="WheelLeggedVMC-Flat-v0",
    entry_point=f"{__name__}.wheel_legged_vmc_flat_env:WheelLeggedVMCFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.mine_env_cfg:MineWheelLeggedVMCFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MineWheelLeggedVMCFlatPPORunnerCfg",
    },
)

gym.register(
    id="WheelLeggedVMC-Legacy-v0",
    entry_point=f"{__name__}.wheel_legged_vmc_flat_env:WheelLeggedVMCFlatEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wheel_legged_vmc_flat_env_cfg:WheelLeggedVMCFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLeggedVMCFlatPPORunnerCfg",
    },
)
