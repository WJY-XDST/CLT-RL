# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause
"""任务注册：同一新车、同一配置和环境，不再登记 Mine 适配层。

保留已使用的两个任务 ID 方便加载新车检查点；任务名不替代控制模式检查。
显式 VMC 运行必须指定 env.leg_control_mode=explicit_vmc，且使用匹配的权重。
"""
import gymnasium as gym
from . import agents

for task_id in ("WheelLeggedJointPD-Flat-v0", "WheelLeggedVMC-Flat-v0"):
    gym.register(
        id=task_id,
        entry_point=f"{__name__}.wheel_legged_vmc_flat_env:WheelLeggedVMCFlatEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.wheel_legged_vmc_flat_env_cfg:WheelLeggedVMCFlatEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WheelLeggedVMCFlatPPORunnerCfg",
        },
    )
