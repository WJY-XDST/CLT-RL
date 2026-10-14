# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause

"""PPO agent configurations for the wheel-legged VMC flat task."""

from __future__ import annotations

from isaaclab.utils import configclass

# 这三类是 LAB 对 RSL-RL 配置的封装：运行器、网络、优化算法。
# 配置不执行训练；train.py 用 agent_cfg.to_dict() 创建库里的 OnPolicyRunner。
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


# LAB @configclass 将这种类属性写法转换为配置对象，提供 to_dict()/copy()/replace()，
# 并处理可变默认值。它不是神经网络装饰器，不会在这里创建 PyTorch 层。
@configclass
class WheelLeggedVMCFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48  # 每个 env 采集 48 个控制步，再做一次 PPO 更新轮次。
    max_iterations = 1000  # 本次学习轮数默认值；CLI/恢复训练会另行解析。
    save_interval = 100  # runner 每隔多少 iteration 保存检查点，不是物理步数。
    experiment_name = "mine_wheel_legged_joint_pd_flat"
    # 键是 actor/critic 的输入角色，值是环境返回的观测组名；这里都读取同一帧 27 维。
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    # wrapper 不提前裁剪，保证环境仍能看到 raw action 并计算饱和惩罚；
    # 最终执行前环境自己限制到 [-1, 1]，None 不代表电机无限幅。
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        # 高斯策略新建时的逐维标准差，不是动作尺度；续训通常加载检查点中的 std。
        # actor 输出均值，训练从分布采样；act_inference() 回放直接用均值。
        init_noise_std=0.2,
        # 这里只给隐藏层尺寸；RSL-RL 从 obs/action 空间推导输入 27 和输出 6。
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # critic 价值估计损失的权重，不是环境 reward 权重。
        use_clipped_value_loss=True,
        clip_param=0.2,  # PPO 新旧动作概率比的裁剪幅度，不是动作 [-1, 1] 限幅。
        entropy_coef=0.001,  # 鼓励探索的熵项系数；它不直接指定 std 的大小。
        num_learning_epochs=5,  # 同一批 rollout 数据重复学习 5 遍。
        num_mini_batches=4,  # 每遍分成 4 个小批量，不是 4 个并行环境。
        learning_rate=3.0e-4,
        schedule="adaptive",  # RSL-RL 根据 KL 与 desired_kl 调整学习率。
        gamma=0.99,  # 对未来每个控制步奖励的折扣。
        lam=0.95,  # GAE 优势估计的平滑系数。
        desired_kl=0.01,
        max_grad_norm=1.0,  # 裁剪网络参数梯度，和电机力矩上限无关。
    )
