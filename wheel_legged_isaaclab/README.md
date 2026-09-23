# Wheel-Legged-Gym (Isaac Lab version)

轮腿机器人强化学习环境，从基于 **Isaac Gym Preview 4**（已停产，不支持 RTX 50 系 / Blackwell GPU）的原工程
[Wheel-Legged-Gym](https://github.com/) 迁移到 **Isaac Lab 2.3.x + Isaac Sim 5.1**（支持 RTX 5070 Ti）。

## 环境要求

- Python 3.11（conda env: `env_isaaclab`）
- Isaac Sim 5.1.0（pip 安装）
- PyTorch 2.7.0 + cu128（支持 Blackwell sm_120）
- Isaac Lab 2.3.0（`release/2.3.0` 分支）

## 目录结构

```
wheel_legged_gym_isaaclab/
├── config/extension.toml        # 扩展元数据
├── setup.py / pyproject.toml    # 打包配置
├── assets/robots/wl/            # 机器人 URDF + 网格（从原工程复制）
└── wheel_legged_gym_isaaclab/
    └── tasks/direct/wheel_legged_vmc_flat/
        ├── wheel_legged_vmc_flat_env.py       # DirectRLEnv：VMC 控制 + FK + 奖励
        ├── wheel_legged_vmc_flat_env_cfg.py   # 环境配置（机器人/地形/奖励系数）
        ├── __init__.py                        # gym.register 注册任务
        └── agents/rsl_rl_ppo_cfg.py           # PPO 训练配置
```

## 使用方法

```bash
# 激活环境
conda activate env_isaaclab

# 安装本扩展（-e 开发模式）
pip install -e /home/aaa/study/wheel_legged_gym_isaaclab

# 训练（需要先进入 IsaacLab 目录或用其 -p 执行器）
cd /home/aaa/study/IsaacLab
./isaaclab.sh -p ../wheel_legged_gym_isaaclab/scripts/rsl_rl/train.py --task=WheelLeggedVMC-Flat-v0 --headless
```

## 迁移说明

- 动作空间：6 维（每侧 `theta0`、`L0`、`wheel_vel`），由 VMC 控制器映射为关节力矩
- 观测空间：27 维（角速度、重力投影、命令、VMC 虚拟坐标 `theta0/L0` 及导数、轮速、动作）
- 奖励：完整移植原工程的 15 项奖励（速度跟踪、基高、姿态、动作率、碰撞等）
- 控制频率：200 Hz 物理 / 50 Hz 策略（decimation=4）

> 注意：原工程使用带观测历史序列编码器（`ActorCriticSequence`），当前迁移先用标准 PPO 打通闭环，
> 序列编码器策略可在后续迭代中加入。
