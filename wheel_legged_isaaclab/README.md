# 自定义映射五连杆轮腿强化学习

当前工程只支持用户导入的自定义闭链车。机器人数据由
`assets/robots/mine/manifest.json` 指定，`robot_model.py` 读取配套 `model.json`。
不存在缺失参数时自动回退到开源示例车的路径。

## 控制模式

27 维观测和 6 维动作保留，但两种控制模式的腿部字段含义不同。

| 模式 | 策略腿部输出 | 执行链 |
| --- | --- | --- |
| `implicit_joint_position` | 四个主动关节目标角 | 隐式关节 PD 加独立重力前馈 |
| `explicit_vmc` | 两侧虚拟腿摆角和腿长参考 | 虚拟腿 PD 加重力前馈，经 Jacobian 转换为力矩 |

两轮输出均为前进为正的轮速参考，按配套 wheel 控制配置执行。
当前平地任务是 `WheelLeggedJointPD-Flat-v0`，使用直接关节角模式；
`WheelLeggedVMC-Flat-v0` 仅保留为任务名别名，指向同一环境、同一配置。
旧 `implicit_joint_reference` IK 模式、对应检查点和入口已移除。

模型几何采用轮端等效的大五连杆，上杆约 0.210 m、下杆约 0.250 m，映射倍率已计入杆长，运行时 scale=1。
正运动学、解析 Jacobian 和虚功力矩映射继续用于状态计算、重力前馈、VMC 和诊断，不是 IK 控制。
腿电机和轮电机命令限幅分别为 35.574 和 4.80944 N·m；这不是持续额定力矩的声明。
任何回放都以该检查点的 `params/env.yaml` 和同轮 overrides 为准，不能只凭输入输出维度加载策略。

## 阅读代码

目录前缀为 `wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/`：

| 文件 | 职责 |
| --- | --- |
| `wheel_legged_vmc_flat_env_cfg.py` | 唯一环境配置：新车、执行器、物理、命令、奖励、观测和终止参数 |
| `wheel_legged_vmc_flat_env.py` | 唯一环境类：所有 LAB 回调、动作、观测、奖励、终止和 reset |
| `env_commands.py` | 较长的命令抽样流程，仅由环境调用，不定义另一套配置或环境 |
| `agents/rsl_rl_ppo_cfg.py` | 策略网络和 PPO 配置 |

阅读主线就是 `env_cfg → env → agents/rsl_rl_ppo_cfg → scripts/rsl_rl/train.py → play.py`。
环境直接继承 `DirectRLEnv`，不再经过四个 Mixin 或 Mine 环境子类。
`robot_model.py` 管理导入数据和车体契约，`five_bar_vmc.py` 管理正解与力矩映射。
训练与可视化入口为 `scripts/rsl_rl/train.py` 和 `play.py`；MuJoCo 接口位于 `sim2sim/`。

## 运行和日志

从 `~/studyRL/src/CLT-RL` 执行：

```bash
./mine_model/run.sh play --dry_run
./mine_model/run.sh play
./mine_model/run.sh test
```

play 优先选择最近优化任务已评估并保留的最佳关节角检查点，
不选未评估的新候选，也不代表所选模型已通过全部运动验收。
关节角对照训练日志位于 `IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat/`；
VMC 实验日志位于 `IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/`。

详见 [工程代码框架与阅读指南](learning/工程代码框架全解析_20261012.md)、
[MuJoCo 接口说明](sim2sim/README.md)。
历史报告用于溯源，不作为当前车辆数据或控制模式的默认来源。
