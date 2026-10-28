# 自定义五连杆轮腿强化学习工程

本目录同时管理自定义闭链机器人工程、配套 Isaac Lab 源码和 Isaac Sim 二进制运行时。
当前训练使用四个主动关节目标角加隐式 PD，并保留独立重力前馈；显式 VMC 仍可选择。
旧开源示例车和 `implicit_joint_reference` IK 控制已经退役，不再提供它们的运行命令。

## 固定版本

- Isaac Sim 二进制版：5.1.0
- Isaac Lab 源码：v2.3.1
- 二进制内置 Python：3.11.13
- PyTorch：2.7.0+cu128
- RSL-RL：3.0.1

运行不依赖旧的 `env_isaaclab` Conda 环境，无需先执行 `conda activate`。
完整工程位置为 `/home/aaa/studyRL/src/CLT-RL`。

## 目录结构

```text
CLT-RL/
├── isaacsim-5.1.0/         # 官方二进制运行时
├── IsaacLab/              # 配套框架源码及训练日志
├── wheel_legged_isaaclab/  # 训练环境、PPO 入口和 MuJoCo 接口
├── mine_model/            # 新车建模、几何证据、诊断和修改报告
├── sim2real/              # 离线部署接口及用户提供的下位机源码
├── run_python.sh          # 二进制 Python 入口
├── train_wheel.sh         # 训练入口
└── play_wheel.sh          # 回放入口
```

`IsaacLab/_isaac_sim` 相对链接到同级 `isaacsim-5.1.0`。移动整个目录可保留这一关系，不要只移动其中一个子目录。
VS Code 可打开 `CLT-RL.code-workspace`，使用相同运行时和源码分析路径。

## 当前训练和回放

从本目录启动新的关节角 PD 对照训练：

```bash
./train_wheel.sh --task WheelLeggedJointPD-Flat-v0 --headless \
  --device cuda:0 --num_envs 8192 --max_iterations 1000 \
  --require_leg_control_mode implicit_joint_position
```

这是新训练命令，不要在后台训练已运行时重复执行。当前课程前 1000 iteration 为站立阶段；
运动学习需要后续训练和独立回放，不能由站立成绩推断。

当前实验日志根目录：

```text
IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat/
```

显式 VMC 实验通常保存于 `IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/`。
任务名带 VMC 不代表执行模式必为 VMC；以配套 `params/env.yaml` 的 `leg_control_mode` 为准。
VMC 与关节角策略的动作和腿部观测含义不同，不能混用检查点。

默认键盘回放优先选择最近优化任务已评估并保留的最佳检查点，不选未评估的新候选：

```bash
./mine_model/run.sh play --dry_run
./mine_model/run.sh play
```

第一条只核对模型和启动命令。第二条启动可视化，WASD 控制运动，Q/E 调整平滑高度命令；
空格仅记录请求，当前没有训练好的跳跃策略。原 IK 的 12600 快照已清理，不是当前可用模型。

GPU 检查：

```bash
./run_python.sh -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 代码和验证说明

- [机器人工程入口](wheel_legged_isaaclab/README.md)
- [当前代码框架和阅读指南](wheel_legged_isaaclab/learning/强化学习框架学习/README.md)
- [本轮精简验证](mine_model/results/framework_cleanup_20261012/validation.json)
- [累计修改记录](mine_model/docs/训练修改记录_latest.md)
- [IK 清理范围和恢复方式](mine_model/docs/IK清理说明_20261011.md)
- [MuJoCo 接口](wheel_legged_isaaclab/sim2sim/README.md)
- [离线部署接口](sim2real/README.md)
- [2026-10-12 历史阶段记录](mine_model/results/joint_pd_optimization_20261012_framework/status.json)

本轮回归核对了控制和观测接口、奖励数值及终止计时；导轨支撑验证不是自由平衡或实机验收。
历史模型报告只代表当时配置。部署需要选定检查点及其配套配置，不能自动跟随变化的 latest。
