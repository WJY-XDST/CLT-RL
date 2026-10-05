# Isaac Lab 二进制版轮腿工程

这是一个彼此自包含的版本目录，用于后续多版本管理。运行时不依赖旧的
`env_isaaclab` Conda 环境，也不需要先执行 `conda activate`。

当前完整路径为：

```text
/home/aaa/studyRL/src/CLT-RL
```

## 固定版本

- Isaac Sim 二进制版：5.1.0
- Isaac Lab 源码：v2.3.1
- 二进制内置 Python：3.11.13
- PyTorch：2.7.0+cu128
- RSL-RL：3.0.1
- 当前自定义闭链机器人任务：`WheelLeggedVMC-Flat-v0`
- 原开源串联腿任务：`WheelLeggedVMC-Legacy-v0`

## 目录结构

```text
CLT-RL/
├── isaacsim-5.1.0/          # 官方 Isaac Sim 二进制运行时
├── IsaacLab/                # Isaac Lab v2.3.1 源码、logs 和 checkpoint
├── wheel_legged_isaaclab/   # 轮腿机器人强化学习工程
├── mine_model/              # 自定义闭链建模、配置、训练评估及报告
├── downloads/               # 官方安装包归档
├── run_python.sh            # 使用二进制版 Python
├── train_wheel.sh           # 训练入口
└── play_wheel.sh            # 回放入口
```

`IsaacLab/_isaac_sim` 是指向同级 `isaacsim-5.1.0` 的相对符号链接。移动整个
`CLT-RL` 文件夹不会破坏该链接，但不要只移动其中一个子目录。

## 常用命令

以下命令都从本目录运行。

### 2026-10-05 最新版本

本次提交包含自定义闭链机器人适配、训练/评估工具、MuJoCo 接口、实机执行器接口、
持续优化记录和自动键盘回放。最新权重快照与对应参数保存于
[`wheel_legged_isaaclab/checkpoints/mine_20261005_model_12600/`](wheel_legged_isaaclab/checkpoints/mine_20261005_model_12600/)。
这是 iteration **12600** 的训练候选模型，不是已通过最终验收或完成 sim2real 的模型。

详见 [版本更改与验证说明](CHANGELOG.md) 和 [键盘控制说明](mine_model/docs/键盘可视化控制说明.md)。
当前机器的最新已评估模型可这样回放：

```bash
./mine_model/run.sh play
```

GitHub 下载的权重请使用快照目录中的显式回放命令；不要把原开源模型的 0.18 m
高度命令用于自定义机器人。Isaac Sim/Isaac Lab、原始 CAD 压缩包、训练历史和日志
均不随这次提交上传，需在本地按固定版本准备。模型目录中保留原训练配置及路径说明。

检查 GPU：

```bash
./run_python.sh -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

无窗口训练 10000 iterations：

```bash
./train_wheel.sh --task WheelLeggedVMC-Legacy-v0 --headless --device cuda:0 --num_envs 4096 --max_iterations 10000
```

从指定 checkpoint 继续训练：

```bash
./train_wheel.sh --headless --device cuda:0 --num_envs 4096 \
  --task WheelLeggedVMC-Legacy-v0 \
  --resume \
  --load_run 2026-09-22_17-34-52_angle_guard_resume_to_29999 \
  --checkpoint model_18100.pt \
  --max_iterations 10000
```

可视化回放，目标速度 `0.5 m/s`、yaw 目标 `0 rad/s`、高度目标 `0.18 m`：

```bash
./play_wheel.sh --device cuda:0 --num_envs 1 \
  --task WheelLeggedVMC-Legacy-v0 --no_keyboard \
  --load_run 2026-09-22_17-34-52_angle_guard_resume_to_29999 \
  --checkpoint model_18100.pt \
  --fixed_command 0.5 0.0 0.18 --print_obs
```

VS Code 可直接打开 `CLT-RL.code-workspace`，其中已经选择二进制
运行时的 Python，并加入 Isaac Lab 与轮腿工程的源码分析路径。

## 已完成的验证

2026-09-22 已完成以下测试：

1. RTX 5070 Ti 可被二进制运行时与 PyTorch CUDA 正确识别。
2. 16 个环境执行 1 个 PPO iteration，正常生成 `model_0.pt`。
3. 从既有 `model_18100.pt` 加载策略，成功执行 10 步 GPU 回放。
4. 回放可读取 27 维 observation，并成功导出 JIT、ONNX 与 MP4。
5. 清除旧 Conda 环境的 pip 安装后，再次完成 1 个 PPO iteration，证明本目录不依赖旧环境。

测试日志位于 `IsaacLab/logs/rsl_rl/wheel_legged_vmc_flat/`。启动时出现的 CPU
governor、IOMMU、MESA 和 deprecated 参数信息属于性能或兼容性警告，本次测试中
没有阻止仿真、训练、checkpoint 加载或导出。

旧 `/home/aaa/miniconda3/envs/env_isaaclab` 中的 `isaacsim*`、`isaaclab*` 和
`wheel_legged_gym_isaaclab` pip 安装记录已清除。原始源码目录与 checkpoint 被保留，
新目录内复制的 246 个既有模型也已通过数量和 `model_18100.pt` 校验和核对。

## 版本管理建议

以后创建新版本时，使用新的同级目录，例如
`isaaclab_binary_v2.4.x/`，并让该目录拥有自己的 Isaac Sim、Isaac Lab、轮腿源码
与日志。不要在不同版本之间共用可编辑安装的 Python 环境，以免导入路径串到另一
个源码副本。
