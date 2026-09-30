# 持续训练与验收

`scripts/rsl_rl/optimize_training.py` 接管一个指定的在训进程，完成后自动执行回放、比较、配置试验和后续训练。所有仿真均为 headless，训练环境数为 12288，不启动窗口或录像。

## 当前门槛

每个随机种子执行 33 个五秒阶段，覆盖 0、±0.2、±0.4、±0.6、±0.8 m/s，以及 0.16、0.18、0.20 m 三种高度。每阶段最后两秒作为稳态窗口。

- 非零速度：绝对误差均值、95 分位数分别除以目标速度绝对值，两者都必须小于 5%。
- 高度：绝对误差均值、95 分位数分别除以目标高度，两者都必须小于 5%。
- 静止：水平合速度的均值和 95 分位数都小于 0.02 m/s，包含横向漂移。
- 机体 pitch/roll、左右腿摆角差：稳态绝对峰值均小于 3°。**这是暂定绝对阈值；零角度目标的相对误差没有定义，不能称作角度误差已小于 5%。**
- 切换：方向性高度超调小于目标高度的 5%，机体倾角峰值小于 10°；第一次初始化落地不作为指令切换超调，但全程任何异常终止或超时都不通过。
- CSV 必须完整、时间连续、数值有限、包含预期的所有指令阶段；任何缺失不能通过。

开发测试使用 seed 43/44/45。通过后追加 seed 46/47/48，并反转速度阶段和高度顺序。全部通过才记为 `complete`，含义是上述仿真工况下达到暂定标准，不代表所有地面、扰动或真实硬件下都完美。额外种子的失败也会用于决定后续试验，不能将反复使用的种子称为未见测试集。

## 选择与训练决策

保留的 3742 模型会用当前控制器、相同初始状态与相同种子重测，作为起始参照。候选必须无重置，最差指标构成的评分改善超过 10%（或首次全部达标），且每个阶段的其他未达标指标不能恶化超过 10%，才会替换参照。已达标指标允许在门槛内变化。失败候选及数据继续留在日志目录中。

有收益时优先继续学习；没有收益时按薄弱目标尝试单项奖励增大或减小，权重范围受限制；每连续四个未改进候选，尝试一次从零训练，检验旧策略偏好。续训每轮 1000 次、学习率 0.0001、重建优化器；从零每轮 2000 次、学习率 0.0003。参数变更通过明确的 Hydra 覆盖参数实现，不覆写正在运行进程的配置。每轮 `plan.json`、实际 `params/env.yaml`、`params/agent.yaml` 和中文训练记录共同保存修改目的、内容、起点与验收结果。

这是一套限定在现有控制器和奖励权重范围内的自动试验流程，不会自行任意重写环境动力学。持续塌陷的训练会提前停止并尝试下一候选；进程崩溃、配置错误、数据损坏或代码被外部修改时记为 `needs_attention`，不会冒充训练达标。机器重启后该进程不会自动启动，应先检查状态和最近 checkpoint，再恢复。

## 状态、停止与留存

状态文件为启动时 `--state-dir` 下的 `state.json`，含 PID、阶段、训练健康统计、保留模型和候选历史。进程是否存在和状态更新时间需要一起检查。不要只看旧 JSON 判断仍在训练。

创建该目录下的 `STOP` 文件，会停止循环及其当前任务；已有 checkpoint 不删除。对该优化器 PID 发送 SIGTERM 也会清理它负责的任务。不要用全局 `pkill python`。

首次启动示例（PID、路径必须使用实际值）：

```bash
python3 wheel_legged_isaaclab/scripts/rsl_rl/optimize_training.py \
  --attach-pid TRAIN_PYTHON_PID --attach-run EXACT_RUN_NAME \
  --attach-log /absolute/train.log \
  --attach-checkpoint /absolute/run/model_1999.pt \
  --state-dir /absolute/new_state_directory --publish
```

同一状态目录通过文件锁拒绝并发接管。再次启动必须先核实旧进程已退出，使用新目录；不可同时运行两个训练循环。

明确提升的模型复制到 `checkpoints/optimized/`，同时留存实际配置、模型 SHA-256、代码哈希与完整分工况结果。`--publish` 只在用户已授权时使用；本任务用户已授权。Git 提交使用中文，只提交选中模型的记录和训练笔记；推送失败会明确记录，本地模型继续保留。后续改善时会再次推送。

离线验收命令：

```bash
python3 wheel_legged_isaaclab/scripts/rsl_rl/assess_tracking.py \
  /absolute/seed43.csv /absolute/seed44.csv /absolute/seed45.csv \
  --output /absolute/assessment.json
python3 -m unittest discover -s wheel_legged_isaaclab/tests -p test_tracking_acceptance.py -v
```
