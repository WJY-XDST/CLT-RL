# wheeled-legged_RL 可取设计迁移总结

日期：2026-09-27
目标工程：`CLT-RL/wheel_legged_isaaclab`
参考工程：`wheeled-legged_RL` 的 `wheelbipe25_v3`、`wheelbipe_V13/V14`、回放诊断及执行器模块。

## 1. 迁移原则

本次没有复制参考工程的大型状态机或整套算法，而是挑选与当前双轮串联腿 VMC 任务兼容、可解释且能独立验证的机制。保持动作维度为 6、策略观测维度为 27、控制频率和 VMC 结构不变，因此必须从零训练，但没有引入新的策略输入接口。

## 2. 已实施改动

### 2.1 航向坐标系水平速度

旧代码将 `root_lin_vel_b[:, :2]` 直接用于 observation 和速度奖励。机身有 roll/pitch 时，机身坐标轴发生倾斜，水平速度与竖直运动会混合。

新代码将世界系 XY 速度投影到机器人 yaw 航向坐标系：

```text
v_forward = dot(v_world_xy, heading_forward_xy)
v_lateral = dot(v_world_xy, heading_left_xy)
```

速度误差统一为：

```text
(cmd_forward - v_forward)^2 + v_lateral^2
```

收益：速度测量不随机身俯仰改变，不能通过倾斜机身伪造前进速度；同时明确惩罚横向漂移。27维观测中原来的两个线速度位置不变，只改变其坐标定义。

### 2.2 逐类观测裁剪与非有限值保护

新增 `ObsClipCfg`，分别约束角速度、航向线速度、虚拟腿角度/角速度、腿长/腿长速度、轮速和动作。裁剪在缩放前执行，并用 `torch.nan_to_num` 清理 NaN/Inf。

这些边界明显宽于正常工作范围，只用于防止异常物理帧污染 PPO rollout，不是新的任务约束。

### 2.3 速度正奖励的姿态/高度门控

正的线速度和 yaw 跟踪奖励乘以：

```text
gate = exp(-upright_error / 0.20) * exp(-height_error^2 / 0.01)
```

其中 `upright_error = gravity_x^2 + gravity_y^2`。只有正奖励被门控；精确速度误差、平方速度误差和 yaw 平方误差始终保留。因此机器人倾斜或趴低时不能继续获得完整速度奖励，也不能通过坏姿态逃避误差惩罚。

### 2.4 分离腿部力矩、轮子功率和轮子力矩

旧奖励统一惩罚全部关节 `sum(torque^2)`，会把静止平衡所需的轮子力矩也当成主要代价。

新奖励拆分为：

- `leg_torques`：腿关节平方力矩，权重 `-1e-4`；
- `wheel_power`：`sum(abs(wheel_torque * wheel_speed))`，权重 `-1e-4`；
- `wheel_torques`：很小的轮子平方力矩，权重 `-1e-5`。

这允许轮子在接近零速时输出必要的平衡力矩，同时惩罚有速度时的持续能耗和漂移。

### 2.5 数值安全终止

新增以下物理状态的有限值检查：根位置、四元数、根线/角速度、关节位置/速度和接触力。任一环境出现 NaN/Inf 时立即终止并重置，避免影响同批次其他环境。策略输出进入 VMC 前也会先清理 NaN/Inf，再裁剪到 `[-1, 1]`。

### 2.6 Reward 分项快照

环境每步保存裁剪并清理后的 `_last_reward_terms`，供回放诊断读取。训练的 episode reward 日志同步更新为 `leg_torques`、`wheel_power`、`wheel_torques`，替代旧的统一 `torques`。

### 2.7 通用姿态惩罚调整

静止回放测得机身稳态 pitch 约为 `-2.16°`、roll 约为 `+1.39°`。为减小固定姿态偏置，同时避免引入只对静止工况生效的特殊奖励，将统一 `orientation` 权重从 `-10.0` 温和提高至 `-15.0`。姿态安全裁剪、动态任务结构和终止阈值保持不变；该修改需要后续训练后才能反映到策略。

续训后的静止诊断显示 pitch 已由约 `-2.16°` 降至约 `-1.06°`，但左右虚拟腿摆角差仍约为 `0.182 rad`。因此保留 `theta_asymmetry_deadband = 0.05 rad`，将该奖励项重命名为 `theta_asymmetry`，并把权重从 `-3.0` 提高至 `-5.0`；不约束左右腿长相等，以保留质心补偿和高低差地形适应能力。

### 2.8 腿长动作边界软惩罚

`model_8997.pt` 的静止回放显示右腿长度动作在完整稳态区间持续超过 `1.0`，环境虽会安全裁剪，但右腿实际长度长期接近 `0.25 m` 上限。新增 `leg_length_action_margin=-2.0` 和 `leg_length_action_threshold=0.90`，仅对动作索引 `1、4` 超过 `±0.90` 的部分施加平方惩罚。该项不要求左右腿长相等，因此仍保留质心补偿及高低差地形所需的独立腿长调节能力。

### 2.9 PPO 稳定性调整

- Actor：`[128, 64, 32]` → `[256, 128, 64]`
- Critic：保持 `[256, 128, 64]`
- 学习率：`1e-3` → `3e-4`
- 目标 KL：`0.005` → `0.01`
- 初始噪声和 entropy 保持较低值，避免再次强化动作饱和与抖动。

目标是降低单次更新过猛的风险，同时给 actor 足够容量表示站立、过渡速度和前进三类连续行为。

### 2.10 回放 CSV 与 HTML 诊断

`play.py` 新增：

```text
--trace_csv PATH
--trace_interval STEPS
--trace_env_id ENV_ID
```

CSV包含目标/实际速度、yaw、机身高度、重力投影、pitch/roll估计、左右虚拟腿角度和长度、轮速、6维动作、6维逻辑顺序力矩及全部reward分项。

回放示例：

```bash
./play_wheel.sh --num_envs 1 --fixed_command 0.0 0.0 0.18 \
  --trace_csv /tmp/clt_stand_trace.csv --trace_interval 1
```

生成自包含HTML：

```bash
./IsaacLab/_isaac_sim/python.sh \
  wheel_legged_isaaclab/scripts/rsl_rl/export_trace_html.py \
  /tmp/clt_stand_trace.csv
```

输出 `/tmp/clt_stand_trace.html`，包含速度、yaw、姿态、高度、轮速曲线和reward统计表。

## 3. 本次没有直接启用的参考功能

- 大范围质量、质心、摩擦和电机随机化：基础策略稳定后再分阶段加入；
- 观测/动作延迟和历史帧网络：会改变策略输入或动态响应，应在名义模型收敛后单独实验；
- M3508+C620力矩-转速曲线：只有确认实际电机、减速比和实测曲线后才能移植；
- `stand_still_deadzone`：未采用静止专用奖励，继续使用统一速度误差目标；
- 左右腿长强制相等：会损害未来高低差地形适应；
- DreamWaQ、HIM、NP3O和跳跃状态机：超出当前平地基础控制阶段。

## 4. 训练与对照要求

由于速度观测语义、奖励结构及网络尺寸均已变化，旧 checkpoint 不应继续训练，新实验必须从零开始。建议保留旧日志作为基线，不删除。

新训练重点观察：

1. `vel_x_heading` 在零速命令下是否收敛到零；
2. `vel_y_heading` 是否保持接近零；
3. pitch/roll是否随训练下降；
4. `wheel_power`是否下降但轮子仍保留平衡力矩；
5. `track_lin_vel`、`lin_vel_error_sq`和`orientation`是否同时改善；
6. 动作0/3是否不再长期饱和。

## 5. 验证状态

- Python语法编译：通过；
- Git空白与补丁检查：通过；
- 旧代码训练服务：已停止；
- 64环境、1 iteration GPU smoke test：通过，完成3072步采样和一次PPO更新；
- 正式训练：已从零启动5000 iterations；
- 运行名：`2026-09-27_12-16-42_migration_coreopt_scratch_5k`；
- 后台服务：`clt-rl-wheel-migration-5k.service`；
- 实时日志：`/tmp/clt_rl_wheel_migration_5k.log`；
- 启动复核：4096环境、27维Actor输入、新reward分项及PPO更新均正常。
