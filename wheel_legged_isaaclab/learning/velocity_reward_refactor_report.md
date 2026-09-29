# 串联轮腿机器人速度奖励与指令课程修改报告

> 2026-09-27 后续偏航修正：在 5000 iteration 回放中发现目标
> `yaw=0` 时实际 yaw 约为 `-0.148 rad/s`，左右轮形成稳定差速。
> 因此将 `yaw_rate_error` 从 `-2.0` 提高到 `-5.0`，将
> `theta_asymmetry_deadband` 从 `0.05 rad` 缩小到 `0.02 rad`，并新增
> `wheel_yaw_rate_error=-1.0`。新增项比较目标 yaw 与由轮速差推算的 yaw，
> 因而允许非零 yaw 指令所需的正常差速，不是强制左右轮永远同速。

> 2026-09-28 组合回放后的速度修正：静止目标下仍有约 `0.09 m/s`
> 正向漂移，`0.5 m/s` 目标下稳态约为 `0.387 m/s`。因此将统一的
> `lin_vel_error_sq` 从 `-5.0` 提高到 `-10.0`，新增基于左右轮平均
> 表面速度的 `wheel_lin_vel_error=-1.0`，并将双向低速过渡样本比例
> 从 `0.15` 提高到 `0.25`。静止比例维持 `0.25`，正常前进样本比例
> 相应为 `0.50`，避免训练分布退化为以静止为主。

## 1. 报告目的

本报告记录串联轮腿机器人 VMC 强化学习任务中，速度观测、速度奖励和速度指令课程经历的三个阶段：

1. 原开源工程 `Wheel-Legged-Gym`；
2. 本次修改前的 Isaac Lab 迁移版本；
3. 本次完成精简后的 Isaac Lab 版本。

报告重点回答以下问题：

- 原版速度奖励实际采用了什么公式；
- 迁移版为解决静止漂移、自旋和姿态问题增加了哪些机制；
- 为什么迁移版能够站立，却容易忽略 `0.5 m/s` 前进指令；
- 本次删除、恢复和保留了哪些设计；
- 每个公式、参数和变量分别表示什么；
- 新版本为什么需要从零训练，以及训练时应观察哪些指标。

本次没有修改 `rsl_rl` 的 PPO、Actor-Critic 或 RolloutStorage 源码。修改范围仅限任务环境、环境配置和已有回放诊断工具。

---

## 2. 版本和文件范围

### 2.1 原开源版本

主要参考文件：

- `Wheel-Legged-Gym/wheel_legged_gym/envs/base/legged_robot.py`
- `Wheel-Legged-Gym/wheel_legged_gym/envs/base/legged_robot_config.py`
- `Wheel-Legged-Gym/wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc.py`
- `Wheel-Legged-Gym/wheel_legged_gym/envs/wheel_legged_vmc/wheel_legged_vmc_config.py`

### 2.2 修改前与修改后的 Isaac Lab 版本

主要文件：

- `wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py`
- `wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env_cfg.py`
- `scripts/rsl_rl/play.py`

---

## 3. 公式符号和物理量说明

本报告所有速度奖励公式使用以下符号。

| 符号 | 代码量 | 含义 | 单位/范围 |
|---|---|---|---|
| `v_x_cmd` | `self._commands[:, 0]` | 期望前向线速度 | m/s |
| `v_yaw_cmd` | `self._commands[:, 1]` | 期望 yaw 角速度 | rad/s |
| `h_cmd` | `self._commands[:, 2]` | 期望机身高度 | m |
| `v_x_h` | `heading_velocity[:, 0]` | 航向坐标系中的实际前向水平速度 | m/s |
| `v_y_h` | `heading_velocity[:, 1]` | 航向坐标系中的实际横向水平速度，左向为正 | m/s |
| `omega_z_b` | `root_ang_vel_b[:, 2]` | 机身坐标系中的实际 yaw 角速度 | rad/s |
| `h` | `self._base_height` | 实际机身高度 | m |
| `g_x_b` | `projected_gravity_b[:, 0]` | 重力在机身前向轴的投影，反映 pitch 倾斜 | 无量纲 |
| `g_y_b` | `projected_gravity_b[:, 1]` | 重力在机身侧向轴的投影，反映 roll 倾斜 | 无量纲 |
| `dt` | `self.step_dt` | 环境控制周期；当前为 `0.02 s`，即 50 Hz | s |
| `sigma_v` | `tracking_sigma` | 指数速度奖励的误差尺度；当前为 `0.25` | `(m/s)^2` |
| `e_v2` | `lin_vel_error` | 水平速度平方误差 | `(m/s)^2` |
| `e_yaw2` | `ang_vel_error` | yaw 角速度平方误差 | `(rad/s)^2` |

航向坐标系只保留机器人在世界水平面内的朝向，不随 pitch 和 roll 倾斜。因此：

- `v_x_h` 表示机器人沿自身航向的真实水平移动；
- `v_y_h` 表示机器人横向漂移；
- 机器人仅靠倾斜机身，不能改变这两个速度的定义。

当前水平速度平方误差为：

```text
e_v2 = (v_x_cmd - v_x_h)^2 + (v_y_h)^2
```

第一项惩罚前向速度跟踪误差，第二项惩罚横向漂移。由于当前任务不下发横向速度指令，横向目标固定为 `0 m/s`。

---

## 4. 第一阶段：原开源工程

### 4.1 原版速度测量

原版用机身坐标系线速度 `base_lin_vel[:, 0]` 跟踪前向指令：

```text
e_original = (v_x_cmd - v_x_body)^2
```

其中：

- `v_x_body` 是机身坐标系 x 轴方向速度；
- 当机身发生较大 pitch 或 roll 时，机身 x 轴不再完全位于世界水平面；
- 原版公式不显式惩罚横向速度。

### 4.2 原版宽指数正奖励

原版主速度奖励为：

```text
r_track_original = exp(-e_original / 0.25)
```

配置权重：

```python
tracking_lin_vel = 1.0
tracking_sigma = 0.25
```

含义：

- 完全跟踪时 `e_original = 0`，奖励原始值为 `1`；
- 误差增大时奖励平滑下降，但始终非负；
- 写入每个控制步时再乘权重和 `dt`。

### 4.3 原版 enhance 项

原版还定义：

```text
r_enhance_original = exp(-e_original / (0.25 * 10)) - 1
```

配置权重：

```python
tracking_lin_vel_enhance = 1.0
```

分母实际为 `2.5`，这是一个很宽、很弱的负奖励：

- 误差为零时等于 `0`；
- 误差增大时缓慢趋近 `-1`；
- 它不是窄带精确跟踪奖励。

### 4.4 原版指令范围

原版基础配置使用：

```python
lin_vel_x = [-1.0, 1.0]
ang_vel_yaw = [-3.14, 3.14]
height = [0.1, 0.25]
heading_command = True
resampling_time = 5.0
```

各量含义：

- `lin_vel_x`：前进/后退目标速度随机范围，单位 m/s；
- `ang_vel_yaw`：yaw 目标角速度范围，单位 rad/s；
- `height`：目标机身高度范围，单位 m；
- `heading_command`：使用目标航向角计算 yaw 角速度指令；
- `resampling_time`：每个环境保持一组指令的时间，单位 s。

原版没有目前这种显式的“静止样本、低速过渡样本、正常前进样本”三段式比例。

---

## 5. 第二阶段：本次修改前的迁移版本

### 5.1 已经完成且合理的迁移改进

修改前的迁移版本已经包含以下合理改进：

1. 使用航向坐标系水平速度 `v_x_h、v_y_h`；
2. 将实际水平线速度加入 27 维 observation；
3. 同时惩罚前向误差和横向漂移；
4. 增加 yaw 平方误差，抑制自旋局部最优；
5. 将腿部力矩、轮子功率、轮子力矩分别正则化；
6. 保留轮子在静止平衡时输出必要力矩的能力；
7. 增加动作、观测、腿长、触地、姿态和数值安全边界。

这些内容本次全部保留。

### 5.2 修改前的速度奖励结构

修改前同时存在三项线速度奖励：

```text
1. r_coarse  = exp(-e_v2 / 0.25)
2. r_precise = exp(-e_v2 / 0.01) - 1
3. r_square  = e_v2
```

对应权重：

```python
tracking_lin_vel = 1.0
tracking_lin_vel_precise = 2.0
lin_vel_error_sq = -5.0
```

每步写入奖励：

```text
R_coarse  = dt * 1.0 * r_coarse
R_precise = dt * 2.0 * r_precise
R_square  = dt * (-5.0) * r_square
```

其中：

- `r_coarse` 是宽指数正奖励；
- `r_precise` 是非常窄的负奖励，只在误差极小时才不饱和；
- `r_square` 是速度平方误差惩罚。

迁移版 `precise` 使用分母 `0.01`，而原版 `enhance` 使用分母 `2.5`。两者相差 250 倍，因此迁移版 `precise` 并不是原版 `enhance` 的等价移植。

### 5.3 修改前的姿态/高度门控

修改前还计算：

```text
e_upright = g_x_b^2 + g_y_b^2
e_height  = (h - h_cmd)^2

gate = exp(-e_upright / 0.20) * exp(-e_height / 0.01)
```

然后：

```text
r_coarse = r_coarse * gate
r_yaw_positive = r_yaw_positive * gate
```

变量含义：

- `e_upright`：机身偏离竖直方向的误差；
- `e_height`：机身高度误差；
- `gate`：位于 `(0, 1]` 的门控系数；
- 姿态或高度变差时，正的速度奖励被同步压低。

这个门控最初用于防止机器人用坏姿态换取速度奖励。但航向水平速度已经不能被机身倾斜伪造，并且任务本身已经有独立的姿态、高度和终止约束。因此门控产生了重复耦合，也会压低加减速时必要短暂 pitch 状态的速度学习信号。

### 5.4 修改前的单项裁剪

修改前：

```text
R_precise >= -2.0 * dt = -0.04
R_square  >= -1.0 * dt = -0.02
```

当 `v_x_cmd = 0.5 m/s`、实际速度约为 `0`、横向速度约为 `0` 时：

```text
e_v2 = (0.5 - 0)^2 = 0.25
R_coarse 约为 +0.00736（尚未考虑 gate）
R_precise 约为 -0.04，已经饱和
R_square 原始值为 -0.025，但被裁剪为 -0.02
```

当目标提高到 `0.8 m/s` 时，两个负奖励仍接近相同的饱和值。这样 `0.5 m/s` 原地不动与 `0.8 m/s` 原地不动之间的差别变小，策略较难从奖励中识别“错误究竟有多严重”。

### 5.5 修改前的指令课程

修改前参数：

```python
standing_only_steps = 48_000
motion_ramp_steps = 96_000
standing_env_fraction = 0.50
transition_env_fraction = 0.20
ranges_transition_lin_vel_x = (-0.3, 0.3)
ranges_lin_vel_x = (0.3, 0.8)
```

各量含义：

- `standing_only_steps`：所有环境仅使用零速指令的环境控制步数；
- `motion_ramp_steps`：从纯静止线性增加到最终移动样本比例所需的控制步数；
- 每个 PPO iteration 包含 48 个环境控制步；
- `standing_env_fraction`：课程完成后，零速环境占全部并行环境的比例；
- `transition_env_fraction`：课程完成后，低速过渡环境占全部环境的比例；
- 剩余比例为正常前进环境。

修改前对应：

- 前 1000 iteration 全部静止；
- 接下来 2000 iteration 缓慢加入移动；
- 课程完成后为 50% 静止、20% 低速过渡、30% 正常前进。

对于一次 5000 iteration 的训练，估算全部经验中：

- 移动样本约占 30%；
- 正常 `0.3～0.8 m/s` 前进样本约占 18%。

这解释了策略为什么能够学会站立，却可能把正速度指令忽略为接近零轮子动作。

### 5.6 修改前的实测现象

在 `v_x_cmd = 0.5 m/s、v_yaw_cmd = 0、h_cmd = 0.18 m` 回放中测得：

- 实际平均前向速度约 `-0.0054 m/s`；
- 左右轮策略动作接近零；
- 机器人主要保持站立，没有形成持续前进；
- `R_precise` 固定约为 `-0.04/step`；
- `R_square` 固定约为 `-0.02/step`。

固定轮子动作开环测试同时证明：

- 左右轮执行通道有效；
- 左右轮同号动作产生一致轮速响应；
- 正负动作产生近似镜像的速度和 pitch 响应；
- 问题主要属于策略与奖励/样本分布，而非轮子执行器卡死。

---

## 6. 第三阶段：本次精简后的版本

### 6.1 删除精细指数项

本次删除：

```python
tracking_lin_vel_precise
tracking_sigma_precise
r_track_lin_precise
```

同时删除：

- episode reward 日志中的 `track_lin_vel_precise`；
- reward 字典中的该项；
- 该项专用裁剪；
- 配置验证中对 `tracking_sigma_precise` 的引用。

原因：它与平方误差项功能重叠，而且在普通移动误差下快速饱和，不能提供误差大小信息。

### 6.2 取消速度姿态/高度门控

本次删除：

```python
velocity_upright_gate_sigma
velocity_height_gate_sigma
velocity_reward_gate
```

速度、姿态和高度现在分别优化：

- 速度由 `track_lin_vel` 和 `lin_vel_error_sq` 管理；
- 姿态由 `orientation`、角速度项和姿态终止管理；
- 高度由 `base_height`、低高度平方惩罚和触地终止管理。

这样不会奖励坏姿态，但允许策略在加速和制动时经历必要的短暂 pitch。

### 6.3 新版线速度奖励

新版只保留两个线速度项：

```text
R_track = dt * exp(-e_v2 / 0.25)
R_error = dt * (-5.0) * e_v2
```

并分别裁剪：

```text
0 <= R_track <= dt
-5 * dt <= R_error <= 0
```

当前 `dt = 0.02 s`，因此：

```text
0 <= R_track <= 0.02
-0.10 <= R_error <= 0
```

`lin_vel_error_clip_multiplier = 5.0` 的含义是：平方速度误差项最多允许产生 `5 * dt` 的单步负奖励。它不是速度误差阈值，也不是奖励权重；奖励权重仍然是 `lin_vel_error_sq = -5.0`。

### 6.4 新版速度奖励数值示例

假设实际前向速度和横向速度均为零：

| `v_x_cmd` | `e_v2` | `R_track` | `R_error` | 两项合计 |
|---:|---:|---:|---:|---:|
| `0.0` | `0.00` | `+0.02000` | `0.00000` | `+0.02000` |
| `0.3` | `0.09` | `+0.01395` | `-0.00900` | `+0.00495` |
| `0.5` | `0.25` | `+0.00736` | `-0.02500` | `-0.01764` |
| `0.8` | `0.64` | `+0.00155` | `-0.06400` | `-0.06245` |

这些数值说明：

- 完全跟踪得到最大正奖励；
- 在较高正速度指令下原地不动会明显变差；
- `0.8 m/s` 原地不动比 `0.5 m/s` 原地不动受到更强惩罚；
- 奖励不再把所有大误差压成同一个值。

静止目标下若仍有 `0.03 m/s` 漂移：

```text
e_v2 = 0.03^2 = 0.0009
R_track 约为 +0.019928
R_error = -0.000090
合计约为 +0.019838
```

完全静止时合计为 `+0.020000`。因此静止漂移仍会降低奖励，但不再额外引入只针对静止状态的特殊目标。

### 6.5 新版 yaw 奖励

yaw 奖励保持不变：

```text
e_yaw2 = (v_yaw_cmd - omega_z_b)^2
R_yaw_track = dt * exp(-e_yaw2 / 0.25)
R_yaw_error = dt * (-2.0) * e_yaw2
```

其中：

- `v_yaw_cmd`：目标 yaw 角速度；当前第一阶段固定为 `0 rad/s`；
- `omega_z_b`：实际机身 yaw 角速度；
- 正指数项鼓励精确跟踪；
- 平方项让持续自旋始终产生负奖励；
- `yaw_rate_error_clip_multiplier = 5.0` 将该负奖励的单步下限设为 `-0.10`。

保留该项是因为历史训练中确实出现过自旋局部最优，它不属于本次需要回退的静止速度补丁。

### 6.6 新版指令课程

新配置：

```python
ranges_lin_vel_x = (0.3, 0.8)
ranges_transition_lin_vel_x = (-0.2, 0.2)
standing_only_steps = 24_000
motion_ramp_steps = 48_000
standing_env_fraction = 0.25
transition_env_fraction = 0.15
```

换算为 PPO iteration：

- `24_000 / 48 = 500`：前 500 iteration 纯站立；
- `48_000 / 48 = 1000`：随后 1000 iteration 逐步增加移动样本；
- iteration 1500 后达到最终比例。

最终并行环境分布：

| 样本类型 | 比例 | 指令范围 |
|---|---:|---|
| 静止 | 25% | `0 m/s` |
| 低速过渡 | 15% | `-0.2～0.2 m/s` |
| 正常前进 | 60% | `0.3～0.8 m/s` |

低速过渡包含少量负速度，其作用是让策略学会跨越零速、制动和消除固定方向动作偏置，而不是把主要任务改成倒车。

对于 5000 iteration 训练，估算全部经验中：

- 所有移动样本约占 60%；
- 正常前进样本约占 48%；
- 静止能力仍由早期纯站立课程和后续 25% 零速环境持续维护。

---

## 7. 三个版本对照表

| 项目 | 原开源版 | 修改前迁移版 | 本次精简版 |
|---|---|---|---|
| 前向速度坐标系 | 机身坐标系 x | 航向水平坐标系 x | 航向水平坐标系 x |
| 横向漂移 | 未纳入线速度误差 | 纳入 | 纳入 |
| 宽指数速度奖励 | 有，权重 1 | 有，权重 1 | 有，权重 1 |
| enhance/precise | 宽分母 2.5、权重 1 | 窄分母 0.01、权重 2 | 删除 |
| 平方速度误差 | 无 | 有，但通用裁剪到 `-0.02` | 有，独立裁剪下限 `-0.10` |
| 速度姿态/高度门控 | 无 | 有 | 删除，目标解耦 |
| yaw 平方误差 | 无 | 有 | 保留 |
| 显式静止样本 | 无独立比例 | 最终 50% | 最终 25% |
| 低速过渡样本 | 无独立比例 | 最终 20%，`±0.3` | 最终 15%，`±0.2` |
| 正常前进样本 | 原范围统一采样 | 最终 30% | 最终 60% |
| 纯站立课程 | 无当前形式 | 1000 iteration | 500 iteration |
| 运动比例爬升 | 无当前形式 | 2000 iteration | 1000 iteration |

---

## 8. 本次明确保留的非速度改进

以下逻辑没有回退：

- 6 维动作在环境边界限制到 `[-1, 1]`；
- 虚拟腿摆角目标限制到 `[-0.20, 0.20] rad`；
- 虚拟腿长目标限制到 `[0.12, 0.25] m`；
- 轮速动作比例 `action_scale_vel = 15 rad/s`；
- 轮半径 `wheel_radius = 0.0675 m`；
- observation 维度保持 27，不改变策略网络输入结构；
- observation 的有限值清理和按物理量裁剪；
- 左右腿摆角差惩罚，但不强制左右腿长相等；
- 腿长动作安全边界惩罚；
- base_link、腿连杆接触和过大 pitch/roll 的持续终止；
- 数值异常立即终止；
- 腿力矩、轮功率、轮力矩分别正则化；
- wheel action 不纳入普通摆角动作饱和惩罚。

这些机制分别负责安全、数值稳定和结构约束，不应通过速度奖励间接实现。

---

## 9. 兼容性和训练策略

### 9.1 checkpoint 结构兼容性

本次没有改变：

- observation 维度：27；
- action 维度：6；
- Actor/Critic 网络结构。

因此旧 checkpoint 在张量形状上可以加载。

### 9.2 为什么仍建议从零训练

虽然形状兼容，但奖励曲线和训练样本分布发生了实质变化：

- 删除了此前长期主导近零速度行为的窄精细项；
- 平方误差重新获得大误差区分能力；
- 正常前进样本从最终 30% 提高到 60%；
- 课程时间缩短。

旧策略已经形成接近零轮子动作的站立局部最优。从旧 checkpoint 续训可能保留该行为偏置，因此建议新实验从随机初始化开始。

---

## 10. 推荐训练与验证流程

建议从零训练 5000 iteration，并保留以下检查点：

| iteration | 主要目的 | 建议测试 |
|---:|---|---|
| 500 | 纯站立阶段结束 | `0 m/s` |
| 1500 | 移动课程完全展开 | `0、0.3、0.5 m/s` |
| 2500 | 检查速度学习趋势 | `0、0.3、0.5、0.8 m/s` |
| 5000 | 完整评估 | 静止、前进、制动和短时负速度 |

每次回放至少记录：

- `v_x_cmd`：目标前向速度；
- `v_x_h`：实际前向速度；
- `v_y_h`：横向漂移速度；
- `omega_z_b`：yaw 角速度；
- pitch、roll；
- `h`：实际机身高度；
- 左右轮角速度；
- 6 维动作；
- `track_lin_vel`；
- `lin_vel_error_sq`；
- `track_ang_vel`；
- `yaw_rate_error`；
- `orientation`；
- episode length 和各类 termination 比例。

建议初步验收标准：

- 零速时稳态 `|v_x_h| < 0.03 m/s`；
- `0.5 m/s` 指令下稳态速度位于 `0.4～0.6 m/s`；
- `0.8 m/s` 指令不会长期原地不动或动作饱和；
- yaw 目标为零时无持续自旋；
- 稳态 pitch 绝对值尽量小于 `8°`；
- 左右轮动作与轮速方向一致，没有单轮持续卡死；
- base_link 和腿连杆没有持续触地。

---

## 11. 风险与后续调整顺序

### 11.1 可能风险

1. 删除窄精细项后，零速漂移可能略有回升；
2. 取消速度姿态门控后，早期探索可能出现更大的瞬时 pitch；
3. 前进样本增加后，训练初期平均 episode length 可能暂时下降。

这些现象不应立即用新的静止专用奖励修补，应先检查它们是否随训练自然改善。

### 11.2 推荐调整顺序

如果 5000 iteration 后仍不理想，按以下顺序一次只改一项：

1. 先检查目标速度是否真实写入 observation；
2. 检查 `v_x_h`、轮速和轮动作是否随指令同向变化；
3. 再微调 `standing_env_fraction`，建议范围 `0.20～0.35`；
4. 再微调 `lin_vel_error_sq`，建议范围 `-3～-7`；
5. 只有速度已经学会但仍有小幅稳态误差时，才考虑加入不超过 `0.25` 权重的精细项；
6. 最后才调整 PPO entropy、学习率或网络结构。

不要同时修改多个奖励和 PPO 参数，否则无法判断改善来自哪一项。

---

## 12. 修改结论

本次修改不是简单增大奖励，而是将速度目标恢复为更清晰的两项结构：

```text
宽指数正奖励 + 未过早饱和的平方速度误差
```

同时让速度、姿态、高度分别由各自奖励负责，并提高正常前进样本比例。新版仍然使用统一的速度误差处理静止、低速和前进，不包含只在零速下激活的特殊惩罚，因此比修改前更容易解释，也更适合作为后续复杂地形和机械臂耦合任务的基础版本。

---

## 13. 修改后的验证结果

已完成以下检查：

1. `wheel_legged_vmc_flat_env.py` 与 `wheel_legged_vmc_flat_env_cfg.py` 通过 Python 语法编译检查；
2. 已删除参数和奖励名称没有残留引用；
3. Git diff 格式检查通过；
4. 使用 GPU、1 个环境和旧 checkpoint 完成 20 控制步无界面冒烟测试；
5. 环境创建、关节映射、27 维网络加载、奖励计算和自动退出均正常；
6. CSV 中已不存在 `reward_track_lin_vel_precise`；
7. CSV 中存在新的 `reward_lin_vel_error_sq`，且不再受 `-0.02` 通用下限限制；
8. 20 步记录中所有数值均为有限值，没有 NaN 或 Inf；
9. 测试完成后没有残留回放进程。

冒烟测试最后一行示例：

```text
v_x_cmd = 0.5 m/s
v_x_h = -0.05530 m/s
reward_track_lin_vel = +0.005825
reward_lin_vel_error_sq = -0.030838
```

平方误差奖励达到 `-0.030838`，已经突破旧版通用裁剪下限 `-0.02`，说明新的独立裁剪分支实际生效。

注意：该测试使用旧 checkpoint，只用于验证代码执行与奖励字段，不能用于判断新奖励训练后的控制性能。
