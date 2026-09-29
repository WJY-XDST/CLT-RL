# Wheel-Legged-Gym Reward 恢复与 Isaac Lab 移植审查报告

日期：2026-09-28

## 1. 本次结论

当前 Isaac Lab 工程的 reward 主体已经恢复为原 `Wheel-Legged-Gym` 的 15 项结构、权重和逐项裁剪方式。此前为解决静止漂移、自旋和动作饱和而叠加的速度平方误差、轮速误差、yaw 平方误差、动作饱和、腿长边界等奖励已从当前训练目标中移除。

同时保留了不属于 reward 塑形、但对仿真安全有必要的迁移改进：

- PPO 输出和环境动作均限制在 `[-1, 1]`；
- 虚拟腿摆角目标硬限制在 `[-0.20, 0.20] rad`；
- 虚拟腿长目标硬限制在 `[0.12, 0.25] m`；
- base、腿连杆触地和持续过大 pitch/roll 会终止 episode；
- 非有限 observation、action 和物理状态具有数值保护；
- 关节索引按名称解析，不依赖 Isaac Sim 的 articulation 内部顺序。

PPO 初始动作标准差由 `0.3` 调整为 `0.2`。

## 2. 当前 reward 定义

设：

- `dt = 0.02 s`：策略控制周期；
- `v_cmd`：目标前向速度；
- `v_x`：航向水平坐标系中的实际前向速度；
- `yaw_cmd`：目标 yaw 角速度；
- `omega_z`：实际机身 yaw 角速度；
- `h_cmd`、`h`：目标和实际机身高度；
- `theta_L`、`theta_R`：左右虚拟腿摆角；
- `g_x`、`g_y`：重力向量在机身 x/y 轴上的投影；
- `a_t`：当前归一化 action；
- `tau`：六个关节的实际施加力矩。

所有普通 reward 项先乘权重和 `dt`，再逐项裁剪到：

```text
[-clip_single_reward * dt, clip_single_reward * dt] = [-0.02, 0.02]
```

| 名称 | 当前公式（乘权重前） | 权重 | 作用 |
|---|---|---:|---|
| `track_lin_vel` | `exp(-(v_cmd-v_x)^2 / 0.25)` | `+1.0` | 宽范围前向速度跟踪正奖励 |
| `track_lin_vel_enhance` | `exp(-(v_cmd-v_x)^2 / (0.25*10))-1` | `+1.0` | 原工程的负型增强项，使非零误差产生额外代价 |
| `track_ang_vel` | `exp(-(yaw_cmd-omega_z)^2 / 0.25)` | `+1.0` | yaw 角速度跟踪 |
| `base_height` | `exp(-(h-h_cmd)^2 / 0.001)` | `+1.0` | 机身高度跟踪 |
| `nominal_state` | `(theta_L-theta_R)^2` | `-0.1` | 左右虚拟腿摆角差 |
| `lin_vel_z` | `v_z^2` | `-2.0` | 抑制上下跳动 |
| `ang_vel_xy` | `omega_x^2+omega_y^2` | `-0.05` | 抑制 roll/pitch 角速度 |
| `orientation` | `g_x^2+g_y^2` | `-10.0` | 保持机身竖直 |
| `dof_vel` | 四个腿关节速度平方和 | `-5e-5` | 抑制腿部高速运动 |
| `dof_acc` | 六个关节加速度平方和 | `-2.5e-7` | 抑制关节加速度 |
| `torques` | `sum(tau^2)` | `-1e-4` | 抑制全部关节力矩 |
| `action_rate` | `sum((a_t-a_(t-1))^2)` | `-0.01` | 抑制动作一阶突变 |
| `action_smooth` | 腿部动作二阶差分平方和 | `-0.01` | 抑制腿部指令抖动；与源工程相同，不含轮子动作 |
| `collision` | 非轮子部件超过接触阈值的数量 | `-1.0` | 惩罚腿连杆和 base 接触 |
| `dof_pos_limits` | 腿关节越过位置限制的距离和 | `-1.0` | 惩罚关节越界 |

## 3. 与原 Wheel-Legged-Gym 完全一致的部分

以下项目已经恢复到源工程：

- 15 个 reward 名称和权重；
- `tracking_sigma = 0.25`；
- `base_height_target = 0.18 m`；
- `tracking_lin_vel_enhance` 的原始公式；
- `nominal_state=-0.1`，不再使用高权重 deadband 摆角差惩罚；
- roll 与 pitch 在通用姿态项中使用相同权重；
- 腿和轮子重新使用统一平方力矩惩罚；
- 每个 reward 项采用对称的 `±dt` 裁剪；
- 不再额外加入 termination reward。

## 4. 相比源工程保留的合理差异

### 4.1 前向速度测量

源工程使用机身坐标系 `base_lin_vel[:, 0]`。当前版本使用航向水平坐标系速度 `heading_velocity[:, 0]`。

这样机身 pitch/roll 不会改变“前进速度”的坐标定义，属于物理含义更明确的迁移改进。奖励仍只使用前向误差，没有额外加入横向速度惩罚。

### 4.2 安全限制与终止

源工程主要依靠 base 接触和倒地判断。当前版本额外保留：

- base 接触持续 `0.15 s` 终止；
- 腿连杆接触持续 `0.20 s` 终止；
- `|pitch| > 30 deg` 持续 `0.25 s` 终止；
- `|roll| > 20 deg` 持续 `0.20 s` 终止；
- 非有限状态立即终止。

这些规则没有向正常动作增加塑形偏置，只限定失败状态，因此适合保留。

### 4.3 动作范围

源工程允许较宽的 PPO 原始输出并依靠执行器裁剪。当前版本把策略动作限制在 `[-1,1]`，同时把腿长完整映射到安全范围。这能避免极端高斯采样直接产生危险参考值。

### 4.4 PPO 探索

源工程初始标准差为 `0.5`、wheel action scale 为 `10 rad/s`；修改前版本为 `0.3`、wheel action scale 为 `15 rad/s`。修改前二者对应的初始轮缘 1-sigma 扰动约为 `0.30 m/s`。

当前改为 `0.2`，对应约：

```text
0.2 * 15 * 0.0675 = 0.2025 m/s
```

仍保留探索，但降低静止阶段由噪声导致的漂移和姿态冲击。

## 5. 本次删除的定制 reward

以下项已删除，不再同时和源 reward 竞争：

- `lin_vel_error_sq`
- `wheel_lin_vel_error`
- `yaw_rate_error`
- `wheel_yaw_rate_error`
- `theta_asymmetry` 及 deadband
- 分离的 `leg_torques`、`wheel_torques`、`wheel_power`
- `action_saturation`
- `leg_length_action_margin`
- `leg_length_below_min`
- `base_height_below_target`
- 额外 `termination=-10`
- 姿态项的 roll 倍率和多级特殊裁剪

删除原因不是这些项一定错误，而是它们经过多轮叠加后改变了源工程的目标比例，难以判断漂移和不前进究竟来自控制、样本分布还是 reward 冲突。

## 6. 全工程移植审查

### 6.1 已确认正确

1. **关节映射不依赖内部顺序**

   运行时 articulation 顺序为：

   ```text
   [lf0, rf0, lf1, rf1, l_wheel, r_wheel]
   ```

   逻辑力矩目标顺序解析为：

   ```text
   [0, 2, 4, 1, 3, 5]
   ```

   与 action 的 `[左腿1, 左腿2, 左轮, 右腿1, 右腿2, 右轮]` 对应正确。

2. **VMC 运动学公式**

   FK、虚拟坐标速度近似及 VMC Jacobian 公式与源工程一致。右腿关节力矩符号已按镜像结构处理。

3. **动作与观测维度**

   action 为 6 维，observation 为 27 维，实际拼接维度一致。所有块在进入策略前都有非有限值处理和物理范围裁剪。

4. **控制频率**

   物理频率 `200 Hz`，`decimation=4`，策略频率 `50 Hz`，与当前 reward 的 `dt=0.02 s` 缩放一致。

5. **二进制运行环境**

   训练实际使用：

   ```text
   Python: CLT-RL/isaacsim-5.1.0/kit/python/bin/python3
   PyTorch: 2.7.0+cu128
   rsl_rl: CLT-RL/isaacsim-5.1.0/kit/python/lib/python3.11/site-packages/rsl_rl
   ```

   VS Code 默认解释器也指向该二进制 Python，训练与代码跳转没有混用 Conda 的 `rsl_rl`。

### 6.2 高优先级待处理项

#### A. 域随机化尚未移植

源工程启用了摩擦、恢复系数、质量、惯量、质心、Kp/Kd、电机强度、默认关节位置、动作延迟和外力推动随机化。当前版本只有 reset 时六维 base 速度 `[-0.5,0.5]` 随机化。

影响：当前策略即使仿真表现良好，也不应直接认为具备 Sim2Real 鲁棒性。

建议：平地基本控制稳定后，分批移植，而不是一次全部打开。优先顺序为摩擦/质量/质心 → 电机强度与增益 → 动作延迟与观测噪声 → 外力推动。

#### B. 摩擦组合可能低于配置直觉

机器人默认材料和地面材料都设置为 `0.5`，组合模式为 `multiply`。接触对的组合摩擦系数会表现为约：

```text
0.5 * 0.5 = 0.25
```

这不是“最终摩擦系数 0.5”。它可能增加打滑、降低速度跟踪能力。

建议：后续单独做摩擦 A/B 测试。若目标最终系数为 `0.5`，可把组合模式改为 `average`，或让其中一侧为 `1.0` 后继续使用 `multiply`。不要和 reward 修改同时进行。

#### C. 源工程的时序策略未移植

源工程使用 `ActorCriticSequence` 和 5 帧 observation history；当前使用单帧 27 维标准 Actor-Critic。当前 observation 用两维实际水平速度替代了原来的两维轮子角度，这能增加速度反馈，但并不等价于原时序编码器。

影响：策略对不可观测量、延迟和速度估计的鲁棒性较弱，不能直接加载源工程 checkpoint。

建议：先用当前标准 PPO验证控制和 reward，再独立建立 history/sequence 版本进行对比，避免同时改变 reward 与网络结构。

#### D. 预生成 USD 缓存路径已过期

`assets/robots/wl/usd/config.yaml` 仍记录旧路径：

```text
/home/aaa/studyRL/src/wheel_legged_isaaclab/...
```

当前工程实际位于：

```text
/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/...
```

而 `force_usd_conversion=false`。现有 USD 可以运行，但后续修改 URDF 时可能继续使用旧缓存，造成“代码/URDF 改了，仿真结构没变”的错觉。

建议：资产发生修改时强制重新转换 USD，并更新或重建 `config.yaml`。

### 6.3 中优先级待处理项

#### E. 初始姿态与零 action 腿长目标不一致

根据当前默认关节角计算：

```text
reset 初始 L0       = 0.236920 m
零 action 的 L0_ref = 0.185000 m
差值                = 0.051920 m
```

因此机器人刚启动时，即使 action 接近 0，VMC 也会主动压缩约 `5.2 cm`。这与观察到的启动腿长变化一致。该默认关节姿态来自源工程，不是本次迁移新引入的错误，但当前更窄的安全映射让这个瞬态更值得关注。

建议：先记录 reset 后前 1 秒的 `L0/L0_ref/force_leg`，再决定是把默认关节姿态改到 `L0≈0.185 m`，还是为 policy 初始化提供与 `0.237 m` 对应的腿长 bias。

#### F. feedforward force 需要实测标定

URDF 总质量为 `12.28 kg`，静态半机身重量约为：

```text
12.28 * 9.81 / 2 = 60.23 N
```

当前每侧 `feedforward_force=50 N`，源工程为 `40 N`。VMC 的几何映射和轮子接触也影响实际需求，所以不能只按重量直接设定，但该值需要用零 action 静态测试标定，而不应只靠 reward 补偿。

#### G. 关节加速度时间基准不同

源工程按策略周期用相邻关节速度计算 `dof_acc`；当前直接使用 Isaac Lab 的 `joint_acc`，通常反映物理步尺度。二者数值尺度可能不同。当前权重很小且有逐项裁剪，风险有限，但它不是严格等价移植。

#### H. 地形与任务覆盖比源工程窄

源工程包含粗糙地形、坡面、台阶、随机高度和 yaw/heading 指令；当前是平面、固定高度、yaw=0 和以直行为主的课程。当前版本适合作为第一阶段，但不能据此评价完整源任务性能。

### 6.4 工程维护问题

- `scripts/verify_env.py` 当前是 0 字节空文件，不能承担回归测试；
- `README.md` 仍写着旧的 pip 安装方式和旧绝对路径；
- 旧报告 `velocity_reward_refactor_report.md` 与 `wheeled_legged_RL_migration_summary.md` 描述的是历史 reward，不能再作为当前配置说明；
- 系统中同时存在二进制环境和 Conda 环境两份 `rsl_rl`，目前运行路径正确，但不要直接使用 Conda Python 启动本工程训练；
- 续训参数 `--max_iterations N` 在当前 RSL-RL 中表示“额外训练 N iteration”，不是“训练到总 iteration=N”。

## 7. 验证结果

完成以下检查：

- 全部 Python 文件 `compileall` 通过；
- 已删除 reward 字段无残留 Python 引用；
- `git diff --check` 通过；
- 二进制 Isaac Sim 5.1 + CUDA GPU 下完成 2 环境、1 iteration 无界面训练；
- observation 被识别为 27 维；
- action 被识别为 6 维；
- PPO 报告 `Mean action noise std: 0.20`；
- 15 个当前 reward 项均成功记录；
- 冒烟训练平均 reward 为 `0.33`，未出现 NaN、维度错误或配置字段缺失。

冒烟测试目录：

```text
logs/rsl_rl/wheel_legged_vmc_flat/2026-09-28_14-52-07_reward_restore_smoke
```

## 8. 后续建议顺序

1. 使用当前 reward **从零训练**，不要续接使用旧定制 reward 训练出的模型；
2. 同时测试固定 `0 m/s` 和固定 `0.5 m/s`，记录前向速度、yaw、pitch、roll、左右 `theta0/L0` 和各 reward；
3. 暂时不要再增加静止专用 reward；
4. 单独做摩擦 `0.25` 与 `0.5` 的 A/B 实验；
5. 处理 reset 腿长瞬态和 feedforward force 标定；
6. 基本控制稳定后再加入 observation noise 和第一批域随机化；
7. 最后对比单帧 PPO 与 5 帧历史/时序策略。

每次只改变一个因素并使用独立 run name，否则无法判断改动效果。
