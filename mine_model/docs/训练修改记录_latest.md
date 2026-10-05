# 新闭链模型从建模到训练的修改记录

生成时间：2026-10-05T17:05:06.934352+08:00；状态：已通过最终验收。

并行环境：None；当前来源权重：`None`。

记录顺序为机械建模、工程适配、诊断与训练、精度优化及完成交付。历史失败与后来通过分别保留；不能把悬空驱动或短测存活解释为训练收敛。

证据边界：早期修改依据现存教程、配置和诊断记录；没有保存的逐行历史差异不补造。逐轮修改原因有管理日志时引用原文，没有时标为未记录。旧开源模型的数据已按用户要求清理，因此不能重建其所有早期原始数据。当前 Git 差异可能包含用户原有修改，不能作为全部改动的作者证明。

## 开链机械模型改为闭链

修改内容：确认 link_002 为 RB；补充 RL3–RS、RL2–RB 及左侧镜像四个 revolute 闭合铰链，保留原始输入，另存 USD 和树形 URDF。

修改原因：URDF 的关节树不能独立表达闭环，需要物理约束与完整 CAD 孔位和轴线。

修改思路：先核对孔轴与左右镜像，再确认默认装配闭合；悬空驱动通过只证明机构与驱动，不证明自由站立。

证据：`mine_model/config/closures.json`；`mine_model/docs/implementation_details.md`；`mine_model/results/bench_drive/report.json`

## 坐标、joint 与五连杆运动学适配

修改内容：工程轴系统一为 X 前、Y 左、Z 上；四个髋关节统一校验以 base_link 为父刚体。主动顺序 LB/LL/LW/RB/RL/RW；FK、Jacobian、逆解采用新几何。

修改原因：旧串联腿角度定义、杆长与传动映射不能直接套到闭链，即使观测和动作维度相同。

修改思路：以实际轮轴位置、数值差分和虚功验证映射；父节点原本已连接 base_link 的部分属于核验，不虚构为重新挂接。

证据：`mine_model/docs/implementation_details.md`；`mine_model/results/parameter_audit`；`mine_model/scripts/audit_model_parameters.py`

## 原策略直接迁移与控制诊断

修改内容：复用原 7738 策略做新旧模型对照，随后保存增益、求解器、步长、承重与导轨诊断。

修改原因：初期自由基座无法站稳，需要区分关节映射错误、驱动承重不足和策略不适配。

修改思路：固定权重、隔离一个变量比较；不能把导轨承重或无重力驱动当成平衡验收。失败尝试作为后续选择的依据。

证据：`mine_model/results/policy_7738_20261003/summary.json`；`mine_model/results/support_rail_confirm_20261004/report.json`；`mine_model/results/solver_audit_20261004/report.json`；`mine_model/results/timestep_audit_20261004/report.json`

## 隐式关节驱动与新模型训练

修改内容：虚拟腿角和长度经五连杆逆解成为真实电机角目标，使用 implicit_joint_reference；初始 Drive 为 Kp300/Kd3，腿轮限幅 10 Nm。

修改原因：显式 VMC 在原限幅下的承重与稳定性未通过，新闭链由 PhysX Drive 与约束联合求解。

修改思路：保留 27 维观测和 6 维动作契约，采用新机构物理参数；新网络训练及后续初始化、续训阶段分别标记。

证据：`mine_model/results/implicit_pid_base_20261004/report.json`；`mine_model/results/implicit_free_base_20261004/report.json`；`mine_model/results/auto_training_implicit_20pct_20261004/README.md`

## 奖励信号与失败判定修正

修改内容：扩大高度、速度、偏航、姿态和动作变化惩罚裁剪范围；增加低根高度终止，调整终止成本。

修改原因：原单项惩罚过早饱和，无法区分轻微误差和蹲塌；终止代价过低时，立即倒地可能比持续纠错更便宜。

修改思路：依据实测饱和率和误差扩大有效信号，保留正常课程；只有真实生存失败才回退平衡或恢复探索。

证据：`mine_model/results/auto_training_implicit_20pct_20261004/reward_saturation_audit_20261004.json`；`mine_model/results/auto_training_implicit_20pct_20261004/代码检查与提前回放.md`

## 速度初始化、轮抖动与精度训练

修改内容：记录速度/腿角初始化，训练轮探索、熵系数和 action_rate 调整；后续冻结腿输出及 actor 特征，仅训练轮输出和 critic。

修改原因：站立策略不能自然推出正确行驶，轮动作出现高频变化，已有站稳能力需要保留。

修改思路：用同权重实测选初始化和单项对照；记录失败增益试验。初始化不是从零 RL，也不是验收成功。

证据：`mine_model/results/speed_response_initialization_2199_20261004/initialization.json`；`mine_model/results/auto_training_implicit_20pct_20261004/wheel_flutter_audit_20261004.json`；`mine_model/results/wheel_head_precision_5399_5600_20261004/trial.json`

## 轮胎碰撞及保护限位

修改内容：轮胎碰撞由 18 边细化至 96 边；base 使用盒体，普通连杆可简化，确认的限位块及小腿接触面保留 CAD 约束。

修改原因：接触多边形影响轮速和机身波动；大量碰撞形状导致内存压力，但机械腿长限位不可被简化掉。

修改思路：先用固定权重比较接触，再做保护块/小腿 CAD 非相交检查。若腿部简化增加误差则回退，不能仅凭吞吐选择。

证据：`mine_model/docs/碰撞简化与机械限位说明_20261004.md`；`mine_model/results/collision_simplification_20261004/collision_comparison.json`；`mine_model/results/collision_simplification_20261004/protected_stops_v2_live_cad/cad_nonpenetration.json`

## 环境数与内存调整

修改内容：早期 256 环境；12288 请求发生 OOM 后选择成功规模，后续用户指定 10240，再因内存压力改为 8192。

修改原因：新闭链刚体、碰撞与克隆负载不同于旧模型，CUDA 计算仍需要主机内存。

修改思路：以成功运行和实测内存选择规模；用户要求停止容量测试后不再重测，规模切换保留来源权重和优化器。

证据：`mine_model/results/auto_training_implicit_20pct_20261004/env_scale_report_20261004.json`；`mine_model/results/collision_simplification_20261004/environment_capacity_report.json`；`mine_model/results/auto_training_implicit_20pct_20261004/before_env8192_after_round38.json`

## MuJoCo sim2sim 验证

修改内容：从完整训练 USD 转为 MuJoCo，保留刚体惯量、joint 与闭链约束，匹配观测/动作、镜像、COM 速度和实际隐式执行器。

修改原因：相同策略在不同接触求解器中可能表现不同；较粗内部积分曾失稳，基础运动稳定也不能代表所有误差通过。

修改思路：用接口测试、相同指令与种子及误差曲线验证；保留偏航超标和机械限位差异，不将两个仿真器视为相同动力学。

证据：`mine_model/docs/新闭链模型_MuJoCo_sim2sim验证_20261004.md`；`mine_model/results/sim2sim_mujoco_6700_20261004/comparison.json`；`mine_model/results/latest_sim2sim_preview.json`

## 10% 精度目标、记录留存和自动完成

修改内容：目标从 20% 收紧到 10%，同步减半绝对容差及姿态门限；保留最新旧模型及默认验证依赖，删除其余旧训练记录。

修改原因：仅改百分比不能收紧零目标工况；最终记录需要覆盖历史失败、真实修改原因与完整交付。

修改思路：逐轮按实际应用参数和评估阈值记录，旧标准结果不能替代新标准；复验、文档、打包与刷盘全部完成后才关机。

证据：`mine_model/config/tracking_acceptance.json`；`mine_model/results/auto_training_implicit_20pct_20261004/acceptance_target_change_20261004.json`

## 人工干预和用户要求的变更

### 干预 1：时间未记录 阶段干预

记录的原因：未单独记录

```json
{
  "checkpoint": "/tmp/tmpl1timlcu/warm_start.pt",
  "checkpoint_sha256": "8198f7e4b4c97810d43e95e8870bb5ce16e263c58f92c7458b17a1c65be228ae",
  "initialization_manifest": "/tmp/tmpl1timlcu/initialization.json",
  "angle_initialization_manifest": "/tmp/tmpl1timlcu/initialization.json"
}
```

## 逐轮参数变化、原因与验证结果

未发生参数变化的续训轮次也记录；达到参数上限不写成已调整。不同验收标准的通过数不能直接视为连续收敛趋势。

### auto_training_20261003 / round_000_pilot

本轮开始前的决策原文：未留存该轮单独说明；请结合上一轮评估及人工干预证据。

|参数|之前|应用后|
|---|---|---|
|commands.grouped_training|null|true|
|commands.heading_command|null|false|
|commands.motion_ramp_steps|null|48000|
|commands.ranges_ang_vel_yaw|null|[-0.3, 0.3]|
|commands.reverse_ramp_start_steps|null|48000|
|commands.reverse_ramp_steps|null|48000|
|commands.standing_only_steps|null|48000|
|commands.yaw_ramp_steps|null|48000|
|commands.yaw_start_steps|null|48000|

验证：iteration 999，存活 6/27，完整精度通过 0/27，重置 247，整体通过 False。

验收标准：Relaxed sim2sim preparation: no first-episode resets; worst-seed steady MAE, final 3 seconds; speed max(0.08 m/s,25% target), yaw max(0.08 rad/s,25% target), height/leg 30 mm, leg angle 0.1 rad, posture p95 10 deg.

最大归一化误差：leg_angle_mae_rad，工况 spin_right，seed 44，误差/门限 = 7.6699。

应用配置：`mine_model/results/auto_training_20261003/round_000_pilot/overrides.json`。

### auto_training_20261003 / round_001_optimization

本轮开始前的决策原文：extend balance phase; restore exploration; strengthen unsafe termination cost

|参数|之前|应用后|
|---|---|---|
|commands.reverse_ramp_start_steps|48000|96000|
|commands.standing_only_steps|48000|96000|
|commands.yaw_start_steps|48000|96000|
|rewards.termination|null|-12.0|

尚无该轮完整评估结果；不能据此称为通过。

应用配置：`mine_model/results/auto_training_20261003/round_001_optimization/overrides.json`。

### auto_training_20pct_20261004 / round_000_pilot

本轮开始前的决策原文：未留存该轮单独说明；请结合上一轮评估及人工干预证据。

|参数|之前|应用后|
|---|---|---|
|commands.grouped_training|null|true|
|commands.heading_command|null|false|
|commands.motion_ramp_steps|null|48000|
|commands.ranges_ang_vel_yaw|null|[-0.3, 0.3]|
|commands.reverse_ramp_start_steps|null|134400|
|commands.reverse_ramp_steps|null|48000|
|commands.standing_only_steps|null|134400|
|commands.yaw_ramp_steps|null|48000|
|commands.yaw_start_steps|null|134400|
|min_root_height|null|0.22|
|reset_velocity_final|null|0.1|
|rewards.base_height_penalty_clip|null|30.0|
|rewards.standing_leg_angle|null|-5.0|
|rewards.termination|null|-12.0|

验证：iteration 2799，存活 0/27，完整精度通过 0/27，重置 945，整体通过 False。

验收标准：20 percent sim2sim preparation: all conditions and seeds survive without reset; final 3-second MAE; speed/yaw max(0.05 absolute,20 percent target); height/length 20 percent target; leg angle 0.04 rad (20 percent of 0.2 rad range); wheel velocity max(0.3 rad/s,20 percent reference); posture p95 5 deg; steady minimum root height at least 80 percent of command. Independent seeds required before export.

应用配置：`mine_model/results/auto_training_20pct_20261004/round_000_pilot/overrides.json`。

### auto_training_20pct_20261004 / round_001_optimization

本轮开始前的决策原文：extend balance phase; restore exploration; strengthen unsafe termination cost

|参数|之前|应用后|
|---|---|---|
|commands.reverse_ramp_start_steps|134400|182400|
|commands.standing_only_steps|134400|182400|
|commands.yaw_start_steps|134400|182400|
|kd_theta|null|0.5|
|kp_theta|null|10.0|
|rewards.base_height_error_sq|null|-1200.0|
|rewards.base_height_penalty_clip|30.0|38.879999999999995|
|rewards.termination|-12.0|-14.399999999999999|

验证：iteration 3799，存活 0/27，完整精度通过 0/27，重置 891，整体通过 False。

验收标准：20 percent sim2sim preparation: all conditions and seeds survive without reset; final 3-second MAE; speed/yaw max(0.05 absolute,20 percent target); height/length 20 percent target; leg angle 0.04 rad (20 percent of 0.2 rad range); wheel velocity max(0.3 rad/s,20 percent reference); posture p95 5 deg; steady minimum root height at least 80 percent of command. Independent seeds required before export.

应用配置：`mine_model/results/auto_training_20pct_20261004/round_001_optimization/overrides.json`。

### auto_training_20pct_20261004 / round_002_optimization

本轮开始前的决策原文：extend balance phase; restore exploration; strengthen unsafe termination cost

|参数|之前|应用后|
|---|---|---|
|commands.reverse_ramp_start_steps|182400|230400|
|commands.standing_only_steps|182400|230400|
|commands.yaw_start_steps|182400|230400|
|rewards.base_height_error_sq|-1200.0|-1440.0|
|rewards.base_height_penalty_clip|38.879999999999995|46.656|
|rewards.termination|-14.399999999999999|-17.279999999999998|

验证：iteration 4799，存活 0/27，完整精度通过 0/27，重置 891，整体通过 False。

验收标准：20 percent sim2sim preparation: all conditions and seeds survive without reset; final 3-second MAE; speed/yaw max(0.05 absolute,20 percent target); height/length 20 percent target; leg angle 0.04 rad (20 percent of 0.2 rad range); wheel velocity max(0.3 rad/s,20 percent reference); posture p95 5 deg; steady minimum root height at least 80 percent of command. Independent seeds required before export.

应用配置：`mine_model/results/auto_training_20pct_20261004/round_002_optimization/overrides.json`。

### auto_training_20pct_20261004 / round_003_optimization

本轮开始前的决策原文：extend balance phase; restore exploration; strengthen unsafe termination cost

|参数|之前|应用后|
|---|---|---|
|commands.reverse_ramp_start_steps|230400|278400|
|commands.standing_only_steps|230400|278400|
|commands.yaw_start_steps|230400|278400|
|rewards.base_height_error_sq|-1440.0|-1728.0|
|rewards.base_height_penalty_clip|46.656|55.987199999999994|
|rewards.termination|-17.279999999999998|-20.735999999999997|

尚无该轮完整评估结果；不能据此称为通过。

应用配置：`mine_model/results/auto_training_20pct_20261004/round_003_optimization/overrides.json`。

## 完成判定和关机顺序

当前目标采用统一 tracking_acceptance.json。9 工况 × 3 种子首先通过，再用独立种子复验，最后同一权重每工况 30 秒延长复测；失败回合不作为稳态样本。

通过后停止追加训练，导出权重、控制契约和参数，生成本记录及最终更改说明，保留 DOCX/PDF 教程、三阶段误差曲线和代码快照，核对文件哈希及 ZIP CRC，sync 刷盘后执行关机。任何验收或文档交付失败都不关机。

Isaac Lab 的准备验收与 MuJoCo sim2sim 验收分开。现存 MuJoCo 验证和未通过项保留，不能把 Isaac Lab 达标写成跨仿真器全部通过。

## 可复查的完整记录

`training_record.json` 保存结构化变更与证据 SHA256；`code_current/` 保存当前相关源代码，`tracked_current_changes.patch` 保存已跟踪文件相对 HEAD 的实际差异，`saved_source_changes.patch` 保存已有源码备份到当前版本的真实累计差异。没有备份的早期代码不补造逐行变化。

# 每次训练后的优化流程

流程：本轮权重固定回放 → 定量误差与失败项 → 诊断与决策 → 选择续训或回退权重 → 计划和实际参数变化 → 下一轮验证。

按评估报告的完整目录关联决策，独立种子或30秒复测失败优先保留其实际依据；相同轮号的接触对照和训练轮分别记录。无改值明确写仅续训，没有历史证据不补造原因。

## auto_training_20261003 / round_000_pilot

已记录训练后决策。

本目录记录的实际训练启动/恢复次数：1。

每次真实启动及恢复的参数、权重、环境数和命令：

```json
[
  {
    "folder": "round_000_pilot",
    "checkpoint": "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_19-36-55_mine_fresh/model_200.pt",
    "num_envs": "256",
    "target_iteration": "1000",
    "wheel_head_only_finetune": false,
    "reset_optimizer": true,
    "reset_exploration_std": "0.2",
    "overrides": {
      "commands.standing_only_steps": 48000,
      "commands.motion_ramp_steps": 48000,
      "commands.reverse_ramp_start_steps": 48000,
      "commands.reverse_ramp_steps": 48000,
      "commands.yaw_start_steps": 48000,
      "commands.yaw_ramp_steps": 48000,
      "commands.ranges_ang_vel_yaw": [
        -0.3,
        0.3
      ],
      "commands.heading_command": false,
      "commands.grouped_training": true
    },
    "agent_overrides": {},
    "command": [
      "/home/aaa/studyRL/src/CLT-RL/run_python.sh",
      "/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/scripts/rsl_rl/train.py",
      "--task",
      "WheelLeggedVMC-Flat-v0",
      "--headless",
      "--num_envs",
      "256",
      "--seed",
      "43",
      "--run_name",
      "mine_auto_r000_auto_training_20261003",
      "--checkpoint_path",
      "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_19-36-55_mine_fresh/model_200.pt",
      "--target_iteration",
      "1000",
      "--reset_optimizer",
      "--reset_exploration_std",
      "0.2",
      "env.commands.standing_only_steps=48000",
      "env.commands.motion_ramp_steps=48000",
      "env.commands.reverse_ramp_start_steps=48000",
      "env.commands.reverse_ramp_steps=48000",
      "env.commands.yaw_start_steps=48000",
      "env.commands.yaw_ramp_steps=48000",
      "env.commands.ranges_ang_vel_yaw=[-0.3, 0.3]",
      "env.commands.heading_command=false",
      "env.commands.grouped_training=true"
    ],
    "evidence": {
      "log": "mine_model/build/logs/auto_training_service.log",
      "line": 2
    }
  }
]
```

本轮 assessment：iteration 999，存活 6/27，精度通过 0/27，重置 247；报告 `mine_model/results/auto_training_20261003/round_000_pilot/replay/assessment.json`。

### 评估后决定下一轮 1

实际决策依据：`/home/aaa/studyRL/src/CLT-RL/mine_model/results/auto_training_20261003/round_000_pilot/replay/assessment.json`。

当时验收标准：Relaxed sim2sim preparation: no first-episode resets; worst-seed steady MAE, final 3 seconds; speed max(0.08 m/s,25% target), yaw max(0.08 rad/s,25% target), height/leg 30 mm, leg angle 0.1 rad, posture p95 10 deg.

结果：精度通过 0/27，重置 247。

最大超差/误差项：spin_right / seed 44 / leg_angle_mae_rad，误差÷门限 7.6699。

失败项及实际数值（未存活、未达到指令、姿态支撑也单独记录）：

```json
[
  {
    "condition": "stand",
    "seed": 43,
    "survived": false,
    "resets": 1,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "stand",
    "seed": 44,
    "survived": false,
    "resets": 1,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "stand",
    "seed": 45,
    "survived": false,
    "resets": 1,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 43,
    "survived": false,
    "resets": 9,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 44,
    "survived": false,
    "resets": 3,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 45,
    "survived": false,
    "resets": 6,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 43,
    "survived": false,
    "resets": 11,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 44,
    "survived": false,
    "resets": 5,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 45,
    "survived": false,
    "resets": 4,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 43,
    "survived": false,
    "resets": 11,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 44,
    "survived": false,
    "resets": 12,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 45,
    "survived": false,
    "resets": 12,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 43,
    "survived": false,
    "resets": 15,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 44,
    "survived": false,
    "resets": 15,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 45,
    "survived": false,
    "resets": 15,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 43,
    "survived": true,
    "resets": 0,
    "exceeded": {
      "yaw_mae_rad_s": {
        "value": 0.23871684477692284,
        "limit": 0.08,
        "ratio": 2.9839605597115355
      },
      "height_mae_m": {
        "value": 0.15895360875998113,
        "limit": 0.03,
        "ratio": 5.298453625332704
      },
      "leg_length_mae_m": {
        "value": 0.1617911664263302,
        "limit": 0.03,
        "ratio": 5.393038880877674
      },
      "leg_angle_mae_rad": {
        "value": 0.7627863157864364,
        "limit": 0.1,
        "ratio": 7.627863157864364
      }
    },
    "upright_support": null,
    "command_reached": true
  },
  {
    "condition": "spin_left",
    "seed": 44,
    "survived": true,
    "resets": 0,
    "exceeded": {
      "yaw_mae_rad_s": {
        "value": 0.23193256731495102,
        "limit": 0.08,
        "ratio": 2.8991570914368876
      },
      "height_mae_m": {
        "value": 0.15893666308052493,
        "limit": 0.03,
        "ratio": 5.297888769350831
      },
      "leg_length_mae_m": {
        "value": 0.1617802099281589,
        "limit": 0.03,
        "ratio": 5.392673664271963
      },
      "leg_angle_mae_rad": {
        "value": 0.7637247975101534,
        "limit": 0.1,
        "ratio": 7.637247975101534
      }
    },
    "upright_support": null,
    "command_reached": true
  },
  {
    "condition": "spin_left",
    "seed": 45,
    "survived": true,
    "resets": 0,
    "exceeded": {
      "yaw_mae_rad_s": {
        "value": 0.23912441366450823,
        "limit": 0.08,
        "ratio": 2.989055170806353
      },
      "height_mae_m": {
        "value": 0.15892633618108484,
        "limit": 0.03,
        "ratio": 5.297544539369495
      },
      "leg_length_mae_m": {
        "value": 0.1618948524558781,
        "limit": 0.03,
        "ratio": 5.396495081862604
      },
      "leg_angle_mae_rad": {
        "value": 0.7622758337167024,
        "limit": 0.1,
        "ratio": 7.622758337167023
      }
    },
    "upright_support": null,
    "command_reached": true
  },
  {
    "condition": "spin_right",
    "seed": 43,
    "survived": true,
    "resets": 0,
    "exceeded": {
      "yaw_mae_rad_s": {
        "value": 0.3353399625059606,
        "limit": 0.08,
        "ratio": 4.191749531324508
      },
      "height_mae_m": {
        "value": 0.1592193100823472,
        "limit": 0.03,
        "ratio": 5.30731033607824
      },
      "leg_length_mae_m": {
        "value": 0.16213170098547905,
        "limit": 0.03,
        "ratio": 5.404390032849302
      },
      "leg_angle_mae_rad": {
        "value": 0.7660501904582099,
        "limit": 0.1,
        "ratio": 7.660501904582098
      }
    },
    "upright_support": null,
    "command_reached": true
  },
  {
    "condition": "spin_right",
    "seed": 44,
    "survived": true,
    "resets": 0,
    "exceeded": {
      "yaw_mae_rad_s": {
        "value": 0.33466462193691826,
        "limit": 0.08,
        "ratio": 4.183307774211478
      },
      "height_mae_m": {
        "value": 0.15922465979658215,
        "limit": 0.03,
        "ratio": 5.307488659886072
      },
      "leg_length_mae_m": {
        "value": 0.16207433200829865,
        "limit": 0.03,
        "ratio": 5.402477733609955
      },
      "leg_angle_mae_rad": {
        "value": 0.7669906639308971,
        "limit": 0.1,
        "ratio": 7.66990663930897
      }
    },
    "upright_support": null,
    "command_reached": true
  },
  {
    "condition": "spin_right",
    "seed": 45,
    "survived": true,
    "resets": 0,
    "exceeded": {
      "yaw_mae_rad_s": {
        "value": 0.33767225175119925,
        "limit": 0.08,
        "ratio": 4.220903146889991
      },
      "height_mae_m": {
        "value": 0.15912520490734783,
        "limit": 0.03,
        "ratio": 5.304173496911594
      },
      "leg_length_mae_m": {
        "value": 0.16200525999463947,
        "limit": 0.03,
        "ratio": 5.400175333154649
      },
      "leg_angle_mae_rad": {
        "value": 0.7665967455920948,
        "limit": 0.1,
        "ratio": 7.665967455920947
      }
    },
    "upright_support": null,
    "command_reached": true
  },
  {
    "condition": "height_low",
    "seed": 43,
    "survived": false,
    "resets": 19,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 44,
    "survived": false,
    "resets": 19,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 45,
    "survived": false,
    "resets": 16,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 43,
    "survived": false,
    "resets": 24,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 44,
    "survived": false,
    "resets": 24,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 45,
    "survived": false,
    "resets": 24,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  }
]
```

诊断与决策原文：extend balance phase; restore exploration; strengthen unsafe termination cost

决策证据：`mine_model/build/logs/auto_training_service.log` 第 9 行。

权重选择：继续本轮评估权重；`/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_20-55-50_mine_auto_r000_auto_training_20261003/model_999.pt`。

规划环境参数改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|48000|96000|
|"commands.standing_only_steps"|48000|96000|
|"commands.yaw_start_steps"|48000|96000|
|"rewards.termination"|null|-12.0|

下一轮首次实际启动环境参数改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|48000|96000|
|"commands.standing_only_steps"|48000|96000|
|"commands.yaw_start_steps"|48000|96000|
|"rewards.termination"|null|-12.0|

下一轮目录最终保存环境配置改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|48000|96000|
|"commands.standing_only_steps"|48000|96000|
|"commands.yaw_start_steps"|48000|96000|
|"rewards.termination"|null|-12.0|

算法、环境数和训练恢复方式改值：该类参数未变化。

下一轮实际启动：round_001_optimization，环境 256，目标 iteration 2000；加载 `/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_20-55-50_mine_auto_r000_auto_training_20261003/model_999.pt`；仅训练轮输出 False；重置优化器 True。

算法覆盖：`{}`。

下一轮验证：待下一轮完整验证。

```json
{
  "status": "待下一轮完整验证"
}
```

## auto_training_20261003 / round_001_optimization

尚无完整评估，未形成可核验的训练后优化流程。

本目录记录的实际训练启动/恢复次数：1。

每次真实启动及恢复的参数、权重、环境数和命令：

```json
[
  {
    "folder": "round_001_optimization",
    "checkpoint": "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_20-55-50_mine_auto_r000_auto_training_20261003/model_999.pt",
    "num_envs": "256",
    "target_iteration": "2000",
    "wheel_head_only_finetune": false,
    "reset_optimizer": true,
    "reset_exploration_std": "0.2",
    "overrides": {
      "commands.standing_only_steps": 96000,
      "commands.motion_ramp_steps": 48000,
      "commands.reverse_ramp_start_steps": 96000,
      "commands.reverse_ramp_steps": 48000,
      "commands.yaw_start_steps": 96000,
      "commands.yaw_ramp_steps": 48000,
      "commands.ranges_ang_vel_yaw": [
        -0.3,
        0.3
      ],
      "commands.heading_command": false,
      "commands.grouped_training": true,
      "rewards.termination": -12.0
    },
    "agent_overrides": {},
    "command": [
      "/home/aaa/studyRL/src/CLT-RL/run_python.sh",
      "/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/scripts/rsl_rl/train.py",
      "--task",
      "WheelLeggedVMC-Flat-v0",
      "--headless",
      "--num_envs",
      "256",
      "--seed",
      "43",
      "--run_name",
      "mine_auto_r001_auto_training_20261003",
      "--checkpoint_path",
      "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_20-55-50_mine_auto_r000_auto_training_20261003/model_999.pt",
      "--target_iteration",
      "2000",
      "--reset_optimizer",
      "--reset_exploration_std",
      "0.2",
      "env.commands.standing_only_steps=96000",
      "env.commands.motion_ramp_steps=48000",
      "env.commands.reverse_ramp_start_steps=96000",
      "env.commands.reverse_ramp_steps=48000",
      "env.commands.yaw_start_steps=96000",
      "env.commands.yaw_ramp_steps=48000",
      "env.commands.ranges_ang_vel_yaw=[-0.3, 0.3]",
      "env.commands.heading_command=false",
      "env.commands.grouped_training=true",
      "env.rewards.termination=-12.0"
    ],
    "evidence": {
      "log": "mine_model/build/logs/auto_training_service.log",
      "line": 11
    }
  }
]
```

## auto_training_20pct_20261004 / round_000_pilot

已记录训练后决策。

本目录记录的实际训练启动/恢复次数：2。

每次真实启动及恢复的参数、权重、环境数和命令：

```json
[
  {
    "folder": "round_000_pilot",
    "checkpoint": "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_22-07-08_mine_auto_r001_auto_training_20261003/model_1800.pt",
    "num_envs": "256",
    "target_iteration": "2800",
    "wheel_head_only_finetune": false,
    "reset_optimizer": true,
    "reset_exploration_std": "0.3",
    "overrides": {
      "commands.standing_only_steps": 134400,
      "commands.motion_ramp_steps": 48000,
      "commands.reverse_ramp_start_steps": 134400,
      "commands.reverse_ramp_steps": 48000,
      "commands.yaw_start_steps": 134400,
      "commands.yaw_ramp_steps": 48000,
      "commands.ranges_ang_vel_yaw": [
        -0.3,
        0.3
      ],
      "commands.heading_command": false,
      "commands.grouped_training": true,
      "rewards.termination": -12.0,
      "rewards.base_height_penalty_clip": 30.0,
      "rewards.standing_leg_angle": -5.0,
      "min_root_height": 0.22,
      "reset_velocity_final": 0.1
    },
    "agent_overrides": {},
    "command": [
      "/home/aaa/studyRL/src/CLT-RL/run_python.sh",
      "/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/scripts/rsl_rl/train.py",
      "--task",
      "WheelLeggedVMC-Flat-v0",
      "--headless",
      "--num_envs",
      "256",
      "--seed",
      "43",
      "--run_name",
      "mine_auto_r000_auto_training_20pct_20261004",
      "--checkpoint_path",
      "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_22-07-08_mine_auto_r001_auto_training_20261003/model_1800.pt",
      "--target_iteration",
      "2800",
      "--reset_optimizer",
      "--reset_exploration_std",
      "0.3",
      "env.commands.standing_only_steps=134400",
      "env.commands.motion_ramp_steps=48000",
      "env.commands.reverse_ramp_start_steps=134400",
      "env.commands.reverse_ramp_steps=48000",
      "env.commands.yaw_start_steps=134400",
      "env.commands.yaw_ramp_steps=48000",
      "env.commands.ranges_ang_vel_yaw=[-0.3, 0.3]",
      "env.commands.heading_command=false",
      "env.commands.grouped_training=true",
      "env.rewards.termination=-12.0",
      "env.rewards.base_height_penalty_clip=30.0",
      "env.rewards.standing_leg_angle=-5.0",
      "env.min_root_height=0.22",
      "env.reset_velocity_final=0.1"
    ],
    "evidence": {
      "log": "mine_model/build/logs/auto_training_20pct_service.log",
      "line": 2
    }
  },
  {
    "folder": "round_000_pilot",
    "checkpoint": "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_22-07-08_mine_auto_r001_auto_training_20261003/model_1800.pt",
    "num_envs": "256",
    "target_iteration": "2800",
    "wheel_head_only_finetune": false,
    "reset_optimizer": true,
    "reset_exploration_std": "0.3",
    "overrides": {
      "commands.standing_only_steps": 134400,
      "commands.motion_ramp_steps": 48000,
      "commands.reverse_ramp_start_steps": 134400,
      "commands.reverse_ramp_steps": 48000,
      "commands.yaw_start_steps": 134400,
      "commands.yaw_ramp_steps": 48000,
      "commands.ranges_ang_vel_yaw": [
        -0.3,
        0.3
      ],
      "commands.heading_command": false,
      "commands.grouped_training": true,
      "rewards.termination": -12.0,
      "rewards.base_height_penalty_clip": 30.0,
      "rewards.standing_leg_angle": -5.0,
      "min_root_height": 0.22,
      "reset_velocity_final": 0.1
    },
    "agent_overrides": {},
    "command": [
      "/home/aaa/studyRL/src/CLT-RL/run_python.sh",
      "/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/scripts/rsl_rl/train.py",
      "--task",
      "WheelLeggedVMC-Flat-v0",
      "--headless",
      "--num_envs",
      "256",
      "--seed",
      "43",
      "--run_name",
      "mine_auto_r000_auto_training_20pct_20261004",
      "--checkpoint_path",
      "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-03_22-07-08_mine_auto_r001_auto_training_20261003/model_1800.pt",
      "--target_iteration",
      "2800",
      "--reset_optimizer",
      "--reset_exploration_std",
      "0.3",
      "env.commands.standing_only_steps=134400",
      "env.commands.motion_ramp_steps=48000",
      "env.commands.reverse_ramp_start_steps=134400",
      "env.commands.reverse_ramp_steps=48000",
      "env.commands.yaw_start_steps=134400",
      "env.commands.yaw_ramp_steps=48000",
      "env.commands.ranges_ang_vel_yaw=[-0.3, 0.3]",
      "env.commands.heading_command=false",
      "env.commands.grouped_training=true",
      "env.rewards.termination=-12.0",
      "env.rewards.base_height_penalty_clip=30.0",
      "env.rewards.standing_leg_angle=-5.0",
      "env.min_root_height=0.22",
      "env.reset_velocity_final=0.1"
    ],
    "evidence": {
      "log": "mine_model/build/logs/auto_training_20pct_service.log",
      "line": 14
    }
  }
]
```

本轮 assessment：iteration 2799，存活 0/27，精度通过 0/27，重置 945；报告 `mine_model/results/auto_training_20pct_20261004/round_000_pilot/replay/assessment.json`。

### 评估后决定下一轮 1

实际决策依据：`/home/aaa/studyRL/src/CLT-RL/mine_model/results/auto_training_20pct_20261004/round_000_pilot/replay/assessment.json`。

当时验收标准：20 percent sim2sim preparation: all conditions and seeds survive without reset; final 3-second MAE; speed/yaw max(0.05 absolute,20 percent target); height/length 20 percent target; leg angle 0.04 rad (20 percent of 0.2 rad range); wheel velocity max(0.3 rad/s,20 percent reference); posture p95 5 deg; steady minimum root height at least 80 percent of command. Independent seeds required before export.

结果：精度通过 0/27，重置 945。

失败项及实际数值（未存活、未达到指令、姿态支撑也单独记录）：

```json
[
  {
    "condition": "stand",
    "seed": 43,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "stand",
    "seed": 44,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "stand",
    "seed": 45,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 43,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 44,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 45,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 43,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 44,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 45,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 43,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 44,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 45,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 43,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 44,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 45,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 43,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 44,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 45,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_right",
    "seed": 43,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_right",
    "seed": 44,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_right",
    "seed": 45,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 43,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 44,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 45,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 43,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 44,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 45,
    "survived": false,
    "resets": 35,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  }
]
```

诊断与决策原文：extend balance phase; restore exploration; strengthen unsafe termination cost

决策证据：`mine_model/build/logs/auto_training_20pct_service.log` 第 21 行。

权重选择：继续本轮评估权重；`/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_00-28-07_mine_auto_r000_auto_training_20pct_20261004/model_2799.pt`。

规划环境参数改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|134400|182400|
|"commands.standing_only_steps"|134400|182400|
|"commands.yaw_start_steps"|134400|182400|
|"rewards.base_height_error_sq"|null|-1200.0|
|"rewards.base_height_penalty_clip"|30.0|38.879999999999995|
|"rewards.termination"|-12.0|-14.399999999999999|

下一轮首次实际启动环境参数改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|134400|182400|
|"commands.standing_only_steps"|134400|182400|
|"commands.yaw_start_steps"|134400|182400|
|"rewards.base_height_error_sq"|null|-1200.0|
|"rewards.base_height_penalty_clip"|30.0|38.879999999999995|
|"rewards.termination"|-12.0|-14.399999999999999|

下一轮目录最终保存环境配置改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|134400|182400|
|"commands.standing_only_steps"|134400|182400|
|"commands.yaw_start_steps"|134400|182400|
|"kd_theta"|null|0.5|
|"kp_theta"|null|10.0|
|"rewards.base_height_error_sq"|null|-1200.0|
|"rewards.base_height_penalty_clip"|30.0|38.879999999999995|
|"rewards.termination"|-12.0|-14.399999999999999|

算法、环境数和训练恢复方式改值：该类参数未变化。

下一轮实际启动：round_001_optimization，环境 256，目标 iteration 3800；加载 `/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_00-28-07_mine_auto_r000_auto_training_20pct_20261004/model_2799.pt`；仅训练轮输出 False；重置优化器 True。

算法覆盖：`{}`。

下一轮验证：同标准同资产同回放协议对比。

```json
{
  "status": "同标准同资产同回放协议对比",
  "comparable": true,
  "before_passed_trials": 0,
  "after_passed_trials": 0,
  "before_worst_error": null,
  "after_worst_error": null
}
```

下一轮结果证据：`mine_model/results/auto_training_20pct_20261004/round_001_optimization/replay/assessment.json`。

## auto_training_20pct_20261004 / round_001_optimization

已记录训练后决策。

本目录记录的实际训练启动/恢复次数：2。

每次真实启动及恢复的参数、权重、环境数和命令：

```json
[
  {
    "folder": "round_001_optimization",
    "checkpoint": "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_00-28-07_mine_auto_r000_auto_training_20pct_20261004/model_2799.pt",
    "num_envs": "256",
    "target_iteration": "3800",
    "wheel_head_only_finetune": false,
    "reset_optimizer": true,
    "reset_exploration_std": "0.3",
    "overrides": {
      "commands.standing_only_steps": 182400,
      "commands.motion_ramp_steps": 48000,
      "commands.reverse_ramp_start_steps": 182400,
      "commands.reverse_ramp_steps": 48000,
      "commands.yaw_start_steps": 182400,
      "commands.yaw_ramp_steps": 48000,
      "commands.ranges_ang_vel_yaw": [
        -0.3,
        0.3
      ],
      "commands.heading_command": false,
      "commands.grouped_training": true,
      "rewards.termination": -14.399999999999999,
      "rewards.base_height_penalty_clip": 38.879999999999995,
      "rewards.standing_leg_angle": -5.0,
      "min_root_height": 0.22,
      "reset_velocity_final": 0.1,
      "rewards.base_height_error_sq": -1200.0
    },
    "agent_overrides": {},
    "command": [
      "/home/aaa/studyRL/src/CLT-RL/run_python.sh",
      "/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/scripts/rsl_rl/train.py",
      "--task",
      "WheelLeggedVMC-Flat-v0",
      "--headless",
      "--num_envs",
      "256",
      "--seed",
      "43",
      "--run_name",
      "mine_auto_r001_auto_training_20pct_20261004",
      "--checkpoint_path",
      "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_00-28-07_mine_auto_r000_auto_training_20pct_20261004/model_2799.pt",
      "--target_iteration",
      "3800",
      "--reset_optimizer",
      "--reset_exploration_std",
      "0.3",
      "env.commands.standing_only_steps=182400",
      "env.commands.motion_ramp_steps=48000",
      "env.commands.reverse_ramp_start_steps=182400",
      "env.commands.reverse_ramp_steps=48000",
      "env.commands.yaw_start_steps=182400",
      "env.commands.yaw_ramp_steps=48000",
      "env.commands.ranges_ang_vel_yaw=[-0.3, 0.3]",
      "env.commands.heading_command=false",
      "env.commands.grouped_training=true",
      "env.rewards.termination=-14.399999999999999",
      "env.rewards.base_height_penalty_clip=38.879999999999995",
      "env.rewards.standing_leg_angle=-5.0",
      "env.min_root_height=0.22",
      "env.reset_velocity_final=0.1",
      "env.rewards.base_height_error_sq=-1200.0"
    ],
    "evidence": {
      "log": "mine_model/build/logs/auto_training_20pct_service.log",
      "line": 23
    }
  },
  {
    "folder": "round_001_optimization",
    "checkpoint": "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_01-22-58_mine_auto_r001_auto_training_20pct_20261004/model_2900.pt",
    "num_envs": "256",
    "target_iteration": "3800",
    "wheel_head_only_finetune": false,
    "reset_optimizer": true,
    "reset_exploration_std": "0.3",
    "overrides": {
      "commands.standing_only_steps": 182400,
      "commands.motion_ramp_steps": 48000,
      "commands.reverse_ramp_start_steps": 182400,
      "commands.reverse_ramp_steps": 48000,
      "commands.yaw_start_steps": 182400,
      "commands.yaw_ramp_steps": 48000,
      "commands.ranges_ang_vel_yaw": [
        -0.3,
        0.3
      ],
      "commands.heading_command": false,
      "commands.grouped_training": true,
      "rewards.termination": -14.399999999999999,
      "rewards.base_height_penalty_clip": 38.879999999999995,
      "rewards.standing_leg_angle": -5.0,
      "min_root_height": 0.22,
      "reset_velocity_final": 0.1,
      "rewards.base_height_error_sq": -1200.0,
      "kp_theta": 10.0,
      "kd_theta": 0.5
    },
    "agent_overrides": {},
    "command": [
      "/home/aaa/studyRL/src/CLT-RL/run_python.sh",
      "/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/scripts/rsl_rl/train.py",
      "--task",
      "WheelLeggedVMC-Flat-v0",
      "--headless",
      "--num_envs",
      "256",
      "--seed",
      "43",
      "--run_name",
      "mine_auto_r001_auto_training_20pct_20261004",
      "--checkpoint_path",
      "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_01-22-58_mine_auto_r001_auto_training_20pct_20261004/model_2900.pt",
      "--target_iteration",
      "3800",
      "--reset_optimizer",
      "--reset_exploration_std",
      "0.3",
      "env.commands.standing_only_steps=182400",
      "env.commands.motion_ramp_steps=48000",
      "env.commands.reverse_ramp_start_steps=182400",
      "env.commands.reverse_ramp_steps=48000",
      "env.commands.yaw_start_steps=182400",
      "env.commands.yaw_ramp_steps=48000",
      "env.commands.ranges_ang_vel_yaw=[-0.3, 0.3]",
      "env.commands.heading_command=false",
      "env.commands.grouped_training=true",
      "env.rewards.termination=-14.399999999999999",
      "env.rewards.base_height_penalty_clip=38.879999999999995",
      "env.rewards.standing_leg_angle=-5.0",
      "env.min_root_height=0.22",
      "env.reset_velocity_final=0.1",
      "env.rewards.base_height_error_sq=-1200.0",
      "env.kp_theta=10.0",
      "env.kd_theta=0.5"
    ],
    "evidence": {
      "log": "mine_model/build/logs/auto_training_20pct_service.log",
      "line": 40
    }
  }
]
```

本轮 assessment：iteration 3799，存活 0/27，精度通过 0/27，重置 891；报告 `mine_model/results/auto_training_20pct_20261004/round_001_optimization/replay/assessment.json`。

### 评估后决定下一轮 2

实际决策依据：`/home/aaa/studyRL/src/CLT-RL/mine_model/results/auto_training_20pct_20261004/round_001_optimization/replay/assessment.json`。

当时验收标准：20 percent sim2sim preparation: all conditions and seeds survive without reset; final 3-second MAE; speed/yaw max(0.05 absolute,20 percent target); height/length 20 percent target; leg angle 0.04 rad (20 percent of 0.2 rad range); wheel velocity max(0.3 rad/s,20 percent reference); posture p95 5 deg; steady minimum root height at least 80 percent of command. Independent seeds required before export.

结果：精度通过 0/27，重置 891。

失败项及实际数值（未存活、未达到指令、姿态支撑也单独记录）：

```json
[
  {
    "condition": "stand",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "stand",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "stand",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_right",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_right",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_right",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  }
]
```

诊断与决策原文：extend balance phase; restore exploration; strengthen unsafe termination cost

决策证据：`mine_model/build/logs/auto_training_20pct_service.log` 第 47 行。

权重选择：继续本轮评估权重；`/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_01-30-15_mine_auto_r001_auto_training_20pct_20261004/model_3799.pt`。

规划环境参数改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|182400|230400|
|"commands.standing_only_steps"|182400|230400|
|"commands.yaw_start_steps"|182400|230400|
|"rewards.base_height_error_sq"|-1200.0|-1440.0|
|"rewards.base_height_penalty_clip"|38.879999999999995|46.656|
|"rewards.termination"|-14.399999999999999|-17.279999999999998|

下一轮首次实际启动环境参数改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|182400|230400|
|"commands.standing_only_steps"|182400|230400|
|"commands.yaw_start_steps"|182400|230400|
|"rewards.base_height_error_sq"|-1200.0|-1440.0|
|"rewards.base_height_penalty_clip"|38.879999999999995|46.656|
|"rewards.termination"|-14.399999999999999|-17.279999999999998|

下一轮目录最终保存环境配置改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|182400|230400|
|"commands.standing_only_steps"|182400|230400|
|"commands.yaw_start_steps"|182400|230400|
|"rewards.base_height_error_sq"|-1200.0|-1440.0|
|"rewards.base_height_penalty_clip"|38.879999999999995|46.656|
|"rewards.termination"|-14.399999999999999|-17.279999999999998|

算法、环境数和训练恢复方式改值：该类参数未变化。

下一轮实际启动：round_002_optimization，环境 256，目标 iteration 4800；加载 `/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_01-30-15_mine_auto_r001_auto_training_20pct_20261004/model_3799.pt`；仅训练轮输出 False；重置优化器 True。

算法覆盖：`{}`。

下一轮验证：同标准同资产同回放协议对比。

```json
{
  "status": "同标准同资产同回放协议对比",
  "comparable": true,
  "before_passed_trials": 0,
  "after_passed_trials": 0,
  "before_worst_error": null,
  "after_worst_error": null
}
```

下一轮结果证据：`mine_model/results/auto_training_20pct_20261004/round_002_optimization/replay/assessment.json`。

## auto_training_20pct_20261004 / round_002_optimization

已记录训练后决策。

本目录记录的实际训练启动/恢复次数：1。

每次真实启动及恢复的参数、权重、环境数和命令：

```json
[
  {
    "folder": "round_002_optimization",
    "checkpoint": "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_01-30-15_mine_auto_r001_auto_training_20pct_20261004/model_3799.pt",
    "num_envs": "256",
    "target_iteration": "4800",
    "wheel_head_only_finetune": false,
    "reset_optimizer": true,
    "reset_exploration_std": "0.3",
    "overrides": {
      "commands.standing_only_steps": 230400,
      "commands.motion_ramp_steps": 48000,
      "commands.reverse_ramp_start_steps": 230400,
      "commands.reverse_ramp_steps": 48000,
      "commands.yaw_start_steps": 230400,
      "commands.yaw_ramp_steps": 48000,
      "commands.ranges_ang_vel_yaw": [
        -0.3,
        0.3
      ],
      "commands.heading_command": false,
      "commands.grouped_training": true,
      "rewards.termination": -17.279999999999998,
      "rewards.base_height_penalty_clip": 46.656,
      "rewards.standing_leg_angle": -5.0,
      "min_root_height": 0.22,
      "reset_velocity_final": 0.1,
      "rewards.base_height_error_sq": -1440.0,
      "kp_theta": 10.0,
      "kd_theta": 0.5
    },
    "agent_overrides": {},
    "command": [
      "/home/aaa/studyRL/src/CLT-RL/run_python.sh",
      "/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/scripts/rsl_rl/train.py",
      "--task",
      "WheelLeggedVMC-Flat-v0",
      "--headless",
      "--num_envs",
      "256",
      "--seed",
      "43",
      "--run_name",
      "mine_auto_r002_auto_training_20pct_20261004",
      "--checkpoint_path",
      "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_01-30-15_mine_auto_r001_auto_training_20pct_20261004/model_3799.pt",
      "--target_iteration",
      "4800",
      "--reset_optimizer",
      "--reset_exploration_std",
      "0.3",
      "env.commands.standing_only_steps=230400",
      "env.commands.motion_ramp_steps=48000",
      "env.commands.reverse_ramp_start_steps=230400",
      "env.commands.reverse_ramp_steps=48000",
      "env.commands.yaw_start_steps=230400",
      "env.commands.yaw_ramp_steps=48000",
      "env.commands.ranges_ang_vel_yaw=[-0.3, 0.3]",
      "env.commands.heading_command=false",
      "env.commands.grouped_training=true",
      "env.rewards.termination=-17.279999999999998",
      "env.rewards.base_height_penalty_clip=46.656",
      "env.rewards.standing_leg_angle=-5.0",
      "env.min_root_height=0.22",
      "env.reset_velocity_final=0.1",
      "env.rewards.base_height_error_sq=-1440.0",
      "env.kp_theta=10.0",
      "env.kd_theta=0.5"
    ],
    "evidence": {
      "log": "mine_model/build/logs/auto_training_20pct_service.log",
      "line": 49
    }
  }
]
```

本轮 assessment：iteration 4799，存活 0/27，精度通过 0/27，重置 891；报告 `mine_model/results/auto_training_20pct_20261004/round_002_optimization/replay/assessment.json`。

### 评估后决定下一轮 3

实际决策依据：`/home/aaa/studyRL/src/CLT-RL/mine_model/results/auto_training_20pct_20261004/round_002_optimization/replay/assessment.json`。

当时验收标准：20 percent sim2sim preparation: all conditions and seeds survive without reset; final 3-second MAE; speed/yaw max(0.05 absolute,20 percent target); height/length 20 percent target; leg angle 0.04 rad (20 percent of 0.2 rad range); wheel velocity max(0.3 rad/s,20 percent reference); posture p95 5 deg; steady minimum root height at least 80 percent of command. Independent seeds required before export.

结果：精度通过 0/27，重置 891。

失败项及实际数值（未存活、未达到指令、姿态支撑也单独记录）：

```json
[
  {
    "condition": "stand",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "stand",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "stand",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p3",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "forward_0p6",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p3",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "reverse_0p6",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_left",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_right",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_right",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "spin_right",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_low",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 43,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 44,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  },
  {
    "condition": "height_high",
    "seed": 45,
    "survived": false,
    "resets": 33,
    "exceeded": {},
    "upright_support": null,
    "command_reached": null
  }
]
```

诊断与决策原文：extend balance phase; restore exploration; strengthen unsafe termination cost

决策证据：`mine_model/build/logs/auto_training_20pct_service.log` 第 55 行。

权重选择：继续本轮评估权重；`/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_02-14-57_mine_auto_r002_auto_training_20pct_20261004/model_4799.pt`。

规划环境参数改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|230400|278400|
|"commands.standing_only_steps"|230400|278400|
|"commands.yaw_start_steps"|230400|278400|
|"rewards.base_height_error_sq"|-1440.0|-1728.0|
|"rewards.base_height_penalty_clip"|46.656|55.987199999999994|
|"rewards.termination"|-17.279999999999998|-20.735999999999997|

下一轮首次实际启动环境参数改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|230400|278400|
|"commands.standing_only_steps"|230400|278400|
|"commands.yaw_start_steps"|230400|278400|
|"rewards.base_height_error_sq"|-1440.0|-1728.0|
|"rewards.base_height_penalty_clip"|46.656|55.987199999999994|
|"rewards.termination"|-17.279999999999998|-20.735999999999997|

下一轮目录最终保存环境配置改值：

|参数|本轮|下一轮|
|---|---|---|
|"commands.reverse_ramp_start_steps"|230400|278400|
|"commands.standing_only_steps"|230400|278400|
|"commands.yaw_start_steps"|230400|278400|
|"rewards.base_height_error_sq"|-1440.0|-1728.0|
|"rewards.base_height_penalty_clip"|46.656|55.987199999999994|
|"rewards.termination"|-17.279999999999998|-20.735999999999997|

算法、环境数和训练恢复方式改值：该类参数未变化。

下一轮实际启动：round_003_optimization，环境 256，目标 iteration 5800；加载 `/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_02-14-57_mine_auto_r002_auto_training_20pct_20261004/model_4799.pt`；仅训练轮输出 False；重置优化器 True。

算法覆盖：`{}`。

下一轮验证：待下一轮完整验证。

```json
{
  "status": "待下一轮完整验证"
}
```

## auto_training_20pct_20261004 / round_003_optimization

尚无完整评估，未形成可核验的训练后优化流程。

本目录记录的实际训练启动/恢复次数：2。

每次真实启动及恢复的参数、权重、环境数和命令：

```json
[
  {
    "folder": "round_003_optimization",
    "checkpoint": "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_02-14-57_mine_auto_r002_auto_training_20pct_20261004/model_4799.pt",
    "num_envs": "256",
    "target_iteration": "5800",
    "wheel_head_only_finetune": false,
    "reset_optimizer": true,
    "reset_exploration_std": "0.3",
    "overrides": {
      "commands.standing_only_steps": 278400,
      "commands.motion_ramp_steps": 48000,
      "commands.reverse_ramp_start_steps": 278400,
      "commands.reverse_ramp_steps": 48000,
      "commands.yaw_start_steps": 278400,
      "commands.yaw_ramp_steps": 48000,
      "commands.ranges_ang_vel_yaw": [
        -0.3,
        0.3
      ],
      "commands.heading_command": false,
      "commands.grouped_training": true,
      "rewards.termination": -20.735999999999997,
      "rewards.base_height_penalty_clip": 55.987199999999994,
      "rewards.standing_leg_angle": -5.0,
      "min_root_height": 0.22,
      "reset_velocity_final": 0.1,
      "rewards.base_height_error_sq": -1728.0,
      "kp_theta": 10.0,
      "kd_theta": 0.5
    },
    "agent_overrides": {},
    "command": [
      "/home/aaa/studyRL/src/CLT-RL/run_python.sh",
      "/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/scripts/rsl_rl/train.py",
      "--task",
      "WheelLeggedVMC-Flat-v0",
      "--headless",
      "--num_envs",
      "256",
      "--seed",
      "43",
      "--run_name",
      "mine_auto_r003_auto_training_20pct_20261004",
      "--checkpoint_path",
      "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_02-14-57_mine_auto_r002_auto_training_20pct_20261004/model_4799.pt",
      "--target_iteration",
      "5800",
      "--reset_optimizer",
      "--reset_exploration_std",
      "0.3",
      "env.commands.standing_only_steps=278400",
      "env.commands.motion_ramp_steps=48000",
      "env.commands.reverse_ramp_start_steps=278400",
      "env.commands.reverse_ramp_steps=48000",
      "env.commands.yaw_start_steps=278400",
      "env.commands.yaw_ramp_steps=48000",
      "env.commands.ranges_ang_vel_yaw=[-0.3, 0.3]",
      "env.commands.heading_command=false",
      "env.commands.grouped_training=true",
      "env.rewards.termination=-20.735999999999997",
      "env.rewards.base_height_penalty_clip=55.987199999999994",
      "env.rewards.standing_leg_angle=-5.0",
      "env.min_root_height=0.22",
      "env.reset_velocity_final=0.1",
      "env.rewards.base_height_error_sq=-1728.0",
      "env.kp_theta=10.0",
      "env.kd_theta=0.5"
    ],
    "evidence": {
      "log": "mine_model/build/logs/auto_training_20pct_service.log",
      "line": 57
    }
  },
  {
    "folder": "round_003_optimization",
    "checkpoint": "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_03-02-58_mine_auto_r003_auto_training_20pct_20261004/model_5100.pt",
    "num_envs": "256",
    "target_iteration": "5800",
    "wheel_head_only_finetune": false,
    "reset_optimizer": true,
    "reset_exploration_std": "0.3",
    "overrides": {
      "commands.standing_only_steps": 278400,
      "commands.motion_ramp_steps": 48000,
      "commands.reverse_ramp_start_steps": 278400,
      "commands.reverse_ramp_steps": 48000,
      "commands.yaw_start_steps": 278400,
      "commands.yaw_ramp_steps": 48000,
      "commands.ranges_ang_vel_yaw": [
        -0.3,
        0.3
      ],
      "commands.heading_command": false,
      "commands.grouped_training": true,
      "rewards.termination": -20.735999999999997,
      "rewards.base_height_penalty_clip": 55.987199999999994,
      "rewards.standing_leg_angle": -5.0,
      "min_root_height": 0.22,
      "reset_velocity_final": 0.1,
      "rewards.base_height_error_sq": -1728.0,
      "kp_theta": 10.0,
      "kd_theta": 0.5
    },
    "agent_overrides": {},
    "command": [
      "/home/aaa/studyRL/src/CLT-RL/run_python.sh",
      "/home/aaa/studyRL/src/CLT-RL/wheel_legged_isaaclab/scripts/rsl_rl/train.py",
      "--task",
      "WheelLeggedVMC-Flat-v0",
      "--headless",
      "--num_envs",
      "256",
      "--seed",
      "43",
      "--run_name",
      "mine_auto_r003_auto_training_20pct_20261004",
      "--checkpoint_path",
      "/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_03-02-58_mine_auto_r003_auto_training_20pct_20261004/model_5100.pt",
      "--target_iteration",
      "5800",
      "--reset_optimizer",
      "--reset_exploration_std",
      "0.3",
      "env.commands.standing_only_steps=278400",
      "env.commands.motion_ramp_steps=48000",
      "env.commands.reverse_ramp_start_steps=278400",
      "env.commands.reverse_ramp_steps=48000",
      "env.commands.yaw_start_steps=278400",
      "env.commands.yaw_ramp_steps=48000",
      "env.commands.ranges_ang_vel_yaw=[-0.3, 0.3]",
      "env.commands.heading_command=false",
      "env.commands.grouped_training=true",
      "env.rewards.termination=-20.735999999999997",
      "env.rewards.base_height_penalty_clip=55.987199999999994",
      "env.rewards.standing_leg_angle=-5.0",
      "env.min_root_height=0.22",
      "env.reset_velocity_final=0.1",
      "env.rewards.base_height_error_sq=-1728.0",
      "env.kp_theta=10.0",
      "env.kd_theta=0.5"
    ],
    "evidence": {
      "log": "mine_model/build/logs/auto_training_20pct_service.log",
      "line": 74
    }
  }
]
```

