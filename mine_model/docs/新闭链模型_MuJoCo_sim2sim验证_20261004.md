# 新闭链模型 MuJoCo sim2sim 验证（2026-10-04）

已完成最新落盘策略 model_6700.pt 的新闭链模型迁移和数值验证。MuJoCo 27/27 个独立工况存活、没有自动重置，能执行站立、正反向运动、左右转向和高度切换；现有精度门限只有 5/27 组全部通过，22 组仅偏航角速度未达标。不能把本轮结论写成“所有数据误差低于 20%”。同一权重在本轮 Isaac Lab 复测中为 27/27 存活、25/27 精度通过。

正式训练仍冻结在内存进度 6787；6700 是最后落盘的检查点。本轮没有恢复训练或改策略权重，也未启动桌面可视化或关机。

## 模型与策略

- 策略：`/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_16-18-34_mine_auto_r037_auto_training_implicit_20pct_20261004_protected_stops_v2_env10240/model_6700.pt`。
- SHA-256：`317cf6dba8358df6bcba2e4c682cacd3a42a6139824c0d5cfac7440266dec307`。
- 当前 MuJoCo 模型：`/home/aaa/studyRL/src/CLT-RL/mine_model/exports/mujoco_protected_stops_v2_20261004/scene_protected_contact.xml`。
- 原始训练资产：`mine_closed_chain_training_protected_stops_v2.usd`，从其完整 USD 层和共享网格读取。
- 15 个机器人刚体，总质量 10.496905 kg；14 个树内铰链和 4 个外部铰链，共保留 18 个真实 revolute。MuJoCo 用沿铰链轴的两个 connect 约束表达每个外部铰链，共 8 个 connect；允许绕轴转动，同时约束孔位和平行轴方向。
- 质量、质心、惯量主值及主轴均直接读取 USD。没有从可视网格重新估计惯量，也没有拿旧机器人 URDF 替代新结构。
- 基底使用已确认的单盒碰撞；普通连杆使用当前训练版本的凸包；限位块和小腿保留原始 CAD 输入，在 MuJoCo 中独立进行 CoACD 凸分解。21 个源碰撞体在分解后会对应更多 MuJoCo 凸片，不能把两者数量混为一谈。

![当前闭链模型的 MuJoCo 站立回放](../results/sim2sim_mujoco_6700_20261004/final_cli_smoke/snapshot.png)

## 接口如何迁移

电机顺序按名称绑定为 `LB_joint, LL_joint, LW_joint, RB_joint, RL_joint, RW_joint`，没有沿用 Isaac Lab 的关节索引。角度相对于 baselink 的 CAD 零位，坐标为 X 前、Y 左、Z 上；右轮仍使用镜像符号。五连杆正/逆运动学直接复用 `five_bar_vmc.py`，包含 upper 94.5 mm、lower 112.5 mm、比例 2.222222 和导出装配偏置。旧配置遗留的 l1/l2 字段只用于旧串联腿分支，新模型走五连杆分支。

保留 27 维观测及其缩放/裁剪、6 维动作、上一次实际施加的长度参考反馈、速度指令 0.8 m/s² 斜坡和质心速度参考点。腿部采用五连杆逆解生成原始电机位置目标，力矩型隐式驱动的刚度/阻尼为 300/3；轮子为阻尼 2 的速度驱动，所有主动关节上限为 10 N·m。新模型没有启用旧显式 VMC 力矩执行分支，没有添加额外航向反馈。

最初直接使用 MuJoCo 5 ms 积分时，0.74 s 出现失稳和厘米级闭链孔位误差。MuJoCo 在仿射驱动力限幅后，其力对速度的导数会被截断；小惯量轮子在该步长下会发生明显数值问题。采用 0.1 ms 内部积分后稳定，另用 0.05 ms 完成站立复核。参考仍每 5 ms 更新、策略仍每 20 ms 推理；没有通过降低/提高驱动增益消除问题。更细步长的站立复核仍有约 0.055 rad/s 偏航 MAE，因此不能把剩余偏航误差全部归因于原来的大步长问题，也不能仅凭现有证据认定具体某个 reward 是原因。

原来限位软接触还出现穿透。最终只对受保护限位块/小腿接触使用 solref=(0.002,1)、solimp=(0.9999,0.9999,0.001)、margin=1 mm；普通接触保持原设置。修正前后两轮完整基础运动的全部记录值逐项差异为 0，因为这些工况未触发限位接触。

## 验证条件及结果

9 个条件：站立、前进 0.3/0.6 m/s、后退 -0.3/-0.6 m/s、左/右旋转 ±0.3 rad/s、高度 0.28/0.32 m。每个条件独立运行 12 s，种子 43/44/45；采用与 Isaac 评估脚本相同的 CPU Torch 初始速度抽样，各轴幅度 0.05。最后 3 s 统计稳态 MAE，保留全部瞬态曲线。MuJoCo 不自动重置，姿态/高度按即时阈值终止，严格程度高于 Isaac 的延时终止。

8 项新旧接口检查全部通过：原始质量和惯量、关节局部锚点和轴、碰撞过滤、实际闭链几何与五连杆正运动学、观测/施加动作/驱动目标与真实 Isaac 源码一致性、质心速度定义和 actor 权重计算。原始日志为 `interface_tests.log`。

下表为三种子、所有工况中最差的稳态指标，不是所有样本的瞬时峰值。腿长、腿摆角和轮速误差均相对于实际驱动参考计算。

| 指标 | Isaac Lab | MuJoCo | 单位 |
|---|---:|---:|---|
| 速度 MAE | 0.02452 | 0.01829 | m/s |
| 偏航角速度 MAE | 0.05221 | 0.06908 | rad/s |
| 高度 MAE | 3.52289 | 3.28288 | mm |
| 腿长 MAE | 6.57757 | 6.55550 | mm |
| 腿摆角 MAE | 0.00399 | 0.00388 | rad |
| 轮速 MAE | 0.53075 | 0.16257 | rad/s |
| 机体倾角 P95 | 2.12621 | 1.99804 | deg |

| MuJoCo 工况 | 最差速度 MAE (m/s) | 最差偏航 MAE (rad/s) | 最差高度 MAE (mm) |
|---|---:|---:|---:|
| stand | 0.01070 | 0.05697 | 2.364 |
| forward_0p3 | 0.01148 | 0.05502 | 2.782 |
| forward_0p6 | 0.01318 | 0.05038 | 3.283 |
| reverse_0p3 | 0.01198 | 0.05800 | 2.224 |
| reverse_0p6 | 0.01829 | 0.05380 | 2.433 |
| spin_left | 0.01038 | 0.05610 | 2.346 |
| spin_right | 0.01081 | 0.06908 | 2.344 |
| height_low | 0.01041 | 0.06219 | 1.454 |
| height_high | 0.01249 | 0.05572 | 3.153 |

精度标准沿用既有验收：速度/偏航 MAE 为 max(0.05 绝对误差, 20% 指令)，高度/腿长为参考的 20%，腿摆角为 0.04 rad，轮速为 max(0.3 rad/s, 20% 参考)，姿态 P95 为 5°。零指令使用绝对门限，不能对零值计算相对百分比。MuJoCo 唯一超标项为偏航；右转约 0.0691 rad/s，约为 0.3 rad/s 指令的 23%。其余不少零偏航工况约 0.055–0.062 rad/s，超过 0.05 rad/s 门限。

![各工况最差种子误差与验收门限之比](../results/sim2sim_mujoco_6700_20261004/transfer_error_comparison.png)

![同权重九工况的 Isaac / MuJoCo 误差曲线](../results/sim2sim_mujoco_6700_20261004/paired_error_curves.png)

完整七项数据、全部三种子的曲线位于 `mujoco_matrix_protected_contact/all_conditions_error_curves.pdf`；逐样本数据位于该目录的 `trace.csv`；逐工况、逐种子门限和通过情况位于 `assessment.json`。`simulator_metrics.csv` 和 `comparison.json` 汇总两端对照数据。

## 闭链和机械限位

基础运动 50 Hz 记录中，最大闭链锚点距离为 0.317 mm，五连杆有效性始终成立，限位内部接触数为 0。普通地面接触的最大记录穿透约 2.863 mm，不能把“限位块没有穿透”扩展为所有接触都没有穿透。

额外固定基底、零重力测试匹配原 Isaac 流程：装配角 0.052219456 rad、2 s 初始保持、5 s 长度斜坡、5 s 保持，目标为 0.12/0.18/0.36/0.42 m。0.18 和 0.36 m 可以跟踪；过短/过长目标由真实限位接触挡住，MuJoCo 实际约 0.1688/0.3681 m。该轮最大闭链孔位距离为 0.119 mm。

原始 CAD 投影检查覆盖左侧 2400、右侧 2400 个真实 MuJoCo 采样姿态，交叠面积均为 0，证明这些样本没有 CAD 实体相交。该结论不覆盖每个内部物理子步或连续时间。MuJoCo 的实际机械极限与此前 Isaac 的约 0.1784–0.1789/0.3542–0.3546 m 仍有约 10/14 mm 差异；两端独立凸分解和约束/接触求解不能视为完全相同。未通过改 CAD 尺寸或加人工关节限位掩盖差异。当前训练基础运动未触发这些机械极限。

![机械限位与腿长斜坡](../results/sim2sim_mujoco_6700_20261004/mechanical_stop_tracking.png)

## 本轮文件和复现

最终模型是 `mine_model/exports/mujoco_protected_stops_v2_20261004/scene_protected_contact.xml`，对应 JSON 保存源 USD 层、OBJ 网格和模型校验和。`scene.xml` 是初始接触参数诊断版本，验证和后续回放应选择 protected_contact 版本。`meshes/` 同时保留视觉 CAD 和物理凸片，原 URDF/USD 均未覆盖。

运行新模型桌面回放时使用：

```bash
cd ~/studyRL/src/CLT-RL
checkpoint='/home/aaa/studyRL/src/CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/2026-10-04_16-18-34_mine_auto_r037_auto_training_implicit_20pct_20261004_protected_stops_v2_env10240/model_6700.pt'
model='mine_model/exports/mujoco_protected_stops_v2_20261004/scene_protected_contact.xml'
./sim2sim_mujoco.sh --checkpoint "$checkpoint" --mjcf "$model" --demo
```

上述命令默认展示九阶段基础运动，每阶段 8 s，自动读取检查点 `params/env.yaml` 和 `agent.yaml`；默认策略周期 20 ms、参考更新 5 ms、MuJoCo 内部积分 0.1 ms。当前没有替用户重新启动可视化。

完整矩阵数值复现：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/env_mujoco/bin/python \
  wheel_legged_isaaclab/sim2sim/evaluate_mine.py \
  --checkpoint "$checkpoint" --mjcf "$model" \
  --seconds 12 --output mine_model/results/sim2sim_recheck
```

新增/修改主要文件：`sim2sim/core.py`、`run.py`、`mine_model.py`、`evaluate_mine.py`、`compare_mine.py`、`test_mine_interface.py`、`test_mine_stops.py`、`audit_stops.py`。迁移代码与验证报告独立于正式训练流程。后续优化应优先定位同权重两端的偏航振荡和执行器/接触响应差异，再做训练或控制层调整；本轮没有用额外航向控制替策略补偿，也没有启动续训。

## 连续运动切换复核

同一最终模型和权重另完成 72 s 连续九阶段回放，3600 次策略步，无姿态/数值失败。每个阶段最后 3 s 统计中，左右虚拟腿摆角差最大峰值为 3.047°，机体 pitch/roll 最大峰值分别为 2.019°/0.116°。左右摆角差描述实际腿姿态的差异，与各自摆角参考的跟踪误差是不同指标，不能相互代替。

![连续基础运动：速度、左右腿摆角、姿态及高度](../results/sim2sim_mujoco_6700_20261004/continuous_basic_motion/analysis/analysis.png)

逐阶段 MAE/峰值和左右摆角差见 `continuous_basic_motion/analysis/phase_metrics.csv`、`analysis.json`；完整 72 s 轨迹见 `continuous_basic_motion/trace.csv`。
