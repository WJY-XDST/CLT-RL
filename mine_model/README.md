# 自定义映射五连杆机器人

这是用户通过开链转闭链导入、并已在实车上用 LQR 和 VMC 控制过的车辆。
它不是早期运行的开源示例车。当前工程已移除示例车资产、串联计算和备用任务入口。

## 当前模型数据

训练数据入口为 ../wheel_legged_isaaclab/assets/robots/mine/manifest.json。
该文件指向新车的 model.json 与 USD。运行时将 CAD 主环映射为轮端大五连杆，
等效上杆约 210 mm、下杆约 250 mm；保留 CAD 装配零位、分支与左右轴向。
质量、惯量、碰撞和闭链关节来自新车 USD，不因几何表达改为大五连杆而缩放车体。

- [旧车清理与大五连杆说明](docs/新车数据清理与大五连杆配置说明.md)
- [当前关节角 PD 奖励基线和验证](docs/关节角PD奖励基线诊断与验证_20261011.md)
- [框架整理与验证](results/framework_cleanup_20261012/validation.json)
- [IK 控制与训练记录清理说明](docs/IK清理说明_20261011.md)
- [当前 VMC 支撑问题定位和求解器修复](docs/下位机硬件参数对齐与VMC站立诊断_20261010.md#追加tgs-外部力矩内部迭代修复)

## 常用命令

从 ~/studyRL/src/CLT-RL 执行：

```bash
./mine_model/run.sh play
./mine_model/run.sh inspect
./mine_model/run.sh bench --headless --seconds 12
./mine_model/run.sh test
```

play 优先选择最近优化任务已评估并保留的最佳直接关节角检查点，
从配套配置生成回放参数，不启动训练，也不自动选择未评估的新候选。
只打印选择结果、不打开窗口：`./mine_model/run.sh play --dry_run`。
单独 evaluate 必须显式指定新车检查点，并提供匹配的执行器配置：

```bash
./mine_model/run.sh evaluate --headless \
  --checkpoint /完整路径/model_XXXXX.pt \
  --overrides_json /完整路径/overrides.json
```

MuJoCo 回放同样要求明确的新车权重和相同资产的审计文件：

```bash
./sim2sim_mujoco.sh --checkpoint /完整路径/model_XXXXX.pt \
  --mjcf mine_model/exports/mujoco_protected_stops_v2_20261004/scene_protected_contact.xml
```

## 控制模式

截至 2026-10-11，新平地训练默认使用 `implicit_joint_position`：策略直接输出
LB、LL、RB、RL 四个主动髋关节目标角，PhysX 隐式位置 PD 产生反馈力矩，
保留下位机对应的 `Jᵀ[F_gravity, 0]` 重力前馈。轮子目标轮速控制不变。
不经过虚拟腿 PD 或 IK；新动作和观测语义不能加载旧虚拟腿权重。
训练状态以当前优化任务的 `status.json` 为准，不以本说明中的历史进程为准。
新模型目录为 `../IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat/`。
修改、硬件限幅来源和本轮有限验证见
[主动关节目标角和隐式 PD 改造说明](docs/主动关节目标角和隐式PD改造说明_20261011.md)。

以下是此前显式 VMC 的历史验证，不能当作新直接关节模式的训练结果。
2026-10-10 的默认模式使用显式 VMC：策略输出 L 和 theta，环境计算
虚拟腿 PD、下位机对应的重力前馈和五连杆 `Jᵀ[F,T]`，下发腿关节力矩。
腿关节内置 PD 为 0/0；轮子仍为隐式轮速驱动。腿/轮限幅分别为 35.574/4.80944 N·m。
新 VMC 运行启用 `enable_external_forces_every_iteration`，修复 TGS 下的支撑失败。
导轨支撑已通过 10 秒测试。第 200 次新策略已通过独立种子 30 秒无导轨回放，
27/27 组无重置；0.30 m 零速站立稳态姿态约 0.35°、前后漂移约 0.0001 m/s。
高度及轮速误差仍未全部达标；运动指令尚未正确跟踪，不能将存活等同于精度验收。

历史 IK 参考策略的训练、评估和导出记录已按保存配置清理，不再支持回放或续训。
`explicit_vmc` 的代码与记录仍保留；测试 VMC 权重请显式传入匹配的
checkpoint 与 overrides，不混用关节角策略。旧 IK 跳跃任务也已移除，当前空格只记录请求，不执行跳跃。

正在运行的进程保留内存中的配置；源码修改不会自动改变其控制器。
新的进程使用大五连杆表达，其几何与原映射数学等价。保留的 VMC 和关节角 PD 权重及参数按版本管理。

## 训练代码在哪里

`mine_model` 不再定义一份环境配置或车辆适配子类。唯一入口为
`../wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env_cfg.py`。
随后读同目录 `wheel_legged_vmc_flat_env.py`，再读 `agents/rsl_rl_ppo_cfg.py`
和 `../wheel_legged_isaaclab/scripts/rsl_rl/train.py`。

这里的脚本只管车辆导入、实验组织、误差检查、回放和记录，不是另一套 RL 环境。
旧 `mine_env_cfg.py`、重复源码快照、过时交付副本、已完成的 IK 清理器和
一次性碰撞变体构建脚本已清理；有效新车模型和机械验证证据保留。

## 目录职责

inputs 和 source 保存用户原始导入文件；config 保存闭合铰链与机械核对数据。
exports 保存新车各版本资产；scripts 提供构建、检查、训练与评估工具。
results 保存训练和验收记录；docs 保存学习说明及优化报告。
归档恢复包位于 archive，只有压缩包，不再保留可被编辑器索引的重复 Python 工程。
训练结果统一在 ../IsaacLab/logs/rsl_rl/。历史虚拟腿模型位于
mine_wheel_legged_vmc_flat，新直接关节 PD 模型位于 mine_wheel_legged_joint_pd_flat。
