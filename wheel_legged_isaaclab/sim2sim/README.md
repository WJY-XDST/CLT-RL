# Isaac Lab -> MuJoCo sim2sim

## 新闭链模型（2026-10-04）

新模型已接入当前部署入口，使用完整训练 USD 的刚体、惯量、树关节及四个外部铰链；
从当前检查点 `params/env.yaml` 自动识别五连杆与隐式位置/速度驱动。
最终转换资产为 `mine_model/exports/mujoco_protected_stops_v2_20261004/scene_protected_contact.xml`。
`scene.xml` 是初始软接触诊断版本。两者均保留在原文件夹，未覆盖原 URDF/USD。

本轮 `model_6700.pt` 的 27 组独立平地回放全部存活，但只有 5/27 组通过完整精度门限，
剩余失败项均为偏航。机械限位的 CAD 采样检查通过，MuJoCo 与 PhysX 的实际限位长度仍有差异。
详见[本轮验证说明](../../mine_model/docs/新闭链模型_MuJoCo_sim2sim验证_20261004.md)。

```bash
# 从工程根目录运行；checkpoint 填入所需新模型检查点路径。
./sim2sim_mujoco.sh --checkpoint "$checkpoint" \
  --mjcf mine_model/exports/mujoco_protected_stops_v2_20261004/scene_protected_contact.xml --demo
```

新模型默认九阶段演示：站立、正反向 0.3/0.6 m/s、左右旋转 0.3 rad/s、高度 0.28/0.32 m。
每阶段 8 秒，不添加航向保持。50 Hz 策略、200 Hz 参考控制保持不变；
MuJoCo 内部积分默认 0.1 ms，以处理薄闭链连杆和限幅驱动的小轮惯量。
腿部 300/3、轮速阻尼 2 和 10 N·m 上限均来自训练配置。
受保护 CAD 采用独立 CoACD 分解；接触与 PhysX 的求解边界在报告中单独说明。

独立矩阵验证用 `evaluate_mine.py`；限位测试与原始 CAD 核对用 `test_mine_stops.py`、`audit_stops.py`。
新旧接口全部检查：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/env_mujoco/bin/python -m unittest discover \
  -s wheel_legged_isaaclab/sim2sim -p 'test*interface.py' -v
```

以下为旧开源串联腿的既有部署和历史验证说明。

直接加载已训练的 RSL-RL actor，在 MuJoCo 中执行同一套 VMC。
默认检查点为 `checkpoints/optimized/yaw_heading_followup_20261001_200205_round0001_improved/model.pt`
（model_6739，SHA-256 `729711fd6c013c56678f159bbfbb344163d487ad3c3a195ab9ebd010f7c09a48`）。

## 环境和可视化

已创建独立环境 `env_mujoco`：Python 3.11.15、MuJoCo 3.14.0、PyTorch 2.14.1+cpu。
激活/退出时自动隔离/恢复 ROS2 的 `PYTHONPATH` 和 `LD_LIBRARY_PATH`。
启动脚本直接调用该环境，无需先激活 Conda，也无需启动 Isaac Sim。

在 CLT-RL 根目录运行：

```bash
cd ~/studyRL/src/CLT-RL
./sim2sim_mujoco.sh --demo
```

45 秒演示包含站立、前进、后退和 0.16/0.18/0.20 m 高度切换。
棋盘格地面提供运动参照，相机跟随机体；关闭窗口结束回放。
需要多轮演示时：`./sim2sim_mujoco.sh --demo --max-steps 9000`。
`MUJOCO_PYTHON` 可以覆盖启动脚本的 Python 路径。

固定命令和指定检查点：

```bash
./sim2sim_mujoco.sh --fixed-command 0.4 0 0.18 --max-steps 1000
./sim2sim_mujoco.sh --checkpoint /absolute/path/model.pt \
  --env-config /absolute/path/env.yaml --agent-config /absolute/path/agent.yaml
```

务必使用检查点对应的 `env.yaml` 和 `agent.yaml`。默认读取权重同目录的配置，缺失时直接报错。
带观测归一化的权重需要先从 Isaac Lab 导出包含 normalizer 的 TorchScript actor，
再以 `--jit --checkpoint /path/policy.pt --env-config ...` 加载。
当前默认检查点没有 normalizer 状态。

## 数值验证

三种高度分别验证站立和正反向 0.8 m/s，每阶段 5 秒，共 45 秒：

```bash
./sim2sim_mujoco.sh --headless --seed 43 --reset-velocity 0.5 \
  --max-steps 2250 --velocity-cycle 0 0.8 -0.8 0 0.8 -0.8 0 0.8 -0.8 \
  --height-cycle 0.16 0.16 0.16 0.18 0.18 0.18 0.20 0.20 0.20
```

转向验证（站立转向与 0.4 m/s 行进转向）：

```bash
./sim2sim_mujoco.sh --headless --max-steps 1500 \
  --velocity-cycle 0 0 0 0.4 0.4 0 --yaw-cycle 0 0.5 -0.5 0.5 -0.5 0
```

每次运行在 `outputs/时间戳/` 保存 `scene.xml`、`trace.csv`、`summary.json`、
`metadata.json`、源配置副本和 `tracking.png`。CSV 包括速度、高度误差（mm）、
姿态角（deg）、虚拟腿状态、原始策略动作和施加力矩。
每阶段统计最后两秒；`phase_complete=false` 表示该阶段未跑满，不能视为完整稳态验证。
`MUJOCO_GL=egl ./sim2sim_mujoco.sh --headless --snapshot` 保存离屏图像。
本机已验证 GLFW 桌面窗口及 EGL 离屏渲染。

对已有回放检测左右虚拟腿摆角、左右摆角差、机体俯仰/横滚、速度和高度：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/env_mujoco/bin/python wheel_legged_isaaclab/sim2sim/analyze_trace.py \
  wheel_legged_isaaclab/sim2sim/outputs/matrix_seed43/trace.csv
```

输出 `analysis/analysis.json`、`phase_metrics.csv` 和 `analysis.png`。默认使用每个阶段最后 2 秒，
报告 `speed_mae_m_s`、`height_mae_mm`、左右腿摆角均值/峰值、摆角差 MAE/峰值以及 pitch/roll 峰值。

姿态超过配置中的 pitch/roll 阈值、机体低于 0.07 m 或出现数值警告时，停止并以退出码 2 标记失败。
没有自动重置。姿态检测为立即停止，严格程度高于 Isaac Lab 原有的延时终止条件。
默认初始根速度为零；`--reset-velocity 0.5` 对质心线速度和世界角速度施加各轴均匀扰动。
种子仅保证 MuJoCo 运行之间可复现，不保证与 Isaac Lab 初始化随机数序列相同。

## 接口和已验证结果

读取顺序：`run.py` -> `core.py` -> `../wheel_legged_gym_isaaclab/vmc.py` -> `test_interface.py`。

保持 27 维观测顺序、观测缩放/裁剪、左右镜像符号、上一步施加的长度动作反馈、
50 Hz 策略与 200 Hz VMC、0.8 m/s² 速度指令斜坡和腿/轮 30/5 N*m 限幅。
复用训练源代码中的 VMC 运动学及 Jacobian；支持历史 angular mapping 配置。
速度使用根刚体质心速度，与 Isaac Lab `root_lin_vel_w` 定义相同。
URDF 原始质量/惯量、关节限位、网格和轮子圆柱碰撞体经 MuJoCo 官方导入器转换，
总质量为 12.28 kg，并关闭自碰撞。接触参数使用 MuJoCo 摩擦系数 0.5。
PhysX 与 MuJoCo 接触算法/材料组合规则不同，当前结果不代表动力学完全一致。

2026-10-02 实测：seed 43/44/45 的三轮 45 秒矩阵回放均完成；
各条件最后两秒的高度 MAE 约 0.20--3.06 mm，速度 MAE 不超过 0.0121 m/s。
另外一轮 30 秒转向回放完成，非零转向阶段 yaw-rate MAE 为 0.0052--0.0274 rad/s；
行进右转仍有约 0.0324 m/s 的速度 MAE。初始及切换瞬态不计入这些稳态指标。
原始证据位于 `outputs/matrix_seed43`、`matrix_seed44`、`matrix_seed45` 和 `yaw_validation`。
这属于平地迁移验证，尚未覆盖坡面、外力和实机。

接口核对测试直接提取 Isaac Lab 原文件方法，对相同状态比较观测、VMC 力矩及动作反馈，
覆盖 80 组随机状态和新旧 Jacobian，另检查质量、碰撞掩码、质心速度参考点及 actor 权重计算。

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/env_mujoco/bin/python -m unittest discover \
  -s wheel_legged_isaaclab/sim2sim -p test_interface.py -v
```

当前 4 项测试通过。重建环境时先安装 CPU PyTorch，再安装本目录 `requirements.txt`；
Gymnasium 已安装在当前环境，但该部署入口本身不依赖 Gymnasium。
