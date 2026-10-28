# Jetson NX 上位机工程

本目录只维护上位机代码、依赖、模型与验证工具。下位机工程位于同级的 [series_balance_rtos/](../series_balance_rtos/)，有独立的 [下位机 README](../series_balance_rtos/README.md)。

当前分工是：**STM32 发送观测；NX 推理并计算六路最终电机力矩；STM32 执行力矩请求**。

**执行链复验状态（2026-10-27）**：PC MuJoCo 对照已发现，原轮速阻尼 4.5 改为 200/500 Hz 的外部保持力矩后存在明显闭环退步。只替换腿部力矩执行时，静止对照维持 20 秒；替换轮子反馈执行时出现失稳。预先站稳后 1 kHz 在这一静止工况维持了 20 秒，冷启动和通信延迟仍需单独验证。当前发布包用于 NX 推理、单机耗时及影子通信测试，实际电机输出方案仍在验证中；参数实验没有修改已发布模型。

网络仍是原来的 27 维输入、6 维输出 Actor。腿动作表示四个主动关节角目标，轮动作表示左右轮速目标。NX 在网络之后执行一次关节位置 PD、轮速 PD 和配套重力前馈，再限制最终总力矩。这里没有加入 IK、虚拟腿位置反馈或额外气弹簧前馈。

当前选择实际更新 7476 次的冻结策略，轮速阻尼采用已独立回放验证的统一值 4.5。名义 735 次试验全部存活，严格 10% 通过 711 次；独立 189 次通过 183 次。模型仍有速度误差未验收；当前数值检查也不代表已经完成实机验证。后台鲁棒性结果写在 `../../mine_model/results/joint_pd_body_plus1p3kg_20261026/source7476_gain4p5_preacceptance_robustness_20261027/`。

## 1. 工程入口

| 路径 | 用途 |
| --- | --- |
| [run_nx.sh](run_nx.sh) | NX 启动入口，先设置 BLAS 单线程再导入 NumPy |
| [requirements-nx.txt](requirements-nx.txt) | NX 运行依赖；不需要 Isaac、PyTorch、CUDA |
| [config/nx.example.json](config/nx.example.json) | 运行配置；串口设备必须明确填写 |
| [bundles/selected_nx_policy/](bundles/selected_nx_policy/) | 冻结 FP32 Actor、配置、观测格式和数值校验样本 |
| [wheelleg_deploy/serial_interface.py](wheelleg_deploy/serial_interface.py) | 打开串口、接收观测、发送力矩、停止和关闭 |
| [wheelleg_deploy/nx_runtime.py](wheelleg_deploy/nx_runtime.py) | 默认 200 Hz 控制循环和 50 Hz Actor 调度 |
| [wheelleg_deploy/torque_control.py](wheelleg_deploy/torque_control.py) | 直接关节位置 PD、轮速 PD、重力前馈、最终力矩限幅 |
| [wheelleg_deploy/observation.py](wheelleg_deploy/observation.py) | 物理数据构造观测的参考函数，以及命令斜坡 |
| [wheelleg_deploy/wire.py](wheelleg_deploy/wire.py) | 正式二进制帧、CRC32、分包与粘包解析 |
| [PROTOCOL.md](PROTOCOL.md) | 上下位机字段、单位、顺序及接入说明 |
| [../series_balance_rtos/control/rl_host.c](../series_balance_rtos/control/rl_host.c)、[rl_host.h](../series_balance_rtos/control/rl_host.h) | 独立下位机工程中的交互模块 |
| [tests/](tests/) | 协议、虚拟串口、回执、超时及推理检查 |
| [tools/verify_selected_bundle.py](tools/verify_selected_bundle.py) | 工作站上与原始 735 次回放、训练辅助函数交叉核对 |
| [verification/](verification/) | 本次离线验证的实际结果 |

完整下位机工程位于 `../series_balance_rtos/`，`Core`、`Usr`、`MDK-ARM`、`project_h7.ioc` 均在该层。目录迁移记录和历史备份位于 `../.directory_migrations/`。

## 2. NX 安装与检查

使用 NX 上的原生 Python 3.8 或更高版本，不要把工作站的 x86 Python 环境复制到 ARM 设备。

```bash
cd /你的路径/sim2real/jetson_nx
python3 -m venv .venv-nx
source .venv-nx/bin/activate
python -m pip install -r requirements-nx.txt
./run_nx.sh check-nx --bundle bundles/selected_nx_policy
./run_nx.sh mcu-config --bundle bundles/selected_nx_policy
./run_nx.sh benchmark-nx --bundle bundles/selected_nx_policy --duration 30 --output nx_timing_001.json
python -m unittest discover -s tests -p 'test_nx*.py' -v
```

`check-nx` 校验权重和配置，完成一次人工观测下的推理与力矩计算，不打开串口。`mcu-config` 打印与当前 bundle 对应的 `RLHost_Config` C 初始化内容，包括完整哈希、力矩上限、动作上限和命令范围。

默认 Python 是 `python3`，可通过 `NX_PYTHON=/完整路径/python` 指定。启动脚本默认限制 NumPy BLAS 为单线程，减少小型单样本 MLP 的调度开销。

`benchmark-nx` 不打开串口，使用人工传感器输入测量 Actor 加 PD、纯 PD、唤醒延迟的均值、P95、P99、最大值和超周期次数。结果保存设备型号及架构；在 PC 上运行会明确标记为 PC 数据，不能当作 NX 的实测性能。它只检验计算和调度预算，不检验闭环运动或真实通信。

## 3. 串口运行

选择现有 **USART3：PD8 TX、PD9 RX；2,000,000 baud、8N1、无流控**。RX DMA 为 DMA2 Stream4，TX DMA 为 DMA2 Stream5。CubeMX 的 `project_h7.ioc` 和生成的 `Core/Src/usart.c` 已同步设置波特率。

NX 的设备名由载板和接线决定，程序不会扫描设备。下例设备名需要替换成实际使用的串口：

```bash
./run_nx.sh run-nx \
  --config config/nx.example.json \
  --port /dev/ttyTHS0 \
  --duration 30 \
  --log bench_001.jsonl
```

默认请求 **shadow 模式**：NX 发送候选力矩，MCU 接口不提供可执行电机请求，用于检查数据、计算耗时、单位、顺序和回执。实际进行输出验证时，`--execute` 才会请求执行；MCU 还须由本机控制逻辑允许执行并确认标定。配置文件不接受自动执行选项。

默认使用 STM32 观测中的速度、偏航角速度和高度命令。需要固定命令时，可明确传入：

```bash
./run_nx.sh run-nx \
  --bundle bundles/selected_nx_policy \
  --port /dev/ttyTHS0 \
  --command 0 0 0.30 \
  --duration 30
```

`--command` 顺序是 `vx[m/s] yaw_rate[rad/s] height[m]`。线速度每个 50 Hz 策略周期按训练加速度限制推进；接收、打印和日志不会额外推进命令。高度表示仿真机身参考点的离地高度，不是直接发送虚拟腿长。

日志拒绝覆盖既有文件。每个 Actor 周期记录输入、有效动作、目标、前馈、总力矩、源观测编号、回执状态和耗时。`--duration 0` 表示持续运行；Ctrl+C 会发送 STOP 后关闭串口。

## 4. 对外串口接口

```python
from wheelleg_deploy.serial_interface import open_robot_serial
from wheelleg_deploy.numpy_policy import NumpyPolicy
from wheelleg_deploy.torque_control import TorqueController

policy = NumpyPolicy("bundles/selected_nx_policy")
controller = TorqueController(policy)
link = open_robot_serial(
    port="/dev/ttyTHS0",
    baudrate=2_000_000,
    contract_sha256=policy.metadata["contract_sha256"],
    execute=False,
)
try:
    frame = link.receive_observation()
    if frame is not None:
        # frame.observation: 27 个已裁剪、缩放的 Actor 输入。
        # frame.actuator_state: 未裁剪 q4、dq4、轮速2。
        result = controller.step(
            frame.observation, frame.actuator_state, infer=True,
            previous_action=frame.observation[21:27],
        )
        link.send_torques(
            result["torque_nm"],
            state_sequence=frame.sequence,
            effective_action=result["effective_action"],
            command=result["command"],
        )
finally:
    link.close()
```

这是接口调用示意。直接使用这些函数时，调用者负责控制频率、新鲜度、限幅和回执；`run-nx` 已将这些环节组合成完整主循环。

`receive_observation()` 返回本次读取到的最新有效样本或 `None`。重复序号、倒序、倒退设备时钟及错误会话不会刷新样本年龄。`send_torques()` 必须关联最近接受的观测编号。`send_stop()` 是独立停止请求；六个零 action 不代表零力矩。

## 5. 控制计算与周期

动作顺序是 `[LB, LL, LW, RB, RL, RW]`；四个腿关节数组顺序是 `[LB, LL, RB, RL]`，左右轮数组顺序是 `[L, R]`。

```text
STM32: IMU + 编码器 + 机身速度估计
    → 27 维已预处理 obs + 10 维未裁剪执行器状态
NX: 50 Hz Actor
    → 四个真实关节角目标 + 两个轮速目标
NX: 默认 200 Hz
    → 腿 τ = Kp(q_ref-q) - Kd*dq + τ_gravity
    → 轮 τ = D(ω_ref-ω)
    → 对合成后的每路力矩限幅
STM32:
    → 校验来源、序号、年龄、范围
    → 换算为电机协议输出
    → 写入实际输出路径后回执
```

| 参数 | 当前值 |
| --- | --- |
| Actor | 27 → 256 → 128 → 64 → 6，ELU，FP32 |
| Actor 周期 | 20 ms，50 Hz |
| NX 力矩计算周期 | 默认 5 ms，200 Hz |
| 训练物理子步 | 5 ms，PhysX 隐式位置 PD |
| 关节增益 | Kp = 300，Kd = 3 |
| 轮速阻尼 | 4.5；检查点原值 3，统一修改有独立回放记录 |
| 腿力矩上限 | 每路 35.574 N·m |
| 轮力矩上限 | 每路 4.80944 N·m |
| 轮速目标上限 | ±18.75 rad/s |
| 速度、yaw rate 范围 | ±0.8 m/s、±0.3 rad/s |
| 高度命令范围 | 0.22534645531～0.37534645531 m |
| 名义整车质量 | 13.64 kg |
| 名义单腿重力前馈基值 | 66.9042 N，再按原训练角度公式映射 |
| 反馈及回执超时 | 默认 30 ms，异常锁存 |

观测经过裁剪，不能直接作为全部 PD 原始反馈，所以 STATUS 额外发送未裁剪的 `q4/dq4/ω2`。NX 核对它们与 obs 对应字段的顺序、单位及缩放。

重力前馈保持训练中的五连杆正解、Jacobian、pitch 补偿及独立 effort 限幅。PD 加前馈后，再限制最终物理力矩。下位机收到的已经是最终 N·m，不应再次叠加 PD、VMC 或重力前馈，也不应再次乘旧 LQR 的 0.3 经验增益。

**NX 显式 PD 与训练中的 PhysX 隐式 PD 存在动力学差异**，串口和 Linux 调度也会增加延迟。算式相符只能证明实现与训练辅助函数一致；后续仍须验证相同延迟及显式执行链下的闭环表现，并实测 NX 耗时。当前没有打开实物串口或使能电机。

## 6. 下位机接入

下位机交互代码集中在 `../series_balance_rtos/control/rl_host.c` 与 `rl_host.h`，已加入下位机的 `MDK-ARM/project_h7.uvprojx`。`usart.c` 用户代码区域只向模块转交 USART3 的接收事件和错误事件。模块处理 DMA 缓冲与 D-Cache，接收中断只排队字节，协议及浮点检查在任务中执行。

当前尚未替换 `joint_task` 的旧输出链。您对齐通信和执行时按以下顺序接入：

1. 用 `mcu-config` 产生配置，初始化 `RLHost_Context`；初始执行许可关闭。
2. 调用 `RLHost_Uart3Start()`，在任务中周期调用 `RLHost_Uart3Poll()`。
3. 从已经对齐坐标、零点和单位的传感器构造 obs 与 raw，按 200 Hz 调用 `RLHost_Uart3SendStatus()`。
4. 在单一电机输出路径读取 `RLHost_GetTorques()` 的六路 N·m 请求，完成实际输出请求写入后调用 `RLHost_MarkApplied()`。

`RLHost_GetTorques()` 失败或 `stop_requested` 置位时，由现有电机任务处理停止。模块不调用 CAN、不切换遥控器模式、不修改 `lqr_flag`，不会自动使能电机。输出互斥、轴向换算及传感器标定由实际接入完成，详见 [PROTOCOL.md](PROTOCOL.md)。

## 7. 工作站导出、核对与打包

导出需要 PyTorch、PyYAML、USD Python；这些不需要装到 NX。检查点旁须保留自己的 `params/env.yaml` 和 `agent.yaml`。

```bash
python -m wheelleg_deploy prepare-nx \
  --checkpoint /完整路径/model_7475.pt \
  --verified-replay /对应冻结回放/canonical27 \
  --output bundles/新的导出目录

python tools/verify_selected_bundle.py \
  --bundle bundles/selected_nx_policy \
  --replay-root /完整735次回放目录 \
  --training-root /完整路径/CLT-RL \
  --output verification/recorded735_actor_and_controller.json

python tools/build_nx_package.py \
  --bundle bundles/selected_nx_policy \
  --output dist/clt_nx_source7476_uart3_20261027.zip
```

导出拒绝覆盖 bundle，也不重写检查点或 Adam。核对分别检查真实回放输入的 Actor 输出、关节目标、重力力矩、协议、回执与超时，并注明尚未验证的环节。

打包只包含 NX 代码、选择的模型、文档、配置、验证结果和两个下位机接口参考文件，不包含完整 STM32 工程、历史模型、Isaac 或虚拟环境。

`prepare/check/dry-run` 及 JSON 离线接口保留用于旧接口核对。当前 NX 串口运行使用 `prepare-nx/check-nx/run-nx`、`wire.py` 和 `clt_nx_torque_numpy_v1`；旧 bundle 不能代替当前包。

## 8. 后续验证

先完成冻结策略在质量、摩擦、增益、噪声及延迟下的测试，再按失败原因决定是否建立域随机化续训分支。续训保留当前策略和优化器，同时复验名义环境。然后用独立种子与参数组合做全工况及持续回放，检查 MuJoCo、NX 推理和真实串口时序，逐步缩小部署差异。
