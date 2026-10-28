# NX—STM32 串口接口 v1

上位机实现对应 `wheelleg_deploy/wire.py`，下位机实现在同级工程的 `../series_balance_rtos/control/rl_host.c/.h`。NX 接收观测，输出六路**最终输出轴力矩 N·m**。下位机负责硬件换算和输出，不再从包中重新执行 PD 或重力前馈。

## 1. 串口与带宽

- STM32：USART3，PD8 = TX，PD9 = RX。
- 既有 DMA：DMA2 Stream4 接收，DMA2 Stream5 发送，Normal 模式。
- 2,000,000 baud，8 数据位，无校验，1 停止位，无 RTS/CTS。
- 建议先使用 200 Hz 状态与力矩循环、50 Hz Actor；波特率已经同步写入 CubeMX 配置及生成代码。

按当前配置，HSE 8 MHz → SYSCLK 480 MHz → AHB 240 MHz → APB1 120 MHz，USART3 取 D2PCLK1。在 16 倍过采样下 BRR 为 60，名义波特率计算值为 2 MHz；实际误差还受时钟和物理链路影响。公式依据见 [ST RM0433 USART 波特率章节](https://www.st.com/resource/en/reference_manual/rm0433-stm32h742-stm32h743-753-and-stm32h750-value-line-advanced-armbased-32bit-mcus-stmicroelectronics.pdf)。

8N1 每字节占 10 bit。STATUS 为 212 字节、TORQUE 为 88 字节，2M 下线上时间分别约 1.06 ms、0.44 ms。200 Hz 时两方向分别约占各自链路容量的 21.2% 和 8.8%；两帧顺序完成的线上时间约 1.5 ms。它不包含传感器采样、DMA、Linux 调度和推理耗时。

USART3_TX 接 NX 的实际 RX，USART3_RX 接 NX 的实际 TX。NX 端的引脚、设备名和逻辑电平按所用载板确认；程序不指定载板，也不扫描串口。当前尚未测量真实接线上的延迟或误码率。

## 2. 公共帧头

所有整数均为无符号小端，所有浮点为小端 IEEE754 float32。禁止直接发送可能含填充的 C 结构体；C 代码使用显式字段读写。

| 字节偏移 | 长度 | 字段 | 内容 |
| --- | --- | --- | --- |
| 0 | 4 | magic | ASCII `WLRL` |
| 4 | 1 | version | 1 |
| 5 | 1 | type | HELLO=1，STATUS=2，TORQUE=3，STOP=4 |
| 6 | 2 | payload_size | 仅 payload 长度 |
| 8 | 4 | session | 本次连接非零随机会话，由 NX 生成 |
| 12 | 4 | sequence | 本发送端递增序号，uint32 回绕 |
| 16 | 4 | time_us | 本发送端单调微秒时钟低 32 位 |
| 20 | payload_size | payload | 对应类型的内容 |
| 20+payload_size | 4 | crc32 | 覆盖整个帧头及 payload，不覆盖 CRC 自身 |

CRC 为 IEEE CRC32，即 Python `zlib.crc32(data) & 0xffffffff`；反射多项式 `0xEDB88320`、初值及最终异或均为 `0xFFFFFFFF`。测试向量 `123456789` 的结果为 `0xCBF43926`。

帧解析器处理分包、粘包、噪声及校验失败，并限制缓冲长度。CRC 和 SHA256 合同号用于发现损坏或配置不一致，不提供身份认证。

## 3. HELLO：NX → STM32

payload 36 字节，整帧 60 字节。

| payload 偏移 | 长度 | 字段 |
| --- | --- | --- |
| 0 | 32 | 合同 SHA256 原始 32 字节，不是 64 字节 ASCII |
| 32 | 4 | 模式：0=shadow，1=请求执行 |

合同号来自选定 bundle 的 `manifest.json.contract_sha256`。它覆盖 Actor、观测、配置，以及“NX 计算最终力矩、MCU 执行”的控制约定。当前值：

```text
bc15a61d2bfc5c68e41776bb947db0a2389b905ea89d9e1342332f2b99be0eff
```

下位机初始化时使用已知的合同号与物理限幅，不能直接回显收到的任意哈希代替配置核对。换模型或修改控制配置后，重新运行：

```bash
./run_nx.sh mcu-config --bundle bundles/selected_nx_policy
```

接收新会话时，下位机清空旧会话待执行请求、状态编号与回执。HELLO 不会清除已经锁存的故障，也不会直接使能电机。

## 4. STATUS：STM32 → NX

payload 188 字节，整帧 212 字节。

| payload 偏移 | 长度 | 字段 |
| --- | --- | --- |
| 0 | 32 | 合同 SHA256 |
| 32 | 4 | flags |
| 36 | 4 | last_applied_torque_sequence |
| 40 | 108 | 27 个已预处理的 Actor 观测 float32 |
| 148 | 40 | 10 个未裁剪的执行器原始 SI 状态 float32 |

flags 定义：

| bit 值 | 名称 | 含义 |
| --- | --- | --- |
| 1 | SENSOR_VALID | IMU、编码器及当前快照有效 |
| 2 | VELOCITY_VALID | 机身水平速度估计有效 |
| 4 | EXECUTION_ACTIVE | 已有实际写入电机输出路径的本会话请求 |
| 8 | FAULT | 下位机报告或接口锁存故障 |
| 16 | CALIBRATION_VERIFIED | 下位机本机逻辑确认轴向、单位及零点已核对 |
| 32 | EXECUTION_AVAILABLE | 本机逻辑允许本接口执行且标定有效、无锁存故障 |

上位机不靠 `lqr_flag` 判断这些状态。请依据实际观测及本机控制状态填写，不能为了让接口运行而固定全部置 1。

### 4.1 Actor 的 27 维观测

先在物理单位下裁剪，再乘 bundle 中的对应 scale，整个链路只做一次。NX 收到后直接使用，不再次乘 scale。

| 索引 | 内容 | 裁剪前单位与约定 |
| --- | --- | --- |
| 0–2 | gyro xyz | rad/s，标准机体系 X 前、Y 左、Z 上 |
| 3–5 | projected gravity xyz | 单位重力方向，水平时 `[0,0,-1]`；不是 m/s² |
| 6 | vx 命令 | m/s |
| 7 | yaw rate 命令 | rad/s；不是 yaw 角 |
| 8 | 高度命令 | m，仿真机身参考点离地高度 |
| 9–12 | `q-offset` | rad，LB、LL、RB、RL，CAD 电机轴约定 |
| 13–16 | dq | rad/s，同上四关节顺序 |
| 17–18 | 机身水平 vx、vy | m/s，只随航向 yaw 旋转的水平坐标系 |
| 19–20 | 左右轮速 | rad/s，两轮均以前进为正 |
| 21–26 | 上一个已写入输出路径请求对应的有效归一化参考 | 顺序 LB、LL、LW、RB、RL、RW，再按 obs_clip.action 裁剪 |

下位机的原 INS 轴名、编码器零点和电机安装方向不自动等于仿真约定。需要先转换到上述坐标与单位，再构造观测。轮速不能直接代替机身水平速度；轮子打滑、机身俯仰和速度估计有效性需要在下位机数据处理中区分。

预处理的可执行参考在 `observation.build_observation()`；比例和裁剪值以配套 `obs_schema.json`、`runtime_config.json` 为准。

线速度命令斜坡由 NX 每个 Actor 周期推进。MCU 发送操作请求，NX 在 TORQUE 中回传当前实际使用的命令；不要在两端重复推进同一斜坡。

### 4.2 未裁剪执行器状态 10 维

```text
[q_LB, q_LL, q_RB, q_RL,
 dq_LB, dq_LL, dq_RB, dq_RL,
 omega_L, omega_R]
```

这里 q 是未经减 offset 的 CAD 原始关节角，单位 rad；dq、omega 是输出轴 rad/s。左右轮均以前进为正。它们不做观测裁剪、不乘观测 scale。

NX 的 PD 使用这 10 个值；Actor 使用上面的 27 个值。NX 会核对 q、dq 和轮速与 obs 对应部分是否满足同一缩放约定，从而发现顺序或重复缩放错误。

### 4.3 回执

`last_applied_torque_sequence` 是最近实际写入电机输出路径的 **NX TORQUE 帧头 sequence**，不是 STATUS 序号。初始为 0，模块只在 `RLHost_MarkApplied()` 后更新。

回执不等于已经实测到相同物理力矩。它表示请求进入了本机执行路径；电流换算、进一步限幅和实际扭矩跟踪仍需要分别验证。

## 5. TORQUE：NX → STM32

payload 64 字节，整帧 88 字节。

| payload 偏移 | 长度 | 字段 |
| --- | --- | --- |
| 0 | 4 | source_observation_sequence，计算所依据的 STATUS 帧头序号 |
| 4 | 24 | 6 个最终输出轴力矩 N·m |
| 28 | 24 | 6 个有效归一化参考，用于回执与下一帧历史动作 |
| 52 | 12 | NX 实际使用的 vx、yaw rate、高度，单位 m/s、rad/s、m |

六列顺序固定为 `[LB, LL, LW, RB, RL, RW]`。轮力矩采用左右均以前进为正；腿力矩采用对应 CAD 原始电机轴正方向。

**执行字段只有力矩。** 有效参考不是第二套要执行的目标，也不是额外力矩。MCU 将 N·m 换算到自身电机协议时，只做一次轴向、电流与传动换算，不再乘旧控制器经验增益。

下位机按自身保存的 STATUS 发送记录检查 source 编号是否存在、采样年龄是否合格；收到同一旧 TORQUE 的重发不能刷新看门狗。这样可以检查完整传感器—NX—执行回路，无需同步两台设备的时钟。

## 6. STOP：NX → STM32

payload 4 字节，整帧 28 字节；内容为 uint32 原因码。当前接口收到有效会话的 STOP 后撤销本会话执行请求、置停止请求，并等待应用输出任务处理停止。

零 action 对应零偏移参考角，仍可能产生 PD 与重力力矩；六路零 action 不能代替 STOP。

## 7. 序号、时钟与超时

每个发送方向单独递增 sequence；有效“更新”满足 `0 < uint32(new-old) < 2^31`。时间戳也按这一方式检查回绕，32 位微秒时钟约 71.6 分钟回绕一次。

NX 只接受本会话、正确合同、有效维度与有限数的状态。重复、倒序或设备时钟不前进，不刷新接收年龄。默认初次状态等待上限 1 秒，之后观测、执行回执以及主循环调度停顿预算为 30 ms。控制计算及串口写入还须满足所配置的单周期预算；超时后发送 STOP 并锁存。

MCU 使用自己的微秒时间检查 source 观测年龄和命令超时，默认都为 30 ms。`RLHost_Poll()` 应在应用任务中持续执行；失去传感器有效性、超时、异常值、超范围请求或 UART 错误会撤销输出并锁存。`RLHost_Init()` 是显式复位接口，应在实际输出关闭时调用。需要重新启动 UART 时，先 `RLHost_Uart3Stop()` 撤销 DMA 绑定，再初始化 context 并调用 `RLHost_Uart3Start()`。

两端时钟没有做绝对同步，NX 的“接收年龄”不能单独证明单向通信延迟。MCU 的源观测年龄检查、实际回执，以及后续设备实测共同用于确认时延。

## 8. 下位机接入点

接口文件只增加 `control/rl_host.c`、`control/rl_host.h`。协议核心可用 `RL_HOST_PORTABLE_TEST` 编译为普通 C99，用于离线核对；正常 STM32 构建包含 USART3 HAL 部分。

初始化步骤示意：

```c
static RLHost_Context rl_host;
/* rl_host_config 内容由 mcu-config 命令生成。 */
RLHost_Init(&rl_host, &rl_host_config);
RLHost_SetPermission(&rl_host, 0, 0);
RLHost_Uart3Start(&rl_host);
```

这段初始化本身不会启动电机。以下调用由您接到当前任务：

| 函数 | 输入 | 输出与作用 |
| --- | --- | --- |
| RLHost_Uart3Poll | context、当前微秒时钟 | 处理 ISR 排队字节及看门狗 |
| RLHost_Uart3SendStatus | obs27、raw10、有效性标志、实际采样时间 | DMA 发送当前状态；TX 忙时不排队旧状态 |
| RLHost_GetTorques | 当前微秒时钟 | 返回是否有新鲜且被本机允许执行的 Nm 请求 |
| RLHost_MarkApplied | 实际写入路径的 TORQUE 序号、时间 | 更新回执与历史动作 |
| RLHost_SetPermission | 本机允许状态、标定确认 | 权限来自本机控制逻辑，失去权限即撤销请求 |
| RLHost_Fault / Disarm | 故障原因或正常撤销 | 撤销待执行请求，置停止状态 |

接收使用已有 Normal DMA 的 Receive-to-Idle 方式，禁用半传输事件；中断复制字节到有界队列并重启 DMA。浮点协议解析在任务中执行。RX/TX 缓冲按 32 字节对齐，进行 D-Cache 维护，启动时检查缓冲位于 DMA 可访问 SRAM。

`usart.c` 的用户代码区域已将 USART3 的 HAL 接收及错误事件转发到该模块，不改变其他串口的用途。`joint_task` 中原有 LQR/VMC 输出尚未替换；接入 RL 时必须由单一路径选择输出来源，不能让旧控制器与 NX 同时写同一电机。

## 9. 已完成的核对与后续工作

2026-10-27 的 PC 闭环对照发现了实际执行差异：同一状态下，NX 与参考伺服的六路力矩最大算式误差约 4.05×10⁻⁶ N·m；但将轮速反馈改为 200/500 Hz 外部计算并保持力矩后，静止表现明显退步，预先站稳也不能消除这两种频率下的失稳。只将腿部改为外部力矩的对照维持了 20 秒。当前正在独立测试通用增益、反馈频率、启动衔接和延迟；这些结果不改变原 735 工况验收门槛，也不证明 NX 或 STM32 硬件已验证。

当前离线测试覆盖 C/Python 逐字节编码、CRC、分包粘包、异常值、超时、会话、源状态年龄、回执，以及真实 pyserial 经过 Linux 虚拟串口收发。虚拟串口不代表 STM32 板端中断、DMA 或物理链路已经测过。

原 735 次 Isaac 回放抽取 5145 个真实输入，与冻结 Actor、训练中的关节目标映射和重力辅助函数交叉核对，结果见 [数值核对记录](verification/recorded735_actor_and_controller_20261027.json)。后续还需完成 NX 实际时序、STM32 工程构建与板端收发、坐标及电机换算、显式力矩闭环和通信故障行为的设备验证。
