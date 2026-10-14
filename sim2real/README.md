# 新车 sim2real 上位机工程

这是独立于 Isaac Sim 的 CPU 策略推理入口，当前仍是离线接口准备，不能直接使能实车。
本版只导出 `implicit_joint_position` 策略：四个主动髋关节目标角和两个轮速请求。
旧 IK 控制的 obs、action、bundle 已停用；维度相同也不兼容。

## 控制分工

下位机采集 IMU、编码器和速度估计，按配套契约构造 obs；上位机只校验、推理和映射参考请求。
下位机内环执行四关节位置 PD 和重力前馈、轮子速度控制、硬件限幅及看门狗。
腿部不再从虚拟摆角/长度进行 IK。重力补偿仍需要五连杆正解和 Jacobian，这些没有删除。

已有传统下位机代码位于 `series_balance_rtos (2)/`，但直接关节 PD 的实机标定和串口协议尚未完成对接。
这里不自动打开串口，不发送电机力矩、电流或使能命令。

## 代码入口

|文件|作用|
|---|---|
|`wheelleg_deploy/policy.py`|冻结明确选定的 actor 和训练配置，拒绝旧 bundle|
|`wheelleg_deploy/schema.py`|定义 27 维观测顺序、单位、裁剪和缩放|
|`wheelleg_deploy/targets.py`|归一化 action 映射为四关节角和两轮速请求|
|`wheelleg_deploy/session.py`|序号、时间、推理超时、故障锁存|
|`wheelleg_deploy/protocol.py`|仅供离线测试的 JSON/CRC 协议模板|
|`wheelleg_deploy/transport.py`|内存回环和禁用真实发送的串口骨架|
|`config/serial.template.json`|待统一的正式串口配置，默认未批准|

## 离线使用

在 `~/studyRL/src/CLT-RL/sim2real` 执行，Python 需要 PyTorch 和 PyYAML，不需要 IsaacLab：

```bash
PY=~/miniconda3/envs/env_mujoco/bin/python
$PY -m wheelleg_deploy prepare \
  --checkpoint /完整路径/model_XXX.pt \
  --output bundles/my_selected_joint_policy
$PY -m wheelleg_deploy check --bundle bundles/my_selected_joint_policy
$PY -m wheelleg_deploy dry-run --bundle bundles/my_selected_joint_policy --steps 5
$PY -m unittest discover -s tests -v
```

检查点旁必须有自己的 `params/env.yaml`、`agent.yaml`，不覆盖既有 bundle，不自动追踪正在变化的 latest。
仅从本地可信检查点导出。人工 obs 和内存回环只验证数据通路，不证明整机运动或实机部署合格。
导出不会将 `hardware_approved`、`policy_precision_accepted` 设为 true。

## 观测顺序

从 0 开始编号，先在物理单位下裁剪，再乘配套 scale；上位机不重复缩放。
下表为当前结构，具体比例和边界以 bundle 的 `obs_schema.json` 为准。

|索引|字段|单位/约定|
|---|---|---|
|0–2|机体系角速度 xyz|rad/s|
|3–5|机体系单位重力方向 xyz|水平为 `[0,0,-1]`|
|6|前向速度命令|m/s|
|7|yaw rate 命令|rad/s，不是 yaw 角|
|8|机身高度命令|m|
|9–12|LB、LL、RB、RL 关节位置减去配套 offset|rad，CAD 原始轴约定|
|13–16|LB、LL、RB、RL 关节速度|rad/s，与位置顺序相同|
|17–18|水平航向坐标系机身 vx、vy|m/s；只随 yaw 转，不随 pitch/roll 转|
|19–20|左右实际轮速|rad/s，两侧均以前进为正|
|21–26|上一周期有效执行动作|归一化 6 维，不是未经限幅的 actor 输出|

轮速不能直接代替机身线速度。编码器零点、轴向、关节位置 offset 和速度估计必须与仿真对齐。
旧 IK 的 9–16 虚拟腿状态不能发送给此策略。

## 动作和物理请求

动作顺序是 `[LB, LL, 左轮, RB, RL, 右轮]`。
actor 输出可能超过 `[-1,1]`；边界先裁剪再映射：

```text
q_request[LB,LL,RB,RL] = offset + clip(action[0,1,3,4], -1, 1) * scale
wheel_request[L,R] = clip(action[2,5], -1, 1) * action_scale_vel
```

输出的 `joint_position_ref_rad[4]` 是四个真实主动关节角请求，不是虚拟腿摆角。
`wheel_speed_ref_rad_s[2]` 两侧前进为正；下位机转换原始轴时只翻转右轮一次。
高度命令通过策略影响动作，不在此映射中插入 IK 或高度闭环。

仿真还按关节物理限位及安全 margin 限制参考，并回算有效 action。
本上位机结果只是请求；下位机必须按核实的实车关节限位处理，使用实际应用的参考构造下一帧 last_actions。
`targets.py` 不是电机控制器，也没有实现最终 MCU 内环。
零 action 对应 nominal 关节参考，既不等于零力矩，也不等于停机。

训练采用关节隐式 PD 加单独的下位机重力补偿，腿部总力矩限幅 35.574 N·m，轮限幅 4.80944 N·m。
配套 gain、零点、前馈、更新周期和限幅必须冻结；真实下位机 PD 是否能等效复现仿真仍需验证。
不能仅因有 Jacobian/重力补偿就把新策略接回旧虚拟腿 `Jᵀ[F,T]` 反馈链。

## 串口和故障保护

正式帧格式、波特率和握手以后统一；JSON/CRC 模板不要求下位机照抄。
不扫描设备、不自动使能。接收骨架要求确认协议与 DTR/RTS 行为；真实 `write()` 始终拒绝执行。
session 校验维度、有限数、schema、序号和单调设备时钟；超时/异常锁存故障，不自行恢复。
CRC/schema hash 不提供身份认证。未同步的两个时钟不能证明首帧绝对年龄，正式协议要解决新鲜度，禁止积压帧追赶输出。

`fallback_request` 只是请求，不代表车辆已安全。下位机独立看门狗、可靠急停、通信断开接管、
硬件限位、电流保护和整机验证必须完成后才能部署。当前没有连接或控制硬件。
