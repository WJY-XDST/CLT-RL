# Isaac Lab -> MuJoCo sim2sim

## 当前控制契约（2026-10-24）

当前 MuJoCo 回放入口支持新模型，使用完整训练 USD 的刚体、惯量、树关节及四个外部铰链；
从检查点 `params/env.yaml` 识别两种执行链：

- `implicit_joint_position`：四个主动关节目标角 + 关节 PD + 单独重力前馈，轮子为目标轮速。
- `explicit_vmc`：虚拟腿 PD + 重力前馈 + `Jᵀ[F,T]`，腿关节没有位置 PD，轮子为目标轮速。

旧 `implicit_joint_reference` 的 IK 控制、训练记录和回放入口已移除；旧检查点会被拒绝。
已完成新质量和解析轮胎的离线转换准备，资产为：

`mine_model/exports/mujoco_measured_mass_analytic_tires_20261024_preacceptance_geometry/scene.xml`。

[离线核验报告](../../mine_model/exports/mujoco_measured_mass_analytic_tires_20261024_preacceptance_geometry/offline_geometry_mass_inertia_verification.json)
核对了 15 个刚体的质量、质心和完整惯量张量，整车约 12.34 kg；
每侧轮子约 0.665 kg，整腿含轮子约 1.935 kg。
21 个启用的 USD 碰撞体均保留；每个车轮包括原轮毂碰撞网格及半径 0.06 m、宽度 0.02 m 的解析圆柱轮胎。
转换器现在处理 Cylinder 的轴向、位姿和缩放，遇到无法表示的形状或空转换会报错。
质量校准覆盖层未单独声明单位时，只在下层资产明确为米的情况下建立临时米制场景，不改写训练 USD、不缩放几何。

离线转换本身只证明几何和惯性数据一致。当前按用户要求，先用保留的7401策略进行 MuJoCo
可靠性诊断和键盘回放；该策略尚未通过 Isaac Lab 全工况10%验收，Sim2Sim结果单独记录。
环境随机化和最终模型验收仍保留原来的独立复验要求。
旧的 `mine_model/exports/mujoco_protected_stops_v2_20261004/` 资产保留为历史诊断数据，不应替代新质量模型的验收资产。

CAD 资产可继续使用，但加载时会按检查点重设执行器类型、增益和力矩限幅；不继承旧 MJCF 的 10 N·m 配置。
轮端动作限幅从保存配置读取：历史检查点缺少该字段时使用 ±1，当前 ±1.25 对应最高目标轮速 18.75 rad/s；
动作历史观测另按保存的 `obs_clip.action` 裁剪，当前仍为 ±1，与训练观测一致。
初始化同时写入全部 14 个主动、被动关节角，保持保存的闭链姿态，
并将腿位置驱动初始目标设为零动作参考角。
关节 PD 与重力前馈相加后再统一限幅，不把前馈加在已限幅的执行器外面。
接口测试检查观测、动作映射、前馈、镜像方向和合力矩限制；接口一致不等于完整 sim2sim 运动验收通过。

```bash
# 从工程根目录运行，checkpoint 填入需要验证的检查点。
./sim2sim_mujoco.sh --checkpoint "$checkpoint" \
  --mjcf mine_model/exports/mujoco_measured_mass_analytic_tires_20261024_preacceptance_geometry/scene.xml --demo
```

新模型默认九阶段演示：站立、正反向 0.3/0.6 m/s、左右旋转 0.3 rad/s、高度 0.28/0.32 m。
每阶段 8 秒，不添加航向保持。50 Hz 策略、200 Hz 参考控制保持不变；
MuJoCo 内部积分默认 0.1 ms，以处理薄闭链连杆和限幅驱动的小轮惯量。
腿 PD、轮速阻尼和力矩上限均从配套配置读取，不在回放中写死。
受保护 CAD 采用独立 CoACD 分解；接触与 PhysX 的求解边界在报告中单独说明。

独立矩阵验证用 `evaluate_mine.py`；原始 CAD 核对用 `audit_stops.py`。
`evaluate_mine.py --cases-json mine_model/config/mujoco_reliability_full_envelope_20261024.json`
可指定当前45工况矩阵，覆盖三档完整安全高度、正反行驶、原地旋转及直行转向；
每工况默认三个独立种子。`assessment.json`保留原始七项门槛，
`assessment_literal10.json`额外逐项检查非零速度、偏航、高度的严格10%相对误差。
直接关节模式的腿角参考遥测来自四个电机目标角的五连杆正运动学，与Isaac Lab一致。
依赖 IK 固定腿长的旧限位测试已移除；几何审计和闭链正运动学测试保留。
新车接口检查：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/env_mujoco/bin/python -m unittest discover \
  -s wheel_legged_isaaclab/sim2sim -p 'test*interface.py' -v
```

## MuJoCo 键盘控制

GUI 回放自动启动 **MuJoCo robot keyboard commands** 控制小窗口。
默认 W/S 速度为 ±0.8 m/s，A/D 偏航角速度为 ±0.3 rad/s。
点击这个控制窗口再按键，不是在终端或 MuJoCo 画面窗口中输入。
独立窗口提供松键和失焦事件，不安装全局键盘监听，也不修改训练代码。

当前保留的7401模型可直接启动：

```bash
./sim2sim_keyboard_7401.sh
```

此入口使用配套的测量质量模型、直接关节目标角PD和重力前馈，保留原检查点与优化器。
点击控制窗口后按键；MuJoCo画面窗口负责显示机器人。

自己的闭链模型可按以下方式启动，替换为需要测试的检查点；
`params/env.yaml` 和 `params/agent.yaml` 自动从同一训练目录读取：

```bash
cd ~/studyRL/src/CLT-RL
./sim2sim_mujoco.sh --checkpoint /完整路径/model_XXXXX.pt \
  --mjcf mine_model/exports/mujoco_measured_mass_analytic_tires_20261024_preacceptance_geometry/scene.xml \
  --fixed-command 0 0 0.30 --keyboard-speed 0.8 --keyboard-yaw-rate 0.3
```

|按键|效果|
|---|---|
|按住 W / S|前进 / 后退，默认目标 `±0.8 m/s`|
|按住 A / D|左转 / 右转，默认目标 yaw rate `±0.3 rad/s`|
|Q / E 单击|降低 / 提高机身高度目标 5 mm|
|Q / E 长按|以 `0.02 m/s` 连续改变高度目标|
|松开 WASD|相应运动轴目标归零，前后命令保留原加速度斜坡|
|松开 Q/E|目标保持，高度命令继续平滑到达目标|
|L / Esc / Stop 按钮|清空运动目标，不强制冻结机器人|
|Nominal 按钮|停止移动，将高度目标恢复为启动时的高度；命令平滑返回|
|空格|停止移动并记录跳跃请求，**当前 MuJoCo 平地回放不执行跳跃**|

允许 W+A 等组合；相反按键同时按住时该轴目标归零。
首次控制按键或 Stop/Nominal 按钮接管后，固定/周期演示不再覆盖手动指令。
失焦清空运动目标，回到控制窗口后先松键再重新按下；停止后仍按住的键不会因重复事件恢复移动。
关闭控制窗口后，已接管的运动目标保持为零，不恢复周期演示；关闭 MuJoCo 画面窗口结束回放。
未接管就关闭控制窗口，则仍按原固定/周期流程运行。

Q/E 调整的是机身高度命令，不是直接锁定两腿长度或改变策略动作。
目标及平滑命令限于检查点配置的 `commands.ranges_height`；当前新车的完整安全范围约为 `0.22535–0.37535 m`，应以所用检查点配置为准。
命令默认以 `0.02 m/s` 斜坡追踪目标，在 50 Hz 策略下每步最多变化 0.4 mm。
手动模式只请求 yaw rate，不额外添加航向保持来掩盖策略残差。
`±0.6 rad/s` 超出当前闭链模型配置的 `±0.3` 训练范围，属于探索测试；
不改变策略动作、驱动力和关节限幅，不能保证这个转向目标能被准确跟踪。

调整参数时使用 MuJoCo 入口的横线参数名，例如：

```bash
./sim2sim_mujoco.sh --checkpoint /完整路径/model_XXXXX.pt \
  --mjcf mine_model/exports/mujoco_protected_stops_v2_20261004/scene_protected_contact.xml \
  --keyboard-speed 0.5 --keyboard-yaw-rate 0.3 \
  --keyboard-height-step 0.005 --keyboard-height-rate 0.02 \
  --keyboard-height-slew-rate 0.02 --max-steps 15000
```

未指定时，普通键盘 GUI 回放最长 30000 个控制步（50 Hz 下为 10 分钟仿真时间）。
`--demo` 保留演示时长，`--max-steps` 可显式覆盖；数值或姿态失败仍会提前停止。
`--no-keyboard` 关闭键盘面板，用于固定/周期对照；`--headless` 自动禁用键盘。
原有 `evaluate_mine.py` 独立矩阵评估不接入手动控制。

控制窗口显示实际速度、pitch/roll 和用户高度目标、平滑命令、实际高度。
`trace.csv` 增加 `command_source`、`keyboard_active`、`keyboard_height_target`、
`keyboard_jump_requests`，保留原有速度、姿态、腿状态和动作字段。
`base_height - cmd_height` 是对送入策略命令的跟踪误差，
`keyboard_height_target - cmd_height` 是未结束的指令斜坡，不要混为同一误差。
手动变化期间的阶段统计不一定对应单一固定工况，查看 `constant_targets` 和原始 CSV。

键盘回归测试可独立运行；可选原生 GUI 测试会自动打开并关闭测试窗口：

```bash
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/miniconda3/envs/env_mujoco/bin/python -m unittest discover \
  -s wheel_legged_isaaclab/sim2sim -p 'test_keyboard*.py' -v
# 在桌面会话内设置 MUJOCO_KEYBOARD_GUI_TEST=1，可同时执行原生 Tk 和 viewer 按键测试。
```

当前仅支持自定义闭链车，必须显式指定新车 checkpoint；已取消其他车的默认权重、URDF 和串联运动学。
车体配置来自检查点配套 env.yaml，几何统一为映射后的大五连杆；不重复乘映射倍率。
