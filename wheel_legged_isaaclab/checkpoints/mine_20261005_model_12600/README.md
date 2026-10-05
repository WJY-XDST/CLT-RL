# 自定义闭链机器人最新模型快照：12600

2026-10-05 上传准备时最新已完整保存的权重是 `model_12600.pt`。它属于第 66 轮精度续训，
是**训练候选模型，不是通过最终验收的部署模型**。后台训练仍继续；本目录是固定快照，
不会随训练自动替换，后续权重须另外发布。

## 文件含义

|文件|用途|
|---|---|
|model_12600.pt|原始 RSL-RL 权重，含 actor/critic、探索参数、优化器状态和 iteration|
|env.yaml / agent.yaml|该训练 run 启动时实际保存的环境/PPO 配置|
|precision_mode.yaml|轮输出精度训练模式及其来源记录|
|training_overrides.json|该轮实际训练覆盖参数，保留原始路径和内容|
|replay_overrides.json|仅将资产路径改为相对仓库路径，适配下方回放命令；控制参数不变|
|manifest.json|来源、SHA256、维度、控制模式及验证范围|
|previous_model_12499_assessment.json|上一个完整评估模型 **12499** 的结果，不是 12600 的验收|

观测 27 维、动作 6 维。腿动作经五连杆逆解转换为关节参考，PhysX 使用隐式关节位置驱动；
轮动作转换为轮速参考，使用隐式速度驱动。不能把该模型套入旧串联腿的显式 VMC 执行契约。

## 键盘可视化

从 `CLT-RL` 根目录执行：

```bash
./mine_model/run.sh play \
  --checkpoint_path wheel_legged_isaaclab/checkpoints/mine_20261005_model_12600/model_12600.pt \
  --overrides_json wheel_legged_isaaclab/checkpoints/mine_20261005_model_12600/replay_overrides.json
```

W/S 前后、A/D 转向、Q/E 调整高度命令、L/Esc 清空移动指令；空格只记录未训练的跳跃请求。
默认零速、零 yaw rate、0.30 m 高度起步。详见 [键盘说明](../../../mine_model/docs/键盘可视化控制说明.md)。

这些入口会切换到 `IsaacLab` 目录运行，因此相对 USD 路径是 `../wheel_legged_isaaclab/...`。
若绕过启动脚本在别的工作目录直接运行 Python，需自己解析资产路径。
原 `env.yaml` 和资产审计 JSON 中的绝对路径是来源记录；不能当作另一台机器天然有效的路径。
在其他目录重新训练/评估前应重新生成解析后的配置与资产链审计，不应修改 CAD/USD 本体冒充相同模型。

## 验证范围

- CPU 加载成功，checkpoint 内 `iter=12600`，17 个网络状态张量中的浮点参数均为有限值。
- SHA256：`a6fc33c7770ff2c3a776c65d737a01bf33217b328ac11fe9563cf4d8ec6cbee3`。
- 工程 127 项 CPU 单元测试通过；这些测试不等于当前策略已经通过运动精度验收。
- 截至本快照，上一个完整评估模型 12499：27/27 存活、零重置，仅 6/27 精度通过，整体未通过。
- 没有对 12600 另做完整三阶段验收、没有进行实机试车，也没有把空格跳跃实际接入执行器。

与权重一起提交运行资产和 MuJoCo 接口，但不提交 Isaac Sim/Isaac Lab SDK 或全部训练日志。
本文命令需要本地安装工程 README 指定版本的运行时。
