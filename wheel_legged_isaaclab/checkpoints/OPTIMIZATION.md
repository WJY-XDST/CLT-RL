# 当前训练与验收入口

旧车的固定模型优化器及旧尺寸验收脚本已于 2026-10-12 退役。当前训练结果在 `CLT-RL/IsaacLab/logs/rsl_rl/`；不要继续调用旧优化器或加载已删除的示例车模型。

当前工程使用自己的映射闭链五连杆，保留两种执行契约：直接关节目标角加隐式 PD 和独立重力前馈，以及显式 VMC。相同的 27 维观测、6 维动作尺寸不意味着模型可以跨控制模式复用。

主要入口：

- `wheel_legged_isaaclab/scripts/rsl_rl/train.py`：完整策略训练与检查点恢复。
- `mine_model/scripts/auto_train_mine.py`：选择明确的训练模式。
- `mine_model/scripts/joint_pd_optimization.py`：当前关节角全策略单变量对照。
- `wheel_legged_isaaclab/scripts/evaluate_mine_policy.py`：冻结模型的 9 工况、3 种子独立回放。
- `mine_model/scripts/summarize_mine_trial.py`：按统一 `tracking_acceptance.json` 判定。

详情见 [工程说明](../../mine_model/README.md)、[训练修改记录](../../mine_model/docs/训练修改记录_latest.md) 和 [完整代码框架](../learning/工程代码框架全解析_20261012.md)。历史模型和记录不会因本次代码清理而删除；各实验状态需要结合实际进程和更新时间判断，不只看旧 JSON。
