# 自定义轮腿检查点位置和使用规则

当前正式训练结果保存在 Isaac Lab 的 logs 下，不把这个目录当成不断变化的 latest：

```text
CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat/
CLT-RL/IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/
```

关节角对照任务为 `WheelLeggedJointPD-Flat-v0`。
检查点目录的 `params/env.yaml`、`agent.yaml` 和同轮 overrides 必须与模型一起保留。
`implicit_joint_position` 与 `explicit_vmc` 的动作和腿部观测含义不同，不能混用权重。

当前运行与独立评估记录见 [训练监测](../../mine_model/results/joint_pd_scratch_20261011_181337/status.json)。
默认键盘回放可在工程根目录用 `./mine_model/run.sh play --dry_run` 核对选择；
模型时间最新、独立评估已完成和全部验收通过是三个不同状态。

原开源示例车和旧 IK 的模型目录已清理，包括 `mine_20261005_model_12600`。
本目录的历史清单和报告不保证其旧路径仍存在，不再推荐加载旧车的 6739、7738 等模型。
显式 VMC 的已确认记录与机械证据保留；没有用目录名中的 implicit 单词判定并删除训练。

恢复旧 IK 记录需要同时恢复其匹配旧源码，当前入口会拒绝旧模式。
清理范围与回收站恢复方式见 [IK 清理说明](../../mine_model/docs/IK清理说明_20261011.md)。
