# 当前训练资产选择

2026-10-04：用户要求停止容量测试并采用 10240 个环境进行正式优化训练。

当前资产：`mine_closed_chain_training_protected_stops_v2.usd`。依赖来源层和共享几何列在同名 JSON 的递归来源清单中，必须一起保留。

底盘用盒；普通连杆合并凸包；左右限位块与小腿接触主板保留 CAD 顶点和拓扑。真实机械限位接触启用，并换闭链求解树使相邻限位接触可求解。关节按名称映射，不能复用旧关节下标。

固定同源权重的27组平地回放中，简化普通腿连杆相对保留原腿碰撞的版本，所有记录数据一致。若后续同权重对照出现误差增加，回退资产 `mine_closed_chain_base_box_stops_v2.usd`，配套参数为 results/collision_simplification_20261004/base_box_stops_v2_overrides.json。

原始 `mine_closed_chain.usd` 和圆轮 `mine_closed_chain_tires_96.usd` 均为来源资产。其他 contacts/groups/stops 命名变体及早期合并版本为诊断产物，不能只根据文件名作为当前训练模型。当前版本的 CAD 几何采样验证、策略回放及容量证据见同名 JSON。

当前并未宣称完整训练目标已经完成；最终交付仍须通过20%三阶段回放和原CAD接触几何校验。
