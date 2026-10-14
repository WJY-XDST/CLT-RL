# 当前训练资产选择

2026-10-14：`assets/robots/mine/manifest.json` 的 `training_usd` 指向
`mine_closed_chain_training_measured_volume_mass_20261014.usda`，当前为 8192 个环境的
`implicit_joint_position` 直接关节角目标 PD 续训。旧检查点回放使用其保存的 USD 路径。

用户实测整车 12.34 kg、每侧整腿含轮 1.935 kg、单轮 0.665 kg。整腿不包含
安装在机身上的髋电机，因此固定机身含髋电机为 8.47 kg、每侧非轮连杆总质量
为 1.27 kg。连杆按原始 URDF 可视 STL 的实体体积估计质量、质心及惯量：
碳纤维板密度暂估 1600 kg/m³，配件有效密度由整腿称重闭合；单根连杆及惯量
尚非实测。机身新增质量暂置于原 CAD 质心，保留原机身惯量，属于待校准假设。
同名 JSON 保存实测约束、材料假设、全部源网格哈希及实际 PhysX 读回数据。

左右轮组质量均为用户确认的 0.665 kg。两轮主惯量取右轮原 CAD 值，按基座坐标
镜像后转换到左轮局部坐标；两侧转轴惯量相同。此惯量是 CAD 估计值，尚非实测。
原 CAD 质心保留；同名 JSON 保存来源哈希与原生 PhysX 核验。

当前 USD 合成链（图中省略共同的 mine_closed_chain 前缀）：

```text
training_measured_volume_mass_20261014.usda
└─ training_matched_wheels_20261014.usda
   └─ training_wheel_mass_20261014.usda
      └─ training_protected_stops_v2.usd
         └─ training_protected.usd
            └─ base_box.usd
               └─ tires_96.usd
                  └─ mine_closed_chain.usd
```

另有 `mine_protected_leg_instanceable_meshes.usd` 和
`mine_base_box_instanceable_meshes.usd` 作为共享几何引用，当前实际加载十个来源文件。
这些层分别保存质量惯量、机械限位、碰撞简化、轮胎及原 CAD 数据，不能只留下入口文件。

目录中的其他 `contacts`、`groups`、`stops` 和 `simple` 版本用于此前调试或历史模型回放，
并非当前全部同时加载。保留原资产与这些版本，避免破坏来源追溯及旧模型复现。

底盘用盒；普通连杆合并凸包；左右限位块与小腿接触主板保留 CAD 顶点和拓扑。
真实机械限位接触启用；关节按名称映射，不复用旧关节下标。

当前未通过完整跟踪验收。标准仍为现有 `tracking_acceptance.json` 的全工况 10%
及各项原有绝对容差，再通过独立持续复验；历史 20% 标准不适用于当前任务。
