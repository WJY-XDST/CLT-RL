# 新闭链模型的 MuJoCo 导出

用于回放和验证的最终模型：`scene_protected_contact.xml`。
同名 JSON 保存源 USD 层、每个 OBJ 和 MJCF 的校验和及关节/惯量审计。
`meshes/` 为共享视觉 CAD 和碰撞凸片。
`scene.xml` / `scene.json` 是最初的软接触诊断版本，保留用于对照；新回放请选择 protected_contact 版本。

模型沿用 15 刚体、18 实际铰链及确认限位 CAD。四个外部铰链由八个 connect 约束表示。
XML 内部积分为 0.1 ms；策略应通过工程 sim2sim wrapper 运行，50 Hz 策略及 200 Hz 参考控制。

本轮 model_6700 基础运动 27/27 存活，但 5/27 精度通过，主要剩余问题为偏航；机械极限仍与 PhysX 有差异。
详见 `mine_model/docs/新闭链模型_MuJoCo_sim2sim验证_20261004.md` 和 `mine_model/results/latest_sim2sim.json`。
