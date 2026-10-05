# 自定义闭链机器人

## 2026-10-05 最新发布入口

当前自己的机器人已独立训练，使用 `protected_stops_v2` 接触资产、8192 环境、
隐式关节参考控制及轮速驱动；当前验收配置为 20% 相对误差，尚未最终达标。
发布时最新已保存模型为 [12600 快照](../wheel_legged_isaaclab/checkpoints/mine_20261005_model_12600/)，
并不是早期直接迁移的旧串联腿权重。

- [版本更改与验证范围](../CHANGELOG.md)
- [键盘回放说明](docs/键盘可视化控制说明.md)
- [持续优化报告发布快照](docs/持续优化报告_20261005快照.md)

```bash
./mine_model/run.sh play  # 本机最新完整评估模型，自动加载配套配置
```

GitHub 固定快照的显式启动命令见模型目录 README。当前后台训练没有因提交重启。
下文是 10 月 3～4 日各阶段建模/试训记录，包含早期资产、4096 环境、10% 目标与旧权重示例；
它们不覆盖本节的最新状态。实时状态以本地 `results/.../status.json` 及实际评估为准。

这里管理 CAD 原始文件、闭链构建、正式导出和验证结果。基础闭链资产为
`exports/export_20261003_181515/mine_closed_chain.usd`，四个闭链关节保存在 USD 中。
当前精度训练另用同目录的 `mine_closed_chain_tires_96.usd` 叠加层，保留原文件。
原始文件按原内容移入 `inputs/`，没有删除或覆盖。

## 从这里开始

- [图文说明 PDF](docs/闭链建模与工程接入说明.pdf)
- [可编辑图文说明 Word](docs/闭链建模与工程接入说明.docx)
- [实现细节与参数](docs/implementation_details.md)
- [VMC 参数和代码核对](results/parameter_audit/README.md)
- [最新策略评估与误差曲线](results/policy_7738_20261003/README.md)
- [正式导出模型](exports/export_20261003_181515/)
- [目录迁移记录](docs/relocation_map.json)

## 目录职责

```text
mine_model/
├── README.md                 本页 总入口
├── run.sh                    统一运行入口
├── inputs/                   原始 URDF ZIP MJCF ZIP USDZ 只读保留
├── source/                   原始 URDF 解压内容 可由 inputs 重建
├── prepared/                 规范化 RB 命名和网格名的中间输入
├── config/                   四个闭合铰链坐标与孔位拟合结果
├── scripts/                  导入 闭链构建 关节检查 导出 计算和分析工具
├── exports/                  按时间版本保存的正式 USD URDF 和网格
├── results/
│   ├── bench_drive/          已通过的双环境悬空 VMC 驱动测试
│   ├── gravity_check/        重力接触检查 平衡未通过
│   ├── drive_preview/        GUI 驱动截图和记录
│   └── policy_7738_20261003/ 本次原策略在新旧模型上的平地评估
├── docs/                     图文说明与 figures 配图
├── archive/                  失败调试与被替代的测试 保留溯源
└── build/                    可再生仿真缓存 场景与 logs 日志
```

只有 `exports/` 是正式交付模型；`prepared/` 是导入中间件，`build/` 是调试场景。
`archive/` 不代表当前通过结果。目录移动记录保存在 `docs/relocation_map.json`。
本次移动后校验了原 URDF ZIP 与正式 USD 的 SHA256，内容未变；工程 manifest 已更新源导出路径。

## 常用命令

在工程根目录 `~/studyRL/src/CLT-RL` 下执行：

```bash
./mine_model/run.sh inspect                         # 闭链关节角度检查 GUI
./mine_model/run.sh bench --num_envs 1 --loop       # 悬空 VMC 驱动演示
./mine_model/run.sh bench --headless --seconds 12  # 双环境驱动验证
./mine_model/run.sh evaluate --headless --robot mine   # 最新 7738 策略 新模型
./mine_model/run.sh evaluate --headless --robot legacy # 同策略 原模型对照
./mine_model/run.sh test                            # 五连杆运动学检查
```

重建闭链调试场景：

```bash
./mine_model/run.sh build --headless --steps 240
./mine_model/run.sh export  # 创建新时间版本 并更新工程 manifest
```

评估入口默认 9 种工况、43/44/45 三个扰动种子、每工况 8 秒，启动立即按 0.8 m/s²
斜坡下发速度。原策略参数不变，采用对应模型当前 VMC 配置；不训练。
初始根速度扰动幅值 0.05，较历史 0.5 的扰动评估更温和。

## 目前状态

闭链结构、五连杆映射和悬空驱动已通过；直接迁移的 7738 策略在新模型上会失稳，
不能把机械驱动测试通过理解为 RL 平地运动通过。详见最新评估目录中的误差曲线和失败记录。
原开源模型仍由 `WheelLeggedVMC-Legacy-v0` 提供，默认 `WheelLeggedVMC-Flat-v0` 使用本模型。

## 当前自动试训与优化

当前流程为 `auto_training_implicit_20pct_20261004`，早期使用256个环境，现按实测切换为4096个环境。
正常USD克隆4096环境短测约29118步/秒，进程组内存峰值17.17 GiB；6144触及22 GiB上限。
详见[环境数实测与建议](results/auto_training_implicit_20pct_20261004/环境数实测与建议.md)。
5500 轮策略配合 96 边轮胎碰撞，在两轮 12 秒验收中均 27/27 通过、零重置；
30 秒回放仍全部存活，但三个工况的偏航误差超限，最大 0.060219 rad/s，容差 0.05 rad/s。
当前在此资产上继续精度训练，尚未通过最终验收，尚未关机。

- [开链转闭链图文教程 Word 十八页](docs/开链转闭链图文教程_v3.docx)
- [开链转闭链图文教程 PDF](docs/开链转闭链图文教程_v3.pdf)
- [实时流程状态](results/auto_training_implicit_20pct_20261004/status.json)
- [当前流程与验收说明](results/auto_training_implicit_20pct_20261004/README.md)
- [闭链文件交付目录](deliveries/)

当前服务 `mine-rl-implicit-20pct-20261004.service` 管理训练、评估、优化、导出与条件关机。
这是从新网络开始的独立实验：五连杆逆解把虚拟腿长和摆角目标转换为电机角目标，
由 PhysX 隐式关节 force Drive 执行，初始关节刚度/阻尼 300/3，腿轮正式限幅 10 Nm。
观测为 27 维、动作 6 维；执行器契约已改变，MuJoCo 必须同步使用逆解和关节 PD。
旧显式 VMC 实验及失败回放完整保留；不能与本实验权重或成功结论混用。

```bash
systemctl --user status mine-rl-implicit-20pct-20261004 --no-pager
cat mine_model/results/auto_training_implicit_20pct_20261004/status.json
```

使用4096个环境继续训练，模型保存在原训练目录旁的 `IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/`。
首次1000轮试训，后续每200轮优化回放9工况和3个种子，保存完整误差曲线。独立种子和 30 秒复测全部通过后，
保存模型、参数、教程、源码及更改说明，校验并刷盘后关机。未通过验收不会关机。
当前继续精度训练，尚无通过全部三阶段验收的模型；状态以实际 replay/assessment.json 为准。
