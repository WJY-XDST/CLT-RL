# sim2real 工程导航

上位机与下位机分开维护，各有独立 README。

| 工程 | 位置 | 说明 |
| --- | --- | --- |
| Jetson NX 上位机 | [jetson_nx/](jetson_nx/) | Python 推理、控制计算、串口接口、模型包和测试。安装与运行见 [上位机 README](jetson_nx/README.md)。 |
| STM32 下位机 | [series_balance_rtos/](series_balance_rtos/) | CubeMX、Keil、传感器和电机执行工程。入口见 [下位机 README](series_balance_rtos/README.md)。 |

通信字段与单位见 [PROTOCOL.md](jetson_nx/PROTOCOL.md)。当前下位机交互模块仍在 `series_balance_rtos/control/rl_host.c` 与 `rl_host.h`。

复制到 NX 时使用 `jetson_nx/`。历史压缩备份和 `.directory_migrations/` 不属于 NX 运行依赖。
