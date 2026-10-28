# STM32 下位机工程

本目录是现有 STM32H750VBT6 固件工程，与 [Jetson NX 上位机](../jetson_nx/README.md) 分开维护。

| 路径 | 用途 |
| --- | --- |
| `project_h7.ioc` | CubeMX 配置入口。 |
| `MDK-ARM/project_h7.uvprojx` | Keil 工程入口。 |
| `Core/`、`Drivers/`、`Middlewares/` | 启动、HAL、外设与系统依赖。 |
| `Usr/` | 应用初始化和任务。 |
| `control/` | 原控制代码及 `rl_host.c`、`rl_host.h` 上下位机交互模块。 |

当前交互接口使用 USART3，PD8 TX、PD9 RX，2,000,000 baud、8N1；CubeMX 配置和 `usart.c` 已同步。协议、字段顺序及单位见 [上位机通信协议](../jetson_nx/PROTOCOL.md)。

`rl_host.c/.h` 提供通信解析、状态回传和力矩请求接口，尚未替换 `joint_task` 的实际电机输出链。目录整理没有修改下位机控制代码，也没有编译、烧录或使能电机。
