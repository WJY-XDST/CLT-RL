# 从训练入口理解 RSL-RL 与 PPO 的调用流程

本篇回答搭建训练框架时最直接的问题：项目怎样准备配置、建立环境、把环境交给 PPO 库、恢复训练状态、开始采样更新，以及把结果保存下来。逐段分析的是项目入口与数据组织；库函数在调用处说明接口，不继续拆库内的网络层创建、张量复制或反向传播实现。

> 阅读方式：沿训练步骤展开本项目自己的配置判断、数据处理、控制映射、奖励和状态恢复逻辑，首次调用处贴真实实现并解释变量与返回去向。Isaac Lab、USD、PyTorch、RSL-RL 的接口及只转调这些接口的薄封装，说明作用、输入、输出、状态变化和调用时机即可，不追踪库内部。

## 阅读目录

- [一 启动程序并准备两份配置](#一-启动程序并准备两份配置)
- [二 建立环境并取得第一批状态](#二-建立环境并取得第一批状态)
- [三 配置策略和 PPO，再创建 runner](#三-配置策略和-ppo再创建-runner)
- [四 续训时恢复什么状态](#四-续训时恢复什么状态)
- [五 调用 learn 开始训练](#五-调用-learn-开始训练)
- [六 一次控制交互的数据怎样流动](#六-一次控制交互的数据怎样流动)
- [七 收满一批后怎样调用 PPO 更新](#七-收满一批后怎样调用-ppo-更新)
- [八 更新结束后记录和保存什么](#八-更新结束后记录和保存什么)
- [九 独立回放怎样使用同一个策略](#九-独立回放怎样使用同一个策略)
- [十 从这些步骤检查自己的强化学习框架](#十-从这些步骤检查自己的强化学习框架)

## 一 启动程序并准备两份配置

### 1.1 先启动 Isaac Sim，再导入依赖它的模块

源码：[train.py](../../scripts/rsl_rl/train.py#L72)，第 72～85 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:72:85 -->

```python
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
```

当前步骤是在准备运行环境。`args_cli` 保存普通命令行参数；`hydra_args` 保存需要交给 Hydra 解析的配置覆盖。把后者交回 `sys.argv`，是让两种参数各走自己的处理入口。`app_launcher` 是负责启动应用的管理对象；`simulation_app` 是启动后的应用句柄，后面程序退出时通过它释放应用资源。

如果启用录像，`enable_cameras` 会提前设好。此时还没有创建机器人环境，也没有创建 Actor、Critic 或 Adam。理解这一点，可以避免把“应用窗口启动成功”误认为“训练已经开始”。

接口源码：[app_launcher.py:59](../../../IsaacLab/source/isaaclab/isaaclab/app/app_launcher.py#L59)。

<!-- api-contract: AppLauncher -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 启动后续 Isaac Lab 物理与场景模块所需的应用。 |
| 输入 | 解析后的命令行参数。 |
| 输出 | AppLauncher 对象，其 app 成员供脚本管理应用生命周期。 |
| 本步骤中的状态变化 | 建立应用上下文，随后才导入并创建任务环境。 |

### 1.2 main 接收环境配置和算法配置

<a id="first-call-update-rsl-rl-cfg"></a>

源码：[train.py](../../scripts/rsl_rl/train.py#L139)，第 139～157 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:139:157 -->

```python
@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.checkpoint_path is not None:
        agent_cfg.resume = True
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    from wheel_legged_gym_isaaclab.leg_control import require_leg_control_mode
    require_leg_control_mode(env_cfg.leg_control_mode, args_cli.require_leg_control_mode)
    print(f'[INFO]: Resolved leg control mode: {env_cfg.leg_control_mode}')
```

`env_cfg` 决定机器人、场景、控制周期、命令、观测和奖励。`agent_cfg` 决定网络与 PPO 的训练参数。两者都是配置对象，不是本批机器人的状态数据。`@hydra_task_config(...)` 负责把任务注册和覆盖后的两份配置交给 main，其接口已在第一篇说明。

第一条赋值用普通 CLI 更新 agent_cfg。先进入项目自己的配置覆盖逻辑，再带着返回值回到 main：

<!-- application-logic: update_rsl_rl_cfg -->

源码：[cli_args.py](../../scripts/rsl_rl/cli_args.py#L72)，第 72～105 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/cli_args.py:72:105 -->

```python
def update_rsl_rl_cfg(agent_cfg: RslRlBaseRunnerCfg, args_cli: argparse.Namespace):
    """Update configuration for RSL-RL agent based on inputs.

    Args:
        agent_cfg: The configuration for RSL-RL agent.
        args_cli: The command line arguments.

    Returns:
        The updated configuration for RSL-RL agent based on inputs.
    """
    # override the default configuration with CLI arguments
    if hasattr(args_cli, "seed") and args_cli.seed is not None:
        # randomly sample a seed if seed = -1
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        agent_cfg.seed = args_cli.seed
    if args_cli.resume is not None:
        agent_cfg.resume = args_cli.resume
    if args_cli.load_run is not None:
        agent_cfg.load_run = args_cli.load_run
    if args_cli.checkpoint is not None:
        agent_cfg.load_checkpoint = args_cli.checkpoint
    if args_cli.learning_rate is not None:
        agent_cfg.algorithm.learning_rate = args_cli.learning_rate
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    # set the project name for wandb and neptune
    if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        agent_cfg.wandb_project = args_cli.log_project_name
        agent_cfg.neptune_project = args_cli.log_project_name

    return agent_cfg
```

输入 agent_cfg 是已取得的算法配置对象，args_cli 保存解析后的命令行参数。函数直接修改 agent_cfg，最后返回的是同一个配置对象，不是重新建立策略或优化器。

seed 分支同时检查属性是否存在和是否给了值。显式 seed=-1 时才抽取一个整数种子，并写回 args_cli.seed 与 agent_cfg.seed；这一抽样只是选择实验种子，不是动作探索。普通 seed 值直接覆盖配置。

后续各个 `is not None` 分支逐项应用 resume、load_run、checkpoint、learning_rate、run_name 和 logger。None 表示命令行没有提供覆盖，因此保留已解析的值；这里不能用简单的真值判断替代所有这些条件。

load_run 选择运行目录，load_checkpoint 选择目录中的权重文件，resume 决定后续是否进入恢复路径。三者提供恢复所需的不同信息；此函数没有执行 runner.load，也没有在这里判断权重和车型是否匹配。

learning_rate 写的是算法配置中的初始值。续训时还要加载 Adam 并同步实际学习率，不能仅用这里的值判断恢复后的更新步幅。最后，只有选用 wandb/neptune 且给了项目名时，才更新相应日志项目字段。返回 main 后继续建立日志位置、环境与 runner。

返回后继续读 main 的赋值。`checkpoint_path` 非空就声明要续训；`scene.num_envs` 决定之后实际创建多少环境；`max_iterations` 决定本次 learn 的更新数量。它们改变的是不同层面的训练组织，不能把增加环境数当成增加迭代数。

`env_cfg.seed = agent_cfg.seed` 使环境采用本次训练种子。`env_cfg.sim.device` 选择仿真设备；后面实际数据张量要与它一致。`require_leg_control_mode()` 检查本次是否采用要求的控制语义，输入是当前和预期模式，正常返回 None，不相容时报错；实现已在第二篇选择驱动增益时展开。本任务的普通训练主线是 implicit_joint_position。

### 1.3 创建日志目录是为了绑定本次输入和输出

`log_root_path` 是该实验的日志根目录，`log_dir` 是当前运行的独立目录。后续检查点、env.yaml、agent.yaml 和控制约定都写到这一目录。目录名用于区分运行；真正判断权重身份还应核对完整路径、内容指纹和配套配置。

这一步结束时，项目已经准备好“用什么环境、用什么算法、在哪里保存结果”，接下来才构造可交互环境。

## 二 建立环境并取得第一批状态

### 2.1 用 env_cfg 创建任务

源码：[train.py](../../scripts/rsl_rl/train.py#L179)，第 179～185 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:179:185 -->

```python

# set the log directory for the environment
env_cfg.log_dir = log_dir

# create isaac environment
env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

```

`env` 接收 Gym 返回的任务环境对象。`args_cli.task` 选择已注册任务；`cfg=env_cfg` 传入刚才整理的环境配置；`render_mode` 只在需要录像时选 rgb_array。当前创建的是物理与任务对象，网络还没有参与。

| 调用 | 输入 | 输出及下一步用途 |
| --- | --- | --- |
| `gym.make(...)` | 任务 ID、env_cfg、渲染模式 | 环境对象 env；进入第二篇分析的 WheelLeggedVMCFlatEnv 构造与 _setup_scene。 |
| `env.unwrapped` | 当前可能已有包装的环境 | 原始任务对象；项目借它访问 common_step_counter 等任务状态。 |

### 2.2 首次 reset 前，先逐项核对恢复输入

这一步还没有创建 runner，因此还不能执行 `runner.load()`。项目先确定检查点身份，读取其配套配置，再检查这份策略是否可以放入刚创建的环境。这里的判断属于应用代码，沿实际调用展开。

#### 2.2.1 确定 resume_path，读取 source_config

源码：[train.py](../../scripts/rsl_rl/train.py#L190)，第 190～199 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:190:199 -->

```python
# save resume path before creating a new log_dir
if agent_cfg.resume:
    if args_cli.checkpoint_path is not None:
        resume_path = os.path.abspath(os.path.expanduser(args_cli.checkpoint_path))
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    source_config = validate_checkpoint_robot(resume_path)
```

显式 `checkpoint_path` 优先：先展开 `~`、转成绝对路径，再检查文件是否存在；否则使用已解析的运行目录与检查点规则取得路径。路径处理接口只做路径解析，Isaac Lab 的 `get_checkpoint_path()` 根据根目录和匹配规则返回检查点路径，这里不进入库的目录查找实现。

`resume_path` 是后面读配套文件、恢复课程和加载网络共同使用的同一个权重路径。`validate_checkpoint_robot()` 的完整项目逻辑已在[第一篇读取检查点配置处](01_环境构建与配置.md#first-call-validate-robot-contract-logic)展开：返回校验后的配置字典 `source_config`，不是模型张量，也没有恢复 Adam。

#### 2.2.2 从调用处进入动作与观测契约比较

<a id="first-call-resume-policy-contract"></a>

源码：[train.py](../../scripts/rsl_rl/train.py#L200)，第 200～201 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:200:201 -->

```python
from wheel_legged_gym_isaaclab.leg_control import require_matching_policy_contract
require_matching_policy_contract(source_config, env_cfg.to_dict())
```

<!-- application-logic: require_matching_policy_contract -->

源码：[leg_control.py](../../wheel_legged_gym_isaaclab/leg_control.py#L54)，第 54～69 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/leg_control.py:54:69 -->

```python
def require_matching_policy_contract(saved_cfg, target_cfg):
    """Equal tensor dimensions do not make virtual-leg and joint-angle policies compatible."""
    old = saved_cfg.get('leg_control_mode', 'explicit_vmc')
    new = target_cfg.get('leg_control_mode', 'explicit_vmc')
    require_leg_control_mode(old)
    require_leg_control_mode(new)
    if old != new:
        raise ValueError('Checkpoint action/observation contract differs; start a fresh policy for the new leg control mode')
    if new == DIRECT_JOINT_POSITION_MODE:
        for name in ('leg_joint_position_offset', 'leg_joint_position_scale', 'leg_joint_target_limit_margin',
                     'leg_joint_gravity_feedforward', 'observation_space', 'action_space'):
            a, b = saved_cfg.get(name), target_cfg.get(name)
            if name in ('leg_joint_position_offset', 'leg_joint_position_scale'):
                a, b = list(a) if a is not None else a, list(b) if b is not None else b
            if a != b:
                raise ValueError(f'Checkpoint direct-joint contract differs at {name}')
```

saved_cfg 是检查点配套配置，target_cfg 是本次配置。old、new 分别取两份控制模式；调用前面已展开的 require_leg_control_mode 先确认名称合法，再要求模式相同。相同的 27 维输入和 6 维输出不能证明虚拟腿语义与直接关节语义相容。

直接关节模式下还逐项比较四角偏置、四角尺度、限位余量、是否启用重力前馈，以及观测和动作空间。偏置/尺度先统一成列表，消除 tuple 与 list 的序列化表示差异；代码没有进行角度单位换算或替换数值。任一项目不同都会报错；正常返回 None。

回到 main，模式、动作映射和观测语义相容后，继续检查重置方案：

源码：[train.py](../../scripts/rsl_rl/train.py#L202)，第 202～203 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:202:203 -->

```python
from wheel_legged_gym_isaaclab.joint_reset import require_matching_reset_profile
require_matching_reset_profile(source_config, env_cfg.to_dict(), args_cli.allow_reset_pose_change)
```

<!-- application-logic: require_matching_reset_profile -->

源码：[joint_reset.py](../../wheel_legged_gym_isaaclab/joint_reset.py#L89)，第 89～93 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_reset.py:89:93 -->

```python
def require_matching_reset_profile(saved, current, allow_change=False):
    before, after = reset_profile(saved), reset_profile(current)
    if before != after and not (allow_change and before == CAD_ZERO_RESET and after == NOMINAL_JOINT_RESET):
        raise ValueError('Reset pose changed; qualify it and explicitly use --allow_reset_pose_change')
    return dict(source_profile=before, target_profile=after, changed=before != after)
```

saved 与 current 是两份环境配置，allow_change 来自本次显式实验选项。reset_profile 在第二篇构造重置模板时已展开，这里用它取得两份配置的重置语义再比较。

before 与 after 分别保存两份配置的重置 profile。默认要求相同；即使 allow_change 为真，也只允许 `CAD_ZERO_RESET → NOMINAL_JOINT_RESET`，即 `cad_zero_v0 → joint_target_nominal_v1` 这一方向，不能据此接受任意 profile 迁移。

返回值是含 source_profile、target_profile、changed 的字典，便于调用方记录迁移情况。当前 train.py 的这次调用没有接收返回值，主要利用检查失败时的异常。该函数不执行 reset、不改关节位置，通过后继续检查资产与求解器设置。

重置检查返回后，还要确认保存配置里的控制模式，以及本次被显式要求的模式一致：

源码：[train.py](../../scripts/rsl_rl/train.py#L204)，第 204～206 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:204:206 -->

```python
source_mode = source_config.get('leg_control_mode')
require_leg_control_mode(source_mode)
require_leg_control_mode(env_cfg.leg_control_mode, source_mode)
```

`source_mode` 是保存的模式字符串。`require_leg_control_mode()` 的完整实现已在[第二篇选择驱动增益处](02_环境执行与控制.md#first-call-synchronize-actuator-limits)展开；这里第一次检查保存模式是否合法，第二次要求当前模式与保存模式相同。它正常返回 None，作用是拒绝不相容输入，不会返回一个新控制器或替换模式。

#### 2.2.3 继续进入资产和外力设置的项目检查

<a id="first-call-resume-physics-contract"></a>

源码：[train.py](../../scripts/rsl_rl/train.py#L207)，第 207～217 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:207:217 -->

```python
from wheel_legged_gym_isaaclab.physics_contract import (
    require_matching_force_iteration_setting, require_matching_robot_asset_setting,
)
require_matching_robot_asset_setting(
    source_config, env_cfg.robot.spawn.usd_path,
    allow_change=args_cli.allow_physics_contract_change,
)
require_matching_force_iteration_setting(
    source_config, env_cfg.enable_external_forces_every_iteration,
    allow_change=args_cli.allow_physics_contract_change,
)
```

下面两个读取函数使用模块约定的字典键：`ROBOT_ASSET_FIELD = 'robot.spawn.usd_path'`，`EXTERNAL_FORCE_FIELD = 'enable_external_forces_every_iteration'`。它们是项目配置字段名，不是 USD 节点对象。

<!-- application-logic: require_matching_robot_asset_setting -->

源码：[physics_contract.py](../../wheel_legged_gym_isaaclab/physics_contract.py#L18)，第 18～26 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/physics_contract.py:18:26 -->

```python
def require_matching_robot_asset_setting(saved_config, current, allow_change=False):
    """Measured mass changes belong to a distinct physics experiment."""
    previous = checkpoint_robot_asset_setting(saved_config)
    if not isinstance(current, str) or not current.strip():
        raise ValueError('Current robot.spawn.usd_path must be nonempty')
    if (Path(previous).expanduser().resolve() != Path(current).expanduser().resolve()
            and not allow_change):
        raise ValueError('Robot asset physics contract changed; explicitly allow the '
                         'physics contract change for a new calibrated-mass experiment')
```

current 是本次 USD 入口路径。函数先从保存配置取 previous，再检查当前路径非空；两条路径都展开并规范化后比较，避免仅因路径写法不同而误判。

路径变化且没有 allow_change 时拒绝继续。这里检查的是资产选择约定；同一路径文件内容是否被改过，还要靠实验记录中的内容指纹检查。取得 previous 的辅助函数也属于项目逻辑：

<!-- application-logic: checkpoint_robot_asset_setting -->

源码：[physics_contract.py](../../wheel_legged_gym_isaaclab/physics_contract.py#L9)，第 9～15 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/physics_contract.py:9:15 -->

```python
def checkpoint_robot_asset_setting(saved_config, overrides=None):
    """Replay historical CAD masses unless a different asset is explicitly requested."""
    value = (overrides or {}).get(ROBOT_ASSET_FIELD,
                                saved_config['robot']['spawn']['usd_path'])
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Checkpoint requires a nonempty robot.spawn.usd_path')
    return value
```

saved_config 提供保存的 robot.spawn.usd_path，overrides 是可选的显式覆盖字典。优先使用覆盖项，否则取保存值；要求结果是非空字符串，返回该路径。函数不创建 Stage、不导入机器人，也不修改文件。

<!-- application-logic: require_matching_force_iteration_setting -->

源码：[physics_contract.py](../../wheel_legged_gym_isaaclab/physics_contract.py#L38)，第 38～43 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/physics_contract.py:38:43 -->

```python
def require_matching_force_iteration_setting(saved_config, current, allow_change=False):
    previous = checkpoint_force_iteration_setting(saved_config)
    current = checkpoint_force_iteration_setting({EXTERNAL_FORCE_FIELD: current})
    if previous != current and not allow_change:
        raise ValueError('External-effort physics contract changed; start fresh or explicitly '
                         'allow the physics contract change for a new experiment')
```

previous 是检查点保存的外力迭代设置，current 是本次请求值。两者都通过同一项目 helper 统一取得合法布尔值，再比较；不一致且没有显式允许物理变化时，恢复入口会报错。它只比较配置，不调用 PhysX 写属性。

<!-- application-logic: checkpoint_force_iteration_setting -->

源码：[physics_contract.py](../../wheel_legged_gym_isaaclab/physics_contract.py#L29)，第 29～35 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/physics_contract.py:29:35 -->

```python
def checkpoint_force_iteration_setting(saved_config, overrides=None):
    """Old weights used PhysX's False default; never silently replay them as True."""
    value = (overrides or {}).get(EXTERNAL_FORCE_FIELD,
                                saved_config.get(EXTERNAL_FORCE_FIELD, False))
    if not isinstance(value, bool):
        raise ValueError(f'{EXTERNAL_FORCE_FIELD} must be a boolean')
    return value
```

读取顺序是“显式 overrides → 检查点字段 → 旧检查点的 False 默认值”。缺字段时保留旧权重当年的设置，避免用今天源码的默认 True 静默改变其物理响应。

value 必须真的是 bool；整数 0/1 不作为布尔配置接受。返回的布尔值供兼容性比较，以及场景设置薄封装使用。到 `_setup_scene()` 中真正写入 USD 时，只需掌握那个库接口的输入、输出与调用时机，见第二篇 2.2。

这些检查都使用完整配置、统一的控制契约与显式变更标志，不根据失败工况选择另一套执行规则。

#### 2.2.4 读取课程时钟，立即进入它调用的校验函数

<a id="first-call-load-curriculum-clock"></a>

源码：[train.py](../../scripts/rsl_rl/train.py#L219)，第 219～225 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:219:225 -->

```python
# Set the command phase BEFORE RSL's wrapper performs its first reset.
# This opt-in clock leaves legacy continuation behavior unchanged.
from wheel_legged_gym_isaaclab.training_state import (
    make_curriculum_clock, load_curriculum_clock, clocked_update_count,
)
clock = load_curriculum_clock(resume_path) if agent_cfg.resume else None
completed_clock_updates = 0
```

恢复训练时调用 `load_curriculum_clock(resume_path)`；新训练先取得 None。`clock` 用于保存课程换算关系，`completed_clock_updates` 先设为 0，随后才可能从完整 Adam 推导已完成次数。先在这里进入读取函数。

<!-- application-logic: load_curriculum_clock -->

源码：[training_state.py](../../wheel_legged_gym_isaaclab/training_state.py#L16)，第 16～22 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/training_state.py:16:22 -->

```python
def load_curriculum_clock(checkpoint):
    """Legacy checkpoints retain their original iteration-driven clock."""
    import yaml
    path = Path(checkpoint).parent / 'params/curriculum_clock.yaml'
    if not path.exists():
        return None
    return make_curriculum_clock(**yaml.safe_load(path.read_text()))
```

checkpoint 是完整权重路径，path 定位同目录 params/curriculum_clock.yaml。没有这个配套文件时返回 None，供调用方保留旧检查点的计时路径；存在时读取 YAML 字典并展开成 make_curriculum_clock 的命名参数。

读取并不是校验完成，接着进入项目定义的字段检查：

<!-- application-logic: make_curriculum_clock -->

源码：[training_state.py](../../wheel_legged_gym_isaaclab/training_state.py#L6)，第 6～13 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/training_state.py:6:13 -->

```python
def make_curriculum_clock(offset_steps, steps_per_update, adam_steps_per_update):
    """A fresh policy may start at the current command phase, not curriculum zero."""
    values = (offset_steps, steps_per_update, adam_steps_per_update)
    if (any(type(value) is not int for value in values)
            or offset_steps < 0 or min(values[1:]) <= 0):
        raise ValueError('Curriculum clock requires nonnegative offset and positive integer rates')
    return dict(offset_steps=offset_steps, steps_per_update=steps_per_update,
                adam_steps_per_update=adam_steps_per_update)
```

offset_steps 是课程控制步偏移，steps_per_update 是一次 PPO 更新的控制步数，adam_steps_per_update 是一次 PPO 更新的优化器小批数。三者必须为真正的整数；偏移允许为 0，后两个速率必须为正。

返回时钟字典供下面的 clocked_update_count 与课程恢复共同使用。控制步与 Adam 小批数承担不同换算：当前 rollout 为 48 个控制步，PPO 每轮有 20 次 Adam 小批更新；8192 这个并行环境数不改变这两个时钟速率。

#### 2.2.5 回到 main：决定沿用、建立还是拒绝课程时钟

源码：[train.py](../../scripts/rsl_rl/train.py#L226)，第 226～235 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:226:235 -->

```python
if args_cli.curriculum_step_offset is not None:
    if agent_cfg.resume:
        raise ValueError('Set a fresh curriculum offset only once; resume inherits it.')
    clock = make_curriculum_clock(args_cli.curriculum_step_offset, int(agent_cfg.num_steps_per_env),
        int(agent_cfg.algorithm.num_learning_epochs) * int(agent_cfg.algorithm.num_mini_batches))
if clock is not None:
    expected = make_curriculum_clock(clock['offset_steps'], int(agent_cfg.num_steps_per_env),
        int(agent_cfg.algorithm.num_learning_epochs) * int(agent_cfg.algorithm.num_mini_batches))
    if clock != expected:
        raise ValueError('Saved curriculum clock does not match the PPO update rates')
```

`curriculum_step_offset` 只在首次声明课程偏移时使用；本次如果同时 resume，代码明确拒绝覆盖，避免续训时悄悄把命令难度拨回起点。非恢复的新训练则调用刚解析过的 `make_curriculum_clock()`，建立 offset、每更新控制步数和每更新 Adam 小批数。

有保存时钟时，`expected` 保留原 offset，但用本次 PPO 的 48 步和 5×4 个小批重建换算关系。字典比较失败就报错；这一步保证接下来从 Adam 计数换算课程进度时，没有把旧速率套在新配置上。没有时钟时整个条件分支跳过，保留旧计时路径。

#### 2.2.6 从 Adam 得到实际更新数，再设置环境课程

<a id="first-call-clocked-update-count"></a>

源码：[train.py](../../scripts/rsl_rl/train.py#L236)，第 236～239 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:236:239 -->

```python
if agent_cfg.resume:
    completed_clock_updates = clocked_update_count(
        torch.load(resume_path, map_location='cpu', weights_only=True), clock,
        allow_initial=os.path.basename(resume_path) == 'model_initial.pt')
```

`torch.load()` 只负责把检查点反序列化到 CPU，输出字典。实际怎样确认完整更新、怎样拒绝不一致的 Adam，由下面的项目函数决定；它不属于 PyTorch 库的默认计数规则。

<!-- application-logic: clocked_update_count -->

源码：[training_state.py](../../wheel_legged_gym_isaaclab/training_state.py#L25)，第 25～39 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/training_state.py:25:39 -->

```python
def clocked_update_count(saved, clock, *, allow_initial=False):
    """Count completed PPO updates from intact Adam, not the last loop label."""
    states = saved['optimizer_state_dict']['state']
    if allow_initial and not states and saved['iter'] == 0:
        return 0
    if not states or any('step' not in state for state in states.values()):
        raise ValueError('Clocked resume requires complete Adam steps')
    steps = {float(state['step']) for state in states.values()}
    if len(steps) != 1:
        raise ValueError('Clocked resume has inconsistent Adam steps')
    step = steps.pop()
    if (not math.isfinite(step) or step <= 0 or not step.is_integer()
            or step % clock['adam_steps_per_update']):
        raise ValueError('Clocked resume requires complete PPO updates')
    return int(step) // clock['adam_steps_per_update']
```

saved 是完整检查点字典，clock 保存项目定义的每次 PPO 更新对应多少 Adam 小批更新。states 取出 optimizer_state_dict 内每个参数的 Adam 状态；当前网络正常包含十七份，它们各自的 step 应一致。

唯一的初始模型例外需要同时满足 allow_initial=True、states 为空和 iter=0，此时返回 0。正常续训不能借这个分支绕过缺失的优化器状态。

接着检查每份状态都有 step，再把各 step 转成 float 放入集合。集合长度必须恰为 1，说明所有参数完成的更新次数相同。`steps.pop()` 取出唯一的值；它不是把十七份 step 相加。随后要求它有限、为正、是整数，且可被 adam_steps_per_update 整除，避免把半次 PPO 更新当成完整更新。

最后整除得到已完成的 PPO 更新数。以本条路线为例，17 份状态都为 step=84020，每次 PPO 使用 20 个 Adam 小批，得到 4201 次实际更新。它不使用 model_4200.pt 的文件名估算，也不把 8192 个并行环境乘进此处的更新计数。

返回的整数供训练入口恢复 runner 和课程时钟；该函数只读检查点，不修改权重或优化器。

源码：[train.py](../../scripts/rsl_rl/train.py#L240)，第 240～242 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:240:242 -->

```python
env.unwrapped.common_step_counter = (clock['offset_steps']
    + completed_clock_updates * clock['steps_per_update'])
print(f'[INFO]: Initial command curriculum step: {env.unwrapped.common_step_counter}', flush=True)
```

`common_step_counter` 是每个环境使用的课程控制步，不是所有环境的样本总数；计算不乘 8192，也不乘 decimation。`offset_steps` 规定课程起点，`completed_clock_updates×steps_per_update` 规定学习至今推进了多少控制步。完成这里以后才创建 RSL-RL wrapper，让它的首次 reset 使用正确阶段的命令。

### 2.3 把任务接口包装成 RSL-RL 能使用的接口

<a id="first-call-rsl-wrapper-constructor"></a>

源码：[train.py](../../scripts/rsl_rl/train.py#L256)，第 256～257 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:256:257 -->

```python
# wrap around environment for rsl-rl
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
```

这里再次给 `env` 赋值，保存的是包装器对象；底层仍然是前面那批物理环境。`agent_cfg.clip_actions` 是包装层的可选动作裁剪值，当前主线为 None，使环境能够接收原始动作并完成自己的执行限幅与饱和统计。

接口源码：[vecenv_wrapper.py:27](../../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py#L27)。

<!-- api-contract: RslRlVecEnvWrapper -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 将 Isaac Lab 的环境接口适配给 RSL-RL。 |
| 输入 | 环境 env，以及 clip_actions。 |
| 输出 | wrapper 对象，提供 num_envs、num_actions、device、step 和 get_observations 等接口。 |
| 本步骤中的状态变化 | 构造过程中主动 reset 一次环境，因此课程、种子和初始姿态应提前准备好。 |

这次 reset 进入项目的 `_reset_idx()`，恢复实际状态与参考；随后初始观测从 `_get_observations()` 得到。应用层需要理解这次调用发生在何时，以及它返回什么，不需要进入 wrapper 的空间重建细节。

## 三 配置策略和 PPO，再创建 runner

### 3.1 项目怎样声明网络与学习参数

源码：[rsl_rl_ppo_cfg.py](../../wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/agents/rsl_rl_ppo_cfg.py#L22)，第 22～54 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/agents/rsl_rl_ppo_cfg.py:22:54 -->

```python
class WheelLeggedVMCFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48  # 每个 env 采集 48 个控制步，再做一次 PPO 更新轮次。
    max_iterations = 1000  # 本次学习轮数默认值；CLI/恢复训练会另行解析。
    save_interval = 100  # runner 每隔多少 iteration 保存检查点，不是物理步数。
    experiment_name = "mine_wheel_legged_joint_pd_flat"
    # 键是 actor/critic 的输入角色，值是环境返回的观测组名；这里都读取同一帧 27 维。
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    # wrapper 不提前裁剪，保证环境仍能看到 raw action 并计算饱和惩罚；
    # 最终执行前环境自己限制到 [-1, 1]，None 不代表电机无限幅。
    clip_actions = None
    policy = RslRlPpoActorCriticCfg(
        # 高斯策略新建时的逐维标准差，不是动作尺度；续训通常加载检查点中的 std。
        # actor 输出均值，训练从分布采样；act_inference() 回放直接用均值。
        init_noise_std=0.2,
        # 这里只给隐藏层尺寸；RSL-RL 从 obs/action 空间推导输入 27 和输出 6。
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,  # critic 价值估计损失的权重，不是环境 reward 权重。
        use_clipped_value_loss=True,
        clip_param=0.2,  # PPO 新旧动作概率比的裁剪幅度，不是动作 [-1, 1] 限幅。
        entropy_coef=0.001,  # 鼓励探索的熵项系数；它不直接指定 std 的大小。
        num_learning_epochs=5,  # 同一批 rollout 数据重复学习 5 遍。
        num_mini_batches=4,  # 每遍分成 4 个小批量，不是 4 个并行环境。
        learning_rate=3.0e-4,
        schedule="adaptive",  # RSL-RL 根据 KL 与 desired_kl 调整学习率。
        gamma=0.99,  # 对未来每个控制步奖励的折扣。
        lam=0.95,  # GAE 优势估计的平滑系数。
        desired_kl=0.01,
        max_grad_norm=1.0,  # 裁剪网络参数梯度，和电机力矩上限无关。
    )
```

这是项目的应用配置，应结合训练步骤逐项理解。`num_steps_per_env=48` 表示每次更新前，每个环境提供 48 次交互；`save_interval` 表示隔多少迭代标签保存；`max_iterations` 是本次学习轮数，续训入口可能重新计算它。

`obs_groups` 把环境返回的 policy 观测组同时交给 Actor 和 Critic。当前每行 27 维，动作每行 6 维；组名的匹配是环境与算法的接口约定。`clip_actions=None` 控制包装层，不解除项目的角度、轮速与力矩限制。

`actor_hidden_dims`、`critic_hidden_dims` 声明各自的隐藏层大小；`activation` 选择激活函数。`init_noise_std` 是新建策略时的初始探索参数，续训通常加载检查点中的 std，不能据配置默认值判断当前真实探索。

`experiment_name` 参与日志根目录命名，不属于 Actor 输入。

算法字段的作用可以按一次学习过程分组：

| 字段 | 为哪一步提供设置 | 构建框架时要理解的含义 |
| --- | --- | --- |
| `gamma`、`lam` | 从奖励估计回报和优势 | 控制未来奖励与优势平滑的权重。 |
| `num_learning_epochs`、`num_mini_batches` | 使用已收集的一批数据更新 | 当前每批 5 轮、每轮 4 个小批，共 20 次 Adam 更新。 |
| `learning_rate`、`schedule`、`desired_kl` | 控制更新步幅 | adaptive 方案会改变实际学习率，续训要保留实际值。 |
| `clip_param` | 限制新旧策略概率比的更新目标 | 不裁剪电机动作，也不是关节角范围。 |
| `entropy_coef` | 探索相关目标 | 影响探索倾向；实际 std 仍要读取模型和测试数据。 |
| `value_loss_coef` | Critic 学习目标 | 属于算法损失权重，不是环境奖励权重。 |
| `use_clipped_value_loss` | Critic 目标计算方式 | 选择库的价值裁剪方案，不限制环境高度或电机动作。 |
| `max_grad_norm` | 优化器更新前的梯度保护 | 与电机力矩限幅属于不同层面。 |

<a id="value-loss-coef-application"></a>

#### `value_loss_coef` 怎样影响本项目的训练

上面的配置中，`tracking_ang_vel` 等字段属于环境的奖励配置，而 `value_loss_coef` 属于 `agent_cfg.algorithm`。两者在训练流程中生效的位置不同，理解这一区别，才能知道应该修改哪一份配置。

沿应用层的数据传递顺序看：

```text
rsl_rl_ppo_cfg.py 中的 RslRlPpoAlgorithmCfg
    → Hydra 合并本轮 agent 配置
    → train.py 中的 agent_cfg.algorithm
    → agent_cfg.to_dict() 与 configured_runner_dict()
    → runner_config["algorithm"]
    → OnPolicyRunner 创建原版 PPO
    → PPO.update() 在更新时读取 value_loss_coef
```

项目配置见 [rsl_rl_ppo_cfg.py](../../wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/agents/rsl_rl_ppo_cfg.py#L41)。`configured_runner_dict()` 是下一节要展开的本项目函数；在普通 PPO 分支中，它保留 `value_loss_coef`，并决定算法类名。库的 `OnPolicyRunner` 接收配置字典、建立策略与算法对象；在这里掌握输入配置流向即可，不需要继续展开库的构造细节。

环境先执行动作，再通过 `_get_rewards()` 计算每个环境的奖励。这些奖励连同观测、动作、结束标记等交给 storage；收满 48 步后，PPO 准备 `returns` 和 `advantages`，才进入网络更新。因此，修改环境 reward 会改变学习目标的原始数据；修改 `value_loss_coef` 则改变已经准备好的价值损失在总优化目标中的权重。

当前普通 PPO 的总目标可用下面的教学写法表示，对应本机库 [ppo.py:315](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py#L315)：

```python
# 教学写法：解释现有 PPO 总损失的三个部分，不是新增项目实现。
loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy
```

| 变量 | 来自哪里 | 在当前训练步骤中的作用 |
| --- | --- | --- |
| `surrogate_loss` | 已采集动作的旧概率、当前策略概率和优势 | 给 Actor 提供策略更新目标。 |
| `value_loss` | Critic 对旧观测的预测与 `returns` | 训练价值估计，供后续回报和优势计算使用。 |
| `value_loss_coef` | 本轮算法配置 | 调整价值损失在总优化目标中的权重。 |
| `entropy` | 当前高斯动作分布 | 描述该动作分布的微分熵，参与探索相关目标。 |
| `entropy_coef` | 本轮算法配置 | 决定熵项的权重，不直接指定六个 std 的数值。 |

例如一次交互已经得到某环境的 reward `0.12`。将 `value_loss_coef` 从 `1` 改成 `5`，不会把这条已采集的 reward 改成 `0.60`，也不会给高度或偏航误差另外乘五。假设当前 `value_loss=0.02`，它对总损失的这一项贡献才会由 `0.02` 变为 `0.10`。这个算例只解释权重关系；未来网络更新会改变行为和新采集的数据，不能据此声称后续训练轨迹保持不变。

本项目的 Actor 和 Critic 是两套参数，不共享隐藏层。因此价值损失不会直接对 Actor 的独立参数求出一份价值梯度。不过，当前 PPO 使用同一个 Adam 管理 Actor、Critic 和 std，并在更新前对全部参数实施一次整体梯度范数裁剪。改变价值损失权重，会改变 Critic 梯度在整体裁剪中的占比，进而可能影响两套网络的实际更新。这里应理解应用配置之间的联系，而不是把“五倍损失权重”直接等同于“五倍参数变化”或“五倍跟踪改善”；Adam 历史、自适应 LR 和后续梯度都会参与实际结果。

`max_grad_norm` 控制的是上述网络梯度范数，`clip_param` 参与 PPO 的概率比与价值裁剪；它们都不修改关节角限位、轮速目标限幅或电机力矩上限。后面分析动作执行时，仍应分别读取任务的动作包络和电机参数。

独立实验可通过 Hydra 覆盖现有字段：

```text
agent.algorithm.value_loss_coef=5.0
```

这是说明覆盖方式的例子，不表示配置文件的默认值已经改成 `5.0`，也不表示该值已通过训练验收。应用层实际使用哪个值，应读取本轮启动参数及保存的 `params/agent.yaml`，再核对创建后的 `runner.alg.value_loss_coef`。同样，续训加载检查点是否保留网络、六个 std、全部 Adam 与实际 LR，需要按第四节的加载流程分别验证；修改这个配置字段本身不要求重建优化器。

读日志还要区分“乘权后的总目标”与“单项原始损失”。当前库的 `value_function` 统计累加的是未乘 `value_loss_coef` 的 `value_loss`，见 [ppo.py:387](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py#L387)。所以系数改为 `5`，日志中的 `value_function` 不会仅因为这次配置改动就自动变为五倍。

应用层检验这类调整时，应同时看价值拟合、实际更新量、探索和独立回放误差。Critic 梯度占比小，或者某一组的价值拟合较差，都是值得验证的现象，不能单独证明它就是某个偏航工况未通过的原因。最终仍由同工况回放、完整验收范围和持续复验判断效果。

#### 先解析系数，再选择算法类

主训练入口在装配 runner_config 之前，先确定本次实际使用的两个系数。这里没有启动辅助训练；这一步是在确定应该实例化普通 PPO 还是显式请求的实验算法。

<a id="first-call-resolve-ppo-coefficients"></a>

<!-- application-logic: resolve_coefficient -->

源码：[joint_mean_guard.py](../../wheel_legged_gym_isaaclab/joint_mean_guard.py#L46)，第 46～59 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_mean_guard.py:46:59 -->

```python
def resolve_coefficient(checkpoint=None, requested=None):
    if requested is not None:
        return validate_coefficient(requested)
    if checkpoint is None:
        return 0.
    path = Path(checkpoint).parent / 'params/agent.yaml'
    if not path.is_file():
        return 0.
    algorithm = yaml.safe_load(path.read_text())['algorithm']
    name = algorithm.get('class_name', 'PPO')
    value = validate_coefficient(algorithm.get('joint_mean_guard_coef', 0.))
    if name not in ('PPO', 'JointMeanGuardPPO') or (name == 'PPO' and value != 0.):
        raise ValueError('Saved algorithm has an incompatible joint mean guard configuration.')
    return value
```

checkpoint 是恢复路径，requested 是命令行显式给出的系数。优先使用非 None 的 requested；没有显式值且没有检查点时返回 0。续训时则尝试读取权重旁 params/agent.yaml 的 algorithm 字典。

name 与 value 分别表示保存的算法名称与辅助系数。算法名称必须相容；普通 PPO 配置不能同时带非零辅助系数。返回的浮点数传回 main，作为 joint_mean_guard_coef。它不读误差轨迹，也不根据某个工况选择值。遇到 validate_coefficient 后，继续看本项目怎样检查这个输入：

<!-- application-logic: validate_coefficient -->

源码：[joint_mean_guard.py](../../wheel_legged_gym_isaaclab/joint_mean_guard.py#L34)，第 34～37 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_mean_guard.py:34:37 -->

```python
def validate_coefficient(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0. <= value <= 10.:
        raise ValueError('Joint mean guard coefficient must be finite and within [0, 10].')
    return float(value)
```

输入必须是有限的 int/float，并排除 bool，允许范围为 `[0,10]`。合法值统一转成 float 返回。这个 helper 只校验系数，不计算损失。

另一项高度辅助系数在 train.py 中以 resolve_height 别名调用；虽然函数同名，但来自另一个项目模块，范围与兼容条件不同：

<!-- application-logic: resolve_coefficient -->

源码：[joint_height_feedback.py](../../wheel_legged_gym_isaaclab/joint_height_feedback.py#L23)，第 23～35 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_height_feedback.py:23:35 -->

```python
def resolve_coefficient(checkpoint=None,requested=None):
    if requested is not None:
        return validate_coefficient(requested)
    if checkpoint is None:
        return 0.
    path=Path(checkpoint).parent/'params/agent.yaml'
    if not path.is_file():
        return 0.
    algorithm=yaml.safe_load(path.read_text())['algorithm']
    value=validate_coefficient(algorithm.get('joint_height_feedback_coef',0.))
    if value and algorithm.get('class_name')!='JointMeanGuardPPO':
        raise ValueError('Height feedback requires the qualified PPO variant')
    return value
```

它同样按“显式 requested → 配套 agent.yaml → 缺省 0”的顺序解析。读取到非零高度辅助系数时，保存的算法必须是 JointMeanGuardPPO；不相容时拒绝加载，避免把配置字段带给错误算法。返回值在 main 中保存为 joint_height_feedback_coef。

<!-- application-logic: validate_coefficient -->

源码：[joint_height_feedback.py](../../wheel_legged_gym_isaaclab/joint_height_feedback.py#L16)，第 16～20 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_height_feedback.py:16:20 -->

```python
def validate_coefficient(value):
    if (isinstance(value,bool) or not isinstance(value,(int,float))
            or not math.isfinite(value) or not 0.<=value<=.25):
        raise ValueError('Height feedback coefficient must be finite and within [0, .25]')
    return float(value)
```

这里允许的范围为 `[0,0.25]`，仍要求数值有限并排除 bool。返回给调用方的是 float。

当前主线解析得到的两个系数都是 0，height_model 因此为 None；main 不生成辅助几何模型，也不注册实验算法。带着这三个值回到下面的 configured_runner_dict，才能从代码确认本次选择的是普通 PPO。

### 3.2 在应用层装配 runner_config

<a id="first-call-configured-runner-dict"></a>

源码：[train.py](../../scripts/rsl_rl/train.py#L274)，第 274～278 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:274:278 -->

```python
runner_config = configured_runner_dict(agent_cfg.to_dict(), joint_mean_guard_coef,joint_height_feedback_coef,height_model)
if agent_cfg.class_name == "OnPolicyRunner":
    runner = OnPolicyRunner(env, runner_config, log_dir=log_dir, device=agent_cfg.device)
else:
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
```

`runner_config` 是交给库的普通字典，`runner` 是接收返回值的训练管理对象。`env` 提供交互接口；`log_dir` 指定保存位置；`agent_cfg.device` 指定算法设备。创建 runner 会建立网络、优化器和数据缓冲，但还没有开始 learn。

<!-- application-logic: configured_runner_dict -->

源码：[joint_mean_guard.py](../../wheel_legged_gym_isaaclab/joint_mean_guard.py#L62)，第 62～84 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_mean_guard.py:62:84 -->

```python
def configured_runner_dict(config, coefficient, height_coefficient=0., height_model=None):
    result = copy.deepcopy(config)
    coefficient = validate_coefficient(coefficient)
    from wheel_legged_gym_isaaclab.joint_height_feedback import validate_coefficient as validate_height
    height_coefficient=validate_height(height_coefficient)
    algorithm = result['algorithm']
    if algorithm.get('class_name', 'PPO') not in ('PPO', 'JointMeanGuardPPO'):
        raise ValueError('Joint mean guard supports the original PPO runner only.')
    if coefficient == 0. and height_coefficient == 0.:
        algorithm['class_name'] = 'PPO'
        algorithm.pop('joint_mean_guard_coef', None)
    else:
        algorithm['class_name'] = 'JointMeanGuardPPO'
        algorithm['joint_mean_guard_coef'] = coefficient
    if height_coefficient:
        if height_model is None:
            raise ValueError('Positive height feedback needs the current forward geometry model')
        algorithm['joint_height_feedback_coef']=height_coefficient
        algorithm['joint_height_feedback_model']=copy.deepcopy(height_model)
    else:
        algorithm.pop('joint_height_feedback_coef',None)
        algorithm.pop('joint_height_feedback_model',None)
    return result
```

这是项目在调用 OnPolicyRunner 前的配置装配逻辑。config 是算法配置字典；coefficient、height_coefficient 是前一步解析得到的两个显式系数；height_model 是可选的前向几何辅助模型。`copy.deepcopy(config)` 得到独立 result，避免下面的字典修改污染原始配置对象。

两个系数先通过本项目校验。algorithm 指向 result 中的算法子字典；现有 class_name 必须属于允许的两种算法，不能在不相容算法上附加这些字段。

当前两个系数都是零，因此进入第一个分支，明确写 `class_name='PPO'` 并删除 joint_mean_guard_coef。后面的 height_coefficient 分支同样走关闭路径，删除高度辅助系数和模型字段。清理这些字段是为了让普通库 PPO 获得它支持的构造参数。

非零分支描述可选辅助算法的配置接口：选择 JointMeanGuardPPO，并在需要高度辅助时要求提供模型、复制模型字典。当前训练没有走这个分支，本篇也不展开该实验算法的更新实现。判断实际算法应看返回的 algorithm.class_name 及显式系数，不能从仓库里存在一个类就推断它已生效。

返回 result 后，main 把它作为 OnPolicyRunner 的 train_cfg。至此应用层完成“本次究竟交给库哪个算法、哪些参数”的选择，下面才阅读库构造接口的作用和输入输出。

接口源码：[on_policy_runner.py:25](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py#L25)。

<!-- api-contract: OnPolicyRunner -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 连接环境、策略、PPO 算法、优化器和训练数据缓冲。 |
| 输入 | env、train_cfg、log_dir、device。 |
| 输出 | runner 对象。 |
| 本步骤中的状态变化 | 内部从真实观测与动作空间建立对象；应用层继续决定是否恢复检查点，再调用 learn。 |

### 3.3 创建后要核对的对象与变量

| 对象或变量 | 保存什么 | 在后续步骤中的用途 |
| --- | --- | --- |
| `runner.alg.policy` | Actor、Critic 和探索参数 | 根据观测产生动作/价值，更新后用于回放。 |
| `runner.alg.optimizer` | Adam 参数组与动量状态 | 执行网络更新，续训时恢复。 |
| `runner.alg.storage` | 当前批次交互与训练目标 | 收满后供 PPO 使用，随后被下一批覆盖。 |
| `runner.current_learning_iteration` | 库的迭代标签 | 日志与保存命名；实际更新次数还需结合 Adam 核对。 |

<a id="first-call-ppo-init-storage"></a>

接口源码：[ppo.py:118](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py#L118)。

<!-- api-contract: PPO.init_storage -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 建立一次更新所需的交互数据缓冲。 |
| 输入 | 训练类型 rl、N、48 个控制步、观测样例、动作形状 [6]。 |
| 输出 | None；建立 storage。 |
| 本步骤中的状态变化 | 观测通常保存为 [48,N,27]，动作保存为 [48,N,6]；当前批次尚未采集。 |

这里了解缓冲用途与尺寸即可。应用层需要让真实观测、配置组名与网络输入语义相符；即使仍是 27 维，把腿状态的含义换掉也会破坏旧权重的输入约定。

## 四 续训时恢复什么状态

### 4.1 先 load，再同步学习率与课程

源码：[train.py](../../scripts/rsl_rl/train.py#L282)，第 282～292 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:282:292 -->

```python
if agent_cfg.resume:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    runner.load(resume_path, load_optimizer=not args_cli.reset_optimizer)
    if args_cli.reset_optimizer:
        print("[INFO]: Loaded policy weights with a fresh optimizer state.")
    else:
        from wheel_legged_gym_isaaclab.training_state import restore_adaptive_learning_rate
        restored_lr = restore_adaptive_learning_rate(runner.alg)
        agent_cfg.algorithm.learning_rate = restored_lr
        print(f"[INFO]: Restored PPO adaptive learning rate from optimizer: {restored_lr}")
```

`resume_path` 指向完整检查点。`reset_optimizer` 为假时，load 同时恢复网络和 Adam；为真会走仅恢复网络的分支。当前连续优化保留完整优化器。

<a id="first-call-restore-adaptive-learning-rate-from-train"></a>

这里首次调用的同步函数属于项目的续训修复逻辑，应立即进入实现：

<!-- application-logic: restore_adaptive_learning_rate -->

源码：[training_state.py](../../wheel_legged_gym_isaaclab/training_state.py#L119)，第 119～132 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/training_state.py:119:132 -->

```python
def restore_adaptive_learning_rate(algorithm):
    """Keep PPO's KL scheduler aligned with the loaded Adam parameter groups.

    RSL-RL loads optimizer state but its separate learning_rate attribute keeps
    the constructor default. Its next adaptive update otherwise overwrites the
    successfully restored optimizer learning rate.
    """
    rates = [float(group['lr']) for group in algorithm.optimizer.param_groups]
    if not rates or any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError('Loaded optimizer learning rate must be finite and positive')
    if any(not math.isclose(rate, rates[0], rel_tol=1.e-12, abs_tol=0.) for rate in rates):
        raise ValueError('Scalar PPO scheduler cannot restore unequal optimizer learning rates')
    algorithm.learning_rate = rates[0]
    return rates[0]
```

`algorithm` 是 runner.alg，optimizer 已由上一行的 runner.load 恢复。rates 按参数组读取实际 lr，并转换为 Python float。它先检查列表非空以及每项有限、严格为正，再检查所有组的学习率一致。

一致性检查对应这里的调度约定：PPO 用单个 algorithm.learning_rate 控制自适应更新。若各参数组各有不同 lr，就不能用 rates[0] 冒充所有组；因此函数抛异常，而不是替调用方随意选一个值。

确认一致后，写 `algorithm.learning_rate = rates[0]`，并返回同一个 float。它没有重建 optimizer、清空 Adam 动量或修改网络。调用方用 restored_lr 接收结果，再写入 agent_cfg.algorithm.learning_rate，让后续调度、实际优化器与保存配置相互一致。


随后入口还把 runner 的迭代计数与环境课程计数对齐。网络、Adam 和课程是三个相关但不同的状态，不能仅凭 load 没报错断言全部恢复正确。

### 4.2 探索参数属于继续学习的状态

`policy` 保存六维动作分布。训练用观测生成分布并采样动作；确定性回放取均值。当前项目的四个腿动作标准差和两个轮动作标准差应分别观察，因为它们乘的物理尺度不同。

训练入口的 std 重置选项、轮 std 设置选项、std 下限是三种不同操作。是否提供了重置选项，必须查实际启动参数；当前续训不能因为看到 init_noise_std 就重新填一遍初始值。

先读保护安装之前的项目分支。传入 `reset_exploration_std` 才会一次性重填整个 std 或 log_std；未传入时保留恢复值。轮动作的单独重填由另一个显式选项触发：

源码：[train.py](../../scripts/rsl_rl/train.py#L323)，第 323～326 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:323:326 -->

```python
if args_cli.wheel_exploration_std is not None:
    from wheel_legged_gym_isaaclab.exploration import set_wheel_exploration_std
    set_wheel_exploration_std(runner.alg.policy, args_cli.wheel_exploration_std)
    print(f"[INFO]: Set wheel-only exploration std to {args_cli.wheel_exploration_std}; leg exploration retained.")
```

<a id="first-call-set-wheel-exploration"></a>

<!-- application-logic: set_wheel_exploration_std -->

源码：[exploration.py](../../wheel_legged_gym_isaaclab/exploration.py#L41)，第 41～52 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/exploration.py:41:52 -->

```python
def set_wheel_exploration_std(policy, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Wheel exploration std must be positive and finite.')
    parameter = getattr(policy, 'std', None)
    use_log = False
    if not isinstance(parameter, torch.Tensor):
        parameter = getattr(policy, 'log_std', None)
        use_log = True
    if not isinstance(parameter, torch.Tensor) or parameter.shape != (6,):
        raise ValueError('Unsupported six-action policy exploration parameter.')
    with torch.no_grad():
        parameter[[2, 5]] = math.log(value) if use_log else value
```

`policy` 是已经建立、续训时已经恢复的策略；`value` 是显式请求的轮动作标准差。函数先要求 value 有限且为正，再取得策略原来的 std 张量；如果没有直接 std，就尝试 log_std 并用 `use_log` 记录表示方式。最终张量必须为六维，防止把本项目动作列约定套到另一种策略上。

`parameter[[2,5]]` 只选择左右轮动作，四个腿列保持原值。直接 std 表示写入 value，log_std 表示写入其对数。`no_grad()` 表示这次赋值不构建训练计算图；它没有关闭参数后续的梯度，也没有创建一份新的 Parameter。

函数正常返回 None，结果体现为原策略张量的两项已改变。它不清空对应的 Adam 动量，也不注册持续回调，所以“一次重填标准差”和“每次更新后实施下限”需要分别考虑、分别记录。当前保留源检查点 std 的续训没有请求这项重填，整段条件分支跳过；不能因为仓库里有该函数就说它在每轮重置探索。

回到 main，接下来才安装每次优化器更新后的下限保护。

现在沿训练入口第一次安装探索保护的调用进入项目实现：

<a id="first-call-positive-exploration-from-train"></a>

源码：[train.py](../../scripts/rsl_rl/train.py#L330)，第 330～332 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:330:332 -->

```python
    exploration_guard = enforce_positive_exploration(
        runner.alg.policy, runner.alg.optimizer, args_cli.min_exploration_std,
        minimum_wheel_std=args_cli.min_wheel_exploration_std)
```

<!-- application-logic: enforce_positive_exploration -->

源码：[exploration.py](../../wheel_legged_gym_isaaclab/exploration.py#L6)，第 6～38 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/exploration.py:6:38 -->

```python
def enforce_positive_exploration(policy, optimizer, minimum_std=1.e-4, *, minimum_wheel_std=None):
    """Project learned noise onto its valid domain after every optimizer step.

    RSL-RL's scalar std is an unconstrained parameter and can cross zero when
    previously frozen outputs are reopened. Preserve parameter identity, the
    actor/critic weights and optimizer state; do not freeze any noise entry.
    Non-finite values are errors rather than silently repaired checkpoints.
    """
    if not math.isfinite(minimum_std) or minimum_std <= 0:
        raise ValueError('Exploration std floor must be positive and finite.')
    if minimum_wheel_std is not None and (
            not math.isfinite(minimum_wheel_std) or minimum_wheel_std < minimum_std):
        raise ValueError('Wheel exploration floor must be finite and at least the global floor.')
    parameter = getattr(policy, 'std', None)
    use_log = False
    if not isinstance(parameter, torch.nn.Parameter):
        parameter = getattr(policy, 'log_std', None)
        use_log = True
    if not isinstance(parameter, torch.nn.Parameter) or parameter.shape != (6,):
        raise ValueError('Unsupported six-action policy exploration parameter.')
    floor = math.log(minimum_std) if use_log else minimum_std
    if minimum_wheel_std is not None:
        floor = parameter.new_full((6,), floor)
        floor[[2, 5]] = math.log(minimum_wheel_std) if use_log else minimum_wheel_std

    def project_noise(*_args, **_kwargs):
        with torch.no_grad():
            if not torch.isfinite(parameter).all():
                raise ValueError('Policy exploration contains non-finite values.')
            parameter.clamp_(min=floor)

    project_noise()
    return optimizer.register_step_post_hook(project_noise)
```

policy 是已经构造并恢复的策略对象，optimizer 是同一个已恢复的 Adam。minimum_std 是全局正下限，minimum_wheel_std 是可选轮动作下限；两者先检查有限性与正值关系。

parameter 取得策略中实际可学习的 std Parameter。若该属性不是 Parameter，函数再尝试 log_std，并用 use_log 记录表示方式；要求最终 Parameter 恰有六个元素。这里选取的是已有参数对象，没有换成新的 Parameter，因此优化器保存的参数引用与动量仍对应原对象。

floor 在直接 std 表示下使用标准差本身，在 log_std 表示下取对数。若单独指定轮下限，就建立六维下限向量，只修改轮动作列 2、5 的下限。四个腿动作列与两个轮动作列的含义来自项目动作契约，不按高度、速度或转向筛选。

内部 `project_noise()` 是本项目的投影操作：先拒绝非有限参数，再在 no_grad 范围内执行原地 clamp_，把低于 floor 的元素抬到 floor。它不冻结参数，也不清空 Adam。调用一次 project_noise() 可立即建立下限，然后将同一回调挂到每次 optimizer.step 完成之后。

`optimizer.register_step_post_hook(project_noise)` 是 PyTorch 接口：输入是回调，返回可移除的 hook handle。main 用 exploration_guard 保存的正是这个句柄；它不是一份诊断字典。探索下限等实验信息会另外写入 precision_mode.yaml。

下限投影可能改变本来低于新下限的 std，因此“恢复后数值仍逐位一致”必须实测。当前路线在基础保护外还记录了统一的腿动作下限，恢复核验会检查网络、全部 Adam、六个 std 和实际 LR。该基础函数与当前 runtime 的下限记录要一起看，不能把全局下限误当成所有动作最终采用的同一个下限。

返回入口后，还会检查全部参数的 requires_grad，确认 Actor 各输出、共享特征、Critic 和探索参数都继续训练。

这一阶段结束后，才具备明确的学习起点：当前网络、完整 Adam、实际 LR、探索状态以及课程位置。

### 4.3 可选的初始检查点为何只属于新训练

<a id="first-call-save-random-initial-checkpoint"></a>

源码：[train.py](../../scripts/rsl_rl/train.py#L428)，第 428～433 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:428:433 -->

```python
if args_cli.save_initial_checkpoint:
    if agent_cfg.resume:
        raise ValueError('Initial random-weight checkpoint is only valid for a fresh policy')
    from wheel_legged_gym_isaaclab.training_state import save_random_initial_checkpoint
    save_random_initial_checkpoint(runner, os.path.join(log_dir, 'model_initial.pt'))
    print('[INFO]: Preserved fresh random weights and empty Adam before training.', flush=True)
```

先检查显式保存选项；如果本次是 resume，就拒绝把恢复后的模型冒充随机初始权重。新训练才把当前 runner 和本次日志目录中的目标文件路径传入下面的项目函数。

<!-- application-logic: save_random_initial_checkpoint -->

源码：[training_state.py](../../wheel_legged_gym_isaaclab/training_state.py#L42)，第 42～47 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/training_state.py:42:47 -->

```python
def save_random_initial_checkpoint(runner, destination):
    """RSL save() needs its writer initialized even before the first learn()."""
    if Path(destination).exists():
        raise FileExistsError('Preserve the existing initial model instead of overwriting it')
    runner._prepare_logging_writer()
    runner.save(str(destination))
```

项目自己的处理有两点：文件已存在就报错，保留原初始模型；首次 learn 尚未发生时，先准备保存依赖的日志 writer，再调用保存接口。`_prepare_logging_writer()` 输入隐含的 runner 状态，准备日志资源；`save(path)` 接收路径并写出检查点，库内实现不用继续展开。

这个 helper 没有产生随机权重、重建网络或执行 Adam；随机初始网络早在创建 runner 时已建立。它返回 None，外部结果是新文件，runner 仍是同一个对象。当前持续恢复路线没有开启该选项，所以不会每个短训块都建立或加载一份随机模型。理解了这个条件分支，再回到下一行的训练调用。

## 五 调用 learn 开始训练

### 5.1 应用层真正开始训练的调用

源码：[train.py](../../scripts/rsl_rl/train.py#L435)，第 435～452 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/train.py:435:452 -->

```python
# 用户／服务正常停止时，等一次完整更新结束再保存，避免信号打断原生物理线程。
from wheel_legged_gym_isaaclab.training_state import (
    CheckpointStopRequested, install_checkpoint_stop, install_episode_log_aggregation,
)
install_episode_log_aggregation(runner)
install_checkpoint_stop(runner, args_cli.stop_file, log_dir)
# run training
try:
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
except CheckpointStopRequested as error:
    print(f'[INFO]: Training stopped at a saved update boundary: {error}', flush=True)
except KeyboardInterrupt:
    path = os.path.join(log_dir, f"model_{runner.current_learning_iteration}_interrupted.pt")
    runner.save(path)
    print(f"[INFO]: Interrupted training checkpoint saved to {path}")
    raise
finally:
    env.close()
```

`agent_cfg.max_iterations` 传给 num_learning_iterations，表示本次 learn 要做多少次更新。应用层不是每个物理步调用 optimizer；它把交互对象和更新设置交给 runner 统一组织。

`init_at_random_ep_len=True` 用于打散开始阶段的回合计数；它不表示机器人已经真实存活了随机时长。因此分析初始失败应看实际物理时间，不能只看这个计数缓存。

这里还有两处本项目自己的管理逻辑需要在首次调用处展开。它们通过库提供的回调位置运行；下面只读项目回调如何处理数据，不进入 RSL-RL 的 log 实现。

<a id="first-call-episode-log-aggregation"></a>

<!-- application-logic: install_episode_log_aggregation -->

源码：[training_state.py](../../wheel_legged_gym_isaaclab/training_state.py#L81)，第 81～90 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/training_state.py:81:90 -->

```python
def install_episode_log_aggregation(runner):
    """Adapt logging in the project, without modifying third-party PPO code."""
    original_log = runner.log

    def log_events(locs, *args, **kwargs):
        records = aggregate_episode_logs(locs.get('ep_infos', []))
        # Do not replace the original list: the runner clears it after logging.
        return original_log(dict(locs, ep_infos=[records] if records else []), *args, **kwargs)

    runner.log = log_events
```

runner 是已经建立的训练管理对象。original_log 保存原来的日志可调用对象；内部 log_events 在写日志前先聚合 ep_infos，再构造一份新的局部变量字典传给原日志函数。最后把 runner.log 绑定到这层项目回调。

records 保存聚合结果。这里保留原始 ep_infos 列表对象，因为库会在记录后清空它；直接换掉原列表可能破坏这个生命周期。被调用的 aggregate_episode_logs 负责真正的项目数据整理，继续进入它：

<!-- application-logic: aggregate_episode_logs -->

源码：[training_state.py](../../wheel_legged_gym_isaaclab/training_state.py#L50)，第 50～78 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/training_state.py:50:78 -->

```python
def aggregate_episode_logs(records):
    """RSL's first-dictionary/mean-of-step-means logger loses sparse events.

    Environment episode fields now contain one element per ended episode.
    Concatenation weights every event equally; ordinary scalar diagnostics keep
    their per-step means. Missing groups remain absent rather than becoming NaN.
    """
    import torch
    result = {}
    for key in dict.fromkeys(key for record in records for key in record):
        values = [torch.as_tensor(record[key]).detach().flatten() for record in records if key in record]
        values = [value for value in values if value.numel()]
        if not values:
            continue
        values = torch.cat([value.to(device=values[0].device, dtype=torch.float32) for value in values])
        result[key] = values.mean()
        if key == 'Episode_Termination/numerical':
            # RSL prints ratios with four decimals; a rare failure can round to
            # zero. Preserve the integer count so the native guard still sees it.
            result['Episode_Count/numerical'] = values.sum()
        if key == 'Episode_Termination/unsafe_contact_or_orientation':
            group = 'all'
        elif key.startswith('Episode_Termination/') and key.endswith('_unsafe'):
            group = key.split('/', 1)[1][:-len('_unsafe')]
        else:
            continue
        result[f'Episode_Count/{group}_ended'] = values.new_tensor(values.numel())
        result[f'Episode_Count/{group}_unsafe'] = values.sum()
    return result
```

records 是一次学习批次收集的回合日志字典列表。第一层循环从所有字典取键的并集，而不是只取第一条记录的键，因此后面才出现的稀疏失败分组不会丢失。

values 为具有当前键的记录取出张量，断开梯度并展平，再排除空项。拼接后每个结束回合都占一个元素，求均值时按真实事件数加权；缺少某个分组的记录不被补成零。对于原本就是每步标量诊断的数据，则继续聚合这些标量。

数值异常还会保留整数总数，避免日志四位小数把稀少异常显示成 0。对不安全结束相关键，group 标记分组名称，额外生成 ended 和 unsafe 计数，分别对应样本数和失败数。函数返回 result 字典，由上面的回调交给原日志接口。

这些操作改变的是日志聚合方式，不会重新计算 reward 或改变环境 reset。

<a id="first-call-cooperative-checkpoint-stop"></a>

<!-- application-logic: install_checkpoint_stop -->

源码：[training_state.py](../../wheel_legged_gym_isaaclab/training_state.py#L97)，第 97～116 行。

<!-- source-excerpt: wheel_legged_isaaclab/wheel_legged_gym_isaaclab/training_state.py:97:116 -->

```python
def install_checkpoint_stop(runner, stop_file, log_dir):
    """Save actor, critic and Adam at the completed-update logging boundary.

    RSL-RL calls log after assigning current_learning_iteration. Do not signal
    native PhysX threads to implement an ordinary user/service stop.
    """
    if stop_file is None:
        return
    marker, destination = Path(stop_file), Path(log_dir)
    original_log = runner.log

    def log_and_check(locals_, *args, **kwargs):
        original_log(locals_, *args, **kwargs)
        if marker.exists():
            path = destination / f'model_{runner.current_learning_iteration}_interrupted.pt'
            runner.save(str(path))
            print(f'[INFO]: Cooperative stop checkpoint saved to {path}', flush=True)
            raise CheckpointStopRequested(str(path))

    runner.log = log_and_check
```

stop_file 是调用方提供的标记路径；为 None 时直接返回，不安装回调。否则 marker 与 destination 分别保存标记路径和实验日志目录，original_log 保留当前日志函数，因此可以与前面的日志聚合回调衔接。

log_and_check 先完成原日志，再检查 marker 是否存在。只有确实存在时才构造完整更新边界的检查点路径，调用 runner.save，并抛出项目定义的 CheckpointStopRequested，由 train.py 的对应 except 接住。标记不存在就正常返回，后续学习继续。

这是显式停止请求的保存机制，不根据奖励、工况或某个训练轮数自行结束总任务。当前短训块正常完成后还需要独立复验与下一块决策；具备此接口不代表当前有人创建了停止标记。

`install_episode_log_aggregation(runner)` 与 `install_checkpoint_stop(runner, stop_file, log_dir)` 是项目的运行管理工具：前者整理回合日志，后者提供在完整更新边界保存退出的能力。stop_file 是传入的信号路径；存在这样的接口，不等于当前有人发出了停止要求。

`CheckpointStopRequested` 对应保存边界的受控返回；KeyboardInterrupt 分支构造 path 保存中断检查点；finally 调用 env.close 释放当前环境。这里描述入口控制流，当前后台训练是否继续应以其实际进程和记录为准。

接口源码：[on_policy_runner.py:61](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py#L61)。

<!-- api-contract: OnPolicyRunner.learn -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 重复执行采样、回报准备、PPO 更新、日志与保存。 |
| 输入 | 本次更新数 num_learning_iterations，以及是否随机初始化回合计数。 |
| 输出 | None；训练状态、日志与检查点随运行更新。 |
| 本步骤中的状态变化 | 策略、Critic、探索参数和 Adam 在更新阶段改变；采样阶段保持同一批策略参数。 |

### 5.2 从一次 learn 看完整时间顺序

```text
读取当前观测
  -> 用当前策略采样动作
  -> 环境执行一个控制步，返回新观测、奖励和结束标记
  -> 保存这次交互
  -> 重复，直到每个环境收集 48 步
  -> 准备回报与优势
  -> 用这批数据完成 PPO 更新
  -> 记录、按间隔保存
  -> 开始下一批
```

这里每一轮更新包含很多控制交互。并行 N 台机器人意味着每次交互产生 N 条样本；它们可以在不同步结束回合，同一批次不要求所有环境同步 reset。

## 六 一次控制交互的数据怎样流动

### 6.1 观测进入库，动作返回应用环境

<a id="first-call-ppo-act"></a>
<a id="first-call-actor-act-helpers"></a>

接口源码：[ppo.py:129](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py#L129)。

<!-- api-contract: PPO.act -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 根据当前观测准备本次动作与更新需要的旧策略信息。 |
| 输入 | obs：含 policy 组的 TensorDict，当前该组为 [N,27]。 |
| 输出 | 原始采样 actions [N,6]。 |
| 本步骤中的状态变化 | 缓存动作、旧价值、旧概率、分布均值/std 和动作前观测，供后续交互记录。 |

`obs` 是本步动作前的状态。`actions` 的四个腿列和两个轮列按项目控制约定解释，不能只因它们同属六维就用同一物理单位。环境收到 raw action 后的执行映射、关节 PD 与重力前馈已在第二篇分析。

### 6.2 wrapper 怎样连接两种 step 返回格式

接口源码：[vecenv_wrapper.py:152](../../../IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py#L152)。

<!-- api-contract: RslRlVecEnvWrapper.step -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 将动作交给 Isaac 环境，再把反馈整理成 RSL-RL 接口。 |
| 输入 | actions [N,6]；若配置了包装层裁剪则先处理。 |
| 输出 | obs、rewards [N]、dones [N]、extras。 |
| 本步骤中的状态变化 | 把环境的 terminated/truncated 合成 done；按配置在 extras 提供 time_outs；观测用 TensorDict 表达。 |

| 名称 | 表示什么 | 后面交给谁 |
| --- | --- | --- |
| `obs` | 环境刚返回的新观测 | 下一步 Actor，以及本批最后的价值估计。 |
| `rewards` | 刚才动作造成的本步反馈 | PPO 交互记录和回报估计。 |
| `dones` | 哪些环境在本步结束回合 | 隔断跨回合的回报递推。 |
| `extras` | 超时标记、回合日志等附加信息 | 算法补偿和训练日志；不会自动成为 Actor 输入。 |

发生自动 reset 时，reward/done 仍对应旧回合最后一步，新观测可以来自新回合。应用层应保持这种时间对应，不能把观测和动作错移一帧。

### 6.3 保存反馈后，才开始下一次动作

<a id="first-call-ppo-process-env-step"></a>

接口源码：[ppo.py:142](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py#L142)。

<!-- api-contract: PPO.process_env_step -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 把当前奖励、结束标记和之前缓存的动作前信息组成完整交互。 |
| 输入 | 新 obs、rewards、dones、extras。 |
| 输出 | None。 |
| 本步骤中的状态变化 | 按当前库约定处理超时补偿并写入 storage，随后准备下一步临时记录。 |

当前库对 time_outs 的补偿使用已缓存的 transition.values；应用层要提供正确的结束类型，不能把真实物理失败标成超时。当前普通 PPO 没有 RND 内在奖励。

<a id="first-call-rollout-transition"></a>
<a id="first-call-rollout-add-transitions"></a>

| 数据记录 | 内容与形状 | 使用时刻 |
| --- | --- | --- |
| 单步 transition | 观测 [N,27]、采样动作 [N,6]、奖励/结束/旧价值、旧分布信息 | 这次交互补齐后写入整批。 |
| 整批 storage | 时间维 48 × 环境维 N × 各字段特征维 | 收满后交给本次 PPO 更新。 |

`add_transitions(transition)` 的作用是保存当前交互，输入是补齐的单步记录，返回 None；`Transition.clear()` 准备下一步临时记录，也返回 None。此处只需掌握数据保存先后和字段属于哪一帧，不展开库内拷贝与隐藏状态处理。

## 七 收满一批后怎样调用 PPO 更新

### 7.1 先准备回报与优势

<a id="first-call-ppo-compute-returns"></a>

接口源码：[ppo.py:171](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py#L171)。

<!-- api-contract: PPO.compute_returns -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 用最后观测和已收集的一批奖励准备训练目标。 |
| 输入 | 本批最后一次交互返回的 obs；批内 rewards/dones/values 已在 storage。 |
| 输出 | None；storage 获得 returns 与 advantages。 |
| 本步骤中的状态变化 | Critic 提供最后状态的价值；结束掩码避免把新回合价值串到旧回合。 |

`returns` 是 Critic 要拟合的目标；`advantages` 表达某个动作相对于原价值估计的好坏。它们不是机器人某个位置误差，也不由下位机 PD 直接计算。应用层提供一致的奖励和结束标记，PPO 库负责准备这些目标。

### 7.2 再调用算法更新

接口源码：[ppo.py:178](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/algorithms/ppo.py#L178)。

<!-- api-contract: PPO.update -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 用刚收满的一批交互更新策略、价值和探索参数。 |
| 输入 | 内部已准备好的 storage，以及 PPO 配置；调用本身无应用层额外参数。 |
| 输出 | 供 runner 记录的平均损失字典；同时修改模型与优化器状态。 |
| 本步骤中的状态变化 | 当前每批执行 5×4=20 次 Adam；库按配置处理策略裁剪、价值损失、熵、KL 调度与梯度裁剪。 |

理解这一步需要看更新什么，而不是继续拆库实现：

| 更新相关量 | 输入信息 | 应用层读结果时关注什么 |
| --- | --- | --- |
| 策略目标 | 旧观测、旧动作概率、当前重新计算的概率、advantages | 独立回放的行为是否改善，是否出现输出饱和。 |
| Critic 目标 | 旧观测与 returns | 价值损失是否正常，有无数据或量纲异常。 |
| 探索相关项 | 当前动作 std 与 entropy_coef | 腿/轮探索是否足够，是否影响稳定性。 |
| KL 调度 | 新旧动作分布、desired_kl | 实际 LR 是否跳变或过小，恢复时是否一致。 |
| Adam | 上述损失产生的参数梯度 | 参数/状态是否有限，完整 step 是否连续。 |

环境的奖励权重与这里的算法损失权重分属两层。改善某个跟踪目标时，应先定位误差来自命令、动作、动力学还是学习，再选择对应参数；不能只看总 reward 增大就认为可验收。

### 7.3 一批到底有多少数据

| 实际环境数 N | 48 步交互总数 | 4 个 minibatch 的单批样本数 | 本批 Adam 调用数 |
| --- | --- | --- | --- |
| 4096 | 196608 | 49152 | 20 |
| 8192 | 393216 | 98304 | 20 |

同样是 200 次更新，环境数改变会改变实际采样总量。当前按用户明确设置直接使用 8192 个环境，不再安排容量试训，也不自动降为 4096。启动后仍需核对实际创建的环境数与采样张量形状，并记录内存、显存和吞吐量；这些运行核验不等同于另起容量测试，也不能用“配置写了 8192”代替实际创建成功的证据。

## 八 更新结束后记录和保存什么

<a id="first-call-rollout-storage-clear"></a>

`RolloutStorage.clear()` 为下一批准备写入位置，输入是当前 storage，返回 None。它不会删除网络或 Adam。下一次 learn 循环继续使用刚更新的策略采样。

`runner.current_learning_iteration` 随更新推进，控制台记录 loss、奖励与近期结束回合统计。回报均值取决于哪些回合结束；真实高度、速度、偏航误差应从物理数据和固定回放另行计算。

`runner.save(path)` 的输入是检查点目标路径和可选 infos，输出 None，写出网络、优化器、迭代信息等内容。配套 env.yaml、agent.yaml 与控制约定由应用层保存，它们是解释本模型的必要输入。保存细节与实际计数核对见第四篇。

一次短训块结束是评估和决策节点。继续、修复或回退都应引用具体检查点和独立数据，保留当前模型与优化器，不能把文件名相同当成同一模型。

## 九 独立回放怎样使用同一个策略

源码：[play.py](../../scripts/rsl_rl/play.py#L463)，第 463～474 行。

<!-- source-excerpt: wheel_legged_isaaclab/scripts/rsl_rl/play.py:463:474 -->

```python
print(f"[INFO]: Loading model checkpoint from: {resume_path}")
# OnPolicyRunner 属于 RSL-RL，不是 LAB：此处借它构建 actor/critic 并加载权重。
# 不调用 runner.learn()，所以回放没有 PPO 参数更新。
if agent_cfg.class_name == "OnPolicyRunner":
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
else:
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
runner.load(resume_path)

# get_inference_policy() 切到 eval 模式，返回 actor.act_inference 的可调用句柄。
# policy(obs) 得到动作均值 [N, 6]，不进行训练时的高斯采样。
policy = runner.get_inference_policy(device=env.unwrapped.device)
```

`resume_path` 指定要检验的模型。`policy` 接收 runner.get_inference_policy 返回的可调用对象，后面把观测传给它获得动作。这里没有调用 learn，因此不能把回放时动作变化理解为参数仍在学习。

接口源码：[on_policy_runner.py:326](../../../IsaacLab/_isaac_sim/kit/python/lib/python3.11/site-packages/rsl_rl/runners/on_policy_runner.py#L326)。

<!-- api-contract: OnPolicyRunner.get_inference_policy -->

| 项目 | 说明 |
| --- | --- |
| 作用 | 取得用于回放的策略调用接口。 |
| 输入 | 推理设备 device。 |
| 输出 | 可调用 policy：输入观测，返回确定性动作均值 [N,6]。 |
| 本步骤中的状态变化 | 使用当前已加载策略，不执行 PPO 更新。 |

训练采样动作包含探索噪声，确定性回放使用均值；两种数据回答的问题不同。独立回放应冻结权重，固定命令、种子、配置与测量窗口，再分别判断存活、姿态、高度、速度和 yaw。第四篇继续分析评估流程。

## 十 从这些步骤检查自己的强化学习框架

### 10.1 按调用顺序追踪一条主线

```text
main(env_cfg, agent_cfg)
  -> 创建并配置环境
  -> 准备首次 reset 所需课程与初始状态
  -> RslRlVecEnvWrapper
  -> runner_config -> OnPolicyRunner
  -> 可选恢复模型、Adam、实际 LR、课程与探索
  -> runner.learn
       -> 动作 -> 项目环境 -> 观测/奖励/结束
       -> 收集 48 步 -> 回报/优势 -> PPO 更新
       -> 日志与保存 -> 下一批
  -> 独立回放 -> 判断改动效果 -> 选择下一步
```

### 10.2 新变量出现时，问清它的生命周期

`env_cfg/agent_cfg/runner_config` 属于配置准备；`env/runner/policy` 是运行对象；`obs/actions/rewards/dones` 是一轮轮交互数据；`storage` 保存当前批；模型、Adam、std、实际 LR 跨批次延续；日志和检查点记录已经发生的结果。按这种生命周期阅读，比把所有变量都当作“训练参数”更容易判断应该在哪一步修改。

### 10.3 用证据判断训练进行到了哪一步

应用启动、场景创建、第一次 reset、首次原生交互、第一次 Adam 更新、检查点写入、独立回放完成是不同的事实。检查训练进度时，应明确看到了哪一步的运行证据。奖励上升只说明某种累计目标变化，最终是否可用仍要回到实际误差与持续复验。

