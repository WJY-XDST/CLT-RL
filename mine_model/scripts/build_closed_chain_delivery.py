"""Bundle the versioned model, corresponding project files and review evidence."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[2]
MINE=ROOT/'mine_model'
asset=ROOT/'wheel_legged_isaaclab/assets/robots/mine'
manifest=json.loads((asset/'manifest.json').read_text())
export=Path(manifest['source_export'])
stamp=datetime.now().astimezone().strftime('%Y%m%d_%H%M%S')
out=MINE/'deliveries'/f'closed_chain_{stamp}'
out.mkdir(parents=True,exist_ok=False)

def copy(source,destination):
    if source.is_dir():
        shutil.copytree(source,destination,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    else:
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)

for name in ('mine_closed_chain.usd','mine_robot.urdf','model.json','closures_cad.json','meshes'):
    copy(export/name,out/'model'/name)
for directory in ('wheel_legged_isaaclab/wheel_legged_gym_isaaclab','wheel_legged_isaaclab/assets/robots/mine','mine_model/scripts','mine_model/config','mine_model/source','mine_model/inputs','mine_model/docs'):
    copy(ROOT/directory,out/'project'/directory)
copy(export,out/'project/mine_model/exports'/export.name)
files=['run_python.sh','train_mine.sh','test_mine_drive.sh','mine_model/run.sh','mine_model/README.md',
       'wheel_legged_isaaclab/scripts/rsl_rl/train.py','wheel_legged_isaaclab/scripts/rsl_rl/cli_args.py',
       'wheel_legged_isaaclab/scripts/test_mine_drive.py','wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
       'wheel_legged_isaaclab/scripts/diagnose_mine_support.py',
       'wheel_legged_isaaclab/tests/test_five_bar_vmc.py','wheel_legged_isaaclab/tests/test_mine_acceptance.py']
for name in files:copy(ROOT/name,out/'project'/name)
for name in ('parameter_audit','bench_implicit_wheels_drives','trial_200_current','trial_200_implicit_drives'):
    for source in (MINE/'results'/name).rglob('*'):
        if source.is_file() and source.suffix in ('.json','.md','.png'):
            copy(source,out/'evidence'/name/source.relative_to(MINE/'results'/name))
# A portable manifest points to the included source copy, not this workstation.
pm=out/'project/wheel_legged_isaaclab/assets/robots/mine/manifest.json'
portable=json.loads(pm.read_text());portable['source_export']='mine_model/exports/'+export.name
pm.write_text(json.dumps(portable,indent=2)+'\n')
tracked=['wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/__init__.py',
         'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py',
         'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env_cfg.py',
         'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/agents/rsl_rl_ppo_cfg.py',
         'wheel_legged_isaaclab/scripts/rsl_rl/train.py']
(out/'tracked_changes.patch').write_bytes(subprocess.check_output(['git','diff','--binary','--',*tracked],cwd=ROOT))
(out/'README.md').write_text('''# 自定义轮腿闭链模型及修改文件\n\nmodel/mine_closed_chain.usd 是完整闭链模型，可在 Isaac Sim 5.1 打开。包含 15 个刚体、14 个树关节和 4 个额外 revolute 闭合铰链。\nmodel/mine_robot.urdf 是配套生成树模型，必须连同 meshes/ 使用。URDF 本身不能保存额外四个闭环，闭链约束以 USD 与 closures_cad.json 为准。\n\n## 关节与几何\n\n主动关节顺序 LB_joint、LL_joint、LW_joint、RB_joint、RL_joint、RW_joint。LB/LL/RB/RL 均以 base_link 为父节点和电机角度参考，q=0 为 CAD 默认装配。\n右侧闭合 RL3↔RS 末端孔、RL2↔RB 中间孔；左侧镜像。闭合铰链只约束机械结构，没有主动驱动。\n坐标 X 前、Y 左、Z 上。主环 94.5/112.5 mm，轮轴放大 20/9；默认虚拟腿长 259.1946 mm。\n\n## 工程修改\n\nproject/ 按 CLT-RL 相对路径保存源码快照和资产，需要现有 Isaac Lab / Isaac Sim 运行环境。它不是包含全部依赖的独立仓库。\nfive_bar_vmc.py 提供新 FK、解析雅可比与虚功力矩映射；mine_env_cfg.py 配置新几何和物理控制参数；wheel_legged_vmc_flat_env.py 负责实际关节映射和控制分支。\n原 vmc.py 的串联公式仅用于 Legacy。默认任务使用新模型，Legacy 任务保留旧机器人。\nCAD USD 不含 importer drive。工程初始化在两个轮关节上建立 PhysX 隐式速度 Drive；腿关节仍用 VMC 力矩，被动关节零驱动。轮阻尼 0.5，腿/轮 effort 上限 10 Nm。\n\n## 训练和验收\n\n自动训练入口 project/mine_model/scripts/auto_train_mine.py：256 环境，从新模型 pilot checkpoint 续接到总计 1000 轮，之后每 1000 轮评估与调整。\n先延长站立学习，再逐渐加入前进、后退和偏航。调整记录存入每轮 overrides.json；质量、惯量、杆长和关节连接不会由自动优化随意修改。\n放宽的 sim2sim 准备标准见 summarize_mine_trial.py：无重置的 12 秒回放，速度/偏航 MAE max(绝对容差,25%目标)，高度/腿长 MAE 30 mm，摆角 MAE 0.1 rad，机体角度 p95 10°。独立扰动种子复测通过后保存 TorchScript、ONNX 和 sim2sim_contract.json。\n闭链与悬空驱动通过不代表自由基座平衡通过。第 200 轮试训策略仍 0/27 存活，证据保存在 evidence/。当前包不含已收敛策略。\n\n## 文件说明\n\ntracked_changes.patch 仅用于审阅已有跟踪文件相对 HEAD 的修改；新增文件在 project/，补丁不能单独完成全部安装。\nFILE_MANIFEST.json 列出交付文件与 SHA256；原 CAD 文件与原导出没有被覆盖。\nproject/mine_model/docs/ 的图文说明记录首次接入过程；本 README 的隐式轮驱动和自动训练设置是后续最新调整。\n''')
inventory={}
for p in sorted(out.rglob('*')):
    if p.is_file():inventory[str(p.relative_to(out))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(out/'FILE_MANIFEST.json').write_text(json.dumps({'generated_at':datetime.now().astimezone().isoformat(),'asset_version':manifest['version'],'files':inventory},indent=2)+'\n')
archive=out.with_suffix('.zip')
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(out.rglob('*')):
        if p.is_file():z.write(p,str(Path(out.name)/p.relative_to(out)))
with zipfile.ZipFile(archive) as z:
    if z.testzip() is not None:raise RuntimeError('Archive CRC verification failed')
print(json.dumps({'directory':str(out),'zip':str(archive),'files':len(inventory)+1,'zip_bytes':archive.stat().st_size},indent=2))
