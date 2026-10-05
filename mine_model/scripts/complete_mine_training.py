"""Finalize accepted weights and explanations; power off only after every gate.

Invoked by the training supervisor after held-out and 30-second export replay.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import zipfile
import sys
from summarize_mine_trial import assess,plot_optimization_history
from build_training_record import build_record,verify_record
from optimization_journal import sync_journal

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'wheel_legged_isaaclab'))
from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--state',type=Path,required=True)
    p.add_argument('--shutdown',action='store_true')
    args=p.parse_args()
    state=json.loads(args.state.read_text())
    if state['state']!='accepted':raise RuntimeError('No accepted training state')
    output=args.state.parent
    export=Path(state['export_directory'])
    acceptance=json.loads((export/'acceptance.json').read_text())
    paths=[Path(acceptance[k]).parent for k in ('training_assessment','heldout_assessment')]
    export_replay=paths[0].parent/'export_replay'
    reports=[assess(p) for p in paths+[export_replay]]
    if not all(r['passed'] and r['resets']==0 for r in reports):raise RuntimeError('Replay acceptance failed')
    criteria=reports[-1]['acceptance_criteria'];percent=100*criteria['relative_error']
    requirements=output/'final_delivery_requirements.json'
    if requirements.exists():
        requested=json.loads(requirements.read_text())
        if criteria['relative_error']>requested['relative_error_target']:
            raise RuntimeError('Acceptance is looser than the requested completion target')
    if any(r['acceptance_criteria']!=criteria for r in reports):
        raise RuntimeError('Acceptance criteria differ between verification stages')
    acceptance.update(criterion=reports[-1]['criterion'],acceptance_criteria=criteria)
    (export/'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
    manifests=[json.loads((p/'manifest.json').read_text()) for p in paths+[export_replay]]
    expected_names={'stand','forward_0p3','forward_0p6','reverse_0p3','reverse_0p6',
                    'spin_left','spin_right','height_low','height_high'}
    for report,manifest in zip(reports,manifests):
        if manifest.get('command_scale',1.)!=1.:
            raise RuntimeError('Diagnostic command amplitude cannot satisfy full acceptance')
        if (report['total_trials']!=27 or len(manifest['cases'])!=9 or len(set(manifest['seeds']))!=3
                or {c['name'] for c in manifest['cases']}!=expected_names):
            raise RuntimeError('Missing full nine-condition three-seed acceptance')
    # Replay names are implementation metadata; commands and seed independence
    # must match across the three runs before promoting a checkpoint.
    if any(m['cases']!=manifests[0]['cases'] for m in manifests[1:]):
        raise RuntimeError('Validation commands differ between runs')
    if set(manifests[0]['seeds']) & set(manifests[1]['seeds']):
        raise RuntimeError('Held-out reset seeds overlap training replay')
    if manifests[1]['seeds']!=manifests[2]['seeds']:
        raise RuntimeError('Extended replay does not use held-out reset seeds')
    checkpoint=Path(acceptance['checkpoint'])
    if len({r['checkpoint_sha256'] for r in reports})!=1 or reports[0]['checkpoint_sha256']!=sha(checkpoint):
        raise RuntimeError('Checkpoint differs between validation runs')
    if manifests[2]['seconds_per_case']<30:
        raise RuntimeError('Missing extended running replay')
    for name in ('policy.pt','policy.onnx','sim2sim_contract.json'):
        if not (export/name).is_file() or not (export/name).stat().st_size:raise RuntimeError('Missing export: '+name)
    contract=json.loads((export/'sim2sim_contract.json').read_text())
    runtime=contract['runtime_config']
    for name in ('policy.pt','policy.onnx'):
        if sha(export/name)!=contract['export_sha256'][name]:raise RuntimeError('Export changed after replay: '+name)
    if len({m['asset_sha256'] for m in manifests})!=1 or sha(runtime['robot']['spawn']['usd_path'])!=manifests[0]['asset_sha256']:
        raise RuntimeError('Robot asset differs from validated replay')
    asset_chain, asset_manifests=verify_asset_chain(runtime['robot']['spawn']['usd_path'])
    if any(m.get('sha256') for m in asset_manifests.values()):
        if (contract.get('asset_chain_sha256') != asset_chain or
                any(m.get('asset_chain_sha256') != asset_chain for m in manifests)):
            raise RuntimeError('Missing or changed complete collision asset snapshot')
    tire_variant=next((m for m in asset_manifests.values() if m.get('overlay_sha256')), None)
    stop_variant=next((m for m in asset_manifests.values() if m.get('confirmed_stop_pairs')), None)
    if stop_variant:
        geometry_path=Path(stop_variant.get('cad_nonpenetration_report', ''))
        if not geometry_path.is_file() or sha(geometry_path)!=stop_variant.get('cad_nonpenetration_sha256'):
            raise RuntimeError('Missing verified original CAD stop contact evidence')
        geometry_report=json.loads(geometry_path.read_text())
        if not geometry_report.get('passed') or geometry_report.get('asset_chain_sha256')!=asset_chain:
            raise RuntimeError('Stop/shank CAD contact geometry validation failed or changed')
    physical_fields=('leg_model','leg_control_mode','leg_joint_stiffness','leg_joint_damping',
        'leg_effort_limit','wheel_effort_limit','five_bar_geometry','wheel_radius','wheel_track_width',
        'wheel_control_mode','wheel_damping','kp_theta','kd_theta','kp_l0','kd_l0','feedforward_force',
        'action_scale_theta','action_scale_l0','action_scale_vel','l0_offset','l0_ref_min','l0_ref_max',
        'height_reference_blend','height_feedback_gain','height_feedback_max_adjustment','obs_scales','decimation','robot')
    physical_fields+=('theta0_ref_min','theta0_ref_max','leg_length_height_offset',
        'forward_support_speed_threshold','forward_support_height_margin','forward_support_min_leg_length')
    resolved=[json.loads((p/'resolved_env.json').read_text()) for p in paths+[export_replay]]
    for cfg in resolved:
        if any(cfg[k]!=runtime[k] for k in physical_fields) or cfg['sim']['dt']!=runtime['sim']['dt']:
            raise RuntimeError('Controller differs between validation runs and export')
    implicit=runtime.get('leg_control_mode')=='implicit_joint_reference'
    if implicit and any(m.get('heading')!='constant requested yaw rate; no added heading feedback' for m in manifests):
        raise RuntimeError('Fixed-command acceptance contains extra heading feedback')
    tutorial=ROOT/('mine_model/docs/开链转闭链图文教程_v3' if implicit else 'mine_model/docs/开链转闭链图文教程_v2')
    qa=json.loads(tutorial.with_suffix('.qa.json').read_text())
    if not qa.get('visually_verified'):raise RuntimeError('Tutorial has not been visually verified')
    for suffix in ('.docx','.pdf'):
        if sha(tutorial.with_suffix(suffix))!=qa['sha256'][suffix]:raise RuntimeError('Tutorial differs from reviewed copy')
    delivery=output/'final_delivery';delivery.mkdir(exist_ok=True)
    journal=sync_journal(output)
    shutil.copy2(journal,delivery/journal.name)
    shutil.copytree(output/'optimization_journal',delivery/'optimization_journal',dirs_exist_ok=True)
    if (output/'optimization_notes.json').exists():
        shutil.copy2(output/'optimization_notes.json',delivery/'optimization_notes.json')
    if requirements.exists():shutil.copy2(requirements,delivery/requirements.name)
    for suffix in ('.docx','.pdf'):shutil.copy2(tutorial.with_suffix(suffix),delivery/(tutorial.name+suffix))
    collision_note=ROOT/'mine_model/docs/碰撞简化与机械限位说明_20261004.md'
    if collision_note.is_file():
        shutil.copy2(collision_note, delivery/collision_note.name)
        evidence=ROOT/'mine_model/results/collision_simplification_20261004'
        for name in ('stop_candidate_and_sweep.png', 'collision_ab_errors.png', 'environment_scale.png',
                     'collision_comparison.json', 'environment_capacity_report.json'):
            if (evidence/name).is_file():
                shutil.copy2(evidence/name, delivery/name)
    for name in ('policy.pt','policy.onnx','sim2sim_contract.json','acceptance.json'):
        shutil.copy2(export/name,delivery/name)
    if stop_variant:
        shutil.copy2(geometry_path, delivery/'cad_nonpenetration.json')
    shutil.copy2(checkpoint,delivery/checkpoint.name)
    source_paths=[
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/five_bar_vmc.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/config_overrides.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/exploration.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/checkpoint_initialization.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/precision_finetune.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/mine_env_cfg.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env_cfg.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/agents/rsl_rl_ppo_cfg.py',
        'wheel_legged_isaaclab/scripts/rsl_rl/train.py','wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
        'wheel_legged_isaaclab/scripts/rsl_rl/play.py',
        'wheel_legged_isaaclab/scripts/rsl_rl/keyboard_control.py',
        'mine_model/scripts/play_keyboard.py',
        'mine_model/docs/键盘可视化控制说明.md',
        'mine_model/run.sh',
        'mine_model/scripts/auto_train_mine.py','mine_model/scripts/complete_mine_training.py',
        'mine_model/scripts/build_training_record.py',
        'mine_model/scripts/optimization_journal.py',
        'mine_model/config/tracking_acceptance.json',
        'mine_model/scripts/benchmark_env_scale.py',
        'mine_model/scripts/build_training_collision_variant.py',
        'mine_model/scripts/build_protected_leg_collision_variant.py',
        'mine_model/scripts/build_stop_contact_variant.py',
        'mine_model/scripts/compare_collision_variants.py',
        'mine_model/scripts/summarize_collision_capacity.py',
        'mine_model/scripts/audit_leg_stop_geometry.py',
        'mine_model/scripts/audit_stop_silhouette.py',
        'mine_model/scripts/audit_live_stop_geometry.py',
        'mine_model/scripts/export_collision_viewer.py',
        'mine_model/scripts/templates/collision_viewer.html',
        'wheel_legged_isaaclab/scripts/test_mine_leg_stops.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/mine_collision_filters.py',
        'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/asset_integrity.py',
        'mine_model/scripts/summarize_mine_trial.py',
        'mine_model/scripts/audit_policy_feedback.py','mine_model/scripts/trial_feedback_initialization.py',
        'mine_model/scripts/trial_wheel_precision.py',
        'mine_model/scripts/build_smooth_tire_variant.py',
        'mine_model/scripts/build_guide.py','mine_model/scripts/build_feedback_figure.py',
        'mine_model/scripts/build_collision_figure.py',
        'mine_model/README.md','mine_model/results/auto_training_implicit_20pct_20261004/README.md',
        'mine_model/run.sh','mine_model/config/closures.json','mine_model/config/hole_candidates.json',
        'mine_model/scripts/run_isaac.py','mine_model/scripts/export_project_model.py',
        'mine_model/scripts/find_closure_axes.py','mine_model/scripts/inspect_joints.py',
        'mine_model/scripts/audit_joint_authoring.py','mine_model/scripts/audit_model_parameters.py',
        'mine_model/scripts/five_bar.py','mine_model/scripts/test_five_bar.py',
    ]
    for relative in source_paths:
        dest=delivery/'code'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,dest)
    shutil.copytree(ROOT/'wheel_legged_isaaclab/assets/robots/mine',delivery/'robot_assets',dirs_exist_ok=True)
    audit=ROOT/'mine_model/results/parameter_audit'
    if audit.is_dir():shutil.copytree(audit,delivery/'parameter_and_joint_audit',dirs_exist_ok=True)
    shutil.copytree(checkpoint.parent/'params',delivery/'training_params',dirs_exist_ok=True)
    shutil.copy2(paths[0].parent/'overrides.json',delivery/'training_overrides.json')
    history=delivery/'optimization_history';history.mkdir(exist_ok=True)
    for file in plot_optimization_history(output):shutil.copy2(file,history/file.name)
    for file in output.glob('round_*/*'):
        if file.name.startswith('overrides') and file.suffix=='.json':
            dest=history/file.parent.name/file.name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,dest)
    for file in output.glob('round_*/replay/assessment.json'):
        dest=history/file.parents[1].name/file.name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest)
    for file in output.glob('round_*/frozen_actor_audit.json'):
        dest=history/file.parent.name/file.name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest)
    for file in output.glob('midpoint_*/replay/assessment.json'):
        dest=history/file.parents[1].name/file.name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,dest)
    for name in ('reward_saturation_audit_20261004.json','wheel_flutter_audit_20261004.json','wheel_collision_outline_audit.json','代码检查与提前回放.md'):
        if (output/name).exists():shutil.copy2(output/name,history/name)
    for file in output.glob('feedback_*audit_*.json'):
        shutil.copy2(file,history/file.name)
    for index,event in enumerate(state.get('interventions',[])):
        for key in ('initialization_manifest','angle_initialization_manifest','baseline_assessment','precision_trial_manifest','comparison_baseline_assessment','collision_variant_manifest','assessment','env_scale_benchmark_manifest','collision_comparison_manifest','cad_nonpenetration_manifest'):
            path=event.get(key)
            if path:
                source=Path(path)
                if not source.is_file():raise RuntimeError('Missing initialization evidence: '+path)
                destination=history/f'intervention_{index:03d}'/(key+source.suffix)
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,destination)
        if event.get('checkpoint_sha256'):
            source=Path(event['checkpoint'])
            if not source.is_file() or sha(source)!=event['checkpoint_sha256']:
                raise RuntimeError('Initialization checkpoint differs from recorded source')
    (history/'interventions.json').write_text(json.dumps(state.get('interventions',[]),indent=2,ensure_ascii=False)+'\n')
    (delivery/'runtime_config.json').write_text(json.dumps(runtime,indent=2,ensure_ascii=False)+'\n')
    report=reports[-1]
    lines=['# 新闭链机器人训练更改说明','',f'完成时间：{datetime.now().astimezone().isoformat()}',
        f'验收 checkpoint：{checkpoint}',f'SHA256：{sha(checkpoint)}','',
        f"最终训练环境数：{state.get('num_envs', 256)}。早期阶段使用256个环境；后续规模切换及同源策略吞吐、内存测试见optimization_history/interventions.json。环境数影响每轮采样量，吞吐提升不等于收敛速度同比提升。",'',
        '## 已修正的问题','',
        '旧高度惩罚在误差超过 31.6 mm 后达到每秒 1 的上限，无法区分轻微偏低与严重蹲塌。新配置使用更宽的高度惩罚范围，并随实测情况调整权重。',
        '增加独立低机身高度失败判定：低于 0.22 m 持续 0.15 秒即终止，避免接触力短暂降低掩盖蹲塌。站立时增加虚拟腿绝对摆角约束。',
        'FK 与原始电机方向经实际力矩脉冲及轮轴位置对照核验；保留 base_link 角度参考、左右镜像、27 维观测和 6 维动作。',
        '固定偏航指令回放不再额外注入航向纠偏，保证实际测试指令与验收目标一致。对已经站稳但轮速参考未能执行的候选，自动流程可启用 standing_wheel_tracking 惩罚；该项仅作用于静止速度和偏航指令，实际系数见 training_overrides.json。',
        '运动课程中首次探索调整对轮动作单独设置0.08探索标准差，保留腿动作分布和优化器；减少站立阶段探索下降导致运动样本不足的问题。该噪声只用于训练，最终导出和验收使用确定性的策略均值。',
        '原速度平方惩罚在约0.183 m/s误差后触及每秒1的单项裁剪。2000轮回放中速度项约43%、静止轮速项约93%的首回合采样饱和。新增速度、偏航、静止轮速独立裁剪范围，随权重同步调整，保留停止与反向运行之间的惩罚区别；这改变训练奖励，不改变观测或执行器契约。',
        '2400轮仍出现约11至16度直行倾角，姿态惩罚在约8度后也达到旧上限3；同步扩大姿态负界。连续跟踪惩罚扩大后，根据50 Hz、gamma=0.99的约2秒有效折扣时域提高终止成本，避免立即倒地比持续纠错更便宜。具体权重和负界见各轮配置记录。',
        'high_height训练日志阈值由旧机体固定0.19 m改为当前高度命令区间的75%分位，新模型0.28–0.32 m区间对应0.31 m；该字段只用于记录终止工况。',
        *(['3000轮曲线显示轮速参考接近22–25 Hz快速摆动。固定2799轮权重的阻尼1.0及0.5对照均完整存活，但没有解决全部误差且轮速跟踪变差，因此正式阻尼保留2.0。已有action_rate包含轮动作，action_smooth仅覆盖腿；从3200阶段将action_rate由-0.01改为-0.5，并新增独立action_rate_penalty_clip、随权重覆盖六维动作全部变化范围。该项仅改变训练奖励，执行器契约不变。'] if (output/'wheel_flutter_audit_20261004.json').exists() else []),
        '自动规划器达到偏航或静止轮速惩罚上限后，不再将未变的参数记为调整。若完整回放全部存活且轮参考单步变化RMS超过1 rad/s，可逐轮增强已有action_rate，最大绝对权重2；没有该实测依据则保持参数继续学习。诊断统计与验收误差分开保存，验收容差保持不变。',
        *([f"精度阶段单独调整PPO算法参数：{json.dumps(state['agent_overrides'],ensure_ascii=False)}。轮探索标准差从运动切换时0.08增长到约0.40–0.57，现有平滑惩罚已降低动作振荡但偏航仍超标，因此将熵系数0.01降为0.001；保留当前权重、优化器和std，由后续学习逐步收敛。最终PPO参数以training_params/agent.yaml为准。"] if state.get('agent_overrides') else []),
        *(['后续精度训练采用wheel-head-only模式：冻结actor共享特征及腿角、腿长输出行0/1/3/4和对应探索参数；只更新轮输出行2/5、轮探索std及critic。清除冻结行的Adam历史，避免零梯度仍被旧动量移动；轮与critic历史保留。该模式仅改变训练梯度，导出仍为普通27维观测、6动作的MLP，推理无额外控制器。训练参数与独立对照证据见training_params/precision_mode.yaml和optimization_history。'] if state.get('wheel_head_only_finetune') else []),
        *(['检查CAD网格发现轮胎碰撞外轮廓只有18边，半径60 mm时每个平面中点比圆轮低约0.912 mm。另存96边圆轮碰撞USD叠加层，保持半径、轴向宽度、原CAD显示网格、15刚体质量/COM/惯量、14个树关节及4个闭合关节。原18边轮胎碰撞关闭，新碰撞外形使用96边及hullVertexLimit=255，避免默认64顶点的凸包简化。原文件保留。固定已训练5500权重，从18边改用96边后首轮27/27全部通过；5500权重的独立12秒复测同样通过，但30秒延长回放存在偏航超标，未标记达标、未关机。随后在96边资产上继续轮输出精度训练，使用实际失败的独立或延长回放报告调整下一轮，保持原验收阈值。最终三阶段验收均使用本次选中权重及同一碰撞变体，没有添加额外航向反馈。资产与来源哈希见robot_assets中的变体清单，底层CAD源USD哈希也由交付门控校验。'] if tire_variant else []),
        *(['原策略持续反向响应时，另建受控续训初始化：actor/critic首层速度命令列6及Adam一阶矩取负；actor腿摆角输出行0、3改为25%原输出与75%CAD零位名义角的混合，只清除对应行优化器历史。其它权重和探索std保留，回放和最终模型使用正常27维输入，没有运行时指令翻转。修改来源与哈希见optimization_history/interventions.json及独立初始化清单。'] if any(e.get('angle_initialization_manifest') for e in state.get('interventions',[])) else []),
        *([f"高度参考混合系数为{runtime['height_reference_blend']}，高度误差反馈增益为{runtime['height_feedback_gain']}。采用轮阻尼2.0及温和高度外环的初始化对照曾27/27完整存活、所有姿态p95不超过5度，但跟踪精度仍超标，仅作为续训起点。最终验收配置见runtime_config.json；MuJoCo需同步腿长参考混合、高度修正和物理限幅。"] if any(e.get('angle_initialization_manifest') for e in state.get('interventions',[])) else []),
        '质量、惯量、几何和正式腿轮力矩上限保持 CAD 新模型配置；30 Nm 测试只属于诊断，不用于正式训练。',
        *(['底盘与普通连杆碰撞简化、保留的限位块和小腿接触面、闭链求解树及动态joint映射、真实接触限制、同权重误差对照和并行环境实测见随包《碰撞简化与机械限位说明_20261004.md》。验收同时记录并核对全部USD来源层及共享网格哈希，避免只核对顶层USD。'] if collision_note.is_file() else []),
        '自动训练先试训1000轮，后续每200轮回放。通过站立精度验收后加入运动；每轮依据实测误差调整奖励和课程。站稳时保留optimizer/std，实际倒地才恢复探索；最佳checkpoint先比较存活数和时长，再比较全部指标的最大及平均归一化误差。初始化分支单独命名并记录真实迭代偏移，避免恢复旧分支较大编号的checkpoint。完整变更见各轮 overrides.json。','',
        '## 最终控制参数','',
        (f"腿部执行为 implicit_joint_reference：虚拟腿目标经五连杆逆解转换为原始电机角目标，PhysX force Drive 联合求解闭环；关节 Kp/Kd 为 {runtime['leg_joint_stiffness']} / {runtime['leg_joint_damping']}，限幅 {runtime['leg_effort_limit']} Nm，轮速阻尼 {runtime['wheel_damping']}。旧显式 VMC 的角度、腿长 PD 和重力前馈不再作为腿电机力矩施加。MuJoCo 复现须同步采用该执行器契约。" if implicit else
         f"摆角 Kp/Kd：{runtime['kp_theta']} / {runtime['kd_theta']}；腿长 Kp/Kd：{runtime['kp_l0']} / {runtime['kd_l0']}；轮速阻尼：{runtime['wheel_damping']}。接入时为 50/3、900/90、0.5，后续调整依据短测和逐轮回放。"),
        '新模型初期 1000 次追加训练后仍为 0/27 完成，探索 std 由 0.3 降到约 0.04。20 组摆角增益短测中，Kp=10 比 Kp=50 减轻塌腿但没有通过站立；据此建立低刚度训练对照，并将新模型 PPO 熵系数由 0.001 提高到 0.01。最终值以随包 training_params 和 runtime_config 为准。','',
        '## 验收口径','',report['criterion'],'',
        '先验证 9 工况和 3 个种子，再用独立 83 84 85 种子复测，最后同一权重每工况运行 30 秒。失败回合没有作为稳态样本；三个验收报告均通过且零重置。',
        f"相对误差目标为 {percent:g}%。速度绝对容差 {criteria['speed_absolute_m_s']:g} m/s，偏航 {criteria['yaw_absolute_rad_s']:g} rad/s，轮速 {criteria['wheel_absolute_rad_s']:g} rad/s；上述指标采用绝对容差与相对阈值中的较大值。高度和腿长为目标的 {percent:g}%，腿角容差 {criteria['relative_error']*criteria['leg_angle_design_range_rad']:g} rad，机身倾角 p95 不超过 {criteria['body_p95_deg']:g} 度，稳态最低高度至少为指令的 {100*criteria['minimum_height_fraction']:g}%。",'',
        '|工况|速度 MAE m/s|偏航 MAE rad/s|高度 MAE mm|腿长 MAE mm|腿角 MAE rad|轮速 MAE rad/s|机身角 p95 度|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for condition in report['conditions']:
        metrics={k:max(t['metrics'][k] for t in condition['trials']) for k in condition['trials'][0]['limits']}
        vals=[metrics['speed_mae_m_s'],metrics['yaw_mae_rad_s'],1000*metrics['height_mae_m'],1000*metrics['leg_length_mae_m'],metrics['leg_angle_mae_rad'],metrics['wheel_speed_mae_rad_s'],metrics['body_p95_deg']]
        lines.append('|'+condition['name']+'|'+'|'.join(f'{x:.4f}' for x in vals)+'|')
    replay_command=['./run_python.sh','wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
                    '--headless','--robot','mine','--checkpoint',str(checkpoint),
                    '--seconds','30','--seeds','83','84','85',
                    '--overrides_json',str(delivery/'training_overrides.json'),
                    '--output',str(output/'reproduction_replay')]
    replay_script=delivery/'replay_accepted.sh'
    replay_script.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(ROOT))+'\n'+shlex.join(replay_command)+'\n')
    replay_script.chmod(0o755)
    lines+=['','## 运行与复现','',
        '复现须加载随包的 training_overrides.json；仅给 play.py 指定 checkpoint 会使用当前默认控制增益，与验收配置可能不一致。以下命令复现全部 9 工况、3 种子和最终控制参数。','',
        '```bash',shlex.join(['bash',str(replay_script)]),'```','',
        'TorchScript 和 ONNX 的观测、动作、VMC、坐标和控制频率定义见 sim2sim_contract.json。Isaac Lab 验收通过之后，新模型 MuJoCo sim2sim 仍需独立进行；本交付不宣称已完成这一步。','',
        '## 证据位置','',*[f'- {p}' for p in paths+[export_replay]],'']
    (delivery/'训练更改说明.md').write_text('\n'.join(lines))
    (delivery/'README.md').write_text('\n'.join([
        '# 新闭链模型验收交付', '',
        f'通过三阶段平地验收的权重：{checkpoint.name}。具体日期与SHA256见训练更改说明和FILE_MANIFEST.json。',
        f'验收为9工况、3种子、首回合零重置；首轮、独立种子、30秒延长回放全部通过。{percent:g}%与绝对容差的定义见训练更改说明。', '',
        '## 目录结构', '',
        f'- {checkpoint.name}：完整PPO checkpoint。',
        '- policy.pt 和 policy.onnx：确定性推理导出。',
        '- sim2sim_contract.json：27维观测、6维动作、坐标系及实际执行器参数。',
        '- training_overrides.json 和 runtime_config.json：验收采用的完整配置。',
        '- robot_assets/：原闭链USD、96边轮胎碰撞叠加USD及参数来源。',
        '- training_replay/、heldout_replay/、extended_replay/：三阶段CSV、验收报告及所有工况误差曲线。',
        '- optimization_history/：逐轮误差趋势、配置与受控修改记录。',
        '- parameter_and_joint_audit/：质量、惯量、关节和闭合孔检查。',
        '- training_params/：训练参数和轮输出冻结模式。',
        '- code/：本次工程源代码快照；其中README内的阶段结果属于历史记录。',
        '- 开链转闭链图文教程_v3.docx 和同名PDF：闭链建模与接入教程。',
        '- 训练更改说明.md：最终更改、验收数字和复现方法。', '',
        '- training_record/：从机械建模开始的逐阶段原因、思路、人工干预、逐轮参数差异、评估结果及代码快照；训练后优化流程.md逐轮串起评估、诊断、改值、权重选择和下一轮验证。最终训练更改说明.md也包含完整流程。', '',
        '## 复现', '',
        '在本机执行 bash replay_accepted.sh。必须同时采用随包USD叠加层与training_overrides.json，不能只加载权重。',
        '资产和脚本保留本机工程路径；迁移到其它电脑时需安装相同Isaac Lab环境并同步修改路径。',
        '新模型的MuJoCo sim2sim尚待独立验证；本包证明Isaac Lab平地验收通过。', '',
    ]))
    for directory,label in zip(paths+[export_replay],('training_replay','heldout_replay','extended_replay')):
        shutil.copytree(directory,delivery/label,dirs_exist_ok=True)
    training_record=build_record(args.state,delivery/'training_record',final=True)
    verify_record(training_record,require_final=True)
    optimization_flow=(training_record/'训练后优化流程.md').read_text()
    change_note=delivery/'训练更改说明.md'
    change_note.write_text(change_note.read_text()+'\n\n'+optimization_flow)
    if optimization_flow not in change_note.read_text():
        raise RuntimeError('Final change explanation lacks full per-round optimization flow')
    files={str(p.relative_to(delivery)):sha(p) for p in delivery.rglob('*') if p.is_file() and p.name!='FILE_MANIFEST.json'}
    (delivery/'FILE_MANIFEST.json').write_text(json.dumps(files,indent=2,ensure_ascii=False))
    archive=output/'accepted_model_and_documents.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for file in delivery.rglob('*'):
            if file.is_file():z.write(file,file.relative_to(output))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('Archive CRC failed')
    status={'completed_at':datetime.now().astimezone().isoformat(),'checkpoint_sha256':sha(checkpoint),
            'delivery':str(delivery),'archive':str(archive),'archive_sha256':sha(archive),
            'replays_passed':True,'documentation_saved':True,'shutdown_requested':args.shutdown,
            'training_record':str(training_record),
            'post_training_optimization':str(training_record/'训练后优化流程.md'),
            'training_record_manifest_sha256':sha(training_record/'record_manifest.json')}
    record=output/'completion.json';record.write_text(json.dumps(status,indent=2,ensure_ascii=False))
    subprocess.run(['/usr/bin/sync'],check=True)
    if args.shutdown:
        if (output/'CANCEL_SHUTDOWN').exists():
            status['shutdown_cancelled']=True;record.write_text(json.dumps(status,indent=2));return
        result=subprocess.run(['/usr/bin/systemctl','--no-ask-password','poweroff'],capture_output=True,text=True)
        if result.returncode:
            status['shutdown_error']=result.stderr;record.write_text(json.dumps(status,indent=2));raise RuntimeError(result.stderr)

if __name__=='__main__':main()
