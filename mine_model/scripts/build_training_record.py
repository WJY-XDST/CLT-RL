"""Create an evidence-backed change history before training completion shutdown."""
import argparse
import difflib
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MINE = ROOT / 'mine_model'
STAGES = [
    ('开链机械模型改为闭链',
     '确认 link_002 为 RB；补充 RL3–RS、RL2–RB 及左侧镜像四个 revolute 闭合铰链，保留原始输入，另存 USD 和树形 URDF。',
     'URDF 的关节树不能独立表达闭环，需要物理约束与完整 CAD 孔位和轴线。',
     '先核对孔轴与左右镜像，再确认默认装配闭合；悬空驱动通过只证明机构与驱动，不证明自由站立。',
     ['config/closures.json', 'docs/implementation_details.md', 'results/bench_drive/report.json']),
    ('坐标、joint 与五连杆运动学适配',
     '工程轴系统一为 X 前、Y 左、Z 上；四个髋关节统一校验以 base_link 为父刚体。主动顺序 LB/LL/LW/RB/RL/RW；FK、Jacobian、逆解采用新几何。',
     '旧串联腿角度定义、杆长与传动映射不能直接套到闭链，即使观测和动作维度相同。',
     '以实际轮轴位置、数值差分和虚功验证映射；父节点原本已连接 base_link 的部分属于核验，不虚构为重新挂接。',
     ['docs/implementation_details.md', 'results/parameter_audit', 'scripts/audit_model_parameters.py']),
    ('原策略直接迁移与控制诊断',
     '复用原 7738 策略做新旧模型对照，随后保存增益、求解器、步长、承重与导轨诊断。',
     '初期自由基座无法站稳，需要区分关节映射错误、驱动承重不足和策略不适配。',
     '固定权重、隔离一个变量比较；不能把导轨承重或无重力驱动当成平衡验收。失败尝试作为后续选择的依据。',
     ['results/policy_7738_20261003/summary.json', 'results/support_rail_confirm_20261004/report.json',
      'results/solver_audit_20261004/report.json', 'results/timestep_audit_20261004/report.json']),
    ('隐式关节驱动与新模型训练',
     '虚拟腿角和长度经五连杆逆解成为真实电机角目标，使用 implicit_joint_reference；初始 Drive 为 Kp300/Kd3，腿轮限幅 10 Nm。',
     '显式 VMC 在原限幅下的承重与稳定性未通过，新闭链由 PhysX Drive 与约束联合求解。',
     '保留 27 维观测和 6 维动作契约，采用新机构物理参数；新网络训练及后续初始化、续训阶段分别标记。',
     ['results/implicit_pid_base_20261004/report.json', 'results/implicit_free_base_20261004/report.json',
      'results/auto_training_implicit_20pct_20261004/README.md']),
    ('奖励信号与失败判定修正',
     '扩大高度、速度、偏航、姿态和动作变化惩罚裁剪范围；增加低根高度终止，调整终止成本。',
     '原单项惩罚过早饱和，无法区分轻微误差和蹲塌；终止代价过低时，立即倒地可能比持续纠错更便宜。',
     '依据实测饱和率和误差扩大有效信号，保留正常课程；只有真实生存失败才回退平衡或恢复探索。',
     ['results/auto_training_implicit_20pct_20261004/reward_saturation_audit_20261004.json',
      'results/auto_training_implicit_20pct_20261004/代码检查与提前回放.md']),
    ('速度初始化、轮抖动与精度训练',
     '记录速度/腿角初始化，训练轮探索、熵系数和 action_rate 调整；后续冻结腿输出及 actor 特征，仅训练轮输出和 critic。',
     '站立策略不能自然推出正确行驶，轮动作出现高频变化，已有站稳能力需要保留。',
     '用同权重实测选初始化和单项对照；记录失败增益试验。初始化不是从零 RL，也不是验收成功。',
     ['results/speed_response_initialization_2199_20261004/initialization.json',
      'results/auto_training_implicit_20pct_20261004/wheel_flutter_audit_20261004.json',
      'results/wheel_head_precision_5399_5600_20261004/trial.json']),
    ('轮胎碰撞及保护限位',
     '轮胎碰撞由 18 边细化至 96 边；base 使用盒体，普通连杆可简化，确认的限位块及小腿接触面保留 CAD 约束。',
     '接触多边形影响轮速和机身波动；大量碰撞形状导致内存压力，但机械腿长限位不可被简化掉。',
     '先用固定权重比较接触，再做保护块/小腿 CAD 非相交检查。若腿部简化增加误差则回退，不能仅凭吞吐选择。',
     ['docs/碰撞简化与机械限位说明_20261004.md',
      'results/collision_simplification_20261004/collision_comparison.json',
      'results/collision_simplification_20261004/protected_stops_v2_live_cad/cad_nonpenetration.json']),
    ('环境数与内存调整',
     '早期 256 环境；12288 请求发生 OOM 后选择成功规模，后续用户指定 10240，再因内存压力改为 8192。',
     '新闭链刚体、碰撞与克隆负载不同于旧模型，CUDA 计算仍需要主机内存。',
     '以成功运行和实测内存选择规模；用户要求停止容量测试后不再重测，规模切换保留来源权重和优化器。',
     ['results/auto_training_implicit_20pct_20261004/env_scale_report_20261004.json',
      'results/collision_simplification_20261004/environment_capacity_report.json',
      'results/auto_training_implicit_20pct_20261004/before_env8192_after_round38.json']),
    ('MuJoCo sim2sim 验证',
     '从完整训练 USD 转为 MuJoCo，保留刚体惯量、joint 与闭链约束，匹配观测/动作、镜像、COM 速度和实际隐式执行器。',
     '相同策略在不同接触求解器中可能表现不同；较粗内部积分曾失稳，基础运动稳定也不能代表所有误差通过。',
     '用接口测试、相同指令与种子及误差曲线验证；保留偏航超标和机械限位差异，不将两个仿真器视为相同动力学。',
     ['docs/新闭链模型_MuJoCo_sim2sim验证_20261004.md',
      'results/sim2sim_mujoco_6700_20261004/comparison.json', 'results/latest_sim2sim_preview.json']),
    ('10% 精度目标、记录留存和自动完成',
     '目标从 20% 收紧到 10%，同步减半绝对容差及姿态门限；保留最新旧模型及默认验证依赖，删除其余旧训练记录。',
     '仅改百分比不能收紧零目标工况；最终记录需要覆盖历史失败、真实修改原因与完整交付。',
     '逐轮按实际应用参数和评估阈值记录，旧标准结果不能替代新标准；复验、文档、打包与刷盘全部完成后才关机。',
     ['config/tracking_acceptance.json',
      'results/auto_training_implicit_20pct_20261004/acceptance_target_change_20261004.json']),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def differences(previous, current):
    return [{'parameter':key, 'before':previous.get(key), 'after':current.get(key)}
            for key in sorted(set(previous) | set(current)) if previous.get(key)!=current.get(key)]


def read_json(path):
    return json.loads(Path(path).read_text())


def command_option(command, name):
    return command[command.index(name)+1] if name in command else None


def supervisor_history(branch, sources):
    """Preserve event order and report paths; round numbers alone are ambiguous."""
    log_names={'auto_training_20261003':'auto_training_service.log',
               'auto_training_20pct_20261004':'auto_training_20pct_service.log',
               'auto_training_implicit_20pct_20261004':'auto_training_implicit_20pct_service.log'}
    path=MINE/'build/logs'/log_names.get(branch, branch+'_service.log')
    history={'plans':[], 'launches':[]}
    if not path.exists():return history
    sources.add(path);context={}
    for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
        try:event=json.loads(line)
        except ValueError:continue
        if not isinstance(event,dict):continue
        context.update(event)
        evidence={'log':str(path.relative_to(ROOT)), 'line':number}
        if event.get('state')=='planning_next_round':
            history['plans'].append(dict(event, basis_assessment=context.get('last_assessment'),
                                         evidence=evidence))
        command=event.get('child_command',[])
        if not any(str(v).endswith('/rsl_rl/train.py') for v in command):continue
        env={};agent={}
        for value in command:
            if value.startswith(('env.','agent.')) and '=' in value:
                key,raw=value.split('=',1)
                try:parsed=json.loads(raw)
                except ValueError:parsed=raw
                (env if key.startswith('env.') else agent)[key.split('.',1)[1]]=parsed
        history['launches'].append({'folder':Path(event['child_log']).parent.name,
            'checkpoint':command_option(command,'--checkpoint_path'),
            'num_envs':command_option(command,'--num_envs'),
            'target_iteration':command_option(command,'--target_iteration'),
            'wheel_head_only_finetune':'--wheel_head_only_finetune' in command,
            'reset_optimizer':'--reset_optimizer' in command,
            'reset_exploration_std':command_option(command,'--reset_exploration_std'),
            'overrides':env,'agent_overrides':agent,'command':command,'evidence':evidence})
    return history


def assessment_summary(path, sources):
    path=Path(path)
    if not path.exists():return None
    report=read_json(path);sources.add(path)
    trials=[t for c in report['conditions'] for t in c['trials']]
    errors=[(t['metrics'][key]/limit,key,c['name'],t['seed'])
            for c in report['conditions'] for t in c['trials'] if t.get('metrics')
            for key,limit in t['limits'].items() if limit>0]
    failed=[]
    for condition in report['conditions']:
        for trial in condition['trials']:
            if trial['passed']:continue
            metrics=trial.get('metrics') or {}
            failed.append({'condition':condition['name'],'seed':trial['seed'],
                'survived':trial.get('survived'),'resets':trial.get('resets'),
                'exceeded':{key:{'value':metrics[key],'limit':limit,'ratio':metrics[key]/limit}
                            for key,limit in trial['limits'].items()
                            if key in metrics and limit>0 and metrics[key]>limit},
                'upright_support':metrics.get('upright_support'),
                'command_reached':metrics.get('command_reached')})
    protocol=None
    manifest=path.parent/'manifest.json'
    if manifest.exists():
        sources.add(manifest);m=read_json(manifest)
        protocol={key:m.get(key) for key in ('cases','seeds','seconds_per_case','command_scale',
                                           'asset_sha256','asset_chain_sha256')}
    return {'iteration':report['iteration'],'checkpoint':report.get('checkpoint'),
        'survivors':report['survivors'],'total_trials':report['total_trials'],
        'resets':report['resets'],'passed':report['passed'],
        'passed_trials':sum(t['passed'] for t in trials),'criterion':report['criterion'],
        'acceptance_criteria':report.get('acceptance_criteria'),'protocol':protocol,
        'worst_error':max(errors) if errors else None,'failed_trials':failed,
        'path':str(path.relative_to(ROOT))}


def attach_optimization_flows(rounds, histories, sources, state):
    indexed={(r['branch'],r['folder']):r for r in rounds}
    # Some controlled validations were run outside the supervisor loop. Only
    # explicit assessment references establish a post-evaluation decision.
    manual_plans=[]
    for index,event in enumerate(state.get('interventions',[])):
        basis=event.get('assessment')
        if not basis:continue
        branch=Path(basis).parent.parent.parent.name
        basis_round=re.match(r'round_(\d+)',Path(basis).parent.parent.name)
        if not basis_round:continue
        launches=histories.get(branch,{}).get('launches',[])
        launch=next((l for l in launches if l['checkpoint']==event.get('checkpoint') and
                     int(re.match(r'round_(\d+)',l['folder'])[1])>=int(basis_round[1])),None)
        if not launch:continue
        manual_plans.append({'branch':branch,'basis_assessment':basis,
            'round':int(re.match(r'round_(\d+)',launch['folder'])[1]),
            'checkpoint':event.get('checkpoint'),'purpose':event.get('reason','未记录'),
            'next_launch_override':launch,
            'evidence':{'state_path':str(MINE/'results'/branch/'status.json'),
                        'intervention_index':index}})
    for item in rounds:
        history=histories[item['branch']]
        item['training_launches']=[l for l in history['launches'] if l['folder']==item['folder']]
        starts=[p for p in history['plans'] if p['round']==item['round'] and
                any(p['evidence']['line']<l['evidence']['line'] for l in item['training_launches'])]
        item['recorded_reason']=starts[-1].get('purpose','') if starts else ''
        decisions=[]
        for plan in history['plans']+[p for p in manual_plans if p['branch']==item['branch']]:
            basis=plan.get('basis_assessment')
            if not basis or Path(basis).parent.parent.name!=item['folder']:continue
            report=assessment_summary(basis,sources)
            launches=[l for l in history['launches']
                      if l['evidence']['line']>plan['evidence'].get('line',float('inf')) and
                      re.match(r'round_(\d+)',l['folder']) and
                      int(re.match(r'round_(\d+)',l['folder'])[1])==plan['round']]
            launch=plan.get('next_launch_override') or (launches[0] if launches else None)
            following=indexed.get((item['branch'],launch['folder'])) if launch else None
            before=read_json(ROOT/item['overrides_path'])
            actual=read_json(ROOT/following['overrides_path']) if following else None
            validation=following.get('assessment') if following else None
            comparison={'status':'待下一轮完整验证'}
            if report and validation:
                same=(report['criterion']==validation['criterion'] and
                      report['acceptance_criteria']==validation['acceptance_criteria'] and
                      report['protocol'] is not None and report['protocol']==validation['protocol'])
                comparison={'status':'同标准同资产同回放协议对比' if same else '标准、资产或回放协议不同，不直接判定改善',
                    'comparable':same,'before_passed_trials':report['passed_trials'],
                    'after_passed_trials':validation['passed_trials'],
                    'before_worst_error':report['worst_error'],'after_worst_error':validation['worst_error']}
                if same and report['worst_error'] and validation['worst_error']:
                    delta=validation['worst_error'][0]-report['worst_error'][0]
                    comparison.update(worst_ratio_delta=delta,
                        conclusion='最大误差/门限降低' if delta<0 else '最大误差/门限升高' if delta>0 else '最大误差/门限不变')
            selected=plan.get('checkpoint')
            selection=('继续本轮评估权重' if report and selected==report['checkpoint'] else
                       '选择其他权重；回退或择优原因以决策原文为准' if report else '缺少评估权重证据')
            settings=('num_envs','agent_overrides','wheel_head_only_finetune',
                      'reset_optimizer','reset_exploration_std')
            previous_launch=item['training_launches'][-1] if item['training_launches'] else None
            settings_changes=(differences({k:previous_launch[k] for k in settings},
                                          {k:launch[k] for k in settings})
                              if previous_launch and launch else None)
            decisions.append({'basis_assessment':report,'basis_path':basis,
                'diagnosis_and_decision':plan.get('purpose','未记录'),
                'decision_evidence':plan['evidence'],'next_round':plan['round'],
                'selected_checkpoint':selected,'checkpoint_selection':selection,
                'planned_parameter_changes':differences(before,plan['overrides']) if 'overrides' in plan else None,
                'launch_parameter_changes':differences(before,launch['overrides']) if launch else None,
                'actual_parameter_changes':differences(before,actual) if actual is not None else None,
                'training_setting_changes':settings_changes,
                'next_launch':launch,'next_folder':following['folder'] if following else None,
                'next_assessment':validation,'next_validation_comparison':comparison})
        if decisions:status='已记录训练后决策'
        elif item.get('assessment'):
            accepted=(state.get('state')=='accepted' and item['assessment'].get('checkpoint')==state.get('checkpoint'))
            status='最终验收通过，结束训练并生成交付' if accepted else '已有评估，未留存关联的后续决策'
        else:status='尚无完整评估，未形成可核验的训练后优化流程'
        item['post_training_optimization']={'status':status,'decisions':decisions}


def optimization_markdown(rounds):
    lines=['# 每次训练后的优化流程','',
        '流程：本轮权重固定回放 → 定量误差与失败项 → 诊断与决策 → 选择续训或回退权重 → 计划和实际参数变化 → 下一轮验证。', '',
        '按评估报告的完整目录关联决策，独立种子或30秒复测失败优先保留其实际依据；相同轮号的接触对照和训练轮分别记录。无改值明确写仅续训，没有历史证据不补造原因。','']
    for item in rounds:
        flow=item['post_training_optimization']
        lines += [f"## {item['branch']} / {item['folder']}",'',flow['status']+'。','']
        launches=item['training_launches']
        lines += [f"本目录记录的实际训练启动/恢复次数：{len(launches)}。"+
                  ('没有训练启动日志，此目录可能为固定权重验证；不计作已执行训练。' if not launches else ''),'']
        if launches:
            lines += ['每次真实启动及恢复的参数、权重、环境数和命令：','',
                      '```json',json.dumps(launches,ensure_ascii=False,indent=2),'```','']
        for label in ('assessment','heldout_replay','export_replay'):
            report=item.get(label)
            if report:
                lines += [f"本轮 {label}：iteration {report['iteration']}，存活 {report['survivors']}/{report['total_trials']}，精度通过 {report['passed_trials']}/{report['total_trials']}，重置 {report['resets']}；报告 `{report['path']}`。",'']
        for decision in flow['decisions']:
            report=decision['basis_assessment']
            lines += [f"### 评估后决定下一轮 {decision['next_round']}",'',
                '实际决策依据：`'+decision['basis_path']+'`。','']
            if report:
                lines += ['当时验收标准：'+report['criterion'],'',
                    f"结果：精度通过 {report['passed_trials']}/{report['total_trials']}，重置 {report['resets']}。",'']
                if report['worst_error']:
                    ratio,metric,case,seed=report['worst_error']
                    lines += [f'最大超差/误差项：{case} / seed {seed} / {metric}，误差÷门限 {ratio:.4f}。','']
                lines += ['失败项及实际数值（未存活、未达到指令、姿态支撑也单独记录）：','',
                    '```json',json.dumps(report['failed_trials'],ensure_ascii=False,indent=2),'```','']
            else:lines += ['依据报告未保存，不能重建具体误差。','']
            evidence=decision['decision_evidence']
            location=(f"`{evidence['log']}` 第 {evidence['line']} 行" if 'log' in evidence else
                      f"`{evidence['state_path']}` interventions[{evidence['intervention_index']}]")
            lines += ['诊断与决策原文：'+decision['diagnosis_and_decision'],'',
                '决策证据：'+location+'。','',
                '权重选择：'+decision['checkpoint_selection']+'；`'+str(decision['selected_checkpoint'])+'`。','']
            for label,key in (('规划环境参数改值','planned_parameter_changes'),
                              ('下一轮首次实际启动环境参数改值','launch_parameter_changes'),
                              ('下一轮目录最终保存环境配置改值','actual_parameter_changes'),
                              ('算法、环境数和训练恢复方式改值','training_setting_changes')):
                changes=decision[key]
                lines += [label+'：'+('未留存对应配置，不能重建改值。' if changes is None else
                                        '该类参数未变化。' if not changes else ''),'']
                if changes:
                    lines += ['|参数|本轮|下一轮|','|---|---|---|']
                    for c in changes:
                        lines.append('|'+ '|'.join(json.dumps(v,ensure_ascii=False).replace('|','\\|')
                                     for v in (c['parameter'],c['before'],c['after']))+'|')
                    lines.append('')
            launch=decision['next_launch']
            if (decision['launch_parameter_changes']==[] and decision['training_setting_changes']==[]
                    and decision['checkpoint_selection']=='继续本轮评估权重'):
                lines += ['环境参数、算法和训练方式均未变化：仅续训，不记录为已完成调参优化。','']
            if launch:
                lines += [f"下一轮实际启动：{launch['folder']}，环境 {launch['num_envs']}，目标 iteration {launch['target_iteration']}；加载 `{launch['checkpoint']}`；仅训练轮输出 {launch['wheel_head_only_finetune']}；重置优化器 {launch['reset_optimizer']}。",'',
                    '算法覆盖：`'+json.dumps(launch['agent_overrides'],ensure_ascii=False)+'`。','']
            comparison=decision['next_validation_comparison']
            lines += ['下一轮验证：'+comparison['status']+'。','',
                '```json',json.dumps(comparison,ensure_ascii=False,indent=2),'```','']
            validation=decision['next_assessment']
            if validation:lines += ['下一轮结果证据：`'+validation['path']+'`。','']
    return '\n'.join(lines)+'\n'


def render_markdown(markdown):
    def inline(text):
        return re.sub(r'`([^`]+)`',r'<code>\1</code>',html.escape(text))
    output=[];code=[];in_code=False;in_table=False
    for line in markdown.splitlines():
        if line.startswith('```'):
            if in_code:
                output.append('<details><summary>查看原始记录</summary><pre>'+html.escape('\n'.join(code))+'</pre></details>')
                code=[]
            in_code=not in_code
            continue
        if in_code:code.append(line);continue
        if line.startswith('|'):
            if not in_table:output.append('<div class="table"><table>');in_table=True
            if re.fullmatch(r'[|:\- ]+',line):continue
            cells=re.split(r'(?<!\\)\|',line.strip('|'))
            output.append('<tr>'+''.join('<td>'+inline(cell.replace('\\|','|'))+'</td>' for cell in cells)+'</tr>')
            continue
        if in_table:output.append('</table></div>');in_table=False
        match=re.match(r'^(#{1,3}) (.*)',line)
        if match:
            level=len(match[1]);output.append(f'<h{level}>'+inline(match[2])+f'</h{level}>')
        elif line:output.append('<p>'+inline(line)+'</p>')
    if in_table:output.append('</table></div>')
    return '\n'.join(output)


def build_record(state_path, destination, final=False):
    state_path=Path(state_path).resolve();destination=Path(destination).resolve()
    state=read_json(state_path)
    if final and state.get('state')!='accepted':
        raise ValueError('Final training record requires accepted weights')
    destination.mkdir(parents=True,exist_ok=True)
    stages=[];sources=set()
    for title,change,reason,thinking,evidence in STAGES:
        available=[MINE/p for p in evidence if (MINE/p).exists()]
        sources.update(p for p in available if p.is_file())
        stages.append(dict(title=title,change=change,reason=reason,thinking=thinking,
                           evidence=[str(p.relative_to(ROOT)) for p in available]))
    rounds=[]
    branches=('auto_training_20261003','auto_training_20pct_20261004',state_path.parent.name)
    histories={branch:supervisor_history(branch,sources) for branch in dict.fromkeys(branches)}
    for branch in dict.fromkeys(branches):
        previous={}
        for folder in sorted((MINE/'results'/branch).glob('round_*')):
            overrides=folder/'overrides.json'
            if not overrides.exists():continue
            sources.add(overrides);current=read_json(overrides)
            match=re.match(r'round_(\d+)',folder.name)
            if not match:continue
            rid=int(match[1])
            item={'branch':branch,'round':rid,'folder':folder.name,
                  'parameter_changes':differences(previous,current),
                  'change_basis':'initial applied settings' if not previous else 'previous recorded round settings',
                  'recorded_reason':'',
                  'overrides_path':str(overrides.relative_to(ROOT))}
            previous=current
            report_path=folder/'replay/assessment.json'
            strict=folder/'replay/assessment_10pct.json'
            if strict.exists():report_path=strict
            if report_path.exists():
                item['assessment']=assessment_summary(report_path,sources)
            for label in ('heldout_replay','export_replay'):
                path=folder/label/'assessment.json'
                if path.exists():
                    item[label]=assessment_summary(path,sources)
            rounds.append(item)
    attach_optimization_flows(rounds,histories,sources,state)
    flow_markdown=optimization_markdown(rounds)
    (destination/'训练后优化流程.md').write_text(flow_markdown)
    (destination/'post_training_optimization.json').write_text(json.dumps(
        [{'branch':r['branch'],'folder':r['folder'],'round':r['round'],
          'training_launches':r['training_launches'],**r['post_training_optimization']} for r in rounds],
        indent=2,ensure_ascii=False)+'\n')
    events=state.get('interventions',[])
    requirements=state_path.parent/'final_delivery_requirements.json'
    if requirements.exists():sources.add(requirements)
    for event in events:
        for value in event.values():
            if isinstance(value,str) and value.startswith(str(ROOT)) and Path(value).is_file():sources.add(Path(value))
    source_files=[MINE/'scripts'/name for name in ('auto_train_mine.py','summarize_mine_trial.py',
                 'complete_mine_training.py','build_training_record.py','optimization_journal.py')]
    source_files += [ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab'/name
                     for name in ('five_bar_vmc.py','precision_finetune.py','config_overrides.py')]
    snapshot=destination/'code_current';snapshot.mkdir(exist_ok=True)
    for path in source_files:
        if path.exists():
            target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(path.read_bytes());sources.add(path)
    diff=subprocess.run(['git','diff','HEAD','--',*[str(p.relative_to(ROOT)) for p in source_files]],
                        cwd=ROOT,capture_output=True,text=True)
    if diff.returncode:raise RuntimeError('Could not capture tracked source changes: '+diff.stderr)
    (destination/'tracked_current_changes.patch').write_text(diff.stdout)
    saved_diffs=[]
    for before in sorted(state_path.parent.glob('before_*/*.py')):
        current=MINE/'scripts'/before.name
        if not current.exists():continue
        saved_diffs.extend(difflib.unified_diff(before.read_text().splitlines(True),current.read_text().splitlines(True),
                           fromfile=str(before.relative_to(ROOT)),tofile=str(current.relative_to(ROOT))))
        sources.add(before)
    (destination/'saved_source_changes.patch').write_text(''.join(saved_diffs))
    data={'status':'final_accepted' if final else 'in_progress','generated_at':datetime.now().astimezone().isoformat(),
          'state_path':str(state_path),'objective':state.get('objective'),'num_envs':state.get('num_envs'),
          'current_checkpoint':state.get('checkpoint'),'acceptance_criteria':state.get('acceptance_criteria'),
          'stages':stages,'interventions':events,'rounds':rounds,
          'coverage_note':'Early narrative uses preserved project documents and diagnostics; unavailable historical source diffs are not reconstructed. Round reasons are quoted from supervisor logs when available. Current git diff may include pre-existing user changes; untracked files are preserved in code_current.',
          'evidence_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(sources)}}
    (destination/'training_record.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    lines=['# 新闭链模型从建模到训练的修改记录','',
           f"生成时间：{data['generated_at']}；状态：{'已通过最终验收' if final else '训练进行中，尚未最终验收'}。",'',
           f"并行环境：{data['num_envs']}；当前来源权重：`{data['current_checkpoint']}`。",'',
           '记录顺序为机械建模、工程适配、诊断与训练、精度优化及完成交付。历史失败与后来通过分别保留；不能把悬空驱动或短测存活解释为训练收敛。', '',
           '证据边界：早期修改依据现存教程、配置和诊断记录；没有保存的逐行历史差异不补造。逐轮修改原因有管理日志时引用原文，没有时标为未记录。旧开源模型的数据已按用户要求清理，因此不能重建其所有早期原始数据。当前 Git 差异可能包含用户原有修改，不能作为全部改动的作者证明。','']
    for stage in stages:
        lines += ['## '+stage['title'],'','修改内容：'+stage['change'],'',
                  '修改原因：'+stage['reason'],'','修改思路：'+stage['thinking'],'',
                  '证据：'+('；'.join('`'+p+'`' for p in stage['evidence']) or '无独立原始证据文件，仅见阶段说明。'),'']
    lines += ['## 人工干预和用户要求的变更','']
    for i,event in enumerate(events,1):
        lines += [f"### 干预 {i}：{event.get('time','时间未记录')} {event.get('kind','阶段干预')}",'',
                  '记录的原因：'+str(event.get('reason',event.get('purpose',event.get('kind','未单独记录')))), '',
                  '```json',json.dumps(event,indent=2,ensure_ascii=False),'```','']
    lines += ['## 逐轮参数变化、原因与验证结果','',
              '未发生参数变化的续训轮次也记录；达到参数上限不写成已调整。不同验收标准的通过数不能直接视为连续收敛趋势。','']
    for item in rounds:
        lines += [f"### {item['branch']} / {item['folder']}",'',
                  '本轮开始前的决策原文：'+(item['recorded_reason'] or '未留存该轮单独说明；请结合上一轮评估及人工干预证据。'), '']
        if item['parameter_changes']:
            lines += ['|参数|之前|应用后|','|---|---|---|']
            for change in item['parameter_changes']:
                values=[change['parameter'],json.dumps(change['before'],ensure_ascii=False),json.dumps(change['after'],ensure_ascii=False)]
                lines.append('|'+ '|'.join(v.replace('|','\\|').replace('\n',' ') for v in values)+'|')
        else:lines.append('参数未变化：继续学习，不能将其描述为一次参数优化。')
        assessment=item.get('assessment')
        if assessment:
            lines += ['',f"验证：iteration {assessment['iteration']}，存活 {assessment['survivors']}/{assessment['total_trials']}，完整精度通过 {assessment['passed_trials']}/{assessment['total_trials']}，重置 {assessment['resets']}，整体通过 {assessment['passed']}。",'',
                      '验收标准：'+assessment['criterion'],'']
            if assessment['worst_error']:
                ratio,metric,case,seed=assessment['worst_error']
                lines += [f'最大归一化误差：{metric}，工况 {case}，seed {seed}，误差/门限 = {ratio:.4f}。','']
        else:lines += ['','尚无该轮完整评估结果；不能据此称为通过。','']
        for label in ('heldout_replay','export_replay'):
            if label in item:lines += [label+'：'+json.dumps(item[label],ensure_ascii=False),'']
        lines += ['应用配置：`'+item['overrides_path']+'`。','']
    lines += ['## 完成判定和关机顺序','',
              '当前目标采用统一 tracking_acceptance.json。9 工况 × 3 种子首先通过，再用独立种子复验，最后同一权重每工况 30 秒延长复测；失败回合不作为稳态样本。', '',
              '通过后停止追加训练，导出权重、控制契约和参数，生成本记录及最终更改说明，保留 DOCX/PDF 教程、三阶段误差曲线和代码快照，核对文件哈希及 ZIP CRC，sync 刷盘后执行关机。任何验收或文档交付失败都不关机。','',
              'Isaac Lab 的准备验收与 MuJoCo sim2sim 验收分开。现存 MuJoCo 验证和未通过项保留，不能把 Isaac Lab 达标写成跨仿真器全部通过。','',
              '## 可复查的完整记录','',
              '`training_record.json` 保存结构化变更与证据 SHA256；`code_current/` 保存当前相关源代码，`tracked_current_changes.patch` 保存已跟踪文件相对 HEAD 的实际差异，`saved_source_changes.patch` 保存已有源码备份到当前版本的真实累计差异。没有备份的早期代码不补造逐行变化。']
    markdown='\n'.join(lines)+'\n\n'+flow_markdown
    (destination/'训练修改记录.md').write_text(markdown)
    # Use only the standard library so finalization has no extra package dependency.
    content=render_markdown(markdown)
    (destination/'training_record.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>新闭链模型训练修改记录</title><style>body{max-width:1200px;margin:32px auto;padding:0 24px;background:#f8fafc;color:#172033;font:15px/1.8 system-ui,sans-serif}h2{margin-top:48px;border-bottom:1px solid #ccd5df}h3{margin-top:32px}pre,code{overflow-wrap:anywhere;white-space:pre-wrap}details{border:1px solid #ccd5df;border-radius:8px;padding:12px;background:white}table{border-collapse:collapse;width:100%;font-size:13px}td{padding:8px;border:1px solid #dce2e9;vertical-align:top;overflow-wrap:anywhere}tr:first-child{background:#e8edf4;font-weight:600}.table{overflow-x:auto}p{overflow-wrap:anywhere}</style>'+content+'</html>')
    manifest={'final':final,'generated_at':data['generated_at'],'round_count':len(rounds),
              'post_training_flow_count':len(rounds),
              'post_training_decision_count':sum(len(r['post_training_optimization']['decisions']) for r in rounds),
              'intervention_count':len(events),'stage_count':len(stages),
              'sha256':{str(p.relative_to(destination)):sha(p) for p in destination.rglob('*')
                        if p.is_file() and p.name!='record_manifest.json'}}
    (destination/'record_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    verify_record(destination,require_final=final)
    (MINE/'docs/training_record_latest.html').write_bytes((destination/'training_record.html').read_bytes())
    (MINE/'docs/训练修改记录_latest.md').write_bytes((destination/'训练修改记录.md').read_bytes())
    return destination


def verify_record(destination,require_final=False):
    destination=Path(destination);manifest=read_json(destination/'record_manifest.json')
    if require_final and not manifest['final']:raise ValueError('Training record is not final')
    if manifest['stage_count']<10 or not manifest['round_count']:
        raise ValueError('Training record lacks chronological stages or round history')
    for relative,expected in manifest['sha256'].items():
        path=destination/relative
        if not path.is_file() or sha(path)!=expected:raise ValueError('Training record changed: '+relative)
    for name in ('训练修改记录.md','training_record.html','training_record.json','tracked_current_changes.patch','saved_source_changes.patch',
                 '训练后优化流程.md','post_training_optimization.json'):
        if name not in manifest['sha256']:raise ValueError('Missing required training record: '+name)
    data=read_json(destination/'training_record.json')
    flows=read_json(destination/'post_training_optimization.json')
    identities=lambda entries:[(r['branch'],r['folder']) for r in entries]
    if (manifest.get('post_training_flow_count')!=manifest['round_count'] or
            len(flows)!=manifest['round_count'] or identities(flows)!=identities(data['rounds'])):
        raise ValueError('Missing per-round post-training optimization coverage')
    if any('post_training_optimization' not in r for r in data['rounds']):
        raise ValueError('Missing post-training optimization fields')
    flow=(destination/'训练后优化流程.md').read_text()
    if flow not in (destination/'训练修改记录.md').read_text():
        raise ValueError('Full optimization flow is missing from change record')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    folder=build_record(args.state,args.output,args.final)
    print(json.dumps(verify_record(folder,args.final),ensure_ascii=False,indent=2))
