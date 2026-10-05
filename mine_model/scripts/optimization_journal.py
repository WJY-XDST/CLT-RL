"""Append evidence-backed changes and validations without restarting training.

Source edits are detected as saved snapshots at the polling interval, not as
individual editor keystrokes. Historical source diffs are never invented.
"""
import argparse
from datetime import datetime
import difflib
import fcntl
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
WATCHED = (
    'mine_model/scripts/auto_train_mine.py',
    'mine_model/scripts/summarize_mine_trial.py',
    'mine_model/scripts/complete_mine_training.py',
    'mine_model/scripts/build_training_record.py',
    'mine_model/scripts/optimization_journal.py',
    'mine_model/scripts/play_keyboard.py',
    'mine_model/run.sh',
    'mine_model/docs/键盘可视化控制说明.md',
    'mine_model/config/tracking_acceptance.json',
    'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
    'wheel_legged_isaaclab/scripts/rsl_rl/play.py',
    'wheel_legged_isaaclab/scripts/rsl_rl/keyboard_control.py',
    'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
    'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/five_bar_vmc.py',
    'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/precision_finetune.py',
    'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/config_overrides.py',
    'wheel_legged_isaaclab/deployment/actuator_bridge.py',
    'wheel_legged_isaaclab/tests/test_optimization_journal.py',
    'wheel_legged_isaaclab/tests/test_keyboard_control.py',
    'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/mine_env_cfg.py',
    'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py',
    'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env_cfg.py',
    'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/agents/rsl_rl_ppo_cfg.py',
)
HEADER = '''# 新模型持续优化报告

本报告只追加，不覆盖过去的说明。实际训练参数变化、源码快照变化、验收标准变更、评估结果分别记录。
每次说明包含依据、修改前后、原因、模型与验证状态。没有参数变化的轮次标记为“仅续训”，不冒称代码优化。

历史参数依据管理日志补录；历史源码没有快照时不补造逐行差异。源码监听记录两次扫描间保存下来的版本，不能还原扫描间的每次编辑。
源码监听覆盖核心环境、奖励、控制、PPO配置、训练、评估、实机接口及自动管理脚本，不扫描整个 Isaac Sim SDK。
源码改动不代表正在运行的 Python 进程已加载：须在后续启动或模型参数中核实。自动训练参数以实际启动命令为准；启动命令也可能因配置校验失败而未开始训练，不能视为成功应用。
原因只引用管理日志或人工记录；没有说明就写“未记录”，不从效果反推修改动机。

## 指标含义与判断口径

|指标|含义及单位|
|---|---|
|speed_mae_m_s|机身前后速度与速度指令的平均绝对误差，m/s|
|yaw_mae_rad_s|机身偏航角速度与目标角速度的平均绝对误差，rad/s|
|height_mae_m|机身高度与高度指令的平均绝对误差，m|
|leg_length_mae_m|左右实际虚拟腿长与应用目标之差，采用较大 MAE，m|
|leg_angle_mae_rad|左右实际虚拟腿摆角与应用目标之差，采用较大 MAE，rad|
|wheel_speed_mae_rad_s|左右实际轮速与应用参考轮速之差，采用较大 MAE，rad/s|
|body_p95_deg|稳态俯仰/横滚绝对值较大者的第 95 百分位，度|

误差÷门限大于 1 表示超标；稳态成绩只统计完整存活的首回合最后 3 秒。零目标使用绝对容差。
27 组存活不等于 27 组跟踪达标。12 秒初测、独立种子复验、30 秒延长复测分别记录；不同标准或资产的数据不直接判断改善。

## 追加记录

'''


def read_json(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def table(changes):
    if not changes:
        return '参数未变化：仅续训，不计为一次调参优化。\n'
    lines = ['|参数|修改前|修改后|', '|---|---|---|']
    for key, before, after in changes:
        cells = [key, json.dumps(before, ensure_ascii=False), json.dumps(after, ensure_ascii=False)]
        lines.append('|' + '|'.join(v.replace('|', '\\|').replace('\n', ' ') for v in cells) + '|')
    return '\n'.join(lines) + '\n'


def delta(before, after):
    return [(key, before.get(key), after.get(key)) for key in sorted(set(before) | set(after))
            if before.get(key) != after.get(key)]


def launch_records(log):
    """Use actual starts, including restarts, rather than planned overrides."""
    context = {}
    for number, line in enumerate(log.read_text(errors='replace').splitlines(), 1):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        context.update(event)
        command = event.get('child_command', [])
        if not any(str(v).endswith('/rsl_rl/train.py') for v in command):
            continue
        def option(name):
            return command[command.index(name) + 1] if name in command else None
        settings = {}
        for value in command:
            if value.startswith(('env.', 'agent.')) and '=' in value:
                key, raw = value.split('=', 1)
                try:
                    settings[key] = json.loads(raw)
                except ValueError:
                    settings[key] = raw
        settings.update(num_envs=option('--num_envs'),
                        wheel_head_only_finetune='--wheel_head_only_finetune' in command,
                        reset_optimizer='--reset_optimizer' in command,
                        reset_exploration_std=option('--reset_exploration_std'),
                        wheel_exploration_std=option('--wheel_exploration_std'))
        yield dict(folder=Path(event['child_log']).parent.name, settings=settings,
                   checkpoint=option('--checkpoint_path'), target=option('--target_iteration'),
                   reason=context.get('purpose', '未记录独立修改原因'),
                   assessment=context.get('last_assessment'), line=number)


def sync_journal(output, report=None, root=ROOT):
    output = Path(output).resolve()
    root = Path(root).resolve()
    report = Path(report) if report else output / '持续优化报告.md'
    report.parent.mkdir(parents=True, exist_ok=True)
    archive = output / 'optimization_journal'
    archive.mkdir(exist_ok=True)
    with (archive / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not report.exists():
            report.write_text(HEADER)
        existing = report.read_text()
        state = read_json(output / 'status.json')
        def append(event, title, body):
            nonlocal existing
            eid = identity(event)
            marker = f'<!-- optimization-event:{eid} -->'
            if marker in existing:
                return
            event['recorded_at'] = datetime.now().astimezone().isoformat()
            atomic_json(archive / (eid + '.json'), event)
            block = f'\n{marker}\n\n### {title}\n\n记录时间：{event["recorded_at"]}。\n\n{body}\n'
            with report.open('a') as stream:
                stream.write(block)
                stream.flush()
            existing += block

        log = root / 'mine_model/build/logs' / f'{output.name.replace("_20261004", "")}_service.log'
        # The existing supervisor has a historic log name without the date.
        if not log.exists():
            log = root / 'mine_model/build/logs' / f'{output.name}_service.log'
        if log.exists():
            previous = {}
            seen_launches = set()
            for launch in launch_records(log):
                key = identity([launch['folder'], launch['settings']])
                if key in seen_launches:
                    continue
                seen_launches.add(key)
                changes = delta(previous, launch['settings'])
                append(dict(kind='training_launch', **launch),
                       launch['folder'] + ('：参数调整/启动配置' if changes else '：仅续训'),
                       f'原因原文：{launch["reason"]}。\n\n' + table(changes) +
                       f'\n来源模型：`{launch["checkpoint"]}`；目标迭代：{launch["target"]}。\n\n'
                       f'决策关联评估：`{launch["assessment"]}`。证据：`{log}` 第 {launch["line"]} 行。\n\n'
                       '此条只证明启动时应用了这些参数，不证明优化有效；后续评估另追加说明。')
                previous = launch['settings']

        for index, event in enumerate(state.get('interventions', [])):
            append(dict(kind='manual_intervention', index=index, evidence=event),
                   f'人工干预 {index + 1}：{event.get("kind", event.get("type", "参数或工程调整"))}',
                   f'原因原文：{event.get("reason", event.get("purpose", "未记录"))}。\n\n'
                   '```json\n' + json.dumps(event, ensure_ascii=False, indent=2) + '\n```\n\n'
                   '原始证据保留；没有独立评估的不宣称改善。')

        notes = output / 'optimization_notes.json'
        if notes.exists():
            for note in read_json(notes):
                append(dict(kind='code_optimization_note', evidence=note), note['title'],
                       '修改内容：' + note['change'] + '\n\n修改原因：' + note['reason'] +
                       '\n\n验证情况：' + note['validation'] + '\n\n涉及文件：' +
                       '；'.join('`' + p + '`' for p in note['files']))

        paths = sorted(output.glob('round_*/*/assessment*.json'))
        for path in paths:
            r = read_json(path)
            failed = [(c['name'], t['seed'], k, t['metrics'][k], limit)
                      for c in r['conditions'] for t in c['trials'] if t.get('metrics')
                      for k, limit in t['limits'].items() if t['metrics'][k] > limit]
            rows = ['|工况|seed|指标|误差|门限|误差÷门限|', '|---|---|---|---|---|---|']
            for case, seed, key, value, limit in failed:
                rows.append(f'|{case}|{seed}|{key}|{value:.6g}|{limit:.6g}|{value / limit:.3f}|')
            passed = sum(t['passed'] for c in r['conditions'] for t in c['trials'])
            append(dict(kind='validation', path=str(path), sha256=identity(r)),
                   f'{path.parent.parent.name} / {path.parent.name}：评估追加',
                   f'模型：`{r["checkpoint"]}`；iteration {r["iteration"]}。\n\n'
                   f'存活 {r["survivors"]}/{r["total_trials"]}，精度通过 {passed}/{r["total_trials"]}，'
                   f'重置 {r["resets"]}，整体通过：{r["passed"]}。\n\n'
                   f'本次标准：{r["criterion"]}。\n\n' +
                   ('\n'.join(rows) if failed else '无存活回合的数值超标项；仍须核对生存、指令到达和支撑要求。') +
                   f'\n\n证据：`{path}`。新结果只追加，不覆盖此前失败或改写当时标准。')

        baseline_path = archive / 'source_baseline.json'
        saved = read_json(baseline_path) if baseline_path.exists() else None
        baseline = saved['files'] if saved else None
        sequence = saved['sequence'] if saved else 0
        current = {}
        sources = archive / 'source_snapshots'
        sources.mkdir(exist_ok=True)
        for relative in WATCHED:
            path = root / relative
            if not path.is_file():
                continue
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            current[relative] = digest
            snapshot = sources / (digest + '.txt')
            if not snapshot.exists():
                snapshot.write_bytes(data)
        changed = delta(baseline or {}, current)
        if baseline is None or changed:
            event = dict(kind='source_baseline' if baseline is None else 'source_change',
                         previous=baseline, current=current, checkpoint=state.get('checkpoint'),
                         baseline_sequence=sequence)
            body = ('建立当前源码基线；早期未保存的逐行改动不补造。' if baseline is None else
                    '检测到已保存源码或监控范围变化。首次纳入监控的文件只建立基线，不代表本次新建；已有基线的文件才能确认修改。修改原因未在此自动推断；请结合人工干预说明与代码差异。')
            if baseline is not None:
                patch = []
                for relative, before, after in changed:
                    left = (sources / (before + '.txt')).read_text(errors='replace') if before else ''
                    right = (sources / (after + '.txt')).read_text(errors='replace') if after else ''
                    patch.extend(difflib.unified_diff(left.splitlines(True), right.splitlines(True),
                                                     fromfile='before/' + relative, tofile='after/' + relative))
                diff_path = archive / (identity(event) + '.patch')
                diff_path.write_text(''.join(patch))
                body += f'\n\n逐行差异：`{diff_path}`。\n\n' + table(changed)
            append(event, '源码基线' if baseline is None else '源码修改：保存差异',
                   body + '\n\n验证状态：尚不能据源码变化宣称训练改善；正在运行的进程不会自动重载。')
            atomic_json(baseline_path, dict(files=current, sequence=sequence + 1))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=float, default=15.)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('interval must be positive')
    while True:
        try:
            report = sync_journal(args.output, args.report)
            print(f'Optimization journal synced: {report}', flush=True)
        except (OSError, ValueError, KeyError) as error:
            if not args.watch:
                raise
            print(f'Journal retry: {error}', flush=True)
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
