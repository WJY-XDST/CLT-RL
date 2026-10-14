"""Measured full-policy coordinate search; physical/controller contracts stay fixed.

Called by auto_train_mine's direct-joint mode. Reuses the existing owned-child
guard and native evaluator. Never freezes legs, changes hardware, drops failed
episodes, deletes models, or restarts an unknown native fault. Poweroff requires
an explicit request, disjoint sustained replays and a verified saved delivery.
"""
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

import resume_after_solver_check_20261011 as runtime
from auto_train_mine import write_json
from diagnose_policy_feedback import diagnose_joint_pd
from summarize_mine_trial import assess_stability, acceptance_criteria, TASK_GOAL_METRICS, REFERENCE_TRACKING_METRICS
from watch_joint_pd_scratch import RUNS, evaluate, validate_contract
from wheel_legged_gym_isaaclab.config_overrides import strip_retired_config_fields

ROOT = runtime.ROOT
FIELDS = ('yaw_rate_error_sq', 'orientation', 'lin_vel_error_sq', 'base_height_sigma', 'base_height_relative_deadband', 'wheel_action_saturation', 'tracking_goal_violation', 'joint_position_action_saturation', 'tracking_goal_component_violation')


def reward_value(cfg, key):
    # Old checkpoints predate the opt-in reward band; their equation is unchanged.
    if key in ('base_height_relative_deadband', 'wheel_action_saturation', 'tracking_goal_violation',
               'tracking_goal_component_violation'):
        return cfg['rewards'].get(key, 0.)
    return cfg['rewards'][key]


def prepared_reward_trial(path):
    """Permit one diagnosed reward intervention, with the existing control arm."""
    if path is None:
        return {}
    change = json.loads(Path(path).read_text())
    if (not isinstance(change, dict) or len(change) != 1
            or any(key not in ('rewards.base_height_relative_deadband', 'rewards.wheel_action_saturation',
                              'rewards.tracking_goal_violation', 'rewards.joint_position_action_saturation',
                              'rewards.tracking_goal_component_violation') for key in change)):
        raise ValueError('Prepared trial supports one diagnosed height, wheel, tracking-goal or joint saturation reward')
    key,value = next(iter(change.items()))
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('Prepared reward value must be finite')
    if key == 'rewards.base_height_relative_deadband' and not 0 < value <= .05:
        raise ValueError('Prepared reward deadband must be positive and at most five percent')
    if key == 'rewards.wheel_action_saturation' and not -2. <= value < 0.:
        raise ValueError('Prepared wheel saturation reward must be negative and no smaller than minus two')
    if key == 'rewards.tracking_goal_violation' and not -2. <= value <= 0.:
        raise ValueError('Prepared tracking-goal reward must be nonpositive and no smaller than minus two')
    if key == 'rewards.joint_position_action_saturation' and not -8. <= value < 0.:
        raise ValueError('Prepared joint saturation reward must be negative and no smaller than minus eight')
    if key == 'rewards.tracking_goal_component_violation' and not -1. <= value < 0.:
        raise ValueError('Prepared component tracking reward must be negative and no smaller than minus one')
    return change


def stop_outcome(output):
    """A diagnostic checkpoint boundary is not an explicit user pause."""
    try:
        request = json.loads((Path(output)/'STOP').read_text())
    except (ValueError, OSError):
        request = {}
    reason = request.get('reason', '') if isinstance(request, dict) else ''
    if isinstance(reason, str) and reason.startswith('agent_diagnosis'):
        return 'agent_diagnosis_required', dict(reason=reason, user_pause_requested=False, goal_remains_active=True)
    return 'paused_user_request', dict(reason='explicit_user_pause', user_pause_requested=True)


def ratios(report, keys=TASK_GOAL_METRICS):
    return {key: max((t['metrics'][key] / t['limits'][key] for c in report['conditions']
                    for t in c['trials'] if t['metrics']), default=1.e6) for key in keys}


def quality(report):
    task = list(ratios(report).values())
    reference = list(ratios(report, REFERENCE_TRACKING_METRICS).values())
    return (report['survivors'], -max(task), -sum(task)/len(task), -max(reference))


def safe_candidate(candidate, control):
    """Reject collapse and substantial cross-task regressions, even if one axis wins."""
    if candidate['survivors'] != candidate['total_trials'] or candidate['resets']:
        return False
    for current, baseline in zip(candidate['conditions'], control['conditions']):
        if current['name'] != baseline['name']:
            raise ValueError('Comparison cases differ')
        for trial, old in zip(current['trials'], baseline['trials']):
            if trial['seed'] != old['seed']:
                raise ValueError('Comparison reset seeds differ')
            if not old['metrics']:
                continue
            for key in TASK_GOAL_METRICS:
                previous = old['metrics'][key]
                margin = 1. if key == 'body_p95_deg' else (.003 if key == 'height_mae_m' else .01)
                allowed = max(trial['limits'][key], previous*1.25, previous+margin)
                if trial['metrics'][key] > allowed:
                    return False
    return True


def learning_choice(results, source_report, control):
    """Choose a measured learning branch, never declare final acceptance.

    Retention checks can reject a useful intermediate policy for a cross-task
    tradeoff. A learning branch must still survive every episode, preserve the
    matched-control regression checks and improve the whole-task score. It is
    evaluated again after actual cumulative updates, not copied back to 2097.
    """
    choice, best = None, source_report
    for result in results:
        measured = json.loads(Path(result['assessment']).read_text())
        finite = all(t['metrics'] and all(math.isfinite(v) for v in t['metrics'].values())
                     for c in measured['conditions'] for t in c['trials'])
        if (finite and safe_candidate(measured, control)
                and quality(measured) > quality(best)):
            choice, best = result, measured
    return choice, best


def validate_learning_source(result):
    """Require intact exact-input diagnosis and the saved full-policy contract."""
    checkpoint = Path(result['checkpoint']).resolve()
    directory = Path(result['assessment']).parent
    measured = json.loads((directory/'assessment.json').read_text())
    diagnosis = json.loads((directory/'joint_pd_command_diagnosis.json').read_text())
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if (not checkpoint.is_relative_to(RUNS) or measured['checkpoint_sha256'] != digest
            or diagnosis['checkpoint_sha256'] != digest
            or diagnosis['actor_inference_parity_max'] > 1.e-4
            or not math.isfinite(diagnosis['actor_inference_parity_max'])
            or diagnosis['physical_contract_changed'] or diagnosis['weights_updated']):
        raise ValueError('Learning continuation has mismatched exact-input evidence')
    validate_contract(checkpoint.parent, 'joint_pd_balance_v1', fresh_weights=False)


def proposal_signature(overrides, std=None, reset_optimizer=False):
    return json.dumps([overrides, std, reset_optimizer], sort_keys=True)


def proposals(cfg, report, tried):
    """At most two independent single-variable trials per matched comparison."""
    errors = ratios(report)
    choices = []
    for metric, key, factors, bound in (
        ('yaw_mae_rad_s', 'yaw_rate_error_sq', (5., 2., 10.), 20.),
        ('body_p95_deg', 'orientation', (2., 1.5, 3.), 80.),
        ('speed_mae_m_s', 'lin_vel_error_sq', (2., 1.5, 3.), 12.),
        ('height_mae_m', 'base_height_sigma', (.5,), .0002),
    ):
        if errors[metric] <= 1.:
            continue
        value = cfg['rewards'][key]
        for factor in factors:
            updated = max(bound, value*factor) if key == 'base_height_sigma' else -min(bound, abs(value)*factor)
            change = {'rewards.'+key: updated}
            signature = proposal_signature(change)
            if updated != value and signature not in tried:
                choices.append((errors[metric], key, change, None, False))
                break
    choices.sort(key=lambda item: item[0], reverse=True)
    result = choices[:2]
    if not result:
        # Explore only after measured parameter trials are exhausted. Keep all
        # weights/heads and controller data; these are separate one-factor arms.
        for std, reset, label in ((.02, False, 'exploration'), (None, True, 'optimizer')):
            if proposal_signature({}, std, reset) not in tried:
                result.append((0., label, {}, std, reset))
    return result


def training_command(checkpoint, label, updates, overrides, *, std=None, reset_optimizer=False, commands=None):
    target = int(checkpoint.stem.split('_')[1]) + updates
    command = [str(ROOT/'run_python.sh'), str(ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py'),
        '--task', 'WheelLeggedJointPD-Flat-v0', '--require_leg_control_mode', 'implicit_joint_position',
        '--checkpoint_path', str(checkpoint), '--target_iteration', str(target),
        '--num_envs', '8192', '--seed', '43', '--run_name', label, '--headless', '--device', 'cuda:0',
        '--min_exploration_std', '.01']
    if std is not None:
        command.extend(['--reset_exploration_std', str(std)])
    if reset_optimizer:
        command.extend(['--reset_optimizer', 'agent.algorithm.learning_rate=0.00005'])
    if any(not key.startswith('rewards.') or key.split('.', 1)[1] not in FIELDS for key in overrides):
        raise ValueError('Autonomous trials may change only the admitted reward fields')
    command.extend('env.'+key+'='+json.dumps(value) for key, value in overrides.items())
    if commands is not None:
        command.extend('env.commands.'+key+'='+json.dumps(value) for key,value in commands.items())
    return command, target-1


def validate_changes(source_cfg, runtime_cfg, overrides):
    """A fresh process must preserve observations, commands and all physical data."""
    source_cfg = strip_retired_config_fields(source_cfg)
    runtime_cfg = strip_retired_config_fields(runtime_cfg)
    if set(runtime_cfg) - set(source_cfg) - {'log_dir'}:
        raise ValueError('Non-reward runtime parameter added')
    for key, value in source_cfg.items():
        if key == 'log_dir':
            continue
        if key == 'rewards':
            expected = dict(value)
            expected.update({name.split('.', 1)[1]: v for name, v in overrides.items()})
            for name,default in (('base_height_relative_deadband',0.),('wheel_action_saturation',0.),
                                 ('wheel_action_saturation_penalty_clip',40.),('tracking_goal_violation',0.),
                                 ('tracking_goal_penalty_clip',40.),('tracking_goal_component_violation',0.),
                                 ('tracking_goal_component_penalty_clip',200.)):
                if name in runtime_cfg[key]:
                    expected.setdefault(name,default)
            if runtime_cfg[key] != expected:
                raise ValueError('Unexpected reward changes outside the prepared comparison')
        elif key == 'commands':
            expected = dict(value)
            for name, default in (('height_expansion_from', None), ('height_expansion_start_steps', 0),
                                  ('height_expansion_ramp_steps', 48000)):
                if name in runtime_cfg[key]:
                    expected.setdefault(name, default)
            if runtime_cfg[key] != expected:
                raise ValueError('Non-reward runtime parameter changed: commands')
        elif runtime_cfg.get(key) != value:
            raise ValueError('Non-reward runtime parameter changed: '+key)


def run(args):
    if args.fresh or not args.checkpoint:
        raise ValueError('Direct-joint optimization requires an existing full-policy checkpoint')
    if not 1 <= args.round_iterations <= 200:
        raise ValueError('Use measured single-variable trials, each between 1 and 200 updates')
    prepared = prepared_reward_trial(args.initial_overrides)
    if args.num_envs not in (None, 8192) or args.max_rounds < 0:
        raise ValueError('This paired experiment uses 8192 environments and a nonnegative round limit')
    output, source = args.output.resolve(), args.checkpoint.resolve()
    if (output.parent != ROOT/'mine_model/results' or not output.name.startswith('joint_pd_optimization_')
            or not source.is_relative_to(RUNS) or not source.is_file()):
        raise ValueError('Output/source must belong to the scoped joint-PD project')
    output.mkdir(parents=True, exist_ok=True)
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    cfg = validate_checkpoint_robot(source)
    if cfg['leg_control_mode'] != 'implicit_joint_position' or cfg['rewards']['profile'] != 'joint_pd_balance_v1':
        raise ValueError('Incompatible policy/vehicle or reward family')
    with (output/'.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = dict(os.environ, PYTHONPATH=str(ROOT/'wheel_legged_isaaclab'),
                           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        runtime.OUTPUT = output
        state_path = output/'status.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if state and state.get('initial_source_sha256') != source_hash:
            raise ValueError('Cannot reuse an optimization directory with a different initial checkpoint')
        if state and state.get('prepared_reward_trial', {}) != prepared:
            raise ValueError('Prepared reward comparison changed; use a new experiment directory')
        if state.get('state') in ('diagnosis_required', 'error'):
            raise ValueError('Inspect the previous fault before manually restarting this directory')
        if state.get('state') in ('simulation_accepted_pending_sim2sim_and_randomization','completed_model_saved'):
            raise ValueError('Verified experiment is complete; inspect its saved delivery rather than retraining')
        watched = [ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
            ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
            ROOT/'mine_model/scripts/joint_pd_optimization.py',
            *[ROOT/'mine_model/scripts'/name for name in ('auto_train_mine.py',
                'complete_mine_training.py','summarize_mine_trial.py',
                'resume_after_solver_check_20261011.py','guard_mcu_vmc_training.py','watch_joint_pd_scratch.py')],
            ROOT/'mine_model/config/tracking_acceptance.json',
            *[ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab'/name
              for name in ('five_bar_vmc.py', 'leg_control.py', 'mcu_hardware.py',
                           'physics_contract.py', 'robot_model.py', 'reward_signals.py',
                           'config_overrides.py', 'training_state.py', 'yaw_training.py', 'action_bias_probe.py', 'height_commands.py')],
            *[ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat'/name
              for name in ('wheel_legged_vmc_flat_env.py', 'wheel_legged_vmc_flat_env_cfg.py',
                           'env_commands.py', 'agents/rsl_rl_ppo_cfg.py')]]
        fingerprints = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in watched}
        if state.get('source_files') and state['source_files'] != fingerprints:
            raise ValueError('Source changed: diagnose before restarting the previous comparison')
        state.update(initial_source=str(source), initial_source_sha256=source_hash,
                     num_envs=8192, full_policy_training=True, updates_per_arm=args.round_iterations,
                     no_hardware_or_controller_changes=True, acceptance_criteria_unchanged=True,
                     source_files=fingerprints,acceptance_criteria=acceptance_criteria(),
                     shutdown_after_success=args.shutdown_after_success, prepared_reward_trial=prepared)
        if state['acceptance_criteria']['relative_error']>.1:
            raise ValueError('Requested completion requires a 10 percent or stricter criterion')
        def save(phase, **values):
            state.update(dict(state=phase, training_active=phase=='training',
                              updated_at=datetime.now().astimezone().isoformat()), **values)
            write_json(state_path, state)
            print(json.dumps(dict(state=phase, **values), ensure_ascii=False), flush=True)
        def job(command, log, training=False):
            if (output/'STOP').exists():
                raise RuntimeError('user_stop_requested')
            save('training_initialization' if training else 'evaluation', command=command, log=str(log))
            runtime.run_child(command, log, environment, training=training, reporter=save,
                              cooperative_stop=training)
        def replay(checkpoint, folder):
            report = evaluate(checkpoint, folder, output/'STOP', progress=1.)
            diagnose_joint_pd(folder, folder/'joint_pd_command_diagnosis.json')
            return report
        def append_record(message):
            with (output/'optimization_report.md').open('a') as stream:
                stream.write(message+'\n\n')
        try:
            if not state.get('best_checkpoint'):
                save('baseline_evaluation')
                report = replay(source, output/'baseline')
                save('baseline_ready', best_checkpoint=str(source), best_assessment=str(output/'baseline/assessment.json'),
                     cycle=0, cache={})
                append_record('# 关节角策略偏航和姿态持续优化\n\n'
                    '以完整静止、前后速度、双向转向和变高度测试为准；每次记录实际参数和独立回放。'
                    '无负载轮通道验证不作为自由平衡验收。所有腿、轮和共享特征继续训练。')
            cycle = state.get('cycle', 0)
            while not args.max_rounds or cycle < args.max_rounds:
                if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p, h in fingerprints.items()):
                    raise RuntimeError('Source changed during the guarded experiment')
                source = Path(state.get('learning_checkpoint',state['best_checkpoint']))
                cfg = validate_checkpoint_robot(source)
                report = json.loads(Path(state.get('learning_assessment',state['best_assessment'])).read_text())
                retained = json.loads(Path(state['best_assessment']).read_text())
                identity = hashlib.sha256(source.read_bytes()).hexdigest()
                cache = state.setdefault('cache', {}).setdefault(identity, {})
                planning = json.loads(Path(state['failed_heldout']).read_text()) if state.get('failed_heldout') else report
                plans = proposals(cfg, planning, set(cache))
                if (prepared and proposal_signature(prepared) not in cache
                        and any(reward_value(cfg, key.split('.', 1)[1]) != value for key, value in prepared.items())):
                    label = ('height_reward_deadband' if 'rewards.base_height_relative_deadband' in prepared
                             else 'wheel_action_saturation')
                    plans = [(0., label, prepared, None, False)]
                if not plans:
                    control_result = cache.get(proposal_signature({}))
                    if control_result:
                        control = json.loads(Path(control_result['assessment']).read_text())
                        learner, measured = learning_choice(list(cache.values()), report, control)
                        if learner:
                            validate_learning_source(learner)
                            state.pop('failed_heldout',None)  # Previous policy's failure is not this policy's plan.
                            save('diagnosed_cumulative_continuation',learning_checkpoint=learner['checkpoint'],
                                 learning_assessment=learner['assessment'],child_pid=None,
                                 learning_ratios=ratios(measured),all_errors_under_10_percent=False,
                                 reason='measured_learning_progress_but_not_retained_or_accepted')
                            append_record('## 试验耗尽后的诊断与连续学习\n\n'
                                f'累计训练起点改为 `{learner["checkpoint"]}`。所有工况存活、无重置，'
                                '精确 actor 输入核验和完整控制合同通过，相对同源控制组无显著退化；'
                                '整体任务评分改善，但不将其标为保留模型或最终达标。'
                                '旧保留模型不覆盖，下一段从该分支实际继续更新，而非回到旧起点。')
                            continue
                    save('diagnosis_required', reason='admitted_single_variable_trials_exhausted_without_improvement',
                         child_pid=None,all_errors_under_10_percent=False, automatic_hardware_change=False,
                         learning_source=str(source),matched_control_checked=bool(control_result))
                    return
                plans.insert(0, (0., 'control', {}, None, False))
                folder = output/f'cycle_{cycle:03d}'
                folder.mkdir(exist_ok=True)
                outcomes = []
                for _, name, delta, std, reset in plans:
                    signature = proposal_signature(delta, std, reset)
                    if signature in cache:
                        outcomes.append(cache[signature])
                        continue  # Never blindly repeat an identical failed trial.
                    arm = folder/name
                    arm.mkdir(exist_ok=False)
                    # Carry only the incumbent's admitted tuned reward values.
                    overrides = {'rewards.'+key: reward_value(cfg, key) for key in FIELDS}
                    overrides.update(delta)
                    label = f'joint_pd_opt_{output.name}_c{cycle:03d}_{name}'
                    command, final = training_command(source, label, args.round_iterations, overrides,
                                                     std=std, reset_optimizer=reset, commands=cfg['commands'])
                    command+=['--stop_file',str(output/'STOP')]
                    save('trial_prepared', cycle=cycle, arm=name, trial_source=str(source),
                         delta=delta, reset_exploration_std=std, reset_optimizer=reset, target_iteration=final+1)
                    write_json(arm/'launch.json', dict(source_checkpoint=str(source), source_sha256=identity,
                        command=command, changed_parameters=delta, reset_exploration_std=std,
                        reset_optimizer=reset, updates=args.round_iterations, full_policy_training=True))
                    job(command, arm/'training.log', training=True)
                    runs = list(RUNS.glob('*_'+label))
                    if len(runs) != 1:
                        raise RuntimeError('Ambiguous or missing owned training directory')
                    checkpoint = runs[0]/f'model_{final}.pt'
                    if not checkpoint.is_file():
                        raise RuntimeError('Target checkpoint missing; do not evaluate an earlier model')
                    validate_contract(runs[0], 'joint_pd_balance_v1', fresh_weights=False, source_checkpoint=source)
                    validate_changes(cfg, validate_checkpoint_robot(checkpoint), overrides)
                    measured = replay(checkpoint, arm/'replay')
                    result = dict(name=name, checkpoint=str(checkpoint), assessment=str(arm/'replay/assessment.json'),
                                  delta=delta, std=std, reset_optimizer=reset)
                    cache[signature] = result
                    outcomes.append(result)
                    save('trial_evaluated', last_result=result, last_ratios=ratios(measured))
                    append_record(f'## 对照轮次 {cycle+1} 的 {name} 分支\n\n'
                        f'修改项：`{json.dumps(delta, ensure_ascii=False)}`；探索重设 `{std}`；优化器重设 `{reset}`。'
                        f'全部 {args.round_iterations} 次更新从同一来源开始，8192 环境，seed 43，控制和命令分布不变。\n\n'
                        f'模型：`{checkpoint}`。回放存活 {measured["survivors"]}/{measured["total_trials"]}；'
                        f'重置 {measured["resets"]}；全指标通过 `{measured["passed"]}`。'
                        f'任务误差与门限比：`{json.dumps(ratios(measured), ensure_ascii=False)}`。')
                control = json.loads(Path(outcomes[0]['assessment']).read_text())
                winner, best = None, retained
                for result in outcomes:
                    measured = json.loads(Path(result['assessment']).read_text())
                    if safe_candidate(measured, control) and safe_candidate(measured, retained) and quality(measured) > quality(best):
                        winner, best = result, measured
                learner, learning_report = learning_choice(outcomes, report, control)
                cycle += 1
                if winner:
                    state.pop('failed_heldout',None)
                    save('promoted_measured_candidate', cycle=cycle, best_checkpoint=winner['checkpoint'],
                         best_assessment=winner['assessment'], winner=winner, best_ratios=ratios(best),
                         learning_checkpoint=winner['checkpoint'],learning_assessment=winner['assessment'])
                    append_record(f'## 对照轮次 {cycle} 的选择\n\n保留 `{winner["name"]}`，其余分支仅留作证据。'
                                  '选择同时核对各工况退化，不按训练总奖励排名。')
                elif learner:
                    validate_learning_source(learner)
                    state.pop('failed_heldout',None)
                    save('continue_measured_learning_branch',cycle=cycle,learning_checkpoint=learner['checkpoint'],
                         learning_assessment=learner['assessment'],learning_ratios=ratios(learning_report),
                         all_errors_under_10_percent=False,child_pid=None)
                    append_record(f'## 对照轮次 {cycle} 的连续学习\n\n'
                        f'从 `{learner["checkpoint"]}` 累计续训；整体任务评分相对本轮来源改善、'
                        '同源控制组退化和存活检查通过。仍有相对旧保留模型的任务权衡，'
                        '因此仅作学习分支，不覆盖保留模型、不放宽最终验收。')
                else:
                    save('planning_different_trial', cycle=cycle, reason='no_candidate_improves_incumbent_without_substantial_regression')
                    append_record(f'## 对照轮次 {cycle} 的选择\n\n没有合格改善，保留原模型；下一轮不重复相同失败设置。')
                if best['passed']:
                    checkpoint = Path(state['best_checkpoint'])
                    heldout = output/f'heldout_cycle_{cycle:03d}'
                    command = [str(ROOT/'run_python.sh'), str(ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
                        '--checkpoint', str(checkpoint), '--output', str(heldout), '--seconds', '30',
                        '--seeds', '183', '184', '185', '--record_policy_inputs', '--headless', '--device', 'cuda:0']
                    job(command, output/f'heldout_cycle_{cycle:03d}.log')
                    validated = assess_stability(heldout,20.)
                    if validated['passed']:
                        extended=output/f'extended_cycle_{cycle:03d}'
                        command=[str(ROOT/'run_python.sh'),str(ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
                            '--checkpoint',str(checkpoint),'--output',str(extended),'--seconds','60',
                            '--seeds','283','284','285','--record_policy_inputs','--headless','--device','cuda:0']
                        job(command,output/f'extended_cycle_{cycle:03d}.log')
                        sustained=assess_stability(extended,40.)
                        if not sustained['passed']:
                            save('extended_failed_continue_diagnosis',all_errors_under_10_percent=False,
                                 failed_heldout=str(extended/'assessment_stability.json'))
                            continue
                        save('simulation_accepted_pending_sim2sim_and_randomization', all_errors_under_10_percent=True,
                             heldout_assessment=str(heldout/'assessment_stability.json'),
                             extended_assessment=str(extended/'assessment_stability.json'),child_pid=None)
                        append_record('## 独立稳态复验\n\n两组不重叠的新种子各 30 秒和 60 秒通过；'
                                      '后 20 秒和 40 秒的每个 3 秒窗口均严格达标，仅确认平地仿真。')
                        command=[str(ROOT/'run_python.sh'),str(ROOT/'mine_model/scripts/complete_mine_training.py'),
                                 '--state',str(state_path),'--require-leg-control-mode','implicit_joint_position']
                        if args.shutdown_after_success:command.append('--shutdown')
                        runtime.run_child(command,output/'completion.log',environment)
                        return
                    save('heldout_failed_continue_diagnosis', all_errors_under_10_percent=False,
                         failed_heldout=str(heldout/'assessment_stability.json'))
            save('configured_round_limit_reached', all_errors_under_10_percent=False)
        except BaseException as error:
            if ((output/'STOP').exists() and isinstance(error, RuntimeError)
                    and str(error) in ('user_stop_requested',
                                      'User stop marker received during replay; no automatic thaw')):
                phase, details = stop_outcome(output)
                save(phase, training_active=False, child_pid=None, automatic_resume=False, **details)
                return
            save('diagnosis_required', reason=repr(error), all_errors_under_10_percent=False,
                 unknown_fault_no_automatic_restart=True)
            raise
