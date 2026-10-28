"""One headless, standing-first fresh PPO chain, measured every 100 updates.

The calibrated asset and full safe height range never change. A standing
readiness check is only a curriculum prerequisite, not tracking acceptance.
All continuations retain the complete policy, learned std, Adam and adaptive
learning rate. There is no hardware, shutdown or parallel simulation path.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path

import torch
import yaml

from auto_train_mine import write_json
from joint_pd_continuation import (ROOT, RUNS, continuation_command, optimizer_step,
    qualified_wheel_mass_asset, sha, tensor_finite, validate_continuation_config)
from joint_pd_optimization import TASK_GOAL_METRICS, REFERENCE_TRACKING_METRICS, ratios
from train_joint_pd_fresh500 import comparison, drift_summary, update_experiment_state
import resume_after_solver_check_20261011 as runtime
from summarize_mine_trial import assess, assess_stability
from watch_joint_pd_scratch import evaluate, numerical_anomaly, validate_contract
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, validate_checkpoint_robot
from wheel_legged_gym_isaaclab.training_state import clocked_update_count, load_curriculum_clock

INTERVAL = 100
BASIC_REWARD_WARMUP_UPDATES = 1500
STAND_CASES = ('stand', 'height_low', 'height_high')
READINESS_METRICS = ('speed_mae_m_s', 'yaw_mae_rad_s', 'body_p95_deg')
RESTORE_SCALES = (0., -.1, -.25, -.5, -.75)


def simple_rewards(original):
    """Restore the historical joint_pd_balance_v1 startup scales, not VMC costs."""
    result = copy.deepcopy(original)
    result.update(orientation=-20., yaw_rate_error_sq=-1., joint_position_action_saturation=-.5,
        wheel_action_saturation=0., tracking_goal_violation=0.,
        tracking_goal_component_violation=0., base_height_relative_deadband=0.)
    return result


def rewards_for_revision(original, revision=None):
    """Apply only the explicitly authorized dense height cost, not late rewards."""
    result = simple_rewards(original)
    if revision is None:
        return result
    overrides = revision.get('overrides')
    if not isinstance(overrides, dict) or set(overrides) != {'base_height_error_sq'}:
        raise ValueError('Height revision may change only base_height_error_sq')
    coefficient = overrides['base_height_error_sq']
    if (type(coefficient) not in (int, float) or not math.isfinite(coefficient)
            or coefficient >= 0.):
        raise ValueError('Height penalty coefficient must be finite and negative')
    result.update(overrides)
    return result


def load_height_penalty_revision(output):
    """Pin the saved full-policy/Adam boundary of an opt-in reward change."""
    path = Path(output)/'height_penalty_revision.json'
    if not path.exists():
        return None
    revision = json.loads(path.read_text())
    rewards_for_revision({}, revision)
    updates = revision['effective_after_actual_ppo_updates']
    source = Path(revision['source_checkpoint']).resolve()
    if (type(updates) is not int or updates <= 0 or not source.is_file()
            or not source.is_relative_to(RUNS.resolve())
            or sha(source) != revision['source_checkpoint_sha256']):
        raise ValueError('Height revision requires an intact owned saved-update boundary')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    if (not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
            or clocked_update_count(saved, load_curriculum_clock(source)) != updates):
        raise ValueError('Height revision must retain all finite weights and complete Adam')
    return revision


def commands_for_stage(original, motion_start=None):
    result = copy.deepcopy(original)
    result.update(ranges_height=list(SAFE_BASE_HEIGHT_RANGE_M), full_height_range_from_start=True,
        height_expansion_from=None, height_expansion_start_steps=0, height_expansion_ramp_steps=48000)
    # Keep normal task ranges so full canonical replay never becomes a zero-only gate.
    # Sampling enables no motion until actual standing evidence releases this clock.
    start = 2_000_000_000 if motion_start is None else motion_start
    for key in ('standing_only_steps', 'reverse_ramp_start_steps', 'yaw_start_steps'):
        result[key] = start
    return result


def standing_readiness(assessment, seconds=20.):
    conditions = [c for c in assessment['conditions'] if c['name'] in STAND_CASES]
    if {c['name'] for c in conditions} != set(STAND_CASES) or len(conditions) != 3:
        raise ValueError('Require all three actual stationary height conditions')
    trials = [t for c in conditions for t in c['trials']]
    if len(trials) != 9 or any(len(c['trials']) != 3 for c in conditions):
        raise ValueError('Standing prerequisite requires three independent seeds per height')
    survived = [t for t in trials if t['survived']]
    ready = all(t['survived'] and t.get('metrics') and all(
        t['metrics'][key] < t['limits'][key] - 1.e-12 for key in READINESS_METRICS) for t in trials)
    return dict(ready=bool(ready), survivors=len(survived), trials=9,
        mean_first_lifetime_s=sum(seconds if t['survived'] else t['first_failure_s'] for t in trials)/9,
        height_relative_mae_percent_by_trial=[
            dict(condition=c['name'], seed=t['seed'],
                 percent=100*t['metrics']['height_mae_m']/c['height'] if t.get('metrics') else None)
            for c in conditions for t in c['trials']],
        readiness_metrics=list(READINESS_METRICS),
        original_limits_used=True, prerequisite_only_not_full_acceptance=True)


def support_stagnation(history):
    recent = history[-3:]
    return (len(recent) == 3 and recent[-1]['actual_ppo_updates'] >= BASIC_REWARD_WARMUP_UPDATES
        and len({x['stage'] for x in recent}) == 1
        and all(x['standing']['survivors'] == 0 for x in recent)
        and max(x['standing']['mean_first_lifetime_s'] for x in recent) < 3.
        and recent[-1]['standing']['mean_first_lifetime_s'] <=
            1.2*recent[0]['standing']['mean_first_lifetime_s'])


def learning_command(source, baseline, cfg, label, target, stop):
    actual = 0 if source is None else clocked_update_count(
        torch.load(source, map_location='cpu', weights_only=True), load_curriculum_clock(source),
        allow_initial=source.name == 'model_initial.pt')
    command = continuation_command(source or baseline, cfg, label, target-actual, cfg['commands'], stop,
        minimum_wheel_std=.03, joint_mean_guard_coef=0., joint_height_feedback_coef=0.)[0]
    if source is None:
        index = command.index('--checkpoint_path')
        del command[index:index+2]
        command[command.index('--target_iteration')+1] = str(target)
        command.extend(['--curriculum_step_offset', '0', '--save_initial_checkpoint'])
    return command


def verify_update(source, checkpoint, target, expected_cfg, baseline):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
            or clocked_update_count(saved, load_curriculum_clock(checkpoint)) != target):
        raise ValueError('Saved full-policy/Adam progress does not match the measured update target')
    initial = source is None or source.name == 'model_initial.pt'
    previous_path = checkpoint.parent/'model_initial.pt' if source is None else source
    previous = torch.load(previous_path, map_location='cpu', weights_only=True)
    if not tensor_finite(previous):
        raise ValueError('Source policy or Adam is nonfinite')
    if initial:
        if previous['optimizer_state_dict']['state']:
            raise ValueError('Fresh random snapshot unexpectedly has Adam history')
        old = torch.load(baseline, map_location='cpu', weights_only=True)
        if torch.equal(previous['model_state_dict']['actor.0.weight'], old['model_state_dict']['actor.0.weight']):
            raise ValueError('New experiment loaded the old actor instead of independent random weights')
    old, new = previous['model_state_dict'], saved['model_state_dict']
    rows = [not torch.equal(old['actor.6.weight'][i], new['actor.6.weight'][i]) for i in range(6)]
    shared = not torch.equal(old['actor.0.weight'], new['actor.0.weight'])
    critic = not torch.equal(old['critic.0.weight'], new['critic.0.weight'])
    if not all(rows) or not shared or not critic:
        raise ValueError('Not all six actor outputs, shared features and critic learned')
    current = validate_checkpoint_robot(checkpoint)
    validate_continuation_config(expected_cfg, current, expected_cfg['commands'])
    validate_contract(checkpoint.parent, fresh_weights=source is None, source_checkpoint=source or baseline)
    algorithm = yaml.safe_load((checkpoint.parent/'params/agent.yaml').read_text())['algorithm']
    if algorithm['class_name'] != 'PPO' or algorithm.get('joint_height_feedback_coef', 0.) or algorithm.get('joint_mean_guard_coef', 0.):
        raise ValueError('Standing-first baseline must use ordinary PPO without auxiliary gradient losses')
    return dict(source_checkpoint=str(previous_path), checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        actual_ppo_updates=target, optimizer_step=optimizer_step(saved), optimizer_states=17,
        all_tensors_finite=True, actor_rows_updated=rows, shared_actor_updated=shared, critic_updated=critic,
        fresh_initialization_only_once=True, continuation_resets_weights=False, continuation_resets_Adam=False,
        calibrated_asset_and_full_height_unchanged=True, physical_control_contract_unchanged=True,
        height_penalty_coefficient=current['rewards'].get('base_height_error_sq', 0.))


def run(baseline, output):
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    if (not baseline.is_file() or not baseline.is_relative_to(RUNS.resolve())
            or output.parent != ROOT/'mine_model/results' or not output.name.startswith('joint_pd_optimization_')
            or not output.name.endswith('_standing_first')):
        raise ValueError('Require a preserved main checkpoint and dedicated standing-first output')
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.controller.lock').open('a') as owned, (RUNS/'.cumulative_learning.lock').open('a') as global_lock:
        for lock in (owned, global_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = validate_checkpoint_robot(baseline)
        reward_revision = load_height_penalty_revision(output)
        active_base_rewards = rewards_for_revision(original['rewards'], reward_revision)
        saved_baseline = torch.load(baseline, map_location='cpu', weights_only=True)
        asset, calibration = qualified_wheel_mass_asset(ASSETS/MANIFEST['wheel_mass_calibration'])
        if not tensor_finite(saved_baseline) or len(saved_baseline['optimizer_state_dict']['state']) != 17:
            raise ValueError('Preserved main model lacks complete finite policy/Adam')
        if original['robot']['spawn']['usd_path'] != asset:
            raise ValueError('Use the owner-confirmed measured-mass asset')
        watched = [*sorted((ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab').rglob('*.py')),
            ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
            ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py', Path(__file__).resolve(),
            ROOT/'mine_model/scripts/joint_pd_continuation.py', ROOT/'mine_model/scripts/train_joint_pd_fresh500.py',
            ROOT/'mine_model/scripts/watch_joint_pd_scratch.py', ROOT/'mine_model/scripts/summarize_mine_trial.py',
            ROOT/'mine_model/scripts/resume_after_solver_check_20261011.py', ROOT/'run_python.sh',
            ROOT/'mine_model/config/tracking_acceptance.json', baseline]
        if reward_revision is not None:
            watched.extend([output/'height_penalty_revision.json',
                            Path(reward_revision['source_checkpoint'])])
        fingerprints = {str(p):sha(p) for p in watched}
        fingerprints.update(calibration['asset_chain_sha256'])
        plan = dict(baseline_checkpoint=str(baseline), baseline_sha256=sha(baseline),
            initial_policy='independent fresh random actor, critic, std and empty Adam; preserved main is never loaded',
            initial_rewards=simple_rewards(original['rewards']), initial_commands=commands_for_stage(original['commands']),
            joint_mean_guard_coef=0., joint_height_feedback_coef=0., check_every_actual_ppo_updates=INTERVAL,
            basic_reward_warmup_actual_PPO_updates=BASIC_REWARD_WARMUP_UPDATES,
            notification_policy='only after two consecutive stationary readiness replays; other monitoring remains internal',
            training_seed=43, num_envs=8192, headless=True, old_main_preserved=True,
            calibration=calibration, full_height_range_from_first_reset=list(SAFE_BASE_HEIGHT_RANGE_M),
            motion_release='two consecutive 100-update replays with 9/9 stationary trials surviving and original zero-speed/yaw/body limits',
            precision_restore='after >=1500 basic updates and confirmed standing: first restore composite -0.1 with motion still disabled; release motion only after two standing-precision replays; no height auxiliary loss',
            component_coefficient_stages=list(RESTORE_SCALES), source_files=fingerprints,
            original_full_acceptance_sha256=sha(ROOT/'mine_model/config/tracking_acceptance.json'),
            causal_mass_or_range_effect_isolated=False, hardware_enabled=False, shutdown=False)
        if reward_revision is not None:
            # Initial/history rewards stay truthful: the new cost begins only
            # after the recorded checkpoint, not at the random initialization.
            plan.update(active_base_rewards=active_base_rewards, height_penalty_revision=reward_revision)
        plan = json.loads(json.dumps(plan))
        if (output/'experiment_plan.json').exists():
            if json.loads((output/'experiment_plan.json').read_text()) != plan:
                raise ValueError('Pinned standing-first experiment changed; inspect before resuming')
        else:
            write_json(output/'experiment_plan.json', plan)
        state = dict(actual_ppo_updates=0, live_ppo_updates=0, stage='standing', motion_start=None,
            precision_index=0, history=[], policy_accepted=False, goal_remains_active=True)
        if (output/'status.json').exists():
            state.update(json.loads((output/'status.json').read_text()))

        def save(phase, **values):
            update_experiment_state(state, phase, **values)
            write_json(output/'status.json', state)
            print(json.dumps(dict(state=phase, **values), ensure_ascii=False), flush=True)

        def unchanged():
            if any(not Path(p).is_file() or sha(p) != digest for p,digest in fingerprints.items()):
                raise ValueError('Pinned source, calibrated asset, original gate or preserved main model changed')
            if (output/'STOP').exists():
                raise RuntimeError('cooperative_stop_marker_present')

        environment = dict(os.environ, PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        runtime.OUTPUT, runtime.anomaly = output, numerical_anomaly
        prefix = 'joint_pd_standing_first_'+output.name
        try:
            candidates, initials = [], []
            for path in RUNS.glob('*_'+prefix+'_*/model_*.pt'):
                if path.name == 'model_initial.pt':
                    initials.append(path)
                    continue
                payload = torch.load(path, map_location='cpu', weights_only=True)
                if not tensor_finite(payload) or len(payload['optimizer_state_dict']['state']) != 17:
                    raise ValueError('Owned checkpoint damaged; never restart random weights automatically')
                candidates.append((clocked_update_count(payload, load_curriculum_clock(path)), path))
            if len(initials) > 1:
                raise ValueError('Multiple random initial snapshots; do not silently choose another initialization')
            actual, source = max(candidates, default=(0, initials[0] if initials else None), key=lambda p:p[0])
            if reward_revision is not None and actual < reward_revision['effective_after_actual_ppo_updates']:
                raise ValueError('Cannot apply a reward revision by rolling back or resetting the policy')
            while True:
                unchanged()
                cfg = copy.deepcopy(original)
                cfg['commands'] = commands_for_stage(original['commands'], state['motion_start'])
                cfg['rewards'] = copy.deepcopy(active_base_rewards)
                cfg['rewards']['tracking_goal_component_violation'] = RESTORE_SCALES[state['precision_index']]
                # Recover a completed measurement boundary before starting the next learning block.
                measured = next((h for h in state['history'] if h['actual_ppo_updates'] == actual), None)
                if actual == 0 or actual % INTERVAL or measured is not None:
                    target = (actual//INTERVAL+1)*INTERVAL
                    block = output/f'updates_{target:04d}'
                    block.mkdir(exist_ok=True)
                    label = prefix+'_u'+str(actual)+'_'+datetime.now().strftime('%H%M%S_%f')
                    command = learning_command(source, baseline, cfg, label, target, output/'STOP')
                    write_json(block/('launch_'+label.rsplit('_',1)[-1]+'.json'), dict(command=command,
                        actual_updates_before=actual, target_actual_updates=target, stage=state['stage'],
                        source_checkpoint=str(source) if source else None, fresh_weights=source is None))
                    save('training', actual_ppo_updates=actual, target_actual_updates=target,
                        checkpoint=str(source) if source else None, baseline_preserved=True, reason=None,
                        height_penalty_coefficient=cfg['rewards'].get('base_height_error_sq', 0.))
                    runtime.run_child(command, block/('training_'+label.rsplit('_',1)[-1]+'.log'), environment,
                        training=True, reporter=save, cooperative_stop=True)
                    unchanged()
                    runs = list(RUNS.glob('*_'+label))
                    if len(runs) != 1:
                        raise ValueError('Owned native run missing or ambiguous')
                    checkpoint = runs[0]/f'model_{target-1}.pt'
                    proof = verify_update(source, checkpoint, target, cfg, baseline)
                    start = yaml.safe_load((runs[0]/'params/curriculum_start.yaml').read_text())
                    if (tuple(start['height_bounds_m']) != SAFE_BASE_HEIGHT_RANGE_M or
                            abs(start['sampled_command_height_min_m']-SAFE_BASE_HEIGHT_RANGE_M[0]) > 1.e-6 or
                            abs(start['sampled_command_height_max_m']-SAFE_BASE_HEIGHT_RANGE_M[1]) > 1.e-6):
                        raise ValueError('Native sampling lost full safe height coverage')
                    if state['motion_start'] is None and (start['motion_progress'] != 0. or any(start[k] != 0. for k in (
                            'sampled_command_speed_min_m_s', 'sampled_command_speed_max_m_s',
                            'sampled_command_yaw_min_rad_s', 'sampled_command_yaw_max_rad_s'))):
                        raise ValueError('Standing-first phase enabled motion before actual readiness')
                    write_json(block/'resume_verification.json', proof)
                    actual, source = target, checkpoint
                block = output/f'updates_{actual:04d}'
                save('independent_replay', actual_ppo_updates=actual, checkpoint=str(source), checkpoint_sha256=sha(source))
                baseline_dir, replay = output/'baseline_replay', block/'replay'
                old = assess(baseline_dir) if (baseline_dir/'completed.json').exists() else evaluate(baseline, baseline_dir, output/'STOP')
                result = assess(replay) if (replay/'completed.json').exists() else evaluate(source, replay, output/'STOP')
                if old['checkpoint_sha256'] != plan['baseline_sha256'] or result['checkpoint_sha256'] != sha(source):
                    raise ValueError('Independent replay belongs to a different saved policy')
                compared = comparison(old, result, drift_summary(baseline_dir), drift_summary(replay))
                standing = standing_readiness(result)
                metric_ratios = ratios(result, (*TASK_GOAL_METRICS, *REFERENCE_TRACKING_METRICS))
                entry = dict(actual_ppo_updates=actual, stage=state['stage'], checkpoint=str(source),
                    standing=standing, metric_limit_ratios=metric_ratios, **compared)
                state['history'] = [h for h in state['history'] if h['actual_ppo_updates'] != actual]+[entry]
                write_json(block/'comparison.json', entry)
                save('milestone_measured', actual_ppo_updates=actual, checkpoint=str(source), standing=standing)
                if result['passed'] and actual >= BASIC_REWARD_WARMUP_UPDATES:
                    passed = True
                    for name,seconds,seeds,stable in (('heldout',30,(183,184,185),20), ('sustained',60,(283,284,285),40)):
                        unchanged()
                        directory = block/name
                        command = [str(ROOT/'run_python.sh'),str(ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
                            '--checkpoint',str(source),'--output',str(directory),'--seconds',str(seconds),
                            '--seeds',*map(str,seeds),'--record_policy_inputs','--diagnose_contacts','--headless','--device','cuda:0']
                        if not (directory/'completed.json').exists():
                            runtime.run_child(command, block/(name+'.log'), environment, reporter=save)
                        passed = bool(assess_stability(directory, stable)['passed']) and passed
                    if passed:
                        save('full_gate_and_sustained_verified_requires_final_package_audit',
                            actual_ppo_updates=actual, independent_sustained_acceptance_verified=True,
                            training_active=False, policy_accepted=False)
                        return
                if support_stagnation(state['history']):
                    raise RuntimeError('three_100_update_replays_show_persistent_early_support_failure_inspect_code_before_more_training')
                recent = state['history'][-2:]
                ready_twice = len(recent)==2 and all(h['standing']['ready'] for h in recent)
                if ready_twice and not (output/'standing_confirmed.json').exists():
                    write_json(output/'standing_confirmed.json', dict(
                        event='standing_confirmed_notify_user', actual_ppo_updates=actual,
                        checkpoint=str(source), checkpoint_sha256=sha(source),
                        consecutive_verifications=[h['actual_ppo_updates'] for h in recent],
                        standing=standing, full_tracking_acceptance_passed=bool(result['passed']),
                        independent_sustained_acceptance_verified=False,
                        original_acceptance_unchanged=True, policy_accepted=False))
                if state['stage'] != 'standing' and any(h['standing']['ready'] for h in state['history'][:-1]) and not standing['ready']:
                    raise RuntimeError('previously_confirmed_standing_lost_preserve_latest_weights_and_diagnose')
                if state['stage'] == 'standing' and ready_twice and actual >= BASIC_REWARD_WARMUP_UPDATES:
                    state.update(stage='standing_precision', precision_index=1)
                    save('standing_confirmed_one_reward_restored_motion_still_disabled',
                         actual_ppo_updates=actual, component_scale=RESTORE_SCALES[1],
                         standing_is_not_full_acceptance=True)
                elif (state['stage'] == 'standing_precision' and ready_twice
                        and all(h['stage'] == 'standing_precision' for h in recent)):
                    state.update(stage='motion_ramp', motion_start=actual*48)
                    save('standing_confirmed_motion_released', actual_ppo_updates=actual,
                         standing_is_not_full_acceptance=True)
                elif (state['motion_start'] is not None and ready_twice and
                        actual*48-state['motion_start'] >= cfg['commands']['motion_ramp_steps'] and
                        result['survivors'] == result['total_trials'] and not result['passed']):
                    previous_quality = max(recent[0]['metric_limit_ratios'].values())
                    current_quality = max(metric_ratios.values())
                    if state['precision_index']==0 or current_quality < .98*previous_quality:
                        if state['precision_index'] < len(RESTORE_SCALES)-1:
                            state.update(stage='precision_restore', precision_index=state['precision_index']+1)
                            save('one_precision_term_restored', actual_ppo_updates=actual,
                                 component_scale=RESTORE_SCALES[state['precision_index']], height_auxiliary_loss_stays_disabled=True)
        except BaseException as error:
            save('agent_diagnosis_required', reason=repr(error), training_active=False,
                 retained_main_and_all_new_checkpoints_preserved=True)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline, args.output)
