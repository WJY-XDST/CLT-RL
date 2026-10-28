"""Bounded, one-reward repair from the latest saved policy and complete Adam.

This is a diagnostic training block, not an acceptance controller. Even a
passing canonical replay must be reviewed against the retained best posture
and height envelope and independently sustained before any acceptance claim.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path

import torch
import yaml

from auto_train_mine import write_json
from assess_height_leg_command_tracking import assess_directory
from diagnose_policy_feedback import diagnose_joint_pd
from diagnose_reward_balance import audit as audit_rewards
from joint_pd_continuation import ROOT, RUNS, optimizer_step, sha, tensor_finite
from joint_pd_continuation import continuation_command, qualified_wheel_mass_asset
from joint_pd_optimization import ratios, TASK_GOAL_METRICS, REFERENCE_TRACKING_METRICS
import resume_after_solver_check_20261011 as runtime
from summarize_mine_trial import assess
from train_joint_pd_standing_first import verify_update
from train_joint_pd_standing_seed import (
    height_posture_regressions, nominal_standing_intact, saved_updates, strict_full_gate,
)
from watch_joint_pd_scratch import numerical_anomaly
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.joint_reset import NOMINAL_JOINT_RESET, reset_profile
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, validate_checkpoint_robot

GATE_SHA = '08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d'
UPDATES = 25
ORIENTATION_SCALE = -400.
METRICS = (*TASK_GOAL_METRICS, *REFERENCE_TRACKING_METRICS)


def corrected_config(saved):
    """Change only the existing orientation scale; never replace a controller."""
    commands, rewards = saved['commands'], saved['rewards']
    if (saved['leg_control_mode'] != 'implicit_joint_position'
            or reset_profile(saved) != NOMINAL_JOINT_RESET
            or tuple(commands['ranges_height']) != SAFE_BASE_HEIGHT_RANGE_M
            or not commands['full_height_range_from_start']
            or any(commands[k] != 0 for k in ('standing_only_steps', 'yaw_start_steps', 'reverse_ramp_start_steps'))
            or commands['motion_ramp_steps'] != 48000
            or rewards['orientation'] != -20.
            or rewards['base_height_error_sq'] != -50.
            or rewards['tracking_goal_component_violation'] != -.1):
        raise ValueError('Require the diagnosed normal-motion, full-height nominal-reset source')
    result = copy.deepcopy(saved)
    result['rewards']['orientation'] = ORIENTATION_SCALE
    return result


def repair_comparison(report, before, protected):
    """Report recovery separately from the original, unchanged acceptance."""
    if any(len(r['conditions']) != 9 or sum(len(c['trials']) for c in r['conditions']) != 27
           for r in (report, before, protected)):
        raise ValueError('Recovery comparison requires all original cases and seeds')
    retained = height_posture_regressions(report, protected)
    source_regressions = height_posture_regressions(report, before)
    def body_values(r):
        return [t['metrics']['body_p95_deg'] for c in r['conditions'] for t in c['trials']
                if t['survived'] and t.get('metrics')]
    def worst(r):
        values = body_values(r)
        # Failed first episodes have no valid steady-window MAE/P95. They
        # must neither crash reporting nor disappear from a full-scope claim.
        return max(values) if len(values) == 27 else None
    old_body, body = worst(before), worst(report)
    values = body_values(report)
    complete = len(values) == 27
    improvement = ((old_body-body)/max(old_body, 1.e-12)
                   if old_body is not None and body is not None else None)
    return dict(
        survivors=report['survivors'], resets=report['resets'],
        nominal_standing_intact=nominal_standing_intact(report),
        worst_body_p95_deg=body, source_worst_body_p95_deg=old_body,
        worst_body_improvement_fraction=improvement,
        surviving_trials_worst_body_p95_deg=max(values) if values else None,
        steady_metrics_missing_trials=27-len(values),
        source_height_posture_regressions=source_regressions,
        best_height_posture_retention_failures=retained,
        best_height_posture_retained=not retained,
        full_original_metric_ratios=ratios(report, METRICS) if complete else dict.fromkeys(METRICS),
        surviving_trials_metric_ratios=ratios(report, METRICS) if values else dict.fromkeys(METRICS),
        full_scope_steady_metrics_available=complete,
        strict_full_original_gate_passed=strict_full_gate(report),
        policy_accepted=False, safe_to_continue_unreviewed=False,
        independent_sustained_verification_completed=False,
        original_acceptance_unchanged=True,
        interpretation='Bounded recovery evidence only; remaining regressions and all original errors stay visible. '
            'Failed first episodes have no steady metrics: full-scope ratios/P95 are null, not zero or a survivors-only success.')


def replay_command(checkpoint, replay):
    return [str(ROOT/'run_python.sh'), str(ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
        '--checkpoint', str(checkpoint), '--output', str(replay), '--seconds', '20', '--seeds', '83', '84', '85',
        '--record_policy_inputs', '--record-reward-terms', '--diagnose_contacts', '--headless', '--device', 'cuda:0']


def prepare(previous, output):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if (previous.parent != ROOT/'mine_model/results' or output.parent != previous.parent
            or not previous.name.startswith('joint_pd_optimization_')
            or not output.name.startswith('joint_pd_optimization_') or previous == output):
        raise ValueError('Require separate owned optimization output directories')
    state = json.loads((previous/'status.json').read_text())
    old_plan = json.loads((previous/'experiment_plan.json').read_text())
    checkpoint = Path(state['checkpoint']).resolve()
    if (state['state'] != 'agent_diagnosis_required' or state['training_active']
            or not checkpoint.is_relative_to(RUNS.resolve())
            or sha(checkpoint) != state['checkpoint_sha256']):
        raise ValueError('Require the latest intact saved diagnostic boundary, not an older policy')
    source_replay = previous/f"updates_{state['additional_ppo_updates']:04d}"/'replay'
    report = json.loads((source_replay/'assessment.json').read_text())
    if (report['checkpoint_sha256'] != sha(checkpoint) or report['survivors'] != 27
            or report['total_trials'] != 27 or report['resets']
            or not nominal_standing_intact(report)
            or max(t['metrics']['body_p95_deg'] for c in report['conditions'] for t in c['trials']) <= 2.5):
        raise ValueError('Require the measured surviving but posture-regressed source')
    source_cfg = validate_checkpoint_robot(checkpoint)
    expected = corrected_config(source_cfg)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    start = saved_updates(checkpoint)
    if (not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
            or optimizer_step(saved) != start*20):
        raise ValueError('Source policy and complete Adam clock must be retained')
    asset, calibration = qualified_wheel_mass_asset(ASSETS/MANIFEST['wheel_mass_calibration'])
    if source_cfg['robot']['spawn']['usd_path'] != asset:
        raise ValueError('Source no longer uses the qualified owner-calibrated asset')
    pins = dict(old_plan['source_files'])
    watched = [checkpoint, *sorted((checkpoint.parent/'params').glob('*.yaml')),
        previous/'status.json', previous/'experiment_plan.json', previous/'protected_height_posture.json',
        *(source_replay/name for name in ('manifest.json', 'assessment.json', 'completed.json', 'trace.csv')),
        Path(__file__).resolve()]
    pins.update({str(p): sha(p) for p in watched})
    pins.update(calibration['asset_chain_sha256'])
    if any(not Path(p).is_file() or sha(p) != digest for p,digest in pins.items()):
        raise ValueError('Previous pinned code, physics, gate or retained weights changed')
    if sha(ROOT/'mine_model/config/tracking_acceptance.json') != GATE_SHA:
        raise ValueError('The original acceptance gate changed')
    output.mkdir(exist_ok=True)
    plan_path = output/'experiment_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if (plan['source_files'] != pins or plan['source_checkpoint'] != str(checkpoint)
                or plan['target_actual_ppo_updates'] != start+UPDATES
                or plan['only_reward_change'] != {'orientation': [-20., ORIENTATION_SCALE]}):
            raise ValueError('Inspect the existing pinned repair before changing it')
    else:
        plan = dict(created_at=datetime.now().astimezone().isoformat(),
            previous_output=str(previous), source_checkpoint=str(checkpoint), source_sha256=sha(checkpoint),
            source_replay=str(source_replay), protected_reference=str(previous/'protected_height_posture.json'),
            source_actual_ppo_updates=start, source_Adam_step=optimizer_step(saved), source_Adam_states=17,
            target_actual_ppo_updates=start+UPDATES, bounded_training_updates=UPDATES,
            run_name='joint_pd_posture_repair_'+output.name+'_u'+str(start)+'_'+datetime.now().strftime('%H%M%S_%f'),
            only_reward_change={'orientation': [-20., ORIENTATION_SCALE]},
            weights_std_Adam_and_adaptive_LR_reset=False, ordinary_full_policy_PPO=True,
            normal_speed_and_yaw_training_retained=True, full_height_range=list(SAFE_BASE_HEIGHT_RANGE_M),
            calibrated_asset=asset, source_files=pins, original_acceptance_sha256=GATE_SHA,
            policy_accepted=False, hardware_enabled=False, shutdown=False)
        write_json(plan_path, plan)
    return plan, expected


def run(previous, output, prepare_only=False):
    output = Path(output).resolve()
    plan, expected = prepare(previous, output)
    status = output/'status.json'
    if status.exists() and json.loads(status.read_text())['state'] != 'prepared':
        raise ValueError('Inspect this already-started repair; never repeat its source updates')
    state = dict(state='prepared', actual_ppo_updates=plan['source_actual_ppo_updates'],
        live_ppo_updates=plan['source_actual_ppo_updates'], checkpoint=plan['source_checkpoint'],
        target_actual_ppo_updates=plan['target_actual_ppo_updates'],
        policy_accepted=False, goal_remains_active=True, training_active=False)

    def save(phase, **values):
        if values.get('latest_update'):
            values['live_ppo_updates'] = values['latest_update']['iteration']+1
        state.update(state=phase, updated_at=datetime.now().astimezone().isoformat(),
                     training_active=phase in ('training_initialization', 'training'))
        state.update(values)
        write_json(status, state)
        print(json.dumps(dict(state=phase, **values)), flush=True)

    def unchanged():
        if any(sha(Path(p)) != digest for p,digest in plan['source_files'].items()):
            raise ValueError('Pinned source, physics, original gate or retained checkpoint changed')

    save('prepared')
    if prepare_only:
        return
    source = Path(plan['source_checkpoint'])
    with (output/'.controller.lock').open('a') as owned, (RUNS/'.cumulative_learning.lock').open('a') as global_lock:
        for handle in (owned, global_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Path(f'/run/user/{os.getuid()}/clt-rl-native-simulation-{os.getuid()}.lock').open('a') as native:
            fcntl.flock(native, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.OUTPUT, runtime.anomaly = output, numerical_anomaly
        environment = dict(os.environ, PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            unchanged()
            command, final, _ = continuation_command(source, validate_checkpoint_robot(source), plan['run_name'],
                UPDATES, expected['commands'], output/'STOP', reward_overrides={'rewards.orientation': ORIENTATION_SCALE},
                minimum_wheel_std=.03, joint_mean_guard_coef=0., joint_height_feedback_coef=0.)
            if any(flag in command for flag in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
                raise ValueError('A repair must retain every saved weight and complete Adam')
            write_json(output/'launch.json', dict(command=command, source_sha256=sha(source)))
            save('training_initialization')
            runtime.run_child(command, output/'training.log', environment, training=True, reporter=save, cooperative_stop=True)
            unchanged()
            runs = list(RUNS.glob('*_'+plan['run_name']))
            if len(runs) != 1:
                raise ValueError('Missing or ambiguous owned native repair run')
            checkpoint = runs[0]/f'model_{final}.pt'
            proof = verify_update(source, checkpoint, plan['target_actual_ppo_updates'], expected, source)
            start = yaml.safe_load((runs[0]/'params/curriculum_start.yaml').read_text())
            if (start['completed_ppo_updates'] != plan['source_actual_ppo_updates']
                    or start['initial_common_step_counter'] != plan['source_actual_ppo_updates']*48
                    or start['motion_progress'] <= 0.
                    or tuple(start['height_bounds_m']) != SAFE_BASE_HEIGHT_RANGE_M
                    or any(abs(start[k]-v)>1.e-6 for k,v in zip(
                        ('sampled_command_height_min_m', 'sampled_command_height_max_m'), SAFE_BASE_HEIGHT_RANGE_M))):
                raise ValueError('The first native reset changed the motion clock or full height range')
            proof['first_reset'] = start
            write_json(output/'resume_verification.json', proof)
            save('independent_replay', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                actual_ppo_updates=plan['target_actual_ppo_updates'], live_ppo_updates=plan['target_actual_ppo_updates'], child_pid=None)
            replay = output/'replay'
            command = replay_command(checkpoint, replay)
            runtime.run_child(command, output/'replay.log', environment, reporter=save)
            unchanged()
            report = assess(replay)
            if report['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('Replay did not use the latest saved repair policy')
            legs = assess_directory(replay)
            write_json(replay/'height_leg_command_assessment.json', legs)
            diagnose_joint_pd(replay, replay/'joint_pd_command_diagnosis.json')
            write_json(replay/'reward_balance_audit.json', audit_rewards(replay))
            comparison = repair_comparison(report,
                json.loads((Path(plan['source_replay'])/'assessment.json').read_text()),
                json.loads(Path(plan['protected_reference']).read_text()))
            comparison.update(actual_leg_static_command_tracking_passed=legs['actual_leg_static_command_tracking_passed'],
                              checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
            write_json(output/'comparison.json', comparison)
            save('agent_review_required', comparison=str(output/'comparison.json'), child_pid=None)
        except BaseException as error:
            candidates = [(saved_updates(p), p) for p in RUNS.glob('*_'+plan['run_name']+'/model_*.pt')]
            actual, checkpoint = max(candidates, key=lambda v:v[0]) if candidates else (plan['source_actual_ppo_updates'], source)
            save('agent_diagnosis_required', reason=repr(error), training_active=False,
                actual_ppo_updates=actual, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                all_weights_std_Adam_and_historical_branches_retained=True)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.previous, args.output, args.prepare_only)
