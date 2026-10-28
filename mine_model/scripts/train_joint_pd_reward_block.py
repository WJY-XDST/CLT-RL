"""One diagnosed reward-scale revision, one bounded block, then full review.

Never resets weights, exploration or Adam; never auto-extends a trial or
accepts a policy. Physical control, normal motion commands and the retained
best height/posture evidence are immutable inputs to each revision.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from auto_train_mine import write_json
from assess_height_leg_command_tracking import assess_directory
from diagnose_policy_feedback import diagnose_joint_pd
from diagnose_reward_balance import audit as audit_rewards
from joint_pd_continuation import ROOT, RUNS, continuation_command, optimizer_step, sha
from joint_pd_continuation import qualified_wheel_mass_asset, tensor_finite
import resume_after_solver_check_20261011 as runtime
from summarize_mine_trial import assess
from train_joint_pd_posture_repair import GATE_SHA, repair_comparison, replay_command
from train_joint_pd_standing_first import verify_update
from train_joint_pd_standing_seed import nominal_standing_intact, saved_updates
from watch_joint_pd_scratch import numerical_anomaly
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.joint_reset import NOMINAL_JOINT_RESET, reset_profile
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, validate_checkpoint_robot


def revised_config(saved, key, value):
    """Only strengthen one existing diagnosed cost, without any action mapping."""
    bounds = {'base_height_error_sq': (-1000., -50.), 'orientation': (-400., -20.)}
    if (key not in bounds or type(value) not in (int, float) or not math.isfinite(value)
            or not bounds[key][0] <= value <= bounds[key][1]
            or not saved['rewards'][key] < 0. or not value < saved['rewards'][key]
            or value/saved['rewards'][key] > 10.):
        raise ValueError('Require one bounded strengthening of an existing cost')
    commands = saved['commands']
    if (saved['leg_control_mode'] != 'implicit_joint_position'
            or reset_profile(saved) != NOMINAL_JOINT_RESET
            or tuple(commands['ranges_height']) != SAFE_BASE_HEIGHT_RANGE_M
            or not commands['full_height_range_from_start']
            or any(commands[k] != 0 for k in ('standing_only_steps', 'yaw_start_steps', 'reverse_ramp_start_steps'))
            or commands['motion_ramp_steps'] != 48000):
        raise ValueError('Preserve normal motion, nominal reset and the current full safe height range')
    if (key == 'base_height_error_sq'
            and abs(value)*(SAFE_BASE_HEIGHT_RANGE_M[1]-SAFE_BASE_HEIGHT_RANGE_M[0])**2
            >= saved['rewards']['base_height_penalty_clip']):
        raise ValueError('Height cost would be flat across the full requested command span')
    result = copy.deepcopy(saved)
    result['rewards'][key] = float(value)
    return result


def quadratic_height_discrepancy(samples, cfg, dt):
    error = samples['height_m']-samples['height_cmd_m']
    expected = np.maximum(cfg['rewards']['base_height_error_sq']*error**2,
                          -cfg['rewards']['base_height_penalty_clip'])*dt
    recorded = samples['reward_base_height_error_sq']
    if not np.isfinite(expected).all() or not np.isfinite(recorded).all():
        raise ValueError('Nonfinite actual height reward telemetry')
    return float(np.max(np.abs(expected-recorded)))


def prepare(previous, output, key, value, updates):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if (previous.parent != ROOT/'mine_model/results' or output.parent != previous.parent
            or not previous.name.startswith('joint_pd_optimization_')
            or not output.name.startswith('joint_pd_optimization_') or previous == output
            or type(updates) is not int or not 1 <= updates <= 100):
        raise ValueError('Require separate owned paths and a bounded update budget')
    state = json.loads((previous/'status.json').read_text())
    old_plan = json.loads((previous/'experiment_plan.json').read_text())
    source = Path(state['checkpoint']).resolve()
    if (state['state'] not in ('agent_review_required', 'agent_diagnosis_required') or state['training_active']
            or not source.is_relative_to(RUNS.resolve()) or sha(source) != state['checkpoint_sha256']):
        raise ValueError('Use only the latest complete saved diagnostic boundary')
    replay = previous/'replay'
    if not (replay/'completed.json').exists():
        replay = previous/f"updates_{state['additional_ppo_updates']:04d}"/'replay'
    report = json.loads((replay/'assessment.json').read_text())
    if (report['checkpoint_sha256'] != sha(source) or report['survivors'] != 27
            or report['total_trials'] != 27 or report['resets'] or not nominal_standing_intact(report)):
        raise ValueError('Require complete independently surviving source evidence')
    assess_directory(replay)
    saved = torch.load(source, map_location='cpu', weights_only=True)
    actual = saved_updates(source)
    if (not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
            or optimizer_step(saved) != actual*20):
        raise ValueError('Preserve every weight and complete Adam clock')
    config = validate_checkpoint_robot(source)
    expected = revised_config(config, key, value)
    asset, calibration = qualified_wheel_mass_asset(ASSETS/MANIFEST['wheel_mass_calibration'])
    if config['robot']['spawn']['usd_path'] != asset:
        raise ValueError('The owner-calibrated asset must remain unchanged')
    protected = Path(old_plan.get('protected_reference', previous/'protected_height_posture.json'))
    diagnosis = previous/'learning_review_after_posture_repair.json'
    if key == 'base_height_error_sq':
        review = json.loads(diagnosis.read_text())
        if (review['checkpoint_sha256'] != sha(source) or review['actual_ppo_updates'] != actual
                or review['all_original_metric_ratios']['height_mae_m'] <= 1.):
            raise ValueError('Height strengthening requires the measured latest height failure review')
    pins = dict(old_plan['source_files'])
    watched = [source, *sorted((source.parent/'params').glob('*.yaml')),
        previous/'status.json', previous/'experiment_plan.json', protected, Path(__file__).resolve(),
        ROOT/'mine_model/scripts/diagnose_reward_balance.py',
        *(replay/name for name in ('manifest.json', 'assessment.json', 'completed.json', 'trace.csv'))]
    if diagnosis.exists():
        watched.append(diagnosis)
    pins.update({str(p): sha(p) for p in watched})
    pins.update(calibration['asset_chain_sha256'])
    if (any(sha(Path(p)) != digest for p,digest in pins.items())
            or sha(ROOT/'mine_model/config/tracking_acceptance.json') != GATE_SHA):
        raise ValueError('Prior sources, calibrated physics, retained weights or original gate changed')
    output.mkdir(exist_ok=True)
    path = output/'experiment_plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if (plan['source_files'] != pins or plan['source_checkpoint'] != str(source)
                or plan['reward_key'] != key or plan['reward_value'] != value
                or plan['bounded_training_updates'] != updates):
            raise ValueError('Do not silently change an already pinned revision')
    else:
        plan = dict(created_at=datetime.now().astimezone().isoformat(),
            previous_output=str(previous), source_checkpoint=str(source), source_sha256=sha(source),
            source_replay=str(replay), protected_reference=str(protected),
            source_actual_ppo_updates=actual, source_Adam_step=optimizer_step(saved), source_Adam_states=17,
            target_actual_ppo_updates=actual+updates, bounded_training_updates=updates,
            reward_key=key, reward_value=float(value), only_reward_change={key:[config['rewards'][key],float(value)]},
            run_name='joint_pd_reward_block_'+output.name+'_u'+str(actual)+'_'+datetime.now().strftime('%H%M%S_%f'),
            weights_std_Adam_and_adaptive_LR_reset=False, ordinary_full_policy_PPO=True,
            normal_speed_and_yaw_training_retained=True, full_height_range=list(SAFE_BASE_HEIGHT_RANGE_M),
            calibrated_asset=asset, source_files=pins, original_acceptance_sha256=GATE_SHA,
            policy_accepted=False, hardware_enabled=False, shutdown=False)
        write_json(path, plan)
    return plan, expected


def run(args, prepare_function=None):
    output = args.output.resolve()
    plan, expected = (prepare_function or prepare)(args.previous, output, args.reward_key, args.value, args.updates)
    status = output/'status.json'
    if status.exists() and json.loads(status.read_text())['state'] != 'prepared':
        raise ValueError('Inspect the already-started revision before continuing its latest checkpoint')
    source = Path(plan['source_checkpoint'])
    state = dict(actual_ppo_updates=plan['source_actual_ppo_updates'],
        live_ppo_updates=plan['source_actual_ppo_updates'], checkpoint=str(source),
        target_actual_ppo_updates=plan['target_actual_ppo_updates'], policy_accepted=False, goal_remains_active=True)

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
            raise ValueError('Pinned code, evidence, physics, original gate or retained checkpoints changed')

    save('prepared')
    if args.prepare_only:
        return
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
                plan['bounded_training_updates'], expected['commands'], output/'STOP',
                reward_overrides={'rewards.'+plan['reward_key']:plan['reward_value']},
                minimum_wheel_std=.03, joint_mean_guard_coef=0., joint_height_feedback_coef=0.)
            if any(f in command for f in ('--reset_optimizer','--reset_exploration_std','--save_initial_checkpoint')):
                raise ValueError('Never reset a revision source')
            write_json(output/'launch.json',dict(command=command,source_sha256=sha(source)))
            save('training_initialization')
            runtime.run_child(command,output/'training.log',environment,training=True,reporter=save,cooperative_stop=True)
            unchanged()
            runs = list(RUNS.glob('*_'+plan['run_name']))
            if len(runs) != 1:
                raise ValueError('Missing or ambiguous owned revision run')
            checkpoint = runs[0]/f'model_{final}.pt'
            proof = verify_update(source,checkpoint,plan['target_actual_ppo_updates'],expected,source)
            start = yaml.safe_load((runs[0]/'params/curriculum_start.yaml').read_text())
            if (start['completed_ppo_updates'] != plan['source_actual_ppo_updates']
                    or start['initial_common_step_counter'] != plan['source_actual_ppo_updates']*48
                    or start['motion_progress'] <= 0. or tuple(start['height_bounds_m']) != SAFE_BASE_HEIGHT_RANGE_M
                    or any(abs(start[k]-v)>1.e-6 for k,v in zip(
                        ('sampled_command_height_min_m','sampled_command_height_max_m'),SAFE_BASE_HEIGHT_RANGE_M))):
                raise ValueError('Native first reset changed the normal clock or full height range')
            proof.update(first_reset=start,only_reward_change=plan['only_reward_change'])
            write_json(output/'resume_verification.json',proof)
            save('independent_replay',checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                actual_ppo_updates=plan['target_actual_ppo_updates'],live_ppo_updates=plan['target_actual_ppo_updates'],child_pid=None)
            replay = output/'replay'
            runtime.run_child(replay_command(checkpoint,replay),output/'replay.log',environment,reporter=save)
            unchanged()
            report = assess(replay)
            if report['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('Replay checkpoint differs from the latest saved revision')
            legs = assess_directory(replay)
            write_json(replay/'height_leg_command_assessment.json',legs)
            diagnose_joint_pd(replay,replay/'joint_pd_command_diagnosis.json')
            reward_audit = audit_rewards(replay)
            samples = np.genfromtxt(replay/'trace.csv',delimiter=',',names=True)
            manifest = json.loads((replay/'manifest.json').read_text())
            resolved = json.loads((replay/'resolved_env.json').read_text())
            error = quadratic_height_discrepancy(samples,resolved,manifest['policy_dt'])
            if error > 1.e-5:
                raise ValueError('Recorded actual quadratic height cost disagrees with its declared coefficient')
            reward_audit['quadratic_height_cost_max_abs_error'] = error
            write_json(replay/'reward_balance_audit.json',reward_audit)
            comparison = repair_comparison(report,
                json.loads((Path(plan['source_replay'])/'assessment.json').read_text()),
                json.loads(Path(plan['protected_reference']).read_text()))
            comparison.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                actual_leg_static_command_tracking_passed=legs['actual_leg_static_command_tracking_passed'])
            write_json(output/'comparison.json',comparison)
            save('agent_review_required',comparison=str(output/'comparison.json'),child_pid=None)
        except BaseException as error:
            candidates = [(saved_updates(p),p) for p in RUNS.glob('*_'+plan['run_name']+'/model_*.pt')]
            actual, checkpoint = max(candidates,key=lambda v:v[0]) if candidates else (plan['source_actual_ppo_updates'],source)
            save('agent_diagnosis_required',reason=repr(error),training_active=False,actual_ppo_updates=actual,
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),all_weights_std_Adam_and_historical_branches_retained=True)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reward-key',choices=('base_height_error_sq','orientation'),required=True)
    parser.add_argument('--value',type=float,required=True)
    parser.add_argument('--updates',type=int,default=25)
    parser.add_argument('--prepare-only',action='store_true')
    run(parser.parse_args())
