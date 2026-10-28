"""Owner-requested model_200 fork: strong height cost, 100 updates, full replay.

The old latest branch is retained, not deleted or silently resumed. The exact
model_200 weights, exploration, 17 Adam states and adaptive LR are loaded from
the existing byte-identical clock-adopted snapshot. This is a review boundary,
never a declaration that standing or one good height completes optimization.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from auto_train_mine import write_json
from assess_height_leg_command_tracking import assess_directory
from diagnose_policy_feedback import diagnose_joint_pd
from diagnose_reward_balance import audit as audit_rewards
from joint_pd_continuation import (
    ROOT, RUNS, continuation_command, optimizer_step, qualified_wheel_mass_asset,
    sha, tensor_finite, validate_continuation_config,
)
import resume_after_solver_check_20261011 as runtime
from summarize_mine_trial import assess
from train_joint_pd_posture_repair import GATE_SHA, repair_comparison, replay_command
from train_joint_pd_reward_block import quadratic_height_discrepancy
from train_joint_pd_standing_first import verify_update
from train_joint_pd_standing_seed import nominal_standing_intact, saved_updates
from watch_joint_pd_scratch import numerical_anomaly
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.joint_reset import NOMINAL_JOINT_RESET, nominal_reset_config
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, validate_checkpoint_robot

SOURCE_SHA = '556ad6b47c108e3cb8522a700c8b0fec143666e3568d4d6de8b4afc6a85349d4'
SOURCE = RUNS/'2026-10-15_standing_seed_recovery_source_snapshot/model_200.pt'
HISTORICAL = RUNS/'2026-10-11_18-14-19_joint_pd_scratch_1k_20261011_181337/model_200.pt'
UPDATES = 100


def branch_config(source, current):
    """Only the already requested mass/reset, normal commands and reward bundle."""
    commands, rewards = current['commands'], current['rewards']
    if (source['leg_control_mode'] != 'implicit_joint_position'
            or current['leg_control_mode'] != 'implicit_joint_position'
            or current.get('joint_reset_profile') != NOMINAL_JOINT_RESET
            or tuple(commands['ranges_height']) != SAFE_BASE_HEIGHT_RANGE_M
            or not commands['full_height_range_from_start']
            or any(commands[k] != 0 for k in ('standing_only_steps', 'yaw_start_steps', 'reverse_ramp_start_steps'))
            or commands['motion_ramp_steps'] != 48000
            or rewards['base_height_error_sq'] != -500. or rewards['orientation'] != -400.
            or rewards['tracking_goal_component_violation'] != -.1
            or rewards['tracking_lin_vel'] <= 0. or rewards['tracking_ang_vel'] <= 0.):
        raise ValueError('Preserve the calibrated direct-PD full-height normal-motion strong-cost trial')
    expected = nominal_reset_config(source)
    expected['robot']['spawn']['usd_path'] = current['robot']['spawn']['usd_path']
    expected['commands'] = copy.deepcopy(commands)
    expected['rewards'] = copy.deepcopy(rewards)
    validate_continuation_config(expected, current, commands)
    return expected


def audit_first_reset(start, initial):
    if (start['completed_ppo_updates'] != initial
            or start['initial_common_step_counter'] != initial*48
            or start['motion_progress'] <= 0.
            or tuple(start['height_bounds_m']) != SAFE_BASE_HEIGHT_RANGE_M
            or any(abs(start[k]-value)>1.e-6 for k,value in zip(
                ('sampled_command_height_min_m', 'sampled_command_height_max_m'), SAFE_BASE_HEIGHT_RANGE_M))):
        raise ValueError('First reset must retain the source clock, full height and normal motion')


def prepare(previous, output):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if (previous.parent != ROOT/'mine_model/results' or output.parent != previous.parent
            or not output.name.startswith('joint_pd_optimization_') or previous == output):
        raise ValueError('Require a new owned source200 trial output')
    old_status = json.loads((previous/'status.json').read_text())
    retained = Path(old_status['checkpoint']).resolve()
    if (old_status['training_active'] or old_status['state'] != 'agent_review_required'
            or not retained.is_relative_to(RUNS.resolve()) or sha(retained) != old_status['checkpoint_sha256']
            or sha(SOURCE) != SOURCE_SHA or sha(HISTORICAL) != SOURCE_SHA):
        raise ValueError('Require the exact owner-selected model_200 and preserved terminal latest branch')
    saved = torch.load(SOURCE, map_location='cpu', weights_only=True)
    if (saved_updates(SOURCE) != 201 or optimizer_step(saved) != 4020
            or not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
            or (saved['model_state_dict']['std'] < .01).any()
            or (saved['model_state_dict']['std'][[2,5]] < .03).any()):
        raise ValueError('Exact model_200 complete Adam and saved exploration must be retained')
    old_cfg, current = validate_checkpoint_robot(SOURCE), validate_checkpoint_robot(retained)
    expected = branch_config(old_cfg, current)
    asset, calibration = qualified_wheel_mass_asset(ASSETS/MANIFEST['wheel_mass_calibration'])
    if expected['robot']['spawn']['usd_path'] != asset:
        raise ValueError('Both owner-confirmed wheel groups must remain 0.665 kg')
    parent_plan = json.loads((previous/'experiment_plan.json').read_text())
    ancestor = Path(parent_plan['previous_output'])
    ancestor_plan = json.loads((ancestor/'experiment_plan.json').read_text())
    protected = Path(ancestor_plan['protected_reference'])
    watched = [SOURCE, HISTORICAL, retained,
        *sorted((SOURCE.parent/'params').glob('*.yaml')), *sorted((retained.parent/'params').glob('*.yaml')),
        previous/'status.json', previous/'comparison.json', previous/'replay/assessment.json', protected,
        ROOT/'run_python.sh', ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
        ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
        ROOT/'mine_model/config/tracking_acceptance.json', Path(__file__).resolve(),
        ROOT/'mine_model/scripts/joint_pd_continuation.py',
        *sorted((ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab').rglob('*.py'))]
    pins = {str(path):sha(path) for path in watched}
    pins.update(calibration['asset_chain_sha256'])
    if sha(ROOT/'mine_model/config/tracking_acceptance.json') != GATE_SHA:
        raise ValueError('The original acceptance gate must not change')
    output.mkdir(exist_ok=True)
    status = output/'status.json'
    if status.exists() and json.loads(status.read_text())['state'] != 'prepared':
        raise ValueError('Never restart a started fork from model_200; inspect its latest saved boundary')
    plan_file = output/'experiment_plan.json'
    if plan_file.exists():
        plan = json.loads(plan_file.read_text())
        if plan['source_files'] != pins or plan['training_config'] != expected:
            raise ValueError('Do not silently alter an already prepared owner-requested fork')
    else:
        label = 'joint_pd_source200_strong_height_'+output.name+'_'+datetime.now().strftime('%H%M%S_%f')
        plan = dict(created_at=datetime.now().astimezone().isoformat(),
            owner_request='从200轮开始用加强惩罚然后测试；每100轮独立全工况回放；不盲目续训，持续优化直到原验收通过',
            source_checkpoint=str(SOURCE), historical_source=str(HISTORICAL), source_sha256=SOURCE_SHA,
            source_actual_ppo_updates=201, source_Adam_step=4020, source_Adam_states=17,
            retained_old_latest_checkpoint=str(retained), retained_old_latest_sha256=sha(retained),
            retained_old_latest_actual_ppo_updates=saved_updates(retained), previous_output=str(previous),
            target_actual_ppo_updates=201+UPDATES, bounded_training_updates=UPDATES,
            run_name=label, training_config=expected, calibrated_asset=asset,
            protected_reference=str(protected), source_files=pins,
            old_branch_discarded_as_training_source_not_deleted=True,
            weights_std_Adam_adaptive_LR_reset=False, ordinary_full_policy_PPO=True,
            height_penalty_coefficient=-500., orientation_coefficient=-400.,
            normal_speed_yaw_and_full_height_from_first_reset=True,
            original_acceptance_sha256=GATE_SHA, policy_accepted=False, goal_remains_active=True,
            review_boundary_is_not_goal_completion=True, hardware_enabled=False, shutdown=False)
        write_json(plan_file, plan)
    overrides = {'robot.spawn.usd_path':asset, 'joint_reset_profile':NOMINAL_JOINT_RESET}
    for group in ('commands', 'rewards'):
        overrides.update({group+'.'+key:value for key,value in expected[group].items()})
    write_json(output/'baseline_overrides.json', overrides)
    write_json(status, dict(state='prepared', checkpoint=str(SOURCE), checkpoint_sha256=SOURCE_SHA,
        actual_ppo_updates=201, training_active=False, policy_accepted=False, goal_remains_active=True))
    return plan


def run(previous, output, prepare_only=False):
    output = Path(output).resolve()
    plan = prepare(previous, output)
    if prepare_only:
        return
    expected, source = plan['training_config'], SOURCE
    state = dict(actual_ppo_updates=201, live_ppo_updates=201, checkpoint=str(source),
        checkpoint_sha256=SOURCE_SHA, target_actual_ppo_updates=plan['target_actual_ppo_updates'],
        policy_accepted=False, goal_remains_active=True)

    def save(phase, **values):
        if values.get('latest_update'):
            values['live_ppo_updates'] = values['latest_update']['iteration']+1
        state.update(state=phase, updated_at=datetime.now().astimezone().isoformat(),
            training_active=phase in ('training_initialization', 'training'))
        state.update(values)
        write_json(output/'status.json', state)
        print(json.dumps(dict(state=phase, **values)), flush=True)

    def unchanged():
        if any(sha(Path(path)) != digest for path,digest in plan['source_files'].items()):
            raise ValueError('Pinned sources, calibrated asset, original gate or retained checkpoints changed')

    def analyze(replay_dir, checkpoint):
        report = assess(replay_dir)
        if report['checkpoint_sha256'] != sha(checkpoint):
            raise ValueError('Independent replay used the wrong frozen checkpoint')
        legs = assess_directory(replay_dir)
        write_json(replay_dir/'height_leg_command_assessment.json', legs)
        diagnose_joint_pd(replay_dir, replay_dir/'joint_pd_command_diagnosis.json')
        reward_audit = audit_rewards(replay_dir)
        cfg = json.loads((replay_dir/'resolved_env.json').read_text())
        manifest = json.loads((replay_dir/'manifest.json').read_text())
        samples = np.genfromtxt(replay_dir/'trace.csv', delimiter=',', names=True)
        error = quadratic_height_discrepancy(samples, cfg, manifest['policy_dt'])
        if error > 1.e-5 or cfg['rewards']['base_height_error_sq'] != -500.:
            raise ValueError('The strengthened native height penalty is not the requested formula')
        reward_audit['quadratic_height_formula_max_abs_error'] = error
        write_json(replay_dir/'reward_balance_audit.json', reward_audit)
        return report, legs

    with (output/'.controller.lock').open('a') as owned, (RUNS/'.cumulative_learning.lock').open('a') as global_lock:
        for lock in (owned, global_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with Path(f'/run/user/{os.getuid()}/clt-rl-native-simulation-{os.getuid()}.lock').open('a') as native:
            fcntl.flock(native, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runtime.OUTPUT, runtime.anomaly = output, numerical_anomaly
        environment = dict(os.environ, PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        try:
            unchanged()
            baseline_dir = output/'initial_replay'
            command = replay_command(source, baseline_dir)+['--overrides_json', str(output/'baseline_overrides.json')]
            write_json(output/'baseline_launch.json', dict(command=command))
            save('frozen_source_transfer_replay', child_pid=None)
            runtime.run_child(command, output/'initial_replay.log', environment, reporter=save)
            before, _ = analyze(baseline_dir, source)
            if before['survivors'] != 27 or before['resets'] or not nominal_standing_intact(before):
                raise ValueError('Calibrated/full-height nominal-reset transfer lost the source standing skill')
            unchanged()
            command, final, _ = continuation_command(source, validate_checkpoint_robot(source), plan['run_name'],
                UPDATES, expected['commands'], output/'STOP',
                reward_overrides={'rewards.'+key:value for key,value in expected['rewards'].items()},
                robot_asset=plan['calibrated_asset'], joint_reset_profile=NOMINAL_JOINT_RESET,
                allow_reset_pose_change=True, minimum_wheel_std=.03,
                joint_mean_guard_coef=0., joint_height_feedback_coef=0.)
            if any(flag in command for flag in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
                raise ValueError('Do not reset any owner-selected model_200 weights, std or Adam')
            write_json(output/'launch.json', dict(command=command, source_sha256=SOURCE_SHA))
            save('training_initialization', child_pid=None)
            runtime.run_child(command, output/'training.log', environment, training=True,
                              reporter=save, cooperative_stop=True)
            unchanged()
            runs = list(RUNS.glob('*_'+plan['run_name']))
            if len(runs) != 1:
                raise ValueError('Missing or ambiguous owned source200 training run')
            checkpoint = runs[0]/f'model_{final}.pt'
            proof = verify_update(source, checkpoint, plan['target_actual_ppo_updates'], expected, source)
            start = yaml.safe_load((runs[0]/'params/curriculum_start.yaml').read_text())
            audit_first_reset(start, 201)
            if proof['optimizer_step'] != 6020:
                raise ValueError('The 100 actual PPO updates did not retain the full source Adam clock')
            proof.update(first_reset=start, historical_source_bytes_unchanged=sha(HISTORICAL)==SOURCE_SHA)
            write_json(output/'resume_verification.json', proof)
            save('independent_replay', checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                 actual_ppo_updates=plan['target_actual_ppo_updates'], live_ppo_updates=plan['target_actual_ppo_updates'], child_pid=None)
            replay_dir = output/'replay'
            runtime.run_child(replay_command(checkpoint, replay_dir), output/'replay.log', environment, reporter=save)
            unchanged()
            report, legs = analyze(replay_dir, checkpoint)
            comparison = repair_comparison(report, before, json.loads(Path(plan['protected_reference']).read_text()))
            comparison.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                actual_leg_static_command_tracking_passed=legs['actual_leg_static_command_tracking_passed'],
                source_initial_replay=str(baseline_dir), actual_training_updates_since_owner_fork=UPDATES,
                review_boundary_is_not_goal_completion=True)
            write_json(output/'comparison.json', comparison)
            save('agent_review_required', comparison=str(output/'comparison.json'), child_pid=None)
        except BaseException as error:
            candidates = [(saved_updates(path),path) for path in RUNS.glob('*_'+plan['run_name']+'/model_*.pt')]
            actual, checkpoint = max(candidates, key=lambda value:value[0]) if candidates else (201, SOURCE)
            save('agent_diagnosis_required', reason=repr(error), training_active=False,
                 actual_ppo_updates=actual, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                 retained_all_weights_std_Adam_and_old_branch=True)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.previous, args.output, args.prepare_only)
