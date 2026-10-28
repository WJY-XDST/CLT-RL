"""One bounded fresh-policy experiment; fixed current reward and full commands.

Old checkpoints are read-only baselines. Training and replay are sequential;
the global controller lock and the native launcher lock forbid parallel runs.
No retuning, automatic extension, hardware deployment or shutdown is provided.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from auto_train_mine import write_json
from joint_pd_continuation import (
    ROOT, RUNS, continuation_command, optimizer_step, qualified_wheel_mass_asset,
    sha, tensor_finite, validate_continuation_config,
)
from joint_pd_optimization import ratios, TASK_GOAL_METRICS, REFERENCE_TRACKING_METRICS
import resume_after_solver_check_20261011 as runtime
from summarize_mine_trial import assess, assess_stability
from verify_joint_pd_resume import verify
from watch_joint_pd_scratch import evaluate, numerical_anomaly, validate_contract
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.joint_height_feedback import resolve_coefficient as height_coefficient
from wheel_legged_gym_isaaclab.joint_mean_guard import resolve_coefficient as guard_coefficient
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, validate_checkpoint_robot
from wheel_legged_gym_isaaclab.training_state import clocked_update_count, load_curriculum_clock

TOTAL_UPDATES = 500
BLOCK_UPDATES = 50
FIRST_REPLAY_UPDATES = 200
METRICS = (*TASK_GOAL_METRICS, *REFERENCE_TRACKING_METRICS)


def update_experiment_state(state, phase, **values):
    """Explicit status fields must not collide with computed keyword defaults."""
    state.update(values)
    state['state'] = phase
    state['training_active'] = values.get('training_active', phase == 'training')
    state['updated_at'] = datetime.now().astimezone().isoformat()
    if phase == 'training' and values.get('latest_update'):
        state['live_ppo_updates'] = int(values['latest_update']['iteration']) + 1
    elif 'actual_ppo_updates' in values:
        state['live_ppo_updates'] = values['actual_ppo_updates']
    return state


def next_replay_target(actual):
    """Let the random policy learn support before the user-requested first test."""
    if type(actual) is not int or not 0 <= actual <= TOTAL_UPDATES:
        raise ValueError('Actual update count must remain inside the bounded experiment')
    if actual < FIRST_REPLAY_UPDATES:
        return FIRST_REPLAY_UPDATES
    return min(TOTAL_UPDATES, (actual // BLOCK_UPDATES + 1) * BLOCK_UPDATES)


def fresh_command(baseline, cfg, label, stop_file, offset):
    command, _, _ = continuation_command(baseline, cfg, label, FIRST_REPLAY_UPDATES,
        cfg['commands'], stop_file, minimum_wheel_std=.03,
        joint_mean_guard_coef=guard_coefficient(baseline),
        joint_height_feedback_coef=height_coefficient(baseline))
    index = command.index('--checkpoint_path')
    del command[index:index + 2]
    command[command.index('--target_iteration') + 1] = str(FIRST_REPLAY_UPDATES)
    command.extend(['--curriculum_step_offset', str(offset), '--save_initial_checkpoint'])
    return command


def drift_summary(directory):
    """World XY displacement, never inferred from wheel reference errors."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    trace = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    trials = []
    for cid, case in enumerate(manifest['cases']):
        if case['speed'] or case['yaw']:
            continue
        for seed in manifest['seeds']:
            data = trace[(trace['case_id'] == cid) & (trace['seed'] == seed)]
            first = data[data['episode'] == 0]
            survived = (len(first) == len(data)
                and not np.any((first['terminated'] > 0) | (first['timeout'] > 0)))
            xy = np.column_stack([first['root_x_w_m'], first['root_y_w_m']])
            tail = first[first['time_s'] >= manifest['seconds_per_case'] - 3.]
            trials.append(dict(condition=case['name'], seed=seed, survived=bool(survived),
                displacement_xy_m=float(np.linalg.norm(xy[-1] - xy[0])),
                path_length_xy_m=float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
                steady_speed_xy_mae_m_s=(float(np.hypot(tail['speed_m_s'], tail['lateral_speed_m_s']).mean())
                                         if survived else None)))
    complete = bool(trials and all(trial['survived'] for trial in trials))
    return dict(trials=trials, all_stationary_trials_survived=complete,
        mean_displacement_xy_m=(float(np.mean([trial['displacement_xy_m'] for trial in trials])) if complete else None),
        max_displacement_xy_m=(max(trial['displacement_xy_m'] for trial in trials) if complete else None),
        definition='Actual world XY displacement over first episode; failed episodes cannot count as improvement')


def comparison(baseline, candidate, old_drift, new_drift):
    complete = (baseline['survivors'] == baseline['total_trials'] == candidate['total_trials']
                == candidate['survivors'] and not baseline['resets'] and not candidate['resets'])
    old, new = ratios(baseline, METRICS), ratios(candidate, METRICS)
    return dict(comparable_all_trials_survived=complete,
        survivors=candidate['survivors'], total_trials=candidate['total_trials'], resets=candidate['resets'],
        baseline_metric_limit_ratios=old, candidate_metric_limit_ratios=new,
        height_worst_relative_error_percent=(10. * new['height_mae_m'] if complete else None),
        height_worst_error_reduction_percent=(100. * (1. - new['height_mae_m'] / old['height_mae_m'])
                                               if complete else None),
        baseline_drift=old_drift, candidate_drift=new_drift,
        full_original_gate_passed=bool(candidate['passed']),
        independent_sustained_acceptance_verified=False,
        policy_accepted=False, acceptance_thresholds_unchanged=True)


def run(baseline, output):
    baseline, output = Path(baseline).resolve(), Path(output).resolve()
    if (not baseline.is_file() or not baseline.is_relative_to(RUNS.resolve())
            or output.parent != ROOT / 'mine_model/results'
            or not output.name.startswith('joint_pd_optimization_') or 'fresh500' not in output.name):
        raise ValueError('Require a saved baseline and a dedicated fresh500 experiment directory')
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.controller.lock').open('a') as own_lock, (RUNS / '.cumulative_learning.lock').open('a') as global_lock:
        for lock in (own_lock, global_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cfg = validate_checkpoint_robot(baseline)
        saved_baseline = torch.load(baseline, map_location='cpu', weights_only=True)
        if not tensor_finite(saved_baseline) or len(saved_baseline['optimizer_state_dict']['state']) != 17:
            raise ValueError('Old baseline must retain finite complete policy and Adam state')
        asset, calibration = qualified_wheel_mass_asset(ASSETS / MANIFEST['wheel_mass_calibration'])
        if cfg['robot']['spawn']['usd_path'] != asset or tuple(cfg['commands']['ranges_height']) != SAFE_BASE_HEIGHT_RANGE_M:
            raise ValueError('Require the current calibrated asset and full protected height range')
        offset = (int(saved_baseline['iter']) + 1) * 48
        watched = [*sorted((ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab').rglob('*.py')),
            ROOT / 'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
            ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
            ROOT / 'run_python.sh', ROOT / 'mine_model/config/tracking_acceptance.json',
            *[ROOT / 'mine_model/scripts' / name for name in (
                'train_joint_pd_fresh500.py', 'joint_pd_continuation.py', 'verify_joint_pd_resume.py',
                'joint_pd_optimization.py', 'watch_joint_pd_scratch.py', 'summarize_mine_trial.py',
                'resume_after_solver_check_20261011.py', 'guard_mcu_vmc_training.py')]]
        fingerprints = {str(path): sha(path) for path in watched}
        fingerprints.update(calibration['asset_chain_sha256'])
        fingerprints[str(baseline)] = sha(baseline)
        plan = dict(baseline_checkpoint=str(baseline), baseline_sha256=sha(baseline),
            baseline_adam_step=optimizer_step(saved_baseline), total_actual_ppo_updates=TOTAL_UPDATES,
            first_future_performance_replay_update=FIRST_REPLAY_UPDATES,
            updates_per_replay_after_first=BLOCK_UPDATES, num_envs=8192, training_seed=43,
            initial_policy='new_random_actor_critic_and_Adam; old branch preserved read-only',
            curriculum_step_offset=offset, full_height_range_from_first_reset=list(SAFE_BASE_HEIGHT_RANGE_M),
            commands=cfg['commands'], rewards=cfg['rewards'], rewards_changed=False,
            joint_mean_guard_coef=guard_coefficient(baseline), joint_height_feedback_coef=height_coefficient(baseline),
            staged_candidate_losses_enabled=False, wheel_mass_calibration=calibration,
            training_and_replay='strictly sequential; two global locks; no old-branch continuation',
            source_files=fingerprints, shutdown=False, hardware_enabled=False,
            acceptance_sha256=sha(ROOT / 'mine_model/config/tracking_acceptance.json'))
        plan = json.loads(json.dumps(plan))
        plan_path = output / 'experiment_plan.json'
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                raise ValueError('Existing fixed experiment differs; inspect before resuming')
        else:
            write_json(plan_path, plan)
        state = dict(actual_ppo_updates=0, history=[], goal_remains_active=True,
                     policy_accepted=False, all_errors_under_10_percent=False)
        if (output / 'status.json').exists():
            state.update(json.loads((output / 'status.json').read_text()))

        def save(phase, **values):
            update_experiment_state(state, phase, **values)
            write_json(output / 'status.json', state)
            print(json.dumps(dict(state=phase, **values), ensure_ascii=False), flush=True)

        def unchanged():
            if any(sha(path) != digest for path, digest in fingerprints.items()):
                raise ValueError('Pinned code, original gate, old checkpoint or asset changed')
            if (output / 'STOP').exists():
                raise RuntimeError('cooperative_stop_marker_present')

        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'wheel_legged_isaaclab') + ':' + str(ROOT / 'mine_model/scripts'),
            PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        runtime.OUTPUT = output
        runtime.anomaly = numerical_anomaly  # Random-policy falls are not numerical faults.
        prefix = 'joint_pd_fresh500_' + output.name

        def measure(checkpoint, actual):
            if actual < FIRST_REPLAY_UPDATES:
                raise ValueError('Performance replay is deferred until 200 complete PPO updates')
            unchanged()
            block = output / f'updates_{actual:04d}'
            block.mkdir(exist_ok=True)
            save('independent_replay', actual_ppo_updates=actual, checkpoint=str(checkpoint),
                 checkpoint_sha256=sha(checkpoint))
            baseline_dir = output / 'baseline_replay'
            old_report = (assess(baseline_dir) if (baseline_dir / 'completed.json').exists()
                          else evaluate(baseline, baseline_dir, output / 'STOP'))
            if old_report['checkpoint_sha256'] != plan['baseline_sha256']:
                raise ValueError('Baseline replay does not belong to the preserved latest main checkpoint')
            replay_dir = block / 'replay'
            measured = (assess(replay_dir) if (replay_dir / 'completed.json').exists()
                        else evaluate(checkpoint, replay_dir, output / 'STOP'))
            if measured['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('Milestone replay belongs to a different checkpoint')
            result = comparison(old_report, measured, drift_summary(baseline_dir), drift_summary(replay_dir))
            write_json(block / 'comparison.json', result)
            state['history'] = [entry for entry in state['history'] if entry['actual_ppo_updates'] != actual]
            state['history'].append(dict(actual_ppo_updates=actual, checkpoint=str(checkpoint), **result))
            save('milestone_measured', actual_ppo_updates=actual, checkpoint=str(checkpoint))

        try:
            checkpoints = []
            for path in RUNS.glob('*_' + prefix + '_*/model_*.pt'):
                if path.name == 'model_initial.pt':
                    continue
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17:
                    raise ValueError('Owned checkpoint is nonfinite; preserve and diagnose instead of resetting')
                count = clocked_update_count(saved, load_curriculum_clock(path))
                checkpoints.append((count, path))
            actual, source = max(checkpoints, default=(0, None), key=lambda entry: entry[0])
            if actual > TOTAL_UPDATES:
                raise ValueError('Experiment exceeded the explicitly bounded update count')
            if not source:
                initial_models = list(RUNS.glob('*_' + prefix + '_*/model_initial.pt'))
                if len(initial_models) > 1:
                    raise ValueError('Ambiguous initial models; do not select or initialize another policy')
                if initial_models:
                    source = initial_models[0]
                    initial = torch.load(source, map_location='cpu', weights_only=True)
                    if (not tensor_finite(initial) or initial['optimizer_state_dict']['state']
                            or clocked_update_count(initial, load_curriculum_clock(source), allow_initial=True) != 0):
                        raise ValueError('Initial-model recovery requires finite random weights and empty Adam')
                    validate_contract(source.parent, fresh_weights=True, source_checkpoint=baseline)
            if source is not None:
                validate_continuation_config(cfg, validate_checkpoint_robot(source), cfg['commands'])
            if (actual >= FIRST_REPLAY_UPDATES and actual % BLOCK_UPDATES == 0 and not any(
                    entry['actual_ppo_updates'] == actual for entry in state['history'])):
                measure(source, actual)  # A saved 500th update must not bypass its replay on recovery.
            while actual < TOTAL_UPDATES:
                unchanged()
                target = next_replay_target(actual)
                block = output / f'updates_{target:04d}'
                block.mkdir(exist_ok=True)
                label = prefix + '_u' + str(actual) + '_' + datetime.now().strftime('%H%M%S_%f')
                command = (fresh_command(baseline, cfg, label, output / 'STOP', offset) if source is None
                    else continuation_command(source, cfg, label, target - actual, cfg['commands'], output / 'STOP',
                        minimum_wheel_std=.03, joint_mean_guard_coef=guard_coefficient(baseline),
                        joint_height_feedback_coef=height_coefficient(baseline))[0])
                write_json(block / ('launch_' + label.rsplit('_', 1)[-1] + '.json'),
                    dict(command=command, actual_updates_before=actual, target_actual_updates=target,
                         source_checkpoint=str(source) if source else None, fresh_weights=source is None))
                save('training', actual_ppo_updates=actual, target_actual_updates=target,
                     checkpoint=str(source) if source else None, baseline_preserved=True, reason=None)
                runtime.run_child(command, block / ('training_' + label.rsplit('_', 1)[-1] + '.log'),
                                  environment, training=True, reporter=save, cooperative_stop=True)
                unchanged()
                runs = list(RUNS.glob('*_' + label))
                if len(runs) != 1:
                    raise ValueError('Native run missing or ambiguous')
                checkpoint = runs[0] / f'model_{target - 1}.pt'
                after = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if (not tensor_finite(after) or len(after['optimizer_state_dict']['state']) != 17
                        or clocked_update_count(after, load_curriculum_clock(checkpoint)) != target):
                    raise ValueError('Saved finite full-policy Adam progress does not match the bounded target')
                validate_contract(runs[0], fresh_weights=source is None, source_checkpoint=source or baseline)
                validate_continuation_config(cfg, validate_checkpoint_robot(checkpoint), cfg['commands'])
                start = yaml.safe_load((runs[0] / 'params/curriculum_start.yaml').read_text())
                if start['motion_progress'] != 1. or tuple(start['height_bounds_m']) != SAFE_BASE_HEIGHT_RANGE_M:
                    raise ValueError('Fresh experiment silently returned to a narrow command phase')
                if not np.allclose([start['sampled_command_height_min_m'], start['sampled_command_height_max_m']],
                                   SAFE_BASE_HEIGHT_RANGE_M, rtol=0., atol=1.e-6):
                    raise ValueError('Actual initial commands did not sample both current height endpoints')
                if source is None or source.name == 'model_initial.pt':
                    initial_path = source if source is not None else runs[0] / 'model_initial.pt'
                    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
                    if (initial['optimizer_state_dict']['state'] or not tensor_finite(initial)
                            or torch.equal(initial['model_state_dict']['actor.0.weight'], saved_baseline['model_state_dict']['actor.0.weight'])):
                        raise ValueError('New experiment did not start with independent fresh weights and empty Adam')
                    old, new = initial['model_state_dict'], after['model_state_dict']
                    rows = [not torch.equal(old['actor.6.weight'][i], new['actor.6.weight'][i]) for i in range(6)]
                    shared = not torch.equal(old['actor.0.weight'], new['actor.0.weight'])
                    critic = not torch.equal(old['critic.0.weight'], new['critic.0.weight'])
                    if not all(rows) or not shared or not critic:
                        raise ValueError('Fresh full-policy parameters did not all update')
                    verification = dict(fresh_actor_critic=True, initial_adam_empty=True, old_checkpoint_unchanged=True,
                        initial_checkpoint=str(initial_path), initial_checkpoint_sha256=sha(initial_path),
                        actual_ppo_updates=target,
                        optimizer_states=17, all_tensors_finite=True, current_full_height_range_from_start=True,
                        actor_rows_updated=rows, shared_actor_updated=shared, critic_updated=critic)
                else:
                    verification = verify(source, checkpoint)
                    if verification['actual_ppo_updates'] != target - actual:
                        raise ValueError('A continuation silently lost or added PPO updates')
                write_json(block / 'resume_verification.json', verification)
                actual, source = target, checkpoint
                save('independent_replay', actual_ppo_updates=actual, checkpoint=str(checkpoint),
                     checkpoint_sha256=sha(checkpoint), optimizer_step=optimizer_step(after))
                measure(checkpoint, actual)
            # Exactly 500 actual updates: no automatic extension or reward search.
            # Passing the ordinary gate alone never declares the persistent goal complete.
            if state['history'] and state['history'][-1]['full_original_gate_passed']:
                passed = True
                for name, seconds, seeds, stable in (
                        ('heldout', 30, (183, 184, 185), 20), ('sustained', 60, (283, 284, 285), 40)):
                    unchanged()
                    save('independent_sustained_revalidation', actual_ppo_updates=actual, verification=name)
                    directory = output / name
                    command = [str(ROOT / 'run_python.sh'), str(ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
                        '--checkpoint', str(source), '--output', str(directory), '--seconds', str(seconds),
                        '--seeds', *map(str, seeds), '--record_policy_inputs', '--diagnose_contacts', '--headless', '--device', 'cuda:0']
                    runtime.run_child(command, output / (name + '.log'), environment, reporter=save)
                    passed = bool(assess_stability(directory, stable)['passed']) and passed
                state['independent_sustained_acceptance_verified'] = passed
            unchanged()
            save('comparison_500_updates_complete', actual_ppo_updates=actual, checkpoint=str(source),
                 next_action='Agent reviews height, actual XY drift and all original gate metrics; no automatic extension',
                 baseline_and_all_new_checkpoints_preserved=True, training_active=False)
        except BaseException as error:
            save('agent_diagnosis_required', reason=repr(error), training_active=False,
                 baseline_and_all_new_checkpoints_preserved=True)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.baseline, args.output)
