"""Bounded full-policy repair with automated anomaly investigation.

Only this supervisor's process groups are stopped. An anomaly saves/interrogates
the latest owned checkpoint and publishes evidence; unknown faults never cause
a blind restart or a hardware/gain change. No model deletion or host shutdown.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import resume_after_solver_check_20261011 as runtime
from diagnose_policy_feedback import diagnose
from guard_mcu_vmc_training import completed_updates, replay_gate
from summarize_mine_trial import assess

ROOT = runtime.ROOT
SOURCE = ROOT / ('IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat/'
                 '2026-10-11_01-16-22_vmc_p128_guarded_to_1000_20261011/model_1000.pt')
OUTPUT = ROOT / 'mine_model/results/vmc_feedback_repair_20261011'
MILESTONES = (1004, *range(1100, 2001, 100))
REPAIR = {
    'rewards.lin_vel_penalty_clip': 80.,
    'rewards.orientation_penalty_clip': 30.,
    'rewards.action_smooth_wheel_multiplier': 5.,
    'rewards.action_smooth_penalty_clip': 3.,
    'rewards.termination': -200.,
}
SOURCE_FILES = [
    ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/reward_signals.py',
    *[ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat' / p
      for p in ('wheel_legged_vmc_flat_env.py', 'wheel_legged_vmc_flat_env_cfg.py', 'env_commands.py',
                'agents/rsl_rl_ppo_cfg.py')],
    Path(__file__).resolve(), ROOT / 'mine_model/scripts/guard_mcu_vmc_training.py',
    ROOT / 'mine_model/scripts/diagnose_policy_feedback.py',
    ROOT / 'mine_model/scripts/resume_after_solver_check_20261011.py',
    ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
    ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/yaw_training.py',
    ROOT / 'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
    ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/training_state.py',
]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_scale(iteration):
    # Do not fail an unpracticed full-range command during early curriculum.
    # Final acceptance explicitly requires scale == 1 and the unchanged 10% gate.
    return min(1., max(0., (iteration - 1000) / 1000.))


def training_command(checkpoint, milestone, label, overrides, first=False):
    command = runtime.training_command(checkpoint, milestone, label, overrides)
    command += ['--min_exploration_std', '.02',
                'agent.algorithm.entropy_coef=0.001']
    if first:
        # Same trained actor/critic, fresh optimizer for changed reward scales.
        # Reset exploration on ALL six outputs, not a wheel-only finetune.
        command += ['--reset_optimizer', '--reset_exploration_std', '.05',
                    'agent.algorithm.learning_rate=0.0001']
    return command


def saved_checkpoint(label):
    runs = list(runtime.LOG_ROOT.glob('*_' + label))
    if len(runs) != 1:
        raise RuntimeError('Cannot uniquely resolve this supervisor\'s checkpoint directory')
    choices = list(runs[0].glob('model_*.pt'))
    if not choices:
        raise RuntimeError('No saved checkpoint in owned training run')
    def number(path):
        return int(path.stem.split('_')[1])
    # Interrupted save is preferred when the same iteration was saved twice.
    return max(choices, key=lambda path: (number(path), 'interrupted' in path.stem))


def feedback_replay_gate(assessment, feedback):
    reason = replay_gate(assessment)
    if reason:
        return reason
    # Do not trust a reconstructed steady tail after a motion condition reset.
    if assessment.get('resets', 0) > 0:
        return 'frozen_replay_reset_requires_diagnosis'
    return feedback['stop_reason']


def evaluate(checkpoint, directory, overrides_path, iteration, environment):
    directory.mkdir(exist_ok=False)
    replay = directory / 'replay'
    command = [str(ROOT / 'run_python.sh'),
               str(ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
               '--checkpoint', str(checkpoint), '--overrides_json', str(overrides_path),
               '--output', str(replay), '--seconds', '30', '--seeds', '83', '84', '85',
               '--curriculum-progress', str(replay_scale(iteration)), '--record_policy_inputs',
               '--headless', '--device', 'cuda:0']
    (directory / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    runtime.run_child(command, directory / 'evaluation.log', environment)
    assessment = assess(replay)
    feedback = diagnose(replay)
    reason = feedback_replay_gate(assessment, feedback)
    result = dict(checkpoint=str(checkpoint), iteration=iteration,
                  curriculum_progress=replay_scale(iteration), stop_reason=reason,
                  survivors=assessment['survivors'], total_trials=assessment['total_trials'],
                  tracking_passed_at_test_scale=assessment['passed'],
                  all_errors_under_10_percent=assessment['passed'] and replay_scale(iteration) == 1.,
                  assessment=str(replay / 'assessment.json'),
                  feedback_diagnosis=str(replay / 'feedback_diagnosis.json'),
                  alerts=feedback['repeated_alerts'])
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def diagnose_training_stop(label, segment, reason, overrides_path, environment):
    """Automatically investigate a saved interrupted model, including direction."""
    record = dict(trigger=reason, automatic_restart=False, hardware_changed=False)
    try:
        checkpoint = saved_checkpoint(label)
        iteration = int(checkpoint.stem.split('_')[1])
        if (OUTPUT / 'STOP').exists():
            record.update(checkpoint=str(checkpoint), diagnosis_skipped='explicit_user_stop')
        else:
            runtime.save('automatic_anomaly_investigation', trigger=reason, checkpoint=str(checkpoint))
            record['replay'] = evaluate(checkpoint, segment / 'anomaly_diagnosis',
                                        overrides_path, iteration, environment)
    except Exception as error:
        record['diagnosis_error'] = str(error)
        record['next_check'] = 'Read owned training/evaluation logs and saved checkpoint; do not restart blindly'
    (segment / 'automatic_diagnosis.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    global OUTPUT, SOURCE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--preserve_optimizer', action='store_true',
                        help='Continue an already repaired checkpoint without resetting std/optimizer')
    parser.add_argument('--milestones', type=int, nargs='+', default=MILESTONES)
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    SOURCE = args.source.resolve()
    runtime.OUTPUT = OUTPUT
    if not OUTPUT.is_relative_to(ROOT / 'mine_model/results'):
        raise ValueError('Output must stay in this project\'s result directory')
    milestones = list(args.milestones)
    if not SOURCE.is_relative_to(runtime.LOG_ROOT) or not SOURCE.is_file():
        raise ValueError('Source must be a saved checkpoint in this project\'s model-specific log root')
    source_iteration = int(SOURCE.stem.split('_')[1])
    if milestones != sorted(set(milestones)) or milestones[0] < 1004 or milestones[-1] > 2000:
        raise ValueError('Ordered bounded milestones required, from >=1004 to <=2000')
    if milestones[0] <= source_iteration:
        raise ValueError('First milestone must advance the saved source checkpoint')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / '.supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUTPUT / 'launch.json').exists():
            raise RuntimeError('Existing launch: inspect it instead of duplicating the run')
        other = subprocess.run(['pgrep', '-f',
            '^[^ ]*python.*(scripts/rsl_rl/train[.]py|evaluate_mine_policy[.]py)'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if other.returncode == 0:
            raise RuntimeError('Another training/evaluation process is active')
        # Reuse the validated PHYSICS, not the old inaccurate movement policy.
        read = lambda path: json.loads(path.read_text())
        reason = runtime.qualified(
            read(runtime.QUALIFICATION / 'qualified_training_screen.json'),
            read(runtime.QUALIFICATION / 'assessment.json'),
            read(runtime.REGRESSION / 'summary.json'), fingerprint(runtime.SOURCE))
        if reason:
            raise RuntimeError(reason)
        hashes = {str(path.relative_to(ROOT)): fingerprint(path) for path in SOURCE_FILES}
        for path in SOURCE_FILES:
            target = OUTPUT / 'source_snapshot' / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        overrides = read(runtime.ASSET_OVERRIDES)
        overrides.update(REPAIR)
        overrides.update({
            'robot.spawn.articulation_props.solver_position_iteration_count': 128,
            'robot.spawn.articulation_props.solver_velocity_iteration_count': 8,
            'enable_external_forces_every_iteration': True,
            'commands.fixed_standing_groups': True, 'commands.standing_env_fraction': .25,
        })
        overrides_path = OUTPUT / 'replay_overrides.json'
        overrides_path.write_text(json.dumps(overrides, indent=2) + '\n')
        (OUTPUT / 'launch.json').write_text(json.dumps(dict(
            started_at=datetime.now().astimezone().isoformat(), source_checkpoint=str(SOURCE),
            source_sha256=fingerprint(SOURCE), source_iteration=source_iteration, num_envs=8192,
            fixed_standing_envs=2048, full_policy_training=True, milestones=milestones,
            hardware_changed=False, geometry_changed=False, controller_gains_changed=False,
            physics_hz=200, vmc_hz=200, policy_hz=50, overrides=overrides,
            first_segment_optimizer=('preserved from repaired checkpoint' if args.preserve_optimizer
                                     else 'fresh Adam, actor AND critic checkpoint retained, lr 1e-4'),
            first_segment_std=('preserved; learned floor .02' if args.preserve_optimizer
                               else 'all six entries .05; learned floor .02; no frozen head'),
            source_hashes=hashes,
            numerical_admission='supplemental FP32 gravity-qualified p128 screen; strict residual retained',
            pause_policy='stop/save owned child and automatically replay+diagnose; no blind restart',
            acceptance_criteria_unchanged=True, models_deleted=False,
        ), indent=2) + '\n')
        environment = dict(os.environ, PYTHONPATH=str(ROOT / 'wheel_legged_isaaclab'),
                           PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
        checkpoint = SOURCE
        for index, milestone in enumerate(milestones):
            if (OUTPUT / 'STOP').exists():
                raise RuntimeError('user_stop_requested')
            if any(fingerprint(ROOT / path) != value for path, value in hashes.items()):
                raise RuntimeError('source_changed_during_guarded_experiment')
            segment = OUTPUT / f'to_{milestone}'
            segment.mkdir()
            label = f'vmc_feedback_repair_to_{milestone}_20261011'
            if list(runtime.LOG_ROOT.glob('*_' + label)):
                raise RuntimeError('Run label already exists: ' + label)
            command = training_command(checkpoint, milestone, label, overrides,
                                       first=index == 0 and not args.preserve_optimizer)
            (segment / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
            runtime.save('training_initialization', checkpoint=str(checkpoint), milestone=milestone,
                         log=str(segment / 'training.log'), full_policy_training=True, num_envs=8192)
            try:
                runtime.run_child(command, segment / 'training.log', environment, training=True)
            except Exception as error:
                diagnosis = diagnose_training_stop(label, segment, str(error), overrides_path, environment)
                runtime.save('stopped_after_automatic_diagnosis', reason=str(error), diagnosis=diagnosis,
                             automatic_restart=False, all_errors_under_10_percent=False)
                return
            updates = completed_updates((segment / 'training.log').read_text(errors='replace'))
            if not updates or 'Tracking/wheel_reference_second_difference_rms_rad_s' not in updates[-1]['metrics']:
                diagnosis = diagnose_training_stop(label, segment, 'required_guard_telemetry_missing', overrides_path, environment)
                runtime.save('stopped_after_automatic_diagnosis', reason='required_guard_telemetry_missing', diagnosis=diagnosis)
                return
            checkpoint = saved_checkpoint(label)
            if checkpoint.name != f'model_{milestone}.pt':
                raise RuntimeError('Target checkpoint missing after normal training exit')
            runtime.save('checkpoint_evaluation', checkpoint=str(checkpoint), milestone=milestone)
            try:
                result = evaluate(checkpoint, segment / 'scheduled_diagnosis', overrides_path, milestone, environment)
            except Exception as error:
                runtime.save('stopped_for_evaluation_diagnosis', reason=str(error), checkpoint=str(checkpoint),
                             automatic_restart=False, logs=str(segment / 'scheduled_diagnosis'))
                return
            runtime.save('checkpoint_diagnosed', **result)
            if result['stop_reason']:
                runtime.save('stopped_after_automatic_diagnosis', **result, automatic_restart=False)
                return
            if result['all_errors_under_10_percent']:
                runtime.save('tracking_passed_simulation_only', **result)
                return
        runtime.save('bounded_repair_complete', **result,
                     next_action='inspect_remaining_precision_errors; no automatic hardware change or export')


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUTPUT.exists():
            runtime.OUTPUT = OUTPUT
            runtime.save('supervisor_stopped_for_diagnosis', reason=str(error), automatic_restart=False)
        raise
