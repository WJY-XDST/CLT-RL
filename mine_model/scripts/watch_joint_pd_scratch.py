"""Monitor one scoped joint-position PPO run without blindly extending training.

Only the service named in launch.json may be frozen. Checkpoint evaluations use
the saved contract, ordinary gravity and no guide. This watcher does not change
rewards, gains, assets, weights, hardware, or start replacement training.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from guard_mcu_vmc_training import anomaly, completed_updates, FATAL, replay_gate

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat'


def numerical_anomaly(updates, content):
    match = FATAL.search(content)
    if match:
        return 'native_or_cuda_error: ' + content[match.start():].splitlines()[0][:500]
    for update in updates[-3:]:
        if any(not math.isfinite(v) for v in update['metrics'].values()):
            return 'nonfinite_training_metric'
        if (update['metrics'].get('Episode_Termination/numerical', 0.) > 0.
                or update['metrics'].get('Episode_Count/numerical', 0.) > 0.):
            return 'numerical_termination'
    # Low initial reward and early random-policy falls are not numerical faults.
    return None


def standing_summary(assessment, seconds=20.):
    standing = next(c for c in assessment['conditions'] if c['name'] == 'stand')
    return {'iteration': assessment['iteration'], 'survivors': standing['survivors'],
            'trials': len(standing['trials']), 'passed': standing['passed'],
            'task_goal_passed': standing.get('task_goal_passed'),
            'reference_tracking_passed': standing.get('reference_tracking_passed'),
            'failed_metrics_by_seed': {str(t.get('seed', i)): t.get('failed_metrics')
                                       for i, t in enumerate(standing['trials'])},
            'evaluation_mode': 'deterministic_policy_mean',
            'mean_first_lifetime_s': sum(seconds if t['survived'] else t['first_failure_s']
                                         for t in standing['trials']) / len(standing['trials']),
            'metrics': [t['metrics'] for t in standing['trials'] if t['metrics']],
            'all_condition_survivors': assessment['survivors'],
            'curriculum_progress': assessment.get('curriculum_progress', 0.),
            'conditions': {c['name']: {
                'speed': c.get('speed', 0.), 'yaw': c.get('yaw', 0.),
                'height': c.get('height'), 'survivors': c['survivors'],
                'trials': len(c['trials']),
                'metrics': [t['metrics'] for t in c['trials'] if t['metrics']],
            } for c in assessment['conditions']},
            'motion_is_diagnostic_not_a_gate_during_standing_curriculum':
                assessment.get('curriculum_progress', 0.) == 0.}


def restore_standing_history(output, history):
    """Enrich recorded milestones, never replay or change an old gate result."""
    restored = []
    for entry in history:
        directory = Path(output) / f'checkpoint_{entry["iteration"]}'
        report = next((p for p in (directory / 'assessment_components.json',
                                   directory / 'assessment.json') if p.exists()), None)
        if report is None:
            restored.append(entry)
            continue
        assessment = json.loads(report.read_text())
        if assessment['iteration'] != entry['iteration']:
            raise ValueError('Saved assessment iteration differs from monitoring history')
        summary = standing_summary(assessment)
        if (summary['survivors'] != entry['survivors'] or summary['passed'] != entry['passed']
                or summary['trials'] != entry['trials']):
            raise ValueError('Saved assessment changes an existing milestone result')
        restored.append(summary)
    return restored


def support_anomaly(history):
    if len(history) < 2:
        return None
    if any(h['survivors'] == h['trials'] for h in history[:-1]) and history[-1]['survivors'] == 0:
        return 'previously_learned_standing_lost'
    # Three independent milestones, not three adjacent noisy PPO updates.
    recent = history[-3:]
    if (len(recent) == 3 and recent[-1]['iteration'] >= 600
            and all(h['survivors'] == 0 for h in recent)
            and max(h['mean_first_lifetime_s'] for h in recent) < 3.
            and recent[-1]['mean_first_lifetime_s'] <= 1.2 * recent[0]['mean_first_lifetime_s']):
        return 'persistent_early_support_failure_without_improvement'
    return None


def learning_anomaly(history):
    """Detect persistent command blindness after the motion curriculum starts.

    Three independent, surviving multi-seed replays are required. Changing
    command magnitudes is handled by normalizing MAE by the tested command.
    This is a diagnosis trigger, not a claim that the controller is broken.
    """
    recent = [h for h in history if h.get('curriculum_progress', 0.) >= .35][-3:]
    if len(recent) < 3:
        return None
    for names, command, metric, minimum in (
            (('forward_0p3', 'forward_0p6', 'reverse_0p3', 'reverse_0p6'),
             'speed', 'speed_mae_m_s', .1),
            (('spin_left', 'spin_right'), 'yaw', 'yaw_mae_rad_s', .05)):
        ratios = []
        for entry in recent:
            conditions = [entry.get('conditions', {}).get(name) for name in names]
            if any(not c or c['survivors'] != c['trials'] or len(c['metrics']) != c['trials']
                   or abs(c[command]) < minimum for c in conditions):
                break
            ratios.append(sum(sum(m[metric] / abs(c[command]) for m in c['metrics']) /
                              c['trials'] for c in conditions) / len(conditions))
        if len(ratios) == 3 and min(ratios) > .85 and ratios[0] - ratios[-1] < .05:
            return f'{command}_command_response_stagnant_three_milestones'
    return None


def checkpoint_milestones(launch):
    final = launch['iterations'] - 1
    milestones = launch.get('checkpoint_milestones', [200, 400, 600, 800, 999])
    if (not milestones or any(type(i) is not int or i < 0 or i > final for i in milestones)
            or sorted(set(milestones)) != milestones or milestones[-1] != final):
        raise ValueError('Milestones must be ordered, unique and include the final checkpoint')
    return milestones


def curriculum_progress(launch, iteration):
    if launch.get('fresh_weights', True):
        return None  # Preserve the original fresh-run full-command diagnostics.
    if (iteration == launch['iterations'] - 1 and
            launch.get('monitoring', {}).get('final_acceptance_requires_full_commands')):
        return 1.  # Final acceptance must not use a curriculum-reduced command.
    steps = iteration * launch['steps_per_env']
    return min(1., max(0., (steps - launch['standing_only_steps']) / launch['motion_ramp_steps']))


def validate_scope(launch, output):
    output = output.resolve()
    fresh = launch['fresh_weights'] is True
    kind = 'scratch' if fresh else 'motion'
    if output.parent != (ROOT / 'mine_model/results').resolve():
        raise ValueError('Watcher output must belong to this joint-PD experiment')
    if not output.name.startswith(f'joint_pd_{kind}_'):
        raise ValueError('Only fresh scratch or explicitly scoped resumed motion training is supported')
    if not re.fullmatch(rf'clt-joint-pd-{kind}-[0-9_]+\.service', launch['unit']):
        raise ValueError('Refuse to control an unrelated or legacy training service')
    if not re.fullmatch(rf'joint_pd_{kind}_1k_[0-9_]+', launch['run_name']):
        raise ValueError('Unexpected run label')
    if type(launch['fresh_weights']) is not bool or launch['leg_control_mode'] != 'implicit_joint_position':
        raise ValueError('Only explicit fresh/resume direct-joint-position training is supported')
    if Path(launch['training_log']).resolve() != output / 'training.log':
        raise ValueError('Training log must belong to this experiment')
    if not fresh:
        checkpoint = Path(launch['source_checkpoint']).resolve()
        if not checkpoint.is_file() or not checkpoint.is_relative_to(RUNS.resolve()):
            raise ValueError('Motion curriculum requires an existing joint-PD source checkpoint')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != launch['source_sha256']:
            raise ValueError('Source checkpoint changed since launch preparation')
        if checkpoint.name != f'model_{launch["source_iteration"]}.pt':
            raise ValueError('Source iteration does not match the identified checkpoint')
        if (any(type(launch[k]) is not int or launch[k] <= 0 for k in
                ('iterations', 'steps_per_env', 'standing_only_steps', 'motion_ramp_steps'))
                or launch['iterations'] <= launch['source_iteration']):
            raise ValueError('Invalid resumed curriculum clock or target')
        checkpoint_milestones(launch)


def unit_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState',
                             '-p', 'FreezerState', '-p', 'MainPID', '-p', 'Result'],
                            capture_output=True, text=True, timeout=15)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def save(output, value):
    value = dict(value, updated_at=datetime.now().astimezone().isoformat())
    temporary = output / 'status.json.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(output / 'status.json')


def freeze_for_diagnosis(unit, output, reason, latest, history):
    state = unit_state(unit)
    if state.get('ActiveState') == 'active' and state.get('FreezerState') != 'frozen':
        subprocess.run(['systemctl', '--user', 'freeze', unit], check=True, timeout=20)
    value = dict(state='paused_for_diagnosis', reason=reason, latest=latest,
                 standing_history=history, checkpoint_and_optimizer_preserved=True,
                 automatic_restart=False, diagnosis_boundary='symptoms are not proof of a physics or PPO code bug')
    save(output, value)
    (output / 'diagnosis.json').write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run_directory(label):
    candidates = [p for p in RUNS.glob('*_' + label) if p.is_dir()]
    if len(candidates) > 1:
        raise ValueError('Ambiguous experiment directory')
    return candidates[0] if candidates else None


def validate_contract(run, reward_profile=None, *, fresh_weights=True, source_checkpoint=None):
    import yaml
    contract = yaml.safe_load((run / 'params/control_contract.yaml').read_text())
    if (contract['leg_control_mode'] != 'implicit_joint_position' or contract['fresh_weights'] is not fresh_weights
            or not contract['leg_joint_gravity_feedforward']
            or contract['joint_pd_total_effort_limit'] != 'PhysxDrivePerformanceEnvelopeAPI:angular'
            or abs(contract['leg_effort_limit'] - 35.574) > 1e-5
            or abs(contract['wheel_effort_limit'] - 4.80944) > 1e-5):
        raise ValueError('Training runtime contract differs from the validated joint-PD and MCU contract')
    if reward_profile is not None:
        from wheel_legged_gym_isaaclab.robot_model import validate_robot_contract
        class Loader(yaml.SafeLoader):
            pass
        Loader.add_constructor('tag:yaml.org,2002:python/tuple', Loader.construct_sequence)
        cfg = yaml.load((run / 'params/env.yaml').read_text(), Loader=Loader)
        validate_robot_contract(cfg)
        if cfg['rewards'].get('profile') != reward_profile:
            raise ValueError('Training reward profile differs from the identified experiment')
    precision = yaml.safe_load((run / 'params/precision_mode.yaml').read_text())
    if precision['mode'] != 'full_policy' or precision['shared_actor_features_frozen']:
        raise ValueError('The joint-PD curriculum must train all actor features and outputs')
    if source_checkpoint:
        previous = yaml.safe_load((Path(source_checkpoint).parent / 'params/control_contract.yaml').read_text())
        for key, value in previous.items():
            if key not in ('fresh_weights',) and contract.get(key) != value:
                raise ValueError('Resumed physical contract changed: ' + key)


def replay_resources_available():
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=15, check=True)
    available = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    return int(result.stdout.splitlines()[0]) >= 2300 and available >= 2500 * 1024


def evaluate(checkpoint, directory, stop_path, progress=None):
    import numpy as np
    import torch
    from summarize_mine_trial import assess
    directory.mkdir(parents=True, exist_ok=True)
    command = [str(ROOT / 'run_python.sh'), str(ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
               '--checkpoint', str(checkpoint), '--output', str(directory), '--seconds', '20',
               '--seeds', '83', '84', '85', '--record_policy_inputs', '--diagnose_contacts',
               '--headless', '--device', 'cuda:0']
    if progress is not None:
        command.extend(['--curriculum-progress', str(progress)])
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'wheel_legged_isaaclab'),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', MPLCONFIGDIR='/tmp/clt_joint_pd_watch_mpl')
    with (directory / 'evaluation.log').open('w') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic()
        try:
            while process.poll() is None:
                if stop_path.exists():
                    raise RuntimeError('User stop marker received during replay; no automatic thaw')
                if time.monotonic() - started > 300:
                    raise TimeoutError('Frozen-policy replay exceeded five minutes')
                time.sleep(2)
            if process.returncode != 0:
                raise RuntimeError('Native replay failed: see evaluation.log')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
    # Native app.close may hide an exception in its exit code: require complete,
    # finite telemetry and the expected sample count instead of trusting exit 0.
    assessment = assess(directory)
    if progress is not None:
        assessment['curriculum_progress'] = progress
        (directory / 'assessment.json').write_text(json.dumps(assessment, indent=2) + '\n')
    data = np.load(directory / 'policy_inputs.npz')
    obs, means = data['observations'], data['means']
    if obs.shape[-1] != 27 or means.shape[-1] != 6 or not np.isfinite(obs).all() or not np.isfinite(means).all():
        raise ValueError('Invalid exact policy inputs or outputs')
    trace = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    failed = trace[trace['terminated'] > 0]
    causes = {key: int((failed[key] > 0).sum()) for key in trace.dtype.names if key.startswith('fail_')}
    diagnostic = dict(joint_action_indices=[0, 1, 3, 4], wheel_action_indices=[2, 5],
                      raw_action_clip_fraction_by_dimension=(np.abs(means) >= 1.).mean(axis=(0, 1)).tolist(),
                      observed_joint_relative_angle_min=obs[:, :, 9:13].min(axis=(0, 1)).tolist(),
                      observed_joint_relative_angle_max=obs[:, :, 9:13].max(axis=(0, 1)).tolist(),
                      termination_counts=causes,
                      limits='actuator torque telemetry is an estimate, not native implicit-drive torque',
                      checkpoint=str(checkpoint), observation_semantics='fixed-q-offset relative angles, not virtual lengths')
    weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
    parameters = weights['model_state_dict']
    if any(not torch.isfinite(v).all() for v in parameters.values() if torch.is_tensor(v)):
        raise ValueError('Checkpoint contains nonfinite policy parameters')
    std = parameters.get('std')
    if std is None and 'log_std' in parameters:
        std = parameters['log_std'].exp()
    diagnostic['exploration_std'] = std.tolist() if std is not None else None
    diagnostic['optimizer_learning_rates'] = [g['lr'] for g in weights['optimizer_state_dict']['param_groups']]
    diagnostic['curriculum_progress'] = progress
    (directory / 'feedback_diagnosis.json').write_text(json.dumps(diagnostic, indent=2) + '\n')
    if causes.get('fail_numerical', 0):
        raise ValueError('Numerical termination in frozen-policy replay; inspect feedback_diagnosis.json')
    return assessment


def monitor(launch, output, interval):
    unit = launch['unit']
    milestones = checkpoint_milestones(launch)
    final = launch['iterations'] - 1
    saved = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
    history = restore_standing_history(output, saved.get('standing_history', []))
    if not history and launch.get('baseline_assessment'):
        baseline = json.loads(Path(launch['baseline_assessment']).read_text())
        if (baseline['checkpoint_sha256'] != launch['source_sha256']
                or baseline['iteration'] != launch['source_iteration']):
            raise ValueError('Baseline assessment belongs to a different source checkpoint')
        history = [standing_summary(dict(baseline, curriculum_progress=0.))]
    checked = {h['iteration'] for h in history}
    previous_iteration = None
    progress_at = time.monotonic()
    contract_checked = False
    while True:
        content = (output / 'training.log').read_text(errors='replace') if (output / 'training.log').exists() else ''
        updates = completed_updates(content)
        latest = updates[-1] if updates else None
        iteration = latest['iteration'] if latest else None
        if iteration != previous_iteration:
            progress_at = time.monotonic()
            previous_iteration = iteration
        reason = (numerical_anomaly(updates, content) if launch['fresh_weights']
                  else anomaly(updates, content))
        state = unit_state(unit)
        run = run_directory(launch['run_name'])
        if reason:
            freeze_for_diagnosis(unit, output, reason, latest, history)
            return
        if (output / 'STOP').exists():
            freeze_for_diagnosis(unit, output, 'user_stop_no_resume', latest, history)
            return
        if state.get('FreezerState') == 'frozen':
            save(output, dict(state='externally_paused_no_resume', latest=latest, standing_history=history))
            return
        if run and (run / 'params/control_contract.yaml').exists() and not contract_checked:
            validate_contract(run, launch.get('reward_profile'), fresh_weights=launch['fresh_weights'],
                              source_checkpoint=launch.get('source_checkpoint'))
            contract_checked = True
        active = state.get('ActiveState') == 'active'
        for milestone in milestones:
            checkpoint = run / f'model_{milestone}.pt' if run else None
            if milestone in checked or iteration is None or iteration < milestone or not checkpoint or not checkpoint.exists():
                continue
            if time.time() - checkpoint.stat().st_mtime < 10:
                if not active:
                    time.sleep(10)
                continue
            if active and not replay_resources_available():
                # Re-evaluate after training releases memory; do not provoke OOM.
                save(output, dict(state='replay_deferred_for_resources', latest=latest, milestone=milestone))
                break
            pid = state.get('MainPID')
            if active:
                subprocess.run(['systemctl', '--user', 'freeze', unit], check=True, timeout=20)
            save(output, dict(state='checkpoint_evaluation', latest=latest, milestone=milestone,
                              run=str(run), standing_history=history))
            assessment = evaluate(checkpoint, output / f'checkpoint_{milestone}', output / 'STOP',
                                  curriculum_progress(launch, milestone))
            history.append(standing_summary(assessment))
            checked.add(milestone)
            reason = support_anomaly(history)
            if not launch['fresh_weights']:
                reason = reason or replay_gate(assessment) or learning_anomaly(history)
            if reason:
                freeze_for_diagnosis(unit, output, reason, latest, history)
                return
            after = unit_state(unit)
            if active:
                if ((output / 'STOP').exists() or after.get('ActiveState') != 'active'
                        or after.get('MainPID') != pid):
                    save(output, dict(state='cancelled_or_service_replaced_no_resume', standing_history=history))
                    return
                subprocess.run(['systemctl', '--user', 'thaw', unit], check=True, timeout=20)
            progress_at = time.monotonic()
        if not active:
            if iteration == final and final not in checked and run and (run / f'model_{final}.pt').exists():
                continue
            finished = iteration == final and run and (run / f'model_{final}.pt').exists()
            save(output, dict(state='completed' if finished else 'training_exited_early_needs_diagnosis',
                              latest=latest, service=state, run=str(run) if run else None,
                              standing_history=history, completed_does_not_mean_tracking_accepted=True))
            return
        if time.monotonic() - progress_at > 900:
            freeze_for_diagnosis(unit, output, 'no_completed_ppo_update_for_15_minutes', latest, history)
            return
        save(output, dict(state='monitoring', latest=latest, run=str(run) if run else None,
                          contract_verified=contract_checked, standing_history=history,
                          checkpoint_milestones=milestones, service=state,
                          metric_context={'training': 'sampled_actions_include_exploration',
                                          'checkpoint_replay': 'deterministic_policy_mean',
                                          'full_acceptance_is_not_body_goal_tracking_alone': True}))
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=15.)
    args = parser.parse_args()
    if not math.isfinite(args.interval) or args.interval < 1.:
        parser.error('interval must be finite and at least one second')
    output = args.output.resolve()
    launch = json.loads((output / 'launch.json').read_text())
    validate_scope(launch, output)
    with (output / '.watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            monitor(launch, output, args.interval)
        except Exception as error:
            saved = json.loads((output / 'status.json').read_text()) if (output / 'status.json').exists() else {}
            freeze_for_diagnosis(launch['unit'], output, 'watch_or_replay_error: ' + repr(error),
                                 saved.get('latest'), saved.get('standing_history', []))
            raise


if __name__ == '__main__':
    main()
