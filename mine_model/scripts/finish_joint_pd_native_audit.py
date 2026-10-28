"""Finish a saved native audit outside Kit, without repeating any PPO update.

Kit's process shutdown is not a reliable return boundary for runpy callers.
This separate CPU owner validates immutable records and full saved Adam, then
optionally runs the unchanged independent canonical policy replay.
"""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from auto_train_mine import write_json
from assess_height_leg_command_tracking import assess_directory
from diagnose_policy_feedback import diagnose_joint_pd
from diagnose_reward_balance import audit as audit_rewards
from joint_pd_continuation import ROOT, RUNS, sha
import resume_after_solver_check_20261011 as runtime
from summarize_mine_trial import assess
from train_joint_pd_posture_repair import GATE_SHA, repair_comparison, replay_command
from train_joint_pd_reward_block import quadratic_height_discrepancy
from train_joint_pd_standing_first import verify_update
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


ALIGNMENT_KEYS = (
    'command_before_act_max_error', 'command_at_reward_max_error',
    'actor_observation_storage_max_error', 'action_storage_max_error',
    'action_native_max_error', 'old_mean_storage_max_error',
    'old_sigma_storage_max_error', 'native_reward_delivery_max_error',
    'timeout_bootstrap_storage_max_error',
)


def summarize_records(records):
    if len(records) != 2:
        raise ValueError('Require exactly the two completed native PPO updates')
    for number, record in enumerate(records, 1):
        errors = record['all_environment_alignment_max_errors']
        steps = record['optimizer_steps']
        if (record['update'] != number or record['total_environments'] != 8192
                or not record['native_alignment_recorded'] or not record['read_only_hooks']
                or record['added_loss_or_backward'] or not record['probe_RNG_unchanged']
                or set(errors) != set(ALIGNMENT_KEYS) or len(steps) != 20
                or any(not math.isfinite(v) or v < 0. or v > 5.e-6 for v in errors.values())):
            raise ValueError('Incomplete or failed native transition alignment')
        for step in steps:
            values = [step[k] for k in (
                'actor_gradient_norm_after_global_clip', 'critic_gradient_norm_after_global_clip',
                'all_parameter_gradient_norm_after_global_clip', 'adaptive_learning_rate',
                'height_reference_loss_change')]
            values += [step[when]['height_reference_normalized_squared_error']
                       for when in ('before', 'after')]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Nonfinite native optimizer audit')
            delta = (step['after']['height_reference_normalized_squared_error']
                     - step['before']['height_reference_normalized_squared_error'])
            if not math.isclose(delta, step['height_reference_loss_change'], abs_tol=1.e-12):
                raise ValueError('Fixed-input optimizer probe delta is inconsistent')
    steps = [step for record in records for step in record['optimizer_steps']]
    delta = np.array([step['height_reference_loss_change'] for step in steps])
    return dict(
        all_environment_alignment_max_errors={
            key:max(record['all_environment_alignment_max_errors'][key] for record in records)
            for key in ALIGNMENT_KEYS},
        transition_alignment_passed=True, optimizer_minibatches=len(steps),
        fixed_input_reference_proxy=dict(
            improved_minibatch_fraction=float(np.mean(delta < 0.)),
            median_normalized_squared_error_change=float(np.median(delta)),
            mean_normalized_squared_error_change=float(delta.mean())),
        all_recorded_probes_RNG_unchanged=True, rewards_and_control_unchanged=True,
        policy_accepted=False, independent_tracking_acceptance_not_proven_by_audit=True,
        interpretation='Paired before/after probes within each optimizer step only. '
            'Different minibatches must not be compared as a learning curve. '
            'FK reference proxies are not actual closed-loop tracking or acceptance.')


def terminal_training_service(unit):
    if not re.fullmatch(r'clt-joint-pd-native-alignment-[A-Za-z0-9-]+\.service', unit):
        raise ValueError('Require the exact owned native-audit service')
    content = subprocess.check_output([
        'systemctl', '--user', 'show', unit, '--property=ActiveState,MainPID,Result'], text=True)
    state = dict(line.split('=', 1) for line in content.splitlines() if '=' in line)
    if state.get('ActiveState') not in ('inactive', 'failed') or state.get('MainPID') != '0':
        raise ValueError('The audited native process is not confirmed terminal')
    return state


def run(output, training_service, replay=False):
    output = Path(output).resolve()
    if (output.parent != ROOT/'mine_model/results'
            or not output.name.startswith('joint_pd_optimization_')):
        raise ValueError('Require the owned existing audit output')
    plan = json.loads((output/'experiment_plan.json').read_text())
    source = Path(plan['source_checkpoint'])
    service_state = terminal_training_service(training_service)
    if (plan['actual_updates'] != 2 or plan['target_actual_ppo_updates'] != plan['source_actual_ppo_updates']+2
            or plan['rewards_changed'] or plan['control_changed']):
        raise ValueError('Require the unchanged two-update audit plan')

    def unchanged():
        if (any(sha(Path(path)) != digest for path, digest in plan['source_files'].items())
                or sha(ROOT/'mine_model/config/tracking_acceptance.json') != GATE_SHA):
            raise ValueError('Pinned sources, asset, gate or preserved weights changed')

    with (output/'.controller.lock').open('a') as owned, (RUNS/'.cumulative_learning.lock').open('a') as global_lock:
        for lock in (owned, global_lock):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        unchanged()
        paths = sorted(output.glob('audit_update_*.json'))
        records = [json.loads(path.read_text()) for path in paths]
        summary = summarize_records(records)
        for record in records:
            with np.load(output/f"rollout_{record['update']:04d}.npz", allow_pickle=False) as arrays:
                if (arrays['observations'].shape != (48, 32, 27)
                        or arrays['actions'].shape != (48, 32, 6)
                        or any(not np.isfinite(arrays[key]).all() for key in arrays.files)
                        or not np.array_equal(arrays['actions'], arrays['native_raw_actions'])):
                    raise ValueError('Incomplete or inconsistent persisted native rollout')
        runs = list(RUNS.glob('*_'+plan['run_name']))
        if len(runs) != 1:
            raise ValueError('Missing or ambiguous owned audit checkpoint')
        checkpoint = runs[0]/f"model_{plan['final_iteration']}.pt"
        proof = verify_update(source, checkpoint, plan['target_actual_ppo_updates'],
                              validate_checkpoint_robot(source), source)
        if proof['optimizer_step'] != plan['target_actual_ppo_updates']*20:
            raise ValueError('The full saved Adam clock is not continuous')
        write_json(output/'resume_verification.json', proof)
        summary.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                       actual_ppo_updates=plan['target_actual_ppo_updates'])
        write_json(output/'audit_summary.json', summary)
        old_status = json.loads((output/'status.json').read_text())
        if not (output/'pre_finalization_status.json').exists():
            write_json(output/'pre_finalization_status.json', old_status)
        write_json(output/'external_finalization_provenance.json', dict(
            native_service=training_service, native_service_state=service_state,
            finalized_by=str(Path(__file__).resolve()), finalizer_sha256=sha(Path(__file__)),
            records_sha256={str(path):sha(path) for path in paths},
            original_plan_and_training_records_unchanged=True, PPO_updates_repeated=0,
            native_close_did_not_return_to_in_process_finalizer=True, policy_accepted=False))
        state = dict(actual_ppo_updates=plan['target_actual_ppo_updates'], checkpoint=str(checkpoint),
                     checkpoint_sha256=sha(checkpoint), training_active=False, goal_remains_active=True,
                     policy_accepted=False)

        def save(phase, **values):
            state.update(state=phase, updated_at=datetime.now().astimezone().isoformat(),
                         replay_active=phase=='independent_replay', **values)
            write_json(output/'status.json', state)
            print(json.dumps(dict(state=phase, **values)), flush=True)

        save('native_audit_verified')
        if not replay:
            return
        replay_dir = output/'replay'
        try:
            if not (replay_dir/'completed.json').exists():
                if (output/'replay.log').exists():
                    raise ValueError('Inspect the incomplete owned replay rather than overwrite it')
                native_path = Path(f'/run/user/{os.getuid()}/clt-rl-native-simulation-{os.getuid()}.lock')
                with native_path.open('a') as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                runtime.OUTPUT = output
                environment = dict(os.environ, PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
                    PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                save('independent_replay')
                runtime.run_child(replay_command(checkpoint, replay_dir), output/'replay.log',
                                  environment, reporter=save)
            unchanged()
            report = assess(replay_dir)
            if report['checkpoint_sha256'] != sha(checkpoint):
                raise ValueError('Independent replay did not use the latest complete checkpoint')
            legs = assess_directory(replay_dir)
            write_json(replay_dir/'height_leg_command_assessment.json', legs)
            diagnose_joint_pd(replay_dir, replay_dir/'joint_pd_command_diagnosis.json')
            reward_audit = audit_rewards(replay_dir)
            samples = np.genfromtxt(replay_dir/'trace.csv', delimiter=',', names=True)
            cfg = json.loads((replay_dir/'resolved_env.json').read_text())
            dt = json.loads((replay_dir/'manifest.json').read_text())['policy_dt']
            discrepancy = quadratic_height_discrepancy(samples, cfg, dt)
            if discrepancy > 1.e-5:
                raise ValueError('Actual native quadratic height reward formula changed')
            reward_audit['quadratic_height_formula_max_abs_error'] = discrepancy
            write_json(replay_dir/'reward_balance_audit.json', reward_audit)
            previous = Path(plan['previous_output'])
            old_plan = json.loads((previous/'experiment_plan.json').read_text())
            comparison = repair_comparison(report,
                json.loads((previous/'replay/assessment.json').read_text()),
                json.loads(Path(old_plan['protected_reference']).read_text()))
            comparison.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                actual_leg_static_command_tracking_passed=legs['actual_leg_static_command_tracking_passed'])
            write_json(output/'comparison.json', comparison)
            save('agent_review_required', comparison=str(output/'comparison.json'), child_pid=None)
        except BaseException as error:
            save('agent_diagnosis_required', reason=repr(error), child_pid=None)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-service', required=True)
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    run(args.output, args.training_service, args.replay)
