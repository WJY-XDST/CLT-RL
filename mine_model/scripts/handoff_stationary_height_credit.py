"""Wait for a proven completed-block height diagnosis, then continue learning.

No stop marker, timer, update budget, training interruption or optimizer reset
is used. The currently running owner keeps its entire training/replay cycle.
"""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time


EXHAUSTION = 'RuntimeError: Bounded stationary repair costs exhausted; diagnose native data before further training'


def validate_completed_block(state, status, plan, proof, completed, journal):
    if state['state'] != 'block_review_complete' or EXHAUSTION not in journal:
        raise ValueError('No verified completed-block height-stagnation diagnosis; inspect the actual owner failure')
    target = plan['target_actual_ppo_updates']
    if (status['training_active'] or status['state'] not in ('agent_review_required', 'agent_diagnosis_required')
            or target-plan['source_actual_ppo_updates'] != 100 or plan['training_motion_scale'] != 0.
            or status['actual_ppo_updates'] != target or proof['actual_ppo_updates'] != target
            or proof['optimizer_states'] != 17 or proof['optimizer_step'] != target*20
            or not proof['all_tensors_finite'] or not all(proof['actor_rows_updated'])
            or len(proof['actor_rows_updated']) != 6 or not proof['shared_actor_updated'] or not proof['critic_updated']
            or status['checkpoint_sha256'] != proof['checkpoint_sha256']
            or completed['checkpoint_sha256'] != proof['checkpoint_sha256']
            or completed['actual_observed_updates'] != 100 or completed['optimizer_minibatches'] != 2000
            or completed['actual_ppo_updates'] != target or completed['PPO_updates_repeated'] != 0
            or not completed['all_probe_RNG_unchanged']
            or not completed['original_control_and_PPO_loss_unchanged']
            or len(completed['all_environment_alignment_max_errors']) != 9
            or any(completed['all_environment_alignment_max_errors'].values())):
        raise ValueError('Require complete finite native learning and unchanged ordinary PPO before handoff')


def inherit(previous_cursor, base, previous_unit):
    previous_cursor, base = Path(previous_cursor).resolve(), Path(base).resolve()
    base.mkdir(exist_ok=False)
    with (previous_cursor.parent/'.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                (base/'handoff_status.json').write_text(json.dumps(dict(
                    state='waiting_without_interrupting_current_reviewed_training',
                    updated_at=datetime.now().astimezone().isoformat(),
                    previous_unit=previous_unit, current_training_interrupted=False,
                    no_total_update_or_runtime_limit=True, goal_remains_active=True))+'\n')
                time.sleep(5)
        import continue_joint_pd_stationary_height_credit100 as next_owner
        from stationary_height_credit_decision import completed_height_trials, completed_output
        state = json.loads(previous_cursor.read_text())
        journal = subprocess.run(['journalctl', '--user', '-u', previous_unit, '--no-pager',
                                  '-o', 'cat', '-n', '100'], check=True, capture_output=True, text=True).stdout
        if state['state'] != 'block_review_complete' or EXHAUSTION not in journal:
            raise ValueError('No verified completed-block height-stagnation diagnosis; inspect the actual owner failure')
        history = state['history']
        height_trials = completed_height_trials(history)
        failed = [row for row in height_trials if not row['promote'] or row['after'] >= .99*row['before']]
        if len({row['coefficient'] for row in failed}) < 2:
            raise ValueError('Require independent evidence from distinct actual-height reward strengths')
        output = completed_output(state)
        if output != Path(history[-1]['output']):
            raise ValueError('Do not skip an unfinished or unreviewed native block')
        status = json.loads((output/'status.json').read_text())
        plan = json.loads((output/'experiment_plan.json').read_text())
        proof = json.loads((output/'resume_verification.json').read_text())
        completed = json.loads((output/'native_learning_signal/completed_external.json').read_text())
        validate_completed_block(state, status, plan, proof, completed, journal)
        if next_owner.core.sha(Path(status['checkpoint'])) != proof['checkpoint_sha256']:
            raise ValueError('Completed checkpoint bytes changed')
        archives = list((output/'native_learning_signal').glob('rollout_*.npz'))
        if len(archives) != 100:
            raise ValueError('Require every native command archive')
        for archive in archives:
            next_owner.verify_rollout_motion(archive, 0.)
        after = next_owner.observed.drift.completed_report(output)
        if len(next_owner.stage.trials(after)) != 27 or after['checkpoint_sha256'] != proof['checkpoint_sha256']:
            raise ValueError('Require all 27 unchanged original independent replay cases')
        if next_owner.core.sha(next_owner.ROOT/'mine_model/config/tracking_acceptance.json') != next_owner.native.guard.GATE_SHA:
            raise ValueError('The original acceptance gate must stay unchanged')
        evidence = dict(state='completed_block_height_diagnosis_verified_starting_successor',
            updated_at=datetime.now().astimezone().isoformat(), previous_unit=previous_unit,
            diagnosis=EXHAUSTION, actual_height_reward_trials=height_trials,
            source_coordinator_sha256=next_owner.core.sha(previous_cursor),
            completed_checkpoint_sha256=proof['checkpoint_sha256'],
            current_training_interrupted=False, no_stop_marker_created=True,
            all_27_original_cases_reviewed=True, native_updates_repeated=0,
            optimizer_reset=False, learned_std_reset=False, no_total_update_or_runtime_limit=True,
            goal_remains_active=True, policy_accepted=False)
        next_owner.write_json(base/'handoff_status.json', evidence)
        return next_owner


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator', type=Path, required=True)
    parser.add_argument('--handoff-directory', type=Path, required=True)
    parser.add_argument('--coordinator', type=Path, required=True)
    parser.add_argument('--environment-revision', type=Path, required=True)
    parser.add_argument('--previous-unit', required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    owner = inherit(args.previous_coordinator, args.handoff_directory, args.previous_unit)
    owner.run(args.previous_coordinator, args.coordinator, args.unit, args.environment_revision)
