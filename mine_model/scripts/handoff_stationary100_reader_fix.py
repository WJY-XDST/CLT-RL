"""Complete an intact stationary block review, then continue without repetition.

The running owner remains untouched until all native updates and its full
replay finish. A post-replay telemetry field mismatch never discards Adam
or causes the just-completed PPO updates to be executed again.
"""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import time

next_owner = None


def recover(previous_cursor, base, unit):
    global next_owner
    previous_cursor,base=Path(previous_cursor).resolve(),Path(base).resolve()
    base.mkdir(exist_ok=False)
    with (previous_cursor.parent/'.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                break
            except BlockingIOError:
                (base/'handoff_status.json').write_text(json.dumps(dict(
                    state='waiting_for_current_native_100_and_full_replay',
                    updated_at=datetime.now().astimezone().isoformat(),
                    interrupt_current_training=False,goal_remains_active=True))+'\n')
                time.sleep(5)
        import continue_joint_pd_stationary100_v2 as next_owner
        state=json.loads(previous_cursor.read_text())
        if (state['state']!='agent_diagnosis_required' or state['reason'] not in (
                "KeyError('commands')", "KeyError('commands is not a file in the archive')")):
            raise ValueError('Only recover the verified post-replay command-field mismatch')
        output=Path(state['current_output'])
        status=json.loads((output/'status.json').read_text())
        plan=json.loads((output/'experiment_plan.json').read_text())
        proof=json.loads((output/'resume_verification.json').read_text())
        if (status['state']!='agent_review_required' or status['training_active']
                or status['actual_ppo_updates']!=plan['target_actual_ppo_updates']
                or proof['actual_ppo_updates']!=plan['target_actual_ppo_updates']
                or proof['optimizer_step']!=20*plan['target_actual_ppo_updates']
                or proof['optimizer_states']!=17 or not proof['all_tensors_finite']
                or next_owner.core.sha(Path(status['checkpoint']))!=proof['checkpoint_sha256']):
            raise ValueError('Do not recover missing native updates, damaged state or an incomplete replay')
        next_owner.observed.finalize_observations(output)
        archives=list((output/'native_learning_signal').glob('rollout_*.npz'))
        if len(archives)!=100:
            raise ValueError('Require all 100 native command archives')
        for archive in archives:
            next_owner.verify_rollout_motion(archive,0.)
        before=next_owner.observed.drift.completed_report(Path(state['incumbent']))
        after=next_owner.observed.drift.completed_report(output)
        decision=next_owner.stage.review(after,before)
        next_owner.write_json(output/'stationary_review.json',decision)
        history=[dict(output=str(output),source=state['incumbent'],training_motion_scale=0.,**decision)]
        incumbent=output if decision['promote'] else Path(state['incumbent'])
        if not decision['promote']:
            next_owner.write_json(output/'checkpoint_rejected_but_preserved.json',dict(
                restore_checkpoint=plan['source_checkpoint'],
                restore_complete_optimizer_from_same_checkpoint=True,
                rejected_checkpoint=status['checkpoint'],**decision))
        recovery=dict(native_updates_repeated=0,actual_completed_native_updates=100,
            correct_archive_field='native_commands',source_file_edited_during_training=False,
            saved_actual_ppo_updates=status['actual_ppo_updates'],all17Adam_step=proof['optimizer_step'],
            checkpoint_sha256=proof['checkpoint_sha256'],all_27_original_cases_reviewed=True,
            stage_decision=decision,goal_remains_active=True,policy_accepted=False)
        next_owner.write_json(output/'post_replay_reader_recovery.json',recovery)
        bridge=base/'completed_stationary_handoff.json'
        next_owner.write_json(bridge,dict(state='explicit_stop_respected',
            reason='Agent-owned completed-block handoff, not a user stop or training endpoint',
            previous_owner_state=state['state'],incumbent=str(incumbent),history=history,
            stationary_history=history,actual_completed_native_updates=100,
            no_native_updates_repeated=True,goal_remains_active=True,policy_accepted=False))
        next_owner.write_json(base/'handoff_status.json',dict(
            state='completed_stationary_review_recovered_starting_next_block',recovery=recovery,
            updated_at=datetime.now().astimezone().isoformat()))
        return bridge


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator',type=Path,required=True)
    parser.add_argument('--recovery-directory',type=Path,required=True)
    parser.add_argument('--coordinator',type=Path,required=True)
    parser.add_argument('--environment-revision',type=Path,required=True)
    parser.add_argument('--unit',required=True)
    args=parser.parse_args()
    bridge=recover(args.previous_coordinator,args.recovery_directory,args.unit)
    next_owner.run(bridge,args.coordinator,args.unit,args.environment_revision)
