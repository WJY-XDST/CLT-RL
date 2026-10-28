"""Replace a failed observer and finish review without restarting native PPO.

The original training controller retains its exited monitor handle. It writes
the normal full200 verification before checking that handle; this coordinator
waits for that verified boundary, joins the replacement observer, and performs
only the existing independent review. It never invokes a training entrypoint.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT=Path(__file__).resolve().parents[2]
SCRIPT=Path(__file__).resolve()


def read(path):
    return json.loads(path.read_text())


def write(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def owner(unit):
    result=subprocess.run(['systemctl','--user','show',unit,'-p','MainPID','-p','SubState','-p','ExecMainStatus'],
                          capture_output=True,text=True,check=True)
    return dict(row.split('=',1) for row in result.stdout.splitlines() if '=' in row)


def run(args):
    p=args.output.resolve()
    proof=read(args.repair_proof)
    assert proof['verified'] and not proof['native_training_stopped_or_restarted']
    assert not (p/'replacement_readonly_review_owner.json').exists()
    monitor=[str(ROOT/'run_python.sh'),str(SCRIPT.with_name('monitor_uniform_motion_learning_v4.py')),
             '--output',str(p),'--recipe',str(args.recipe.resolve())]
    with (p/'learning_monitor_replacement_v4.log').open('x') as log:
        child=subprocess.Popen(monitor,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT)
        write(p/'replacement_readonly_review_owner.json',dict(
            observed_at=datetime.now(timezone.utc).isoformat(),coordinator_pid=os.getpid(),
            replacement_monitor_pid=child.pid,original_training_unit=args.training_unit,
            original_training_process_not_touched=True,new_PPO_updates=0,policy_accepted=False))
        final=p/'final_checkpoint_verification_readonly.json'
        while not final.exists():
            code=child.poll()
            assert code in (None,0),('Inspect replacement observer; never restart PPO',code)
            state=owner(args.training_unit)
            assert state['MainPID']!='0',('Native owner ended without full200 proof',state)
            time.sleep(15.)
        completed=read(final)
        assert completed['verified'] and completed['all200_native_updates_and_original_alignment_verified']
        assert completed['actual_ppo_updates']==args.target_updates
        assert completed['all17_Adam_step']==args.target_updates*20
        code=child.wait(timeout=1800)
        assert code==0,('Inspect replacement observer outputs without repeating PPO',code)
    # Do not race an original controller that might still be finalizing.
    while owner(args.training_unit)['MainPID']!='0':time.sleep(5.)
    status=owner(args.training_unit)
    assert status['SubState']=='failed' and status['ExecMainStatus']=='1'
    prior=read(p/'controller_inspection_required.json')
    assert prior['error']=='AssertionError(1)',('Unexpected controller failure needs inspection',prior)
    assert not (p/'independent_review').exists(),'Do not duplicate an existing/partial replay'
    review=[str(ROOT/'run_python.sh'),str(SCRIPT.with_name('review_uniform_motion_block_v4.py')),
            '--output',str(p),'--recipe',str(args.recipe.resolve())]
    with (p/'review_controller_observer_recovery.log').open('x') as log:
        subprocess.run(review,cwd=ROOT,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,check=True)
    marker=p/'independent_review/complete_stage_review_with_global_envelope189.json'
    assert read(marker)['complete']
    write(p/'observer_failure_recovered_all200_and_extended_review.json',dict(
        complete=True,observed_at=datetime.now(timezone.utc).isoformat(),actual_updates=args.target_updates,
        native_PPO_repeated=False,original_four_and_new_envelope189_complete=True,
        all_original_numeric_readers_unchanged=True,policy_accepted=False,goal_remains_active=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--repair-proof',type=Path,required=True)
    parser.add_argument('--training-unit',required=True)
    parser.add_argument('--target-updates',type=int,required=True)
    args=parser.parse_args()
    try:run(args)
    except BaseException as error:
        write(args.output/'readonly_review_recovery_inspection_required.json',dict(
            error=repr(error),native_training_not_restarted=True,policy_accepted=False,
            goal_remains_active=True))
        raise
