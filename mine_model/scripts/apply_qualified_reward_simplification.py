"""Apply the measured duplicate-reward trial after a complete learning block.

This local handoff keeps the most recent saved weights/Adam and immediately
continues. It accepts only the already-qualified source/test bundle below.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from launch_joint_pd_continuation import ROOT, launch, newest_owned_checkpoint

EVIDENCE = ROOT/'mine_model/results/reward_simplification_20261015'
PREVIOUS = ROOT/'mine_model/results/joint_pd_optimization_20261015_000630_400331_mass_recovery_cumulative'
FILES = {
    'joint_pd_optimization.py': 'mine_model/scripts/joint_pd_optimization.py',
    'joint_pd_continuation.py': 'mine_model/scripts/joint_pd_continuation.py',
    'test_tracking_goal_feedback.py': 'wheel_legged_isaaclab/tests/test_tracking_goal_feedback.py',
    'test_duplicate_goal_removal.py': 'wheel_legged_isaaclab/tests/test_duplicate_goal_removal.py',
    'test_native_asset_proof.py': 'wheel_legged_isaaclab/tests/test_native_asset_proof.py',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_record(value):
    (EVIDENCE/'transition_status.json').write_text(json.dumps(value,indent=2)+'\n')


def main():
    stage=EVIDENCE/'staged_helpers'
    expected=json.loads((EVIDENCE/'qualified_transition_plan.json').read_text())
    for name,checksum in expected['staged_sha256'].items():
        if digest(stage/name) != checksum:
            raise ValueError('Qualified staged source changed: '+name)
    proof=json.loads((EVIDENCE/'native_reward_isolation_verification.json').read_text())
    if (proof['physical_telemetry_max_abs_difference'] != 0.
            or proof['all_other_rewards_max_abs_difference'] != 0.
            or not proof['diagnostic_rejected_by_acceptance']):
        raise ValueError('Frozen native reward isolation not qualified')
    if '487 passed, 3 skipped, 62 subtests passed' not in (EVIDENCE/'staged_regression_tests.log').read_text():
        raise ValueError('Complete staged regression did not pass')
    while True:
        if (ROOT/'mine_model/results/joint_pd_continuation_service_watch/PAUSE').exists():
            write_record(dict(state='explicit_user_pause_no_transition'))
            return
        if (PREVIOUS/'STOP').exists():
            write_record(dict(state='existing_stop_requires_inspection_no_transition'))
            return
        state=json.loads((PREVIOUS/'status.json').read_text())
        phase=state['state']
        assessment=PREVIOUS/'block_0000/replay/assessment.json'
        verified=PREVIOUS/'block_0000/resume_verification.json'
        if phase in ('agent_diagnosis_required','paused_user_request'):
            write_record(dict(state='diagnosis_requires_inspection_no_transition',previous_state=state))
            return
        if assessment.exists() and verified.exists():
            result=json.loads(assessment.read_text())
            update=json.loads(verified.read_text())
            if result['iteration'] != 4824 or update['actual_updates'] != 100:
                raise ValueError('Require the completed new-mass 100-update control block')
            if result['passed']:
                write_record(dict(state='baseline_passed_continue_original_sustained_verification'))
                return
            break
        write_record(dict(state='waiting_for_completed_mass_control_block',latest=state.get('latest_update',{}).get('iteration')))
        time.sleep(5)
    for name,checksum in expected['production_sha256_before'].items():
        if digest(ROOT/name) != checksum:
            raise ValueError('Running production source changed before qualified handoff: '+name)
    if digest(ROOT/'mine_model/config/tracking_acceptance.json') != expected['acceptance_sha256']:
        raise ValueError('Original acceptance gate changed')
    marker=PREVIOUS/'STOP'
    marker.write_text(json.dumps(dict(reason='agent_diagnosis_qualified_duplicate_reward_removal',
                                     checkpoint_before_exit=True))+'\n')
    # ExecStop waits for the owned controller's cooperative checkpoint save.
    record=json.loads((PREVIOUS/'launch_record.json').read_text())
    subprocess.run(['systemctl','--user','stop',record['unit']],check=True,timeout=180)
    source,_=newest_owned_checkpoint(PREVIOUS)
    backup=EVIDENCE/'production_before_transition'
    backup.mkdir(exist_ok=True)
    for name,relative in FILES.items():
        destination=ROOT/relative
        if destination.exists():
            shutil.copyfile(destination,backup/name)
        temporary=destination.with_suffix('.qualified-new')
        shutil.copyfile(stage/name,temporary)
        os.replace(temporary,destination)
    environment=dict(os.environ,PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
                     PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (EVIDENCE/'production_regression_tests.log').open('w') as log:
        tested=subprocess.run([str(ROOT/'run_python.sh'),'-m','pytest',
            'wheel_legged_isaaclab/tests','-q'],cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    if tested.returncode:
        for name,relative in FILES.items():
            destination=ROOT/relative
            if (backup/name).exists():
                shutil.copyfile(backup/name,destination)
            elif destination.exists():
                destination.unlink()
        resumed=launch(PREVIOUS)
        write_record(dict(state='candidate_regression_failed_original_training_restored',
                          checkpoint=str(source),active_directory=resumed['directory']))
        raise RuntimeError('Candidate regression failed; original code and continued training restored')
    override=EVIDENCE/'remove_duplicate_worst_reward.json'
    override.write_text(json.dumps({'rewards.tracking_goal_violation':0.})+'\n')
    resumed=launch(PREVIOUS,checkpoint=source,reward_overrides=override)
    write_record(dict(state='qualified_reward_trial_training_started',source_checkpoint=str(source),
        optimizer_step_before=resumed['optimizer_step_before'],new_directory=resumed['directory'],unit=resumed['unit'],
        weights_reset=False,optimizer_reset=False,only_reward_change={'rewards.tracking_goal_violation':[-.5,0.]},
        checkpoint_save_interval=5,native_asset_proof_required=True,acceptance_unchanged=True,
        policy_accepted=False,baseline_assessment=str(assessment)))
    print(json.dumps(json.loads((EVIDENCE/'transition_status.json').read_text()),indent=2))


if __name__ == '__main__':
    main()
