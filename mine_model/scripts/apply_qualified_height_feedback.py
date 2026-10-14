"""At a saved block boundary, continue with qualified forward-height training."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from launch_joint_pd_continuation import ROOT,launch,newest_owned_checkpoint

EVIDENCE=ROOT/'mine_model/results/joint_height_feedback_20261015'
PREVIOUS=ROOT/'mine_model/results/joint_pd_optimization_20261015_004022_041465_mass_recovery_cumulative'
FILES=('wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_height_feedback.py',
       'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_mean_guard.py',
       'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
       'mine_model/scripts/joint_pd_continuation.py','mine_model/scripts/auto_train_mine.py',
       'wheel_legged_isaaclab/tests/test_joint_height_feedback_ppo.py',
       'wheel_legged_isaaclab/tests/test_joint_height_proxy.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value):
    (EVIDENCE/'transition_status.json').write_text(json.dumps(value,indent=2)+'\n')


def main():
    stage=EVIDENCE/'stage'
    plan=json.loads((EVIDENCE/'qualified_transition_plan.json').read_text())
    for path,digest in plan['staged_files'].items():
        if sha(stage/path)!=digest:raise ValueError('Qualified candidate changed: '+path)
    qualification=json.loads((EVIDENCE/'native_verification.json').read_text())
    if not (qualification['all_tensors_finite'] and qualification['control_contract_unchanged']
            and qualification['native_actual_updates']==3 and qualification['optimizer_states']==17):
        raise ValueError('Native PPO qualification incomplete')
    if '495 passed, 3 skipped, 62 subtests passed' not in (EVIDENCE/'staged_regression_tests.log').read_text():
        raise ValueError('Qualified candidate tests not passed')
    assessment=PREVIOUS/'block_0001/replay/assessment.json'
    while not assessment.exists():
        if (PREVIOUS/'STOP').exists() or (ROOT/'mine_model/results/joint_pd_continuation_service_watch/PAUSE').exists():
            save(dict(state='stop_marker_respected_no_transition'));return
        state=json.loads((PREVIOUS/'status.json').read_text())
        if state['state'] in ('agent_diagnosis_required','paused_user_request'):
            save(dict(state='prior_fault_requires_inspection_no_transition'));return
        save(dict(state='waiting_for_completed_second_reward_block',iteration=state.get('latest_update',{}).get('iteration')))
        time.sleep(5)
    measured=json.loads(assessment.read_text())
    if measured['passed']:
        save(dict(state='original_policy_passed_no_height_intervention'));return
    worst=max(t['metrics']['height_mae_m']/t['limits']['height_mae_m']
              for c in measured['conditions'] for t in c['trials'] if t['metrics'])
    if worst < 1.5:
        save(dict(state='height_improving_keep_reward_trial',height_ratio=worst));return
    for path,digest in plan['production_files'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Running source changed before transition: '+path)
    if sha(ROOT/'mine_model/config/tracking_acceptance.json')!=plan['acceptance_sha256']:
        raise ValueError('Original gate changed')
    record=json.loads((PREVIOUS/'launch_record.json').read_text())
    (PREVIOUS/'STOP').write_text(json.dumps(dict(reason='agent_diagnosis_qualified_forward_height_loss',checkpoint_before_exit=True))+'\n')
    subprocess.run(['systemctl','--user','stop',record['unit']],check=True,timeout=180)
    source,_=newest_owned_checkpoint(PREVIOUS)
    backup=EVIDENCE/'production_before_transition';backup.mkdir(exist_ok=True)
    for relative in FILES:
        destination=ROOT/relative
        previous=backup/relative;previous.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():shutil.copyfile(destination,previous)
        temporary=destination.with_suffix('.qualified-new')
        shutil.copyfile(stage/relative,temporary);os.replace(temporary,destination)
    environment=dict(os.environ,PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
                     PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with (EVIDENCE/'production_regression_tests.log').open('w') as log:
        result=subprocess.run([str(ROOT/'run_python.sh'),'-m','pytest','wheel_legged_isaaclab/tests','-q'],
                              cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=60)
    if result.returncode:
        for relative in FILES:
            destination=ROOT/relative
            if (backup/relative).exists():shutil.copyfile(backup/relative,destination)
            elif destination.exists():destination.unlink()
        resumed=launch(PREVIOUS)
        save(dict(state='candidate_tests_failed_original_learning_restored',directory=resumed['directory']))
        raise RuntimeError('Production candidate tests failed; original training restored')
    resumed=launch(PREVIOUS,checkpoint=source,height_feedback_coefficient=.1)
    save(dict(state='qualified_height_feedback_training_started',source_checkpoint=str(source),
              optimizer_step_before=resumed['optimizer_step_before'],new_directory=resumed['directory'],
              unit=resumed['unit'],height_coefficient=.1,weights_reset=False,optimizer_reset=False,
              control_contract_unchanged=True,rewards_and_acceptance_unchanged=True,
              diagnostic_checkpoint_not_used=True,prior_height_ratio=worst,policy_accepted=False))


if __name__=='__main__':main()
