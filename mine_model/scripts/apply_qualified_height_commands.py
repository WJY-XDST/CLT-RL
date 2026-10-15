"""Activate qualified command learning at a saved, independently replayed block."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from auto_train_mine import write_json
from launch_joint_pd_continuation import ROOT,launch,newest_owned_checkpoint

EVIDENCE=ROOT/'mine_model/results/joint_height_command_qualification_20261015'
FILES=('wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_height_feedback.py',
       'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_mean_guard.py',
       'wheel_legged_isaaclab/tests/test_joint_height_commands.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value):
    write_json(EVIDENCE/'transition_status.json',value)


def read_if_complete(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return None


def main():
    plan=json.loads((EVIDENCE/'qualified_transition_plan.json').read_text())
    previous=Path(plan['previous_directory'])
    record=json.loads((previous/'launch_record.json').read_text())
    pause=ROOT/'mine_model/results/joint_pd_continuation_service_watch/PAUSE'
    stage=EVIDENCE/'stage'
    native=json.loads((EVIDENCE/'native_verification.json').read_text())
    if not (native['native_actual_updates']==3 and native['optimizer_states']==17
            and native['all_tensors_finite'] and native['control_contract_unchanged']):
        raise ValueError('Native update qualification incomplete')
    if '514 passed, 3 skipped, 62 subtests passed' not in (EVIDENCE/'staged_regression_tests.log').read_text():
        raise ValueError('Staged regression qualification incomplete')
    for relative,digest in native['stage_source_sha256'].items():
        if sha(stage/relative)!=digest:
            raise ValueError('Qualified candidate changed: '+relative)
    assessment=previous/'block_0000/replay/assessment.json'
    while True:
        if pause.exists() or (previous/'STOP').exists():
            save(dict(state='stop_marker_respected_no_transition'));return
        pointer=json.loads((ROOT/'mine_model/results/joint_pd_active_resume.json').read_text())
        if Path(pointer['directory'])!=previous:
            save(dict(state='active_training_changed_no_transition'));return
        paired=read_if_complete(EVIDENCE/'paired_native_replay_comparison.json')
        measured=read_if_complete(assessment)
        if paired and measured:
            break
        state=json.loads((previous/'status.json').read_text())
        service=subprocess.run(['systemctl','--user','is-active',record['unit']],capture_output=True,text=True)
        if service.stdout.strip() not in ('active','activating'):
            save(dict(state='prior_service_not_live_requires_inspection',controller_state=state.get('state')));return
        save(dict(state='waiting_for_paired_replays_and_complete_main_block',
                  iteration=state.get('latest_update',{}).get('iteration')))
        time.sleep(5)
    control,candidate=(paired['reports'][key] for key in ('control','candidate'))
    if (control['checkpoint']!=native['source_checkpoint'] or candidate['checkpoint']!=native['checkpoint']
            or sha(native['source_checkpoint'])!=native['source_sha256']
            or sha(native['checkpoint'])!=native['checkpoint_sha256']):
        raise ValueError('Paired replay and native update checkpoints do not match')
    if (candidate['survivors']!=27 or candidate['total_trials']!=27 or candidate['resets']
            or control['survivors']!=27 or control['resets']):
        save(dict(state='candidate_survival_not_qualified_original_training_kept'));return
    regressions={key:candidate['ratios'][key]-value for key,value in control['ratios'].items()
                 if candidate['ratios'][key]>value+max(.05,.05*value)}
    if regressions:
        save(dict(state='candidate_replay_regression_original_training_kept',regressions=regressions));return
    if measured['survivors']!=27 or measured['total_trials']!=27 or measured['resets']:
        save(dict(state='main_survival_requires_diagnosis_no_transition'));return
    height_ratio=max(t['metrics']['height_mae_m']/t['limits']['height_mae_m']
                     for c in measured['conditions'] for t in c['trials'])
    # This only selects a training experiment; the original ten-percent
    # acceptance gate remains untouched and is never replaced by this check.
    if measured['passed'] or height_ratio<1.5:
        save(dict(state='current_height_learning_improving_keep_current',height_ratio=height_ratio,policy_accepted=False));return
    # Recheck after the asynchronous wait as well as at startup.
    for relative,digest in native['stage_source_sha256'].items():
        if sha(stage/relative)!=digest:
            raise ValueError('Qualified candidate changed while awaiting replay: '+relative)
    for relative,digest in plan['production_files'].items():
        if sha(ROOT/relative)!=digest:
            raise ValueError('Running source changed before activation: '+relative)
    if sha(ROOT/'mine_model/config/tracking_acceptance.json')!=plan['acceptance_sha256']:
        raise ValueError('Original acceptance gate changed')
    write_json(previous/'STOP',dict(reason='agent_diagnosis_qualified_height_command_learning',checkpoint_before_exit=True))
    subprocess.run(['systemctl','--user','stop',record['unit']],check=True,timeout=180)
    source,_=newest_owned_checkpoint(previous)
    backup=EVIDENCE/'production_before_transition';backup.mkdir()
    for relative in FILES:
        destination=ROOT/relative
        if destination.exists():
            saved=backup/relative;saved.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(destination,saved)
    try:
        for relative in FILES:
            destination=ROOT/relative;temporary=destination.with_suffix('.qualified-new')
            shutil.copyfile(stage/relative,temporary);os.replace(temporary,destination)
        environment=dict(os.environ,PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
                         PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (EVIDENCE/'production_regression_tests.log').open('w') as log:
            result=subprocess.run([str(ROOT/'run_python.sh'),'-m','pytest','wheel_legged_isaaclab/tests','-q'],
                                  cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        if result.returncode:
            raise RuntimeError('Production candidate regression tests failed')
        if pause.exists():
            save(dict(state='explicit_pause_respected_after_saved_transition',checkpoint=str(source)));return
        resumed=launch(previous,checkpoint=source,reward_overrides=record.get('reward_overrides_path'),
                       height_feedback_coefficient=record.get('height_feedback_coefficient'))
    except BaseException:
        for relative in FILES:
            destination=ROOT/relative;saved=backup/relative
            if saved.exists():shutil.copyfile(saved,destination)
            elif destination.exists():destination.unlink()
        if not pause.exists():
            resumed=launch(previous,checkpoint=source,reward_overrides=record.get('reward_overrides_path'),
                           height_feedback_coefficient=record.get('height_feedback_coefficient'))
            save(dict(state='candidate_failed_original_training_restored',directory=resumed['directory']))
        raise
    save(dict(state='qualified_height_command_learning_started',source_checkpoint=str(source),
              optimizer_step_before=resumed['optimizer_step_before'],new_directory=resumed['directory'],
              unit=resumed['unit'],training_profile=native['training_profile'],weights_reset=False,
              optimizer_reset=False,control_contract_unchanged=True,rewards_and_acceptance_unchanged=True,
              diagnostic_checkpoint_not_used=True,prior_height_ratio=height_ratio,policy_accepted=False))


if __name__=='__main__':main()
