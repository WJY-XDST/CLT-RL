"""Recover lost owned training services without hiding numerical diagnoses.

PAUSE is an explicit persistent user control. Transient process loss and reboot
recover from verified files; numerical faults and measured stagnation retain
their evidence for diagnosis. No policy reset, gate change or hardware path.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from launch_joint_pd_continuation import ROOT, launch

WATCH = ROOT/'mine_model/results/joint_pd_continuation_service_watch'
POINTER = ROOT/'mine_model/results/joint_pd_active_resume.json'
RECOVERABLE = {'training', 'training_initialization', 'checkpoint_evaluation',
               'block_evaluated', 'best_measured_checkpoint_saved'}


def decision(service, state, marker, *, paused=False):
    if paused:
        return 'explicit_user_pause'
    if service.get('ActiveState') in ('active', 'activating', 'deactivating'):
        return 'service_running'
    phase = state.get('state')
    reason = marker.get('reason', '')
    if state.get('all_errors_under_10_percent') or phase in (
            'simulation_accepted_pending_sim2sim_and_randomization', 'completed_model_saved'):
        return 'verify_delivery'
    if phase in ('agent_diagnosis_required', 'diagnosis_required', 'error') or reason.startswith('agent_diagnosis'):
        return 'diagnosis_required'
    if reason and reason != 'host_shutdown':
        return 'explicit_user_pause'
    if phase in RECOVERABLE or (phase == 'paused_user_request' and reason == 'host_shutdown'):
        return 'recover_verified_checkpoint'
    return 'inspect_unknown_state'


def write_state(value):
    value = dict(value, updated_at=datetime.now().astimezone().isoformat())
    temporary=WATCH/'status.new'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(WATCH/'status.json')


def once():
    record = json.loads(POINTER.read_text())
    output = Path(record['directory']).resolve()
    if output.parent != ROOT/'mine_model/results' or not output.name.startswith('joint_pd_optimization_'):
        raise ValueError('Watch only scoped continuation output')
    unit = record['unit']
    if not unit.startswith('clt-joint-pd-') or not unit.endswith('.service'):
        raise ValueError('Watch only owned joint-PD services')
    response = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'ActiveState', '-p', 'MainPID'],
                              capture_output=True, text=True, timeout=15)
    service = dict(line.split('=',1) for line in response.stdout.splitlines() if '=' in line)
    state = json.loads((output/'status.json').read_text()) if (output/'status.json').exists() else {}
    marker = json.loads((output/'STOP').read_text()) if (output/'STOP').exists() else {}
    action = decision(service, state, marker, paused=(WATCH/'PAUSE').exists())
    result = dict(state=action, directory=str(output), unit=unit, service=service,
                  checkpoint=state.get('checkpoint'), latest_update=state.get('latest_update'))
    if action == 'recover_verified_checkpoint':
        for name, digest in state.get('source_files',{}).items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError('Source or physics changed since the qualified run: '+name)
        next_run = launch(output, reward_overrides=record.get('reward_overrides_path'),
                          height_feedback_coefficient=record.get('height_feedback_coefficient'))
        result.update(state='recovered_verified_checkpoint', new_directory=next_run['directory'],
                      new_checkpoint=next_run['source_checkpoint'], optimizer_step=next_run['optimizer_step_before'])
    write_state(result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--pause',action='store_true',help='Persist an explicit pause and request a saved stop')
    args=parser.parse_args()
    WATCH.mkdir(parents=True,exist_ok=True)
    if args.pause:
        (WATCH/'PAUSE').write_text('explicit_user_pause\n')
        record=json.loads(POINTER.read_text())
        subprocess.run([str(ROOT/'run_python.sh'),str(ROOT/'mine_model/scripts/auto_train_mine.py'),
                        '--stop-output',record['directory']],check=True,timeout=180)
        return
    with (WATCH/'.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            try:
                result=once()
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
                result=dict(state='watch_diagnosis_required',reason=repr(error))
                write_state(result)
            if args.once:
                print(json.dumps(result,ensure_ascii=False,indent=2))
                return
            time.sleep(15)


if __name__ == '__main__':
    main()
