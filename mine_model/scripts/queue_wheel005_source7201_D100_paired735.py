"""Run the original qualified735 pairing after this exact7401 review unit."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
RESULTS = ROOT / 'mine_model/results'
PREPARATION = RESULTS / 'joint_pd_wheel005_source7201_D100_qualification_20261024'
POINTER = RESULTS / 'joint_pd_wheel005_source7201_D100_prepared_capacity_pointer_20261024.json'
SOURCE_REQUEST = RESULTS / 'joint_pd_AF7001_wheelD075_wheel005_qualification_20261024/paired735_after_full_reviews_request.json'
REVIEW = RESULTS / 'joint_pd_wheel005_source7201_D100_review7401_20261024'
AFTER_UNIT = 'clt-joint-pd-wheel005-source7201-D100-full-reviews-20261024.service'
STATE = PREPARATION / 'paired735_queue_status.json'
SOURCE_SHA = '3964b3a21906dad8160f08d8ca4e1d2310a6bcc2d179277729a63e8a92756785'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def run():
    output = Path(read(POINTER)['planned_production_output'])
    request_path = PREPARATION / 'paired735_after_full_reviews_request.json'
    destination = output / 'paired_source7201_candidate_D100_7401_all735_readonly_20261024.json'
    assert not request_path.exists() and not destination.exists()
    while True:
        result = subprocess.run(['systemctl', '--user', 'show', AFTER_UNIT,
            '-p', 'MainPID', '-p', 'Result', '-p', 'SubState'],
            text=True, capture_output=True, check=True)
        owner = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        if owner['MainPID'] == '0':
            assert owner['Result'] == 'success' and owner['SubState'] == 'exited', owner
            break
        save(STATE, dict(observed_at=datetime.now().astimezone().isoformat(),
            state='waiting_for_live_owned_complete_review', unit=AFTER_UNIT,
            unit_state=owner, simulator_started=False, optimizer_executed=False, policy_accepted=False))
        time.sleep(15.)
    final_path = output / 'final_checkpoint_verification_readonly_20261019.json'
    complete_path = REVIEW / 'complete_all_scopes_wheel005_D100.json'
    final, complete = read(final_path), read(complete_path)
    assert final['verified'] and final['actual_ppo_updates'] == 7401 and final['all17_Adam_step'] == 148020
    assert complete['completed'] and complete['checkpoint_sha256'] == final['checkpoint_sha256'] == sha(final['checkpoint'])
    assert all(complete[key] for key in ('original735_static810_original120_complete',
        'signed_fullspeed_height_complete', 'small_yaw120_complete'))
    prior = read(SOURCE_REQUEST)
    source = prior['candidate']
    assert read(source['final_verification'])['checkpoint_sha256'] == SOURCE_SHA
    assert read(source['final_verification'])['actual_ppo_updates'] == 7201
    candidate = dict(final_verification=str(final_path),
        raw_reports=[str(output / 'all_errors_all657_source7401_readonly_20261024.json'),
            str(REVIEW / 'known_interior/all54_raw_verified.json'),
            str(REVIEW / 'retained_left_interior/all24_raw_verified.json')],
        original657_input_specification=str(output / 'original_all657_tracking_input_specification_20261024.json'),
        known54_directory=str(REVIEW / 'known_interior/candidate'),
        left24_directory=str(REVIEW / 'retained_left_interior/candidate'))
    collector = SCRIPT.with_name('audit_paired_retained735_tracking.py')
    waiter = SCRIPT.with_name('collect_paired735_after_owned_review.py')
    qualification = Path(prior['collector_qualification'])
    assert read(qualification)['qualified']
    assert read(qualification)['source_pins'][str(collector)] == sha(collector)
    paths = [SCRIPT, POINTER, SOURCE_REQUEST, collector, waiter, qualification,
        final_path, complete_path, output / 'experiment_plan.json',
        SCRIPT.with_name('review_wheel005_source7201_D100_block200.py')]
    for spec in (source, candidate):
        paths += [Path(path) for path in (spec['final_verification'],
            *spec['raw_reports'], spec['original657_input_specification'])]
    request = dict(source=source, candidate=candidate, output=str(destination),
        wait_status=str(output / 'paired735_after_full_reviews_status.json'),
        collector_qualification=str(qualification), waiter_script_sha256=sha(waiter),
        collector_script_sha256=sha(collector), after_unit=AFTER_UNIT,
        complete_all_scopes=str(complete_path), candidate_checkpoint_sha256=final['checkpoint_sha256'],
        request_source_pins={str(path): sha(path) for path in paths},
        created_at=datetime.now().astimezone().isoformat(),
        can_select_source=False, can_extend_training=False, policy_accepted=False)
    with request_path.open('x') as stream:
        json.dump(request, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    subprocess.run([str(ROOT / 'run_python.sh'), str(waiter), '--request', str(request_path)],
        cwd=ROOT, env=os.environ.copy(), check=True)
    report = read(destination)
    assert report['verified'] and report['source_checkpoint_sha256'] == SOURCE_SHA
    assert report['candidate_checkpoint_sha256'] == final['checkpoint_sha256']
    save(STATE, dict(state='paired735_complete_agent_decision_required',
        report=str(destination), summaries=report['summaries'], policy_accepted=False))


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        save(STATE, dict(observed_at=datetime.now().astimezone().isoformat(),
            state='agent_inspection_required', error=repr(error),
            PPO_restarted=False, policy_accepted=False))
        raise
