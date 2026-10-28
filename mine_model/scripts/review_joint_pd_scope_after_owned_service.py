"""Wait for one verified owned training service, then run frozen scope review.

No training, restart, checkpoint replacement or policy promotion is available.
An upstream failure is terminal for this reviewer and requires inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_pins(request):
    for path, digest in request['source_pins'].items():
        assert sha(path) == digest, path


def state(unit):
    assert unit.startswith('clt-') and unit.endswith('.service')
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=LoadState,ActiveState,SubState,MainPID,ExecMainStatus,Result'],
        text=True, capture_output=True, check=True)
    values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    assert values['LoadState'] == 'loaded', values
    return values


def complete_upstream(request, service):
    assert service['SubState'] == 'exited' and service['ActiveState'] == 'active', service
    assert service['MainPID'] == '0' and service['ExecMainStatus'] == '0', service
    assert service['Result'] == 'success', service
    owner = Path(request['upstream_output'])
    status = read(owner / 'status.json')
    assert status['state'] == 'complete50_independently_reviewed_agent_decision_required'
    assert status['training_active'] is False and status['policy_accepted'] is False
    verified = read(owner / 'final_checkpoint_verification_readonly.json')
    assert verified['verified'] and verified['all50_native_updates_verified']
    assert verified['actual_ppo_updates'] == status['actual_ppo_updates'] == request['expected_actual_updates']
    checkpoint = Path(verified['checkpoint']).resolve()
    assert checkpoint == Path(status['checkpoint']).resolve()
    assert sha(checkpoint) == verified['checkpoint_sha256']
    assert verified['all17_Adam_step'] == request['expected_actual_updates'] * 20
    assert read(owner / 'original11_health50_complete.json')['qualified']
    paired = read(owner / 'complete50_matched189_agent_decision_required.json')
    assert paired['complete'] and paired['checkpoint_sha256'] == sha(checkpoint)
    assert paired['full735_and_sustained_qualification_still_required']
    return checkpoint, owner / 'final_checkpoint_verification_readonly.json'


def run(request_path, *, validate_ready_only=False):
    request = read(request_path)
    assert request['schema'] == 'frozen_same735_after_owned_service_v1'
    verify_pins(request)
    status_path = Path(request['status_path'])
    previous = None
    while True:
        service = state(request['upstream_unit'])
        if service['ActiveState'] == 'active' and service['SubState'] == 'running':
            assert int(service['MainPID']) > 0
            if previous != service:
                write(status_path, dict(state='waiting_for_exact_verified_upstream_service',
                    observed_at=datetime.now(timezone.utc).isoformat(), unit=request['upstream_unit'],
                    service=service, simulator_started=False, optimizer_executed=False,
                    policy_accepted=False, original735_scope_retained=True))
                previous = service
            if validate_ready_only:
                raise AssertionError('The requested upstream is not terminal yet')
            time.sleep(30.)
            continue
        checkpoint, verification = complete_upstream(request, service)
        break
    verify_pins(request)
    ready = dict(state='upstream_full_state_original11_and_paired189_verified',
        observed_at=datetime.now(timezone.utc).isoformat(), unit=request['upstream_unit'],
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        actual_updates=request['expected_actual_updates'], policy_accepted=False,
        simulator_started=False, optimizer_executed=False)
    write(status_path, ready)
    if validate_ready_only:
        return ready
    output = Path(request['output']).resolve()
    assert not output.exists(), 'Inspect existing review output; never repeat it automatically'
    helper = ROOT / 'mine_model/scripts/review_saved_joint_pd_same_scope.py'
    qualify = [str(ROOT / 'run_python.sh'), str(helper), 'qualify', '--output', str(output),
        '--recipe', request['recipe'], '--checkpoint', str(checkpoint),
        '--verification', str(verification), '--original-scope', request['original_scope']]
    subprocess.run(qualify, cwd=ROOT, check=True)
    write(status_path, dict(ready, state='original735_frozen_review_running',
        qualified_scope=str(output / 'qualification.json'), simulator_started=True))
    subprocess.run([str(ROOT / 'run_python.sh'), str(helper), 'run', '--output', str(output),
                    '--before-report', request['before_report']], cwd=ROOT, check=True)
    report = read(output / 'paired_original735_comparison.json')
    assert report['verified'] and report['pairs'] == 735
    assert report['checkpoint_sha256'] == sha(checkpoint)
    write(status_path, dict(ready, state='complete_original735_agent_decision_required',
        original735_report=str(output / 'complete_frozen_command_matrix.json'),
        paired_comparison=str(output / 'paired_original735_comparison.json'),
        summary=report['after']['summary'], simulator_started=True,
        independent_sustained_still_required=True, policy_promoted=False))
    verify_pins(request)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--validate-ready-only', action='store_true')
    args = parser.parse_args()
    try:
        run(args.request.resolve(), validate_ready_only=args.validate_ready_only)
    except BaseException as error:
        request = read(args.request)
        write(Path(request['status_path']), dict(state='inspection_required_no_automatic_restart',
            observed_at=datetime.now(timezone.utc).isoformat(), error=repr(error),
            upstream_unit=request['upstream_unit'], checkpoints_preserved=True,
            optimizer_executed=False, policy_accepted=False))
        raise
