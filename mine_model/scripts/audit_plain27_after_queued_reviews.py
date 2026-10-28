"""Verify raw independent reviews and all original errors after one exact200 block.

Wait only on the named live review units. A reader failure never restarts PPO,
changes a gate or turns a failed physical trial into a passing metric.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import audit_completed_joint_pd_retention_raw as raw
import audit_completed_plain27_errors_with_failures as errors
import review_plain27_single_reward_block200 as owner

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'


def specification(request):
    request = Path(request).resolve()
    data = json.loads(request.read_text())
    if (request.parent != RESULTS or data['script_sha256'] != sha(SCRIPT)
            or any(sha(p) != h for p, h in data['source_pins'].items())):
        raise ValueError('Require the intact owned raw/error reader request')
    full_path, moving_path = [Path(data[k]) for k in ('full_review_launch', 'moving_review_launch')]
    full, moving = [json.loads(p.read_text()) for p in (full_path, moving_path)]
    training_launch = Path(full['training_launch'])
    launch = json.loads(training_launch.read_text())
    bound = owner.build(training_launch, full['static_qualification'])
    if (full['script'] != str(owner.SCRIPT) or full['script_sha256'] != sha(owner.SCRIPT)
            or not full['queued'] or not moving['queued']
            or moving['after_review_unit'] != full['unit']
            or moving['training_output'] != launch['output']
            or data['training_output'] != launch['output']
            or data['source_actual_updates'] != bound.source_clock
            or data['target_actual_updates'] != bound.target_clock
            or data['target_all17_Adam_step'] != 20 * bound.target_clock
            or Path(data['static_output']) != Path(full['static_output'])
            or moving['target_actual_updates'] != bound.target_clock):
        raise ValueError('Require the exact declared full51/static/moving review chain')
    return data, full, moving, launch


def qualify(request, destination):
    data, full, moving, launch = specification(request)
    destination = Path(destination).resolve()
    reference = json.loads(Path(data['source_error_reference']).read_text())
    execution = json.loads(Path(data['source_execution_reference']).read_text())
    if (destination.parent != RESULTS or destination.exists()
            or reference['checkpoint_sha256'] != launch['source_checkpoint_sha256']
            or reference['total_trials'] != 333 or reference['original_metrics_undefined_trials'] != 9
            or reference['original231_detailed_reference_parity_trials'] != 231
            or reference['all7_metrics_and_limits_raw_reproduction_max_error']
            or not execution['verified'] or execution['checkpoint_sha256'] != launch['source_checkpoint_sha256']
            or reference['optimizer_executed'] or execution['optimizer_executed']):
        raise ValueError('Require the completed same-source original333/failed-metrics reader qualification')
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        request=str(Path(request).resolve()), request_sha256=sha(request),
        source_actual_updates=data['source_actual_updates'], target_actual_updates=data['target_actual_updates'],
        target_all17_Adam_step=data['target_all17_Adam_step'],
        original231_detailed_metrics_limits_flags_reproduced=True,
        all9_failed_survival_trials_and_None_metrics_retained=True,
        required_full27_four51_and_independent_raw_readers_unchanged=True,
        source_pins={str(p): sha(p) for p in (SCRIPT, raw.SCRIPT, errors.SCRIPT, owner.SCRIPT,
            Path(request), Path(data['source_error_reference']), Path(data['source_execution_reference']))},
        optimizer_executed=False, simulator_started=False, future_model_performance_unproven=True,
        policy_accepted=False)
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def wait_unit(unit, waiter):
    while True:
        # A transient observation error is retried. It is not a process exit.
        try:
            output = subprocess.check_output(['systemctl', '--user', 'show', unit,
                '-p', 'MainPID', '-p', 'LoadState', '-p', 'ActiveState', '-p', 'SubState', '-p', 'Result'], text=True)
        except subprocess.CalledProcessError as error:
            write_json(waiter, dict(observed_at=datetime.now().astimezone().isoformat(),
                state='unit_observation_retry', unit=unit, error=repr(error), PPO_restarted=False))
            time.sleep(10.)
            continue
        state = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
        pid = int(state.get('MainPID', 0))
        if pid and Path('/proc', str(pid)).exists():
            write_json(waiter, dict(observed_at=datetime.now().astimezone().isoformat(),
                state='verified_wait', unit=unit, actual_live_MainPID=pid, authoritative_unit_state=state,
                PPO_restarted=False, policy_accepted=False))
            time.sleep(10.)
            continue
        if state.get('ActiveState') in ('activating', 'reloading'):
            time.sleep(10.)
            continue
        if state.get('SubState') == 'exited' and state.get('Result') == 'success':
            return state
        raise RuntimeError('Review unit is terminal or missing; inspect without restarting PPO: ' + json.dumps(state))


def run(request, qualification):
    data, full, moving, launch = specification(request)
    proof = json.loads(Path(qualification).read_text())
    if (not proof['qualified'] or proof['request_sha256'] != sha(request)
            or any(sha(p) != h for p, h in proof['source_pins'].items())):
        raise ValueError('Preserve the qualified original raw/error readers')
    training = Path(data['training_output'])
    status_path = training / 'post_review_integrity_status.json'
    wait_unit(full['unit'], status_path)
    all_errors_path = training / f"all_error_types_source{data['target_actual_updates']}_readonly_20261019.json"
    if all_errors_path.exists():
        all_errors = json.loads(all_errors_path.read_text())
    else:
        all_errors = errors.audit(Path(full['training_launch']), full['static_qualification'], all_errors_path)
    if (all_errors['actual_ppo_updates'] != data['target_actual_updates']
            or any(sha(p) != h for p, h in all_errors['source_pins'].items())):
        raise ValueError('Existing raw error report belongs to another checkpoint')
    static_path = Path(data['static_output']) / 'static_retention.json'
    raw_path = Path(data['static_raw_output'])
    static = None
    if static_path.exists():
        static = (json.loads(raw_path.read_text()) if raw_path.exists()
            else raw.audit(training, Path(data['static_output']), raw_path))
        if any(sha(p) != h for p, h in static['source_pins'].items()):
            raise ValueError('Existing independent raw verification changed')
    else:
        status = json.loads((Path(data['static_output']) / 'status.json').read_text())
        if status.get('state') != 'not_run_canonical_static_failed':
            raise ValueError('Resolve missing independent static evidence after the review unit completed')
    wait_unit(moving['unit'], status_path)
    moving_dir = Path(moving['output'])
    moving_report = json.loads((moving_dir / 'motion_height_steps_summary.json').read_text())
    moving_raw_path = moving_dir / 'motion_height_steps_raw_integrity_readonly_20261019.json'
    moving_raw = json.loads(moving_raw_path.read_text())
    if (not moving_raw['verified'] or moving_report['checkpoint_sha256'] != all_errors['checkpoint_sha256']
            or moving_raw['checkpoint_sha256'] != all_errors['checkpoint_sha256']
            or any(sha(p) != h for p, h in moving_raw['source_pins'].items())):
        raise ValueError('Moving-height source or raw integrity differs')
    summary = dict(observed_at=datetime.now().astimezone().isoformat(), state='agent_review_required',
        target_actual_updates=data['target_actual_updates'], target_all17_Adam_step=data['target_all17_Adam_step'],
        checkpoint_sha256=all_errors['checkpoint_sha256'], all_errors=str(all_errors_path),
        all_error_scope_counts=all_errors['scope_counts'],
        static_raw_manifest=str(raw_path) if static is not None else None,
        static_retention_verified=bool(static and static['static_retention_verified']),
        moving_height_summary=str(moving_dir / 'motion_height_steps_summary.json'),
        moving_height_original_readiness=moving_report['supplemental_steps_ready'],
        moving_height_results=moving_report['results'], moving_raw_verified=True,
        source_pins={str(p): sha(p) for p in (SCRIPT, Path(request), Path(qualification),
            all_errors_path, moving_raw_path)},
        next_training_requires_agent_review=True, PPO_restarted=False, optimizer_executed=False,
        acceptance_changed=False, policy_accepted=False, goal_remains_active=True)
    with (training / 'post_review_integrity_and_errors_readonly_20261019.json').open('x') as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    write_json(status_path, summary)
    return summary


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--request', type=Path, required=True)
    cli.add_argument('--qualification', type=Path, required=True)
    cli.add_argument('--qualify-only', action='store_true')
    args = cli.parse_args()
    try:
        result = qualify(args.request, args.qualification) if args.qualify_only else run(args.request, args.qualification)
    except BaseException as error:
        data = json.loads(args.request.read_text())
        write_json(Path(data['training_output']) / 'post_review_integrity_error.json', dict(
            observed_at=datetime.now().astimezone().isoformat(), error=repr(error),
            requires_agent_inspection=True, PPO_restarted=False, optimizer_executed=False,
            acceptance_changed=False, policy_accepted=False))
        raise
    print(json.dumps({k: result[k] for k in ('qualified', 'state', 'target_actual_updates', 'policy_accepted') if k in result}))
