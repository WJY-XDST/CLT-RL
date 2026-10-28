"""Measure current plain-PPO capacity before a new reviewed training block.

Wait on owned jobs, then run isolated five-update probes from one frozen
production checkpoint. No capacity checkpoint is a training continuation
source. A failure or low-memory abort affects only its own diagnostic unit.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
BACKEND = ROOT / 'mine_model/experiments/joint_pd_yawcost200_capacity_runtime.py'
POLICY = ROOT / 'mine_model/config/training_environment_capacity.json'
GIB = 1024 ** 3


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def now():
    return datetime.now().astimezone().isoformat()


def unit_state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit, '-p', 'LoadState',
        '-p', 'MainPID', '-p', 'Result', '-p', 'SubState', '-p', 'ActiveState',
        '-p', 'MemoryCurrent', '-p', 'MemoryPeak', '-p', 'MemorySwapCurrent',
        '-p', 'ControlGroup'], capture_output=True, text=True, check=True)
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def resources():
    memory = {line.split(':')[0]: int(line.split()[1]) * 1024
              for line in Path('/proc/meminfo').read_text().splitlines()}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    total, used, free = map(int, gpu.stdout.splitlines()[0].split(','))
    return dict(observed_at=now(), host_available_bytes=memory['MemAvailable'],
        host_total_bytes=memory['MemTotal'], swap_free_bytes=memory['SwapFree'],
        gpu_total_mib=total, gpu_used_mib=used, gpu_free_mib=free)


def intact(request):
    for path, digest in request['source_pins'].items():
        if sha(path) != digest:
            raise ValueError('Capacity request input changed: ' + path)


def probe_command(request, request_path, count, folder):
    argv = list(request['prototype_command'])
    if (argv[:3] != [str(ROOT / 'run_python.sh'), request['production_runtime'], 'native']
            or argv[argv.index('--checkpoint_path') + 1] != request['source_checkpoint']
            or argv[argv.index('--num_envs') + 1] != '4096'
            or any(x in argv for x in ('--reset_optimizer', '--reset_exploration_std',
                                      '--save_initial_checkpoint'))):
        raise ValueError('Require the exact successful native production command')
    boundary = argv.index('--')
    original_arguments = argv[boundary + 1:]
    arguments = list(original_arguments)
    values = {'--checkpoint_path': request['source_checkpoint'],
        '--target_iteration': request['source_actual_updates'] + request['probe_updates'],
        '--num_envs': count,
        '--run_name': 'capacity_only_' + folder.name,
        '--stop_file': folder / 'CAPACITY_ONLY_STOP'}
    for flag, value in values.items():
        if arguments.count(flag) != 1:
            raise ValueError('Expected one native option: ' + flag)
        arguments[arguments.index(flag) + 1] = str(value)
    if count not in (4096, 8192):
        raise ValueError('Unsupported requested capacity')
    # Check every other CLI value, including all physics/reward/PD settings.
    restored = list(arguments)
    for flag in values:
        restored[restored.index(flag) + 1] = original_arguments[original_arguments.index(flag) + 1]
    if restored != original_arguments:
        raise ValueError('Capacity command changed beyond counts and isolated output clocks')
    return [str(ROOT / 'run_python.sh'), str(BACKEND), '--capacity-request', str(request_path),
        '--observation-output', str(folder / 'native_learning_signal'),
        '--capacity-num-envs', str(count), '--', *arguments]


def require_capacity_selection(receipt_path, *, block_id, source, num_envs):
    """Future block entries must call this with their own fresh block ID.

    A historic8192 success, another source or another block cannot satisfy
    the user-requested check. This admits a scale, never a tracking model.
    """
    receipt = read(receipt_path)
    request = read(receipt['request'])
    intact(request)
    if (not receipt['completed'] or receipt['block_id'] != block_id
            or request['block_id'] != block_id
            or receipt['request_sha256'] != sha(receipt['request'])
            or receipt['source_checkpoint_sha256'] != sha(source)
            or request['source_checkpoint_sha256'] != sha(source)
            or receipt['selected_num_envs'] != num_envs
            or not receipt['fresh_capacity_check_required_for_each_new_training_block']
            or receipt['capacity_checkpoints_promoted']
            or not receipt['production_checkpoint_and_optimizer_unchanged']):
        raise ValueError('Each future training block needs its own exact-source capacity measurement')
    return dict(qualified=True, observed_at=now(), block_id=block_id,
        capacity_receipt=str(Path(receipt_path).resolve()), capacity_receipt_sha256=sha(receipt_path),
        source_checkpoint_sha256=sha(source), selected_num_envs=num_envs,
        standing_environments=num_envs // 4, motion_environments=num_envs * 3 // 4,
        sampled_transitions_per_update=num_envs * 48, original_Adam_minibatches_per_update=20,
        architecture_or_optimizer_reset=False, tracking_acceptance_proven=False)


def wait_owned_jobs(request, output):
    while True:
        intact(request)
        states = {unit: unit_state(unit) for unit in request['after_units']}
        waiting = []
        for unit, state in states.items():
            if state.get('LoadState') == 'not-found' or state.get('SubState') in ('failed', 'dead'):
                raise RuntimeError('Inspect owned prerequisite; never restart training: ' + unit)
            if not (state.get('MainPID') == '0' and state.get('Result') == 'success'
                    and state.get('SubState') == 'exited'):
                waiting.append(unit)
        write(output / 'wait_state.json', dict(observed_at=now(), state='waiting' if waiting else 'ready',
            waiting_units=waiting, unit_states=states, simulator_started=False,
            production_training_interrupted=False))
        if not waiting:
            for item in request['required_complete_reports']:
                report = read(item['path'])
                if report.get(item['completion_key']) is not True:
                    raise ValueError('Complete all current reviews before capacity testing')
            return
        time.sleep(15.)


def verify_finished_probe(folder, request, count):
    import torch
    import yaml
    from train_joint_pd_standing_seed import saved_updates
    from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
    from joint_pd_continuation import tensor_finite
    complete = read(folder / 'native_learning_signal/completed.json')
    load = read(folder / 'native_learning_signal/source200_runtime_load.json')
    if (not complete['completed'] or complete['actual_observed_updates'] != request['probe_updates']
            or complete['optimizer_minibatches'] != request['probe_updates'] * 20
            or any(v != 0 for v in complete['all_environment_alignment_max_errors'].values())
            or not complete['all_probe_RNG_unchanged']
            or load['checkpoint_sha256'] != request['source_checkpoint_sha256']
            or not load['all_model_and_optimizer_tensors_bitwise_preserved']):
        raise ValueError('Full PPO, native alignment and retained state must all be verified')
    for index in range(1, 6):
        audit = read(folder / ('native_learning_signal/audit_update_' + str(index).zfill(4) + '.json'))
        if audit['total_environments'] != count or len(audit['optimizer_steps']) != 20:
            raise ValueError('Verify the actual full environment count and original Adam updates')
    runs = list((ROOT / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat').glob(
        '*_capacity_only_' + folder.name))
    if len(runs) != 1:
        raise ValueError('Require one isolated capacity run directory')
    after_path = runs[0] / ('model_' + str(request['source_actual_updates'] + 4) + '.pt')
    saved = torch.load(after_path, map_location='cpu', weights_only=True)
    actual = saved_updates(after_path)
    states = saved['optimizer_state_dict']['state']
    if (actual != request['source_actual_updates'] + 5 or len(states) != 17
            or any(float(v['step']) != actual * 20 for v in states.values()) or not tensor_finite(saved)
            or saved['model_state_dict']['actor.0.weight'].shape[1] != 27
            or not saved.get('infos', {}).get('source200_stationary_height', {}).get('capacity_only_probe')):
        raise ValueError('Require a finite capacity-only saved checkpoint and all17 retained clocks')
    before = validate_checkpoint_robot(Path(request['source_checkpoint']))
    after = validate_checkpoint_robot(after_path)
    expected = json.loads(json.dumps(before))
    expected['log_dir'] = after['log_dir']
    for key in ('scene', 'terrain'):
        expected[key]['num_envs'] = count
    if after != expected:
        raise ValueError('Saved environment changed beyond parallelism and output directory')
    agent_before = yaml.safe_load((Path(request['source_checkpoint']).parent / 'params/agent.yaml').read_text())
    agent_after = yaml.safe_load((after_path.parent / 'params/agent.yaml').read_text())
    if agent_before != agent_after:
        raise ValueError('Capacity test must use exactly the saved PPO agent configuration')
    return dict(checkpoint=str(after_path), checkpoint_sha256=sha(after_path), actual_updates=actual,
        all17_Adam_step=actual * 20, original_minibatches=100,
        alignment_max_errors=complete['all_environment_alignment_max_errors'],
        retained_state_bitwise_at_load=True, capacity_only_profile=True,
        source_checkpoint_unchanged=sha(request['source_checkpoint']) == request['source_checkpoint_sha256'],
        only_environment_count_and_output_configuration_changed=True)


def run_probe(request, request_path, output, count, policy):
    folder = ROOT / 'mine_model/results' / ('joint_pd_optimization_capacity_' + output.name + '_n' + str(count))
    folder.mkdir(exist_ok=False)
    unit = 'clt-pd-capacity-' + str(count) + '-' + output.name.rsplit('_', 1)[-1]
    before = resources()
    budget = min(int(policy['maximum_probe_memory_gib'] * GIB),
        before['host_available_bytes'] - int(policy['minimum_host_reserve_gib'] * GIB)
        + min(before['swap_free_bytes'], int(policy['maximum_probe_swap_gib'] * GIB)))
    if budget < 12 * GIB:
        return dict(num_envs=count, passed=False, state='not_started_insufficient_current_host_memory',
            folder=str(folder), resources_before=before, retry_next_block=True,
            no_claim_absolute_capacity_limit=True)
    argv = probe_command(request, request_path, count, folder)
    log = folder / 'training.log'
    command = ['systemd-run', '--user', '--unit=' + unit, '--property=Type=exec',
        '--property=RemainAfterExit=yes', '--property=WorkingDirectory=' + str(ROOT),
        '--property=MemoryMax=' + str(budget),
        '--property=MemorySwapMax=' + str(int(policy['maximum_probe_swap_gib'] * GIB)),
        '--property=OOMPolicy=stop', '--property=StandardOutput=append:' + str(log),
        '--property=StandardError=append:' + str(log),
        '--setenv=PYTHONPATH=' + ':'.join(str(ROOT / p) for p in
            ('wheel_legged_isaaclab', 'mine_model/scripts', 'mine_model/experiments')),
        '--setenv=OMP_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1',
        '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=PYTHONDONTWRITEBYTECODE=1',
        '--setenv=PYTHONUNBUFFERED=1', *argv]
    record = dict(num_envs=count, unit=unit, folder=str(folder), command=command,
        started_at=now(), resources_before=before, memory_budget_bytes=budget,
        state='launching', passed=False, capacity_weights_may_be_promoted=False)
    write(folder / 'run.json', record)
    launched = subprocess.run(command, capture_output=True, text=True)
    record['launch'] = dict(returncode=launched.returncode, stdout=launched.stdout, stderr=launched.stderr)
    if launched.returncode:
        record['state'] = 'launch_failed'
        write(folder / 'run.json', record)
        return record
    started = time.monotonic()
    severe_pressure_since = None
    while True:
        state = unit_state(unit)
        snapshot = resources()
        for key in ('MemoryCurrent', 'MemoryPeak', 'MemorySwapCurrent'):
            if state.get(key, '').isdigit():
                record[key + '_observed_peak'] = max(record.get(key + '_observed_peak', 0), int(state[key]))
        record['gpu_used_mib_peak'] = max(record.get('gpu_used_mib_peak', 0), snapshot['gpu_used_mib'])
        record['host_available_bytes_min'] = min(record.get('host_available_bytes_min', snapshot['host_available_bytes']), snapshot['host_available_bytes'])
        record.update(state='running', unit_state=state, observed_at=now(), latest_resources=snapshot)
        if snapshot['host_available_bytes'] < .75 * GIB:
            severe_pressure_since = severe_pressure_since or time.monotonic()
        else:
            severe_pressure_since = None
        if severe_pressure_since is not None and time.monotonic() - severe_pressure_since >= 10:
            record['resource_abort'] = 'under0.75GiB host memory for10s; only this diagnostic unit stopped'
            subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, text=True, check=True)
            state = unit_state(unit)
        write(folder / 'run.json', record)
        if state.get('MainPID') == '0' or state.get('SubState') in ('failed', 'dead', 'exited'):
            break
        time.sleep(2.)
    record.update(wall_seconds=time.monotonic() - started, ended_at=now(), final_unit_state=state)
    text = log.read_text(errors='replace') if log.exists() else ''
    rows = re.findall(r'Computation:\s+([\d.]+) steps/s \(collection:\s*([\d.]+)s, learning\s*([\d.]+)s\)', text)
    steady = rows[1:] or rows
    record['logged_updates'] = len(re.findall(r'Learning iteration\s+\d+/', text))
    record['finite_error_log'] = not bool(re.search(
        r'(?i)(?:loss|tracking/|reward)[^\n]*\b(?:nan|inf)\b|CUDA error|OutOfMemoryError|PhysX error|\[Error\] \[omni\.physx(?:\.tensors)?\.plugin\]', text))
    if steady:
        record['steady_transition_steps_per_second'] = sum(float(v[0]) for v in steady) / len(steady)
        record['steady_update_seconds'] = sum(float(v[1]) + float(v[2]) for v in steady) / len(steady)
    clean = (state.get('MainPID') == '0' and state.get('Result') == 'success'
        and state.get('SubState') == 'exited' and not record.get('resource_abort'))
    record['minimum_gpu_headroom_mib'] = before['gpu_total_mib'] - record.get('gpu_used_mib_peak', before['gpu_used_mib'])
    record['required_gpu_headroom_retained'] = record['minimum_gpu_headroom_mib'] >= policy['minimum_gpu_reserve_mib']
    if clean and record['finite_error_log'] and record['required_gpu_headroom_retained']:
        try:
            record['native_verification'] = verify_finished_probe(folder, request, count)
            record['passed'] = True
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            record['verification_error'] = repr(error)
    record['state'] = 'passed' if record['passed'] else 'failed_or_not_verified'
    record['short_probe_does_not_prove_long_term_stability_or_tracking_acceptance'] = True
    intact(request)
    write(folder / 'run.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request_path = args.request.resolve()
    request = read(request_path)
    output = Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    policy = read(POLICY)
    if request['probe_updates'] != 5 or request['counts'] != [8192, 4096]:
        raise ValueError('Use the declared original full-rollout capacity comparison')
    intact(request)
    wait_owned_jobs(request, output)
    results = []
    for count in request['counts']:
        results.append(run_probe(request, request_path, output, count, policy))
        write(output / 'progress.json', dict(observed_at=now(), results=results,
            current_request=str(request_path), production_not_interrupted=True))
    good = [r for r in results if r['passed']]
    preferred = next((r for r in good if r['num_envs'] == 8192), None)
    fallback = next((r for r in good if r['num_envs'] == 4096), None)
    selected = preferred or fallback
    summary = dict(completed=True, observed_at=now(), block_id=request['block_id'],
        request=str(request_path), request_sha256=sha(request_path),
        source_checkpoint=request['source_checkpoint'], source_checkpoint_sha256=request['source_checkpoint_sha256'],
        all_results=results, selected_num_envs=selected['num_envs'] if selected else None,
        preferred8192_verified=bool(preferred), fallback4096_verified=bool(fallback),
        selection_reason='Prefer8192 when the complete unchanged PPO probe passes; keep both throughput measurements.',
        fresh_capacity_check_required_for_each_new_training_block=True,
        production_checkpoint_and_optimizer_unchanged=True, capacity_checkpoints_promoted=False,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    if preferred and fallback:
        summary['8192_to4096_transition_throughput_ratio'] = preferred.get('steady_transition_steps_per_second', 0) / fallback['steady_transition_steps_per_second']
        summary['batch_sizes'] = {'8192': 8192 * 48, '4096': 4096 * 48}
        summary['equal_update_counts_are_not_equal_sample_counts'] = True
    write(output / 'complete.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'all_results'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
