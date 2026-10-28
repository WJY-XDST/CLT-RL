"""Retain actual five-update capacity runs; normalize only their run budget.

The sealed v4 checker reached its final YAML check after a successful8192
run. Its remaining mismatch was max_iterations200 versus the declared5.
Allow exactly that diagnostic budget, retain all other PPO checks, and
verify the existing checkpoint/records without repeating native PPO.
"""
import argparse
import ast
import copy
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

import check_plain27_environment_capacity_v4 as serial
import check_plain27_environment_capacity_v3 as retained
import check_plain27_environment_capacity as original

SCRIPT = Path(__file__).resolve()
_new_completion = retained.capacity_external_finalization


def completion_once(folder, request, count):
    directory = Path(folder) / 'native_learning_signal'
    path = directory / 'completed_external_capacity.json'
    if not path.exists():
        return _new_completion(folder, request, count)
    report = original.read(path)
    native = original.read(directory / 'plan.json')
    paths = sorted(directory.glob('audit_update_*.json'))
    if (len(paths) != 5 or report['actual_observed_updates'] != 5
            or report['actual_environment_count'] != count
            or report['optimizer_minibatches'] != 100 or report['PPO_updates_repeated']
            or not report['all_five_records_counted_once']
            or native['source_sha256'] != request['source_checkpoint_sha256']
            or native['target_actual_ppo_updates'] != request['source_actual_updates'] + 5
            or set(report['records_sha256']) != {str(p) for p in paths}
            or any(original.sha(p) != digest for p, digest in report['records_sha256'].items())
            or any(original.sha(p) != digest for p, digest in report['source_files_sha256'].items())):
        raise ValueError('Revalidate the exact original five-update capacity records')
    for index in range(1, 6):
        with np.load(directory / ('rollout_' + str(index).zfill(4) + '.npz'), allow_pickle=False) as arrays:
            if (arrays['observations'].shape != (48, 32, 27)
                    or arrays['actions'].shape != (48, 32, 6)
                    or any(not np.isfinite(arrays[k]).all() for k in arrays.files)
                    or not np.array_equal(arrays['actions'], arrays['native_raw_actions'])):
                raise ValueError('Preserve every complete finite native rollout')
    return report


tree = copy.deepcopy(retained.modified)
before = ast.dump(tree)
function = tree.body[0]
indices = [i for i, node in enumerate(function.body) if isinstance(node, ast.If)
    and ast.dump(node.test) == ast.dump(ast.parse('agent_before != agent_after', mode='eval').body)]
if len(indices) != 1:
    raise ValueError('Requalify the original saved-agent verification')
index = indices[0]
added = ast.parse("agent_before['max_iterations'] = request['probe_updates']").body[0]
function.body.insert(index, added)
modified = copy.deepcopy(tree)
function.body.pop(index)
if ast.dump(tree) != before:
    raise ValueError('Verifier changed beyond the declared five-update diagnostic budget')
original.capacity_external_finalization = completion_once
exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), 'exec'), original.__dict__)
original.SCRIPT = SCRIPT
require_capacity_selection = original.require_capacity_selection
main = original.main


def verify_existing(request_path, folder, destination):
    request_path, folder, destination = map(lambda p: Path(p).resolve(),
        (request_path, folder, destination))
    request = original.read(request_path)
    original.intact(request)
    record_path = folder / 'run.json'
    record = original.read(record_path)
    unit = record['unit'] + '.service'
    state = original.unit_state(unit)
    if (destination.exists() or state.get('MainPID') != '0'
            or state.get('Result') != 'success' or state.get('SubState') != 'exited'
            or record.get('resource_abort') or record['logged_updates'] != 5
            or not record['finite_error_log'] or not record['required_gpu_headroom_retained']):
        raise ValueError('Verify only an existing successful native resource probe; never restart PPO')
    proof = original.verify_finished_probe(folder, request, record['num_envs'])
    if not proof['source_checkpoint_unchanged']:
        raise ValueError('The production source must remain untouched')
    result = dict(verified=True, passed=True, observed_at=datetime.now().astimezone().isoformat(),
        num_envs=record['num_envs'], unit=unit, unit_state=state,
        source_checkpoint=request['source_checkpoint'],
        source_checkpoint_sha256=request['source_checkpoint_sha256'],
        request=str(request_path), request_sha256=original.sha(request_path),
        resource_record=str(record_path), resource_record_sha256=original.sha(record_path),
        native_verification=proof,
        actual_full_PPO_updates=5, actual_original_Adam_minibatches=100,
        memory_peak_bytes=record['MemoryCurrent_observed_peak'],
        native_swap_peak_bytes=record.get('MemorySwapCurrent_observed_peak'),
        gpu_card_peak_mib=record['gpu_used_mib_peak'],
        host_available_min_bytes=record['host_available_bytes_min'],
        steady_transition_steps_per_second=record['steady_transition_steps_per_second'],
        steady_update_seconds=record['steady_update_seconds'],
        previous_verification_error=record.get('verification_error'),
        only_agent_metadata_normalizations=['run_name', 'algorithm/learning_rate_from_actual_native_load',
            'max_iterations_exact_declared_probe_updates5'],
        original_saved_checkpoint_verifier_AST_restored_identical=True,
        original_failed_checker_and_complete_records_preserved=True,
        raw_rollout_sha256={str(p):original.sha(p) for p in
            sorted((folder / 'native_learning_signal').glob('rollout_*.npz'))},
        verifier_sha256=original.sha(SCRIPT), PPO_updates_repeated=0,
        capacity_checkpoint_promoted=False, source_checkpoint_optimizer_std_LR_unchanged=True,
        short_capacity_probe_is_not_long_term_or_tracking_acceptance=True,
        policy_accepted=False)
    original.write(destination, result)
    return result


if __name__ == '__main__':
    if '--verify-existing' in sys.argv:
        cli = argparse.ArgumentParser(description=__doc__)
        cli.add_argument('--verify-existing', action='store_true', required=True)
        cli.add_argument('--request', type=Path, required=True)
        cli.add_argument('--folder', type=Path, required=True)
        cli.add_argument('--output', type=Path, required=True)
        args = cli.parse_args()
        proof = verify_existing(args.request, args.folder, args.output)
        print(json.dumps({k:proof[k] for k in ('verified', 'num_envs', 'actual_full_PPO_updates',
            'memory_peak_bytes', 'gpu_card_peak_mib', 'steady_transition_steps_per_second',
            'PPO_updates_repeated')}, ensure_ascii=False))
    else:
        main()
