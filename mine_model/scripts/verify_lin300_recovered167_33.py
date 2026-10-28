"""Original full200 verifier routed to immutable167+33 native segment files.

Only audit-path and segment-local update metadata are adapted. Every original
gradient, RNG, state, clock, configuration and finite-value predicate remains.
"""
import ast
import copy
import inspect
import json
from pathlib import Path

from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import verify_completed_plain27_joint_pd_trial as original
import recover_lin300_native33_20261020 as recovery

SCRIPT = Path(__file__).resolve()
VIEW = ROOT / 'mine_model/results/joint_pd_optimization_20261020_recovered_source6201_lin300_200'


def module(path_for, local_index, *, script=SCRIPT):
    api = adapter.private_module(original.SCRIPT, '_original_full200_segmented_verifier')
    api.SCRIPT, api.segmented_audit_path, api.segmented_local_index = script, path_for, local_index
    tree = ast.parse(inspect.getsource(original.verify))
    before, changes = ast.dump(tree), []
    expected_path = ast.dump(ast.parse("directory / f'native_learning_signal/audit_update_{index:04d}.json'", mode='eval').body)
    expected_index = ast.dump(ast.parse("record['update'] != index", mode='eval').body)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and ast.dump(node.value) == expected_path:
            changes.append((node, 'value', copy.deepcopy(node.value)))
            node.value = ast.parse('segmented_audit_path(directory, index)', mode='eval').body
        elif isinstance(node, ast.Compare) and ast.dump(node) == expected_index:
            changes.append((node, 'comparators', copy.deepcopy(node.comparators)))
            node.comparators = [ast.parse('segmented_local_index(index)', mode='eval').body]
    if len(changes) != 2:
        raise ValueError('Requalify the exact two segment-metadata AST locations')
    modified = copy.deepcopy(tree)
    for node, field, value in reversed(changes):
        setattr(node, field, value)
    if ast.dump(tree) != before:
        raise ValueError('Original verifier changed outside segment-path/local-index metadata')
    exec(compile(ast.fix_missing_locations(modified), str(original.SCRIPT), 'exec'), api.__dict__)
    return api, dict(original_verifier=str(original.SCRIPT), original_verifier_sha256=sha(original.SCRIPT),
        metadata_locations_changed=2, restored_original_AST_identical=True,
        original_all200_predicates_unchanged=True, raw_audits_rewritten_or_renumbered=False)


def local(index):
    if not 1 <= index <= 200:
        raise ValueError('Only the exact retained200 block')
    return index if index <= 167 else index - 167


def path_for(directory, index):
    if Path(directory) != VIEW:
        raise ValueError('Only the qualified recovered block view')
    segment = recovery.OLD if index <= 167 else recovery.OUTPUT
    return segment / 'native_learning_signal' / f'audit_update_{local(index):04d}.json'


def verify(directory, destination=None):
    directory = Path(directory).resolve()
    if directory != VIEW:
        raise ValueError('Do not route an unrelated block to these retained segments')
    integrity = json.loads((VIEW / 'recovery_integrity.json').read_text())
    if (not integrity['retained200_verified'] or integrity['old_retained_updates'] != 167
            or integrity['new_observed_updates'] != 33 or integrity['all_executed_updates'] != 200
            or integrity['PPO_updates_repeated'] or not integrity['all17_resume_load_bitwise_preserved']
            or integrity['raw_records_rewritten_or_renumbered']
            or [r['actual_update'] for r in integrity['checkpoint_rows']] != list(range(6202, 6402))):
        raise ValueError('Require an exact immutable200 checkpoint/Adam/rollout ledger')
    recovery.intact(integrity['source_pins'])
    for index, row in enumerate(integrity['checkpoint_rows'], 1):
        if (Path(row['audit']) != path_for(directory, index)
                or row['segment_local_update'] != local(index)
                or row['all17_Adam_step'] != (6201 + index) * 20):
            raise ValueError('A raw segment record was incorrectly routed')
    api, _ = module(path_for, local)
    return api.verify(directory, destination)


def qualify_fixture(output):
    """Exercise both verifiers against the same real completed200 fixture."""
    fixture = recovery.OLD.parent / 'joint_pd_optimization_20261020_175930_562396_source6001_orientation060_200'
    api, proof = module(lambda directory, index: directory / f'native_learning_signal/audit_update_{index:04d}.json',
        lambda index: index, script=original.SCRIPT)
    before = original.verify(fixture, output / 'original_fixture_verification.json')
    after = api.verify(fixture, output / 'segmented_adapter_fixture_verification.json')
    before.pop('observed_at')
    after.pop('observed_at')
    if before != after:
        raise ValueError('The complete real-data verifier result changed beyond timestamp')
    if ([local(i) for i in range(1, 201)] != list(range(1, 168)) + list(range(1, 34))
            or path_for(VIEW, 167).parent.parent != recovery.OLD
            or path_for(VIEW, 168).parent.parent != recovery.OUTPUT):
        raise ValueError('Exact167/33 routing failed')
    proof.update(real_completed200_full_verification_result_equal=True,
        fixture=str(fixture), actual_updates=6201, all17_Adam_step=124020,
        segment_boundary_167_168_and_final200_checked=True,
        optimizer_executed=False, simulator_started=False)
    return proof
