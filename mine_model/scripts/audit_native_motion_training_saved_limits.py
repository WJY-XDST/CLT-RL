"""Preserve all native physical metrics; correct the wheel-clipping counter."""
import argparse
import ast
import copy
import json
from pathlib import Path

from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
import audit_native_motion_training_profile as profile_reader

SCRIPT = Path(__file__).resolve()


def implementation(limit):
    original = copy.deepcopy(profile_reader._changed)
    target = ast.dump(ast.parse("np.abs(data['actions'][..., [2, 5]]) > 1.", mode='eval').body)
    counts = dict(bound=0, destination=0)
    changed = copy.deepcopy(original)
    for node in ast.walk(changed):
        if isinstance(node, ast.Compare) and ast.dump(node) == target:
            node.comparators = [ast.Name(id='SAVED_WHEEL_LIMIT', ctx=ast.Load())]
            counts['bound'] += 1
        if isinstance(node, ast.Constant) and node.value == 'native_motion_training_prefix_':
            node.value = 'native_motion_training_saved_limits_prefix_'
            counts['destination'] += 1
    if counts != dict(bound=1, destination=1):
        raise ValueError('Requalify the single counter boundary and new output basename')
    restored = copy.deepcopy(changed)
    for node in ast.walk(restored):
        if (isinstance(node, ast.Compare) and len(node.comparators) == 1
                and isinstance(node.comparators[0], ast.Name) and node.comparators[0].id == 'SAVED_WHEEL_LIMIT'):
            node.comparators = [ast.Constant(value=1.)]
        if isinstance(node, ast.Constant) and node.value == 'native_motion_training_saved_limits_prefix_':
            node.value = 'native_motion_training_prefix_'
    if ast.dump(restored) != ast.dump(original):
        raise ValueError('Original physical errors, masks, flags, windows or guards changed')
    scope = dict(profile_reader._api.__dict__, SAVED_WHEEL_LIMIT=limit, __file__=str(SCRIPT))
    return changed, scope


def audit(output, through):
    output = Path(output).resolve()
    plan = json.loads((output / 'experiment_plan.json').read_text())
    limit = checkpoint_wheel_action_limit(plan['training_config'])
    guard = profile_reader.verified_exploration_guard(plan, output)
    tree, scope = implementation(limit)
    scope['verified_guard_plan'] = {'exploration_guard': guard}
    exec(compile(ast.fix_missing_locations(tree), str(profile_reader.original.__file__), 'exec'), scope)
    path, report = scope['audit'](output, through)
    report.update(saved_wheel_action_limit=limit,
        wheel_clipping_counter_uses_saved_execution_bound=True,
        original_physical_metrics_masks_and_all9_alignment_checks_preserved=True,
        declared_exploration_guard_used=guard, original_numeric_reader_AST_restored_identical=True,
        only_legacy_clipping_counter_and_output_basename_changed=True,
        supersedes_unit_bound_wheel_clipping_counter_only=True)
    report['source_sha256'].update({str(p): sha(p) for p in
        (SCRIPT, profile_reader.SCRIPT, Path(profile_reader.original.__file__))})
    original_path = output / f'native_motion_training_prefix_{through:04d}_readonly.json'
    if original_path.exists():
        legacy = json.loads(original_path.read_text())
        differences = []
        for window, groups in report['groups'].items():
            for group, data in groups.items():
                a, b = dict(data), dict(legacy['groups'][window][group])
                corrected, old = a.pop('clipped_wheel_action_fraction'), b.pop('clipped_wheel_action_fraction')
                if a != b:
                    raise ValueError('Native physical metrics changed beyond the single clipping counter')
                differences.append(dict(window=window, population=group,
                    legacy_unit_bound_fraction=old, saved_bound_fraction=corrected))
        report.update(legacy_report_preserved=str(original_path),
            all_other_group_fields_bitwise_equal_to_legacy_report=True, corrected_counters=differences)
        report['source_sha256'][str(original_path)] = sha(original_path)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return path, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--through', type=int, required=True)
    args = parser.parse_args()
    path, report = audit(args.output, args.through)
    print(json.dumps(dict(report=str(path), wheel_action_limit=report['saved_wheel_action_limit'],
        original_metrics_preserved=report['original_physical_metrics_masks_and_all9_alignment_checks_preserved'])))
