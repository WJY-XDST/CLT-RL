"""Use the recorded native floor profile without altering pinned plan metadata."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path

import audit_native_motion_training_prefix as original
import plain27_sampler_metadata_adapter as adapter

SCRIPT = Path(__file__).resolve()
_api = adapter.private_module(original.__file__, '_native_physical_profile_metadata_reader')


def verified_exploration_guard(plan, output):
    record = json.loads((Path(output) / 'native_learning_signal/leg_exploration_floor_runtime.json').read_text())
    profile = plan['profile']
    keys = ('minimum_leg_std', 'minimum_wheel_std', 'global_minimum_std')
    if (record['profile'] != profile or not record['projection_after_every_optimizer_step']
            or not record['parameter_identity_preserved'] or record['added_loss_or_backward']
            or record['before_std'] != record['after_std']
            or not record['all17_Adam_bitwise_preserved']
            or not record['actor_and_critic_bitwise_preserved']
            or not record['wheel_std_bitwise_preserved']
            or any(not 0. < profile[k] < 1. for k in keys)):
        raise ValueError('Require the intact recorded native exploration profile')
    changes = {k: [plan['exploration_guard'][k], profile[k]] for k in keys
        if plan['exploration_guard'][k] != profile[k]}
    declared = plan.get('only_exploration_change', {})
    if (changes and changes != declared) or plan['exploration_guard']['saved_initial_std_reset']:
        raise ValueError('Native floor differs beyond the exact authorized exploration change')
    return {k: profile[k] for k in keys}


_tree = ast.parse(inspect.getsource(original.audit))
_before, _edits = ast.dump(_tree), []
for _node in ast.walk(_tree):
    if (isinstance(_node, ast.Subscript) and isinstance(_node.value, ast.Name)
            and _node.value.id == 'plan' and isinstance(_node.slice, ast.Constant)
            and _node.slice.value == 'exploration_guard'):
        _edits.append((_node, copy.deepcopy(_node)))
if len(_edits) != 2:
    raise ValueError('Requalify the exact two declared floor metadata lookups')
for _node, _ in _edits:
    _node.value = ast.Name(id='verified_guard_plan', ctx=ast.Load())
_changed = copy.deepcopy(_tree)
for _node, _old in _edits:
    _node.value = _old.value
if ast.dump(_tree) != _before:
    raise ValueError('Original physical metrics, filters or data checks changed')
_api.__file__ = str(SCRIPT)
exec(compile(ast.fix_missing_locations(_changed), original.__file__, 'exec'), _api.__dict__)


def audit(output, through):
    output = Path(output).resolve()
    plan = json.loads((output / 'experiment_plan.json').read_text())
    guard = verified_exploration_guard(plan, output)
    # This is an in-memory reader lookup, not a rewrite of the native plan.
    _api.verified_guard_plan = {'exploration_guard': guard}
    target, report = _api.audit(output, through)
    report.update(declared_exploration_guard_used=guard,
        preserved_legacy_plan_exploration_guard=plan['exploration_guard'],
        native_recorded_profile_matches_plan_profile=True,
        original_numeric_reader_AST_restored_identical=True,
        only_two_floor_metadata_lookups_adapted=True,
        pinned_native_plan_unmodified=True, acceptance_changed=False)
    for path in (SCRIPT, Path(original.__file__), adapter.SCRIPT,
            output / 'native_learning_signal/leg_exploration_floor_runtime.json'):
        report['source_sha256'][str(path)] = original.sha(path)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return target, report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    path, result = audit(args.output, args.through)
    print(json.dumps(dict(report=str(path), committed_updates=result['committed_updates'],
        declared_guard=result['declared_exploration_guard_used'],
        native_std_range=result['native_wheel_std_range'],
        original_metrics_unchanged=True, training_modified=False)))
