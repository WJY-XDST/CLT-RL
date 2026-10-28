"""Use the original yaw residual reader with either saved wheel limit.

The historical source uses 1.0, whereas the newer source used 1.25. Only
that admission predicate changes; clipping, masks and numerical metrics
continue to read the actual saved configuration without alteration.
"""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path

import audit_native_sampled_yaw_reference_components as original
import plain27_sampler_metadata_adapter as adapter


SCRIPT = Path(__file__).resolve()
api = adapter.private_module(original.SCRIPT, "_historical_saved_limit_yaw_reader")
api.SCRIPT = SCRIPT
tree = ast.parse(inspect.getsource(original.audit))
before, changes = ast.dump(tree), []
for node in ast.walk(tree):
    if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name)
            and node.left.id == "limit" and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and isinstance(node.comparators[0], ast.Constant)
            and node.comparators[0].value == 1.25):
        changes.append((node, copy.deepcopy(node.ops), copy.deepcopy(node.comparators)))
        node.ops = [ast.In()]
        node.comparators = [ast.Tuple(elts=[ast.Constant(1.), ast.Constant(1.25)], ctx=ast.Load())]
assert len(changes) == 1
modified = copy.deepcopy(tree)
for node, ops, comparators in changes:
    node.ops, node.comparators = ops, comparators
assert ast.dump(tree) == before
exec(compile(ast.fix_missing_locations(modified), str(original.SCRIPT), "exec"), api.__dict__)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--through", type=int, required=True)
    args = cli.parse_args()
    path, report = api.audit(args.directory, args.through)
    # Keep the original numerical report intact and record the one admission
    # change in a separate receipt, including both original source hashes.
    receipt = dict(verified=True, numerical_report=str(path),
        source_pins={str(p): original.sha(p) for p in (SCRIPT, original.SCRIPT, path)},
        original_AST_restored_identical=True, changed_admission_predicates=1,
        admitted_saved_wheel_action_limits=[1., 1.25],
        numerical_formulas_masks_and_saved_limit_clipping_unchanged=True,
        training_modified=False, optimizer_executed=False, policy_accepted=False)
    destination = path.with_name(f"historical_yaw_reader_adapter_prefix_{args.through:04d}_readonly.json")
    assert not destination.exists()
    destination.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(verified=True, report=str(path), adapter_receipt=str(destination),
        closure_max_error=report["signed_component_closure_max_error_rad_s"],
        training_modified=False, policy_accepted=False)))
