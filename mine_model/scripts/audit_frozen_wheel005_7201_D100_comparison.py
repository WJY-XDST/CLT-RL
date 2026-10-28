"""Reuse original paired frozen-drive metrics for the declared .75/1.0 test.

Only the drive-factor guards, owned routes and report labels are rebound.
Original physical scorers, literal10 algebra, windows, failures and None
remain unchanged. A completed prefix is explicitly not acceptance.
"""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import audit_frozen_AF7001_wheel_damping_comparison as original
import plain27_sampler_metadata_adapter as adapter
import run_frozen_wheel005_7201_D100_diagnostic as driver


SCRIPT = Path(__file__).resolve()
BINDING = driver.OUTPUT / 'D100_original_comparison_binding_qualification_20261024.json'


def private_reader():
    api = adapter.private_module(original.SCRIPT, '_wheel005_D100_original_comparison')
    tree = ast.parse(inspect.getsource(original.audit))
    before, changes, counts = ast.dump(tree), [], {}
    labels = {
        'baseline05_native_qualification.json': 'baseline075_native_qualification.json',
        'damping075_': 'damping100_',
        'source05': 'source075',
        'frozen075': 'frozen100',
    }
    # In this original function these four float nodes are exactly the two
    # expected wheel-drive assignments and the reported factor pair. None is
    # a performance threshold or part of a numeric metric calculation.
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg in labels:
            changes.append((node, 'arg', node.arg))
            node.arg = labels[node.arg]
            counts['owned_result_keyword_label'] = counts.get('owned_result_keyword_label', 0) + 1
            continue
        if not isinstance(node, ast.Constant):
            continue
        value = node.value
        if type(value) is float and value in (.5, .75):
            new, label = ({.5: .75, .75: 1.}[value], 'declared_drive_factor_guard')
        elif isinstance(value, str) and value in labels:
            new, label = labels[value], 'owned_report_label'
        else:
            continue
        changes.append((node, 'value', copy.deepcopy(value)))
        node.value = new
        counts[label] = counts.get(label, 0) + 1
    assert counts['declared_drive_factor_guard'] == 4
    assert counts['owned_result_keyword_label'] == 2
    assert sum(value == .5 for _, _, value in changes if type(value) is float) == 1
    assert sum(value == .75 for _, _, value in changes if type(value) is float) == 3
    modified = copy.deepcopy(tree)
    for node, field, value in changes:
        setattr(node, field, value)
    assert ast.dump(tree) == before
    for name in ('relative', 'strict', 'summary'):
        old, new = getattr(original, name), getattr(api, name)
        assert old.__code__.co_code == new.__code__.co_code
        assert old.__code__.co_consts == new.__code__.co_consts
    proof = dict(qualified=True,
        original_audit_AST_restored_identical=True,
        changed_nodes=counts,
        only_float_changes_are_declared_drive_guard_and_factor_metadata=True,
        original_relative_strict_and_summary_bytecode_and_constants_identical=True,
        all_numeric_performance_thresholds_and_failure_semantics_unchanged=True,
        source_checkpoint_sha256=driver.SOURCE_SHA,
        controller_factor={'wheel_damping': [.75, 1.]},
        source_pins={str(path): driver.sha(path) for path in
            (SCRIPT, original.SCRIPT, adapter.SCRIPT, driver.SCRIPT)},
        PPO_updates=0, simulator_started=False, policy_accepted=False)
    if BINDING.exists():
        assert driver.read(BINDING) == proof
    else:
        assert driver.read_and_check_preparation()['qualified_CPU_only']
        with BINDING.open('x') as stream:
            json.dump(proof, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')

    def checked_preparation():
        value = driver.read_and_check_preparation()
        assert all(driver.sha(path) == digest for path, digest in proof['source_pins'].items())
        return dict(value, source_pins={**value['source_pins'], **proof['source_pins'],
                                      str(BINDING): driver.sha(BINDING)})

    api.SCRIPT = SCRIPT
    api.driver = SimpleNamespace(**dict(driver.__dict__,
        read_and_check_preparation=checked_preparation))
    exec(compile(ast.fix_missing_locations(modified), str(original.SCRIPT), 'exec'), api.__dict__)
    assert api.audit.__globals__ is api.__dict__
    return api


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--allow-completed-prefix-only', action='store_true')
    args = cli.parse_args()
    private_reader().audit(args.output,
        require_complete=not args.allow_completed_prefix_only)
