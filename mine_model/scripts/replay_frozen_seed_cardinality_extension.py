"""Diagnostic-only seed-count extension of the original frozen evaluator.

The original interface intentionally accepts three seeds. Preserve its files
and standard reviews; change only parser cardinality and case/seed indexing
in a private syntax tree. A three-seed native parity check must qualify any
later larger reset-recovery experiment before its results are interpreted.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import plain27_sampler_metadata_adapter as adapter

ALLOWED = {ROOT / 'mine_model/experiments' / name for name in (
    'joint_pd_source6601_targeted_turn030_runtime.py',
    'joint_pd_source6801_targeted030_yaw0075_runtime.py')}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extend_tree(tree):
    tree = copy.deepcopy(tree)
    original = ast.dump(tree)
    parsers = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
        and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--seeds']
    if len(parsers) != 1:
        raise ValueError('Requalify the original one seed parser')
    nargs = [k for k in parsers[0].keywords if k.arg == 'nargs']
    if len(nargs) != 1 or not isinstance(nargs[0].value, ast.Constant) or nargs[0].value.value != 3:
        raise ValueError('The preserved evaluator must explicitly require three seeds')
    nargs[0].value = ast.Constant(value='+')
    declarations = [('seeds[i % 3]', 'seeds[i % len(seeds)]'),
        ('eid // 3', 'eid // len(seeds)'), ('seeds[eid % 3]', 'seeds[eid % len(seeds)]'),
        ('round(args.seconds / env.step_dt) * len(cases) * 3',
         'round(args.seconds / env.step_dt) * len(cases) * len(seeds)'),
        ('len(cases) * 3', 'len(cases) * len(seeds)')]
    table = {ast.dump(ast.parse(a, mode='eval').body): ast.parse(b, mode='eval').body
        for a, b in declarations}
    counts = {key: 0 for key in table}
    changed_locations = {}

    def location(node):
        return tuple(getattr(node, k, None) for k in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'))

    class Replace(ast.NodeTransformer):
        def visit(self, node):
            key = ast.dump(node)
            if key in table:
                counts[key] += 1
                changed_locations[location(node)] = copy.deepcopy(node)
                return ast.copy_location(copy.deepcopy(table[key]), node)
            return super().visit(node)

    changed = Replace().visit(tree)
    if list(counts.values()) != [1, 1, 1, 1, 2] or len(changed_locations) != 6:
        raise ValueError('Requalify only the declared reset and trace indexing expressions')

    class Restore(ast.NodeTransformer):
        def visit(self, node):
            value = changed_locations.get(location(node))
            return ast.copy_location(copy.deepcopy(value), node) if value is not None else super().visit(node)

    restored = Restore().visit(copy.deepcopy(changed))
    restored_parser = [n for n in ast.walk(restored) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
        and n.args and isinstance(n.args[0], ast.Constant) and n.args[0].value == '--seeds'][0]
    next(k for k in restored_parser.keywords if k.arg == 'nargs').value = ast.Constant(value=3)
    if ast.dump(restored) != original:
        raise ValueError('Physics, policy, reward, reset formula or trace content changed')
    evidence = dict(restored_original_AST_identical=True, seed_parser_nargs=[3, '+'],
        only_seed_cardinality_expressions_changed=[dict(before=a, after=b) for a, b in declarations],
        declared_occurrence_counts=list(counts.values()),
        original_three_seed_indexing_mathematically_identical=True,
        wheel_probe_mode_not_supported_by_extension=True)
    return ast.fix_missing_locations(changed), evidence


def replay(runtime_path, arguments):
    runtime_path = Path(runtime_path).resolve()
    arguments = list(arguments)
    if runtime_path not in ALLOWED or '--seeds' not in arguments or '--output' not in arguments:
        raise ValueError('Require an owned frozen runtime with explicit seeds and output')
    forbidden = ('--wheel_probe', '--bias_probe_json', '--overrides_json', '--export_dir',
        '--heading-hold', '--heading_hold', '--wheel-reference-limit-probe',
        '--physics_hz', '--control_hz', '--curriculum_progress')
    if any(a.startswith('--repair_') or a in forbidden for a in arguments):
        raise ValueError('Only normal frozen control and a declared command/reset diagnosis are allowed')
    i = arguments.index('--seeds') + 1
    j = next((k for k in range(i, len(arguments)) if arguments[k].startswith('--')), len(arguments))
    seeds = [int(v) for v in arguments[i:j]]
    if len(seeds) not in (3, 4096) or len(set(seeds)) != len(seeds) or any(not 0 <= v < 2**31 for v in seeds):
        raise ValueError('Only the original three-seed fixture or the owned4096-seed stress is admitted')
    output = Path(arguments[arguments.index('--output') + 1]).resolve()
    if output.exists() or not output.is_relative_to(ROOT / 'mine_model/results'):
        raise ValueError('Preserve previous frozen data in an owned fresh directory')
    runtime = adapter.private_module(runtime_path, '_frozen_seed_cardinality_runtime')
    original_tree = runtime.evaluator_tree
    tree, evaluator_path = original_tree()
    changed, evidence = extend_tree(tree)
    runtime.evaluator_tree = lambda: (changed, evaluator_path)
    receipt = dict(wrapper_sha256=sha(SCRIPT), runtime=str(runtime_path), runtime_sha256=sha(runtime_path),
        evaluator=str(evaluator_path), evaluator_sha256=sha(evaluator_path),
        seeds=seeds, seed_count=len(seeds), seed_count_extension=evidence,
        optimizer_steps_executed=0, policy_accepted=False)
    with output.with_name(output.name + '_seed_cardinality_adapter.json').open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return runtime.replay(arguments)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--runtime', type=Path, required=True)
    args, remaining = cli.parse_known_args()
    replay(args.runtime, remaining)
