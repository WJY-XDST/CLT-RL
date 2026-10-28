"""Bind the original external pair reader to the admitted native population.

Only its one population comparison changes. Numeric alignment, optimizer,
finite-value, RNG and fixed-input-delta checks remain the original operations.
Install into a private future external-finalizer module, never a live process.
"""
import ast
import copy
import inspect
from pathlib import Path

SCRIPT = Path(__file__).resolve()


def install(external_api, admitted_num_envs):
    if type(admitted_num_envs) is not int or admitted_num_envs not in (4096, 8192):
        raise ValueError('Require the actual independently admitted native population')
    original = external_api.summarize_pair
    tree = ast.parse(inspect.getsource(original))
    before = ast.dump(tree)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
               and ast.unparse(node.left) == "record['total_environments']"]
    if (len(matches) != 1 or len(matches[0].comparators) != 1
            or len(matches[0].ops) != 1 or not isinstance(matches[0].ops[0], ast.NotEq)
            or not isinstance(matches[0].comparators[0], ast.Constant)
            or matches[0].comparators[0].value != 8192):
        raise ValueError('Requalify the exact original single population guard')
    comparison = matches[0].comparators[0]
    comparison.value = admitted_num_envs
    changed = copy.deepcopy(tree)
    comparison.value = 8192
    if ast.dump(tree) != before:
        raise ValueError('Original numeric audit operations changed beyond the population guard')
    namespace = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(changed), str(SCRIPT), 'exec'), namespace)
    external_api.summarize_pair = namespace[original.__name__]
    return dict(original_pair_reader=str(Path(inspect.getfile(original)).resolve()),
                original_AST_restored_identical=True, admitted_native_num_envs=admitted_num_envs,
                population_comparison_count=1, numeric_RNG_and_optimizer_checks_unchanged=True,
                original_reader_globals_unmodified=True, training_logic_changed=False)
