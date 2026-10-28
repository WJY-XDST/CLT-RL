"""Retarget source/target clock and sampler metadata, preserving original ASTs.

This helper does not modify the robot controller, observations, PPO, reward
formulas or the original numeric acceptance readers.
"""
import ast
import copy
import importlib.util
import inspect
from pathlib import Path

SCRIPT = Path(__file__).resolve()


def private_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def retarget(module, names, *, integers=None, strings=None, exact_strings=None,
             keyword_names=None, comparisons=None):
    integers, strings = integers or {}, strings or {}
    exact_strings, keyword_names = exact_strings or {}, keyword_names or {}
    comparisons = comparisons or {}
    evidence = []
    for name in names:
        function = getattr(module, name)
        tree = ast.parse(inspect.getsource(function))
        before, changes, counts = ast.dump(tree), [], {}

        def change(node, field, value, label):
            changes.append((node, field, copy.deepcopy(getattr(node, field))))
            setattr(node, field, value)
            counts[label] = counts.get(label, 0) + 1

        class Adapter(ast.NodeVisitor):
            def visit_Constant(self, node):
                table = integers.get(name, integers.get('*', {}))
                if type(node.value) is int and node.value in table:
                    change(node, 'value', table[node.value], 'declared_clock_or_reset_count_metadata')
                elif isinstance(node.value, str):
                    if node.value in exact_strings:
                        value = exact_strings[node.value]
                    else:
                        value = node.value
                        for old, new in strings.items():
                            value = value.replace(old, new)
                    if value != node.value:
                        change(node, 'value', value, 'declared_label_or_command_option_metadata')

            def visit_keyword(self, node):
                if node.arg in keyword_names:
                    change(node, 'arg', keyword_names[node.arg], 'declared_lineage_key_metadata')
                self.generic_visit(node)

        Adapter().visit(tree)
        for expression, old, new in comparisons.get(name, []):
            expected = ast.dump(ast.parse(expression, mode='eval').body)
            matches = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
                and len(n.ops) == 1 and isinstance(n.ops[0], ast.NotEq)
                and ast.dump(n.left) == expected and len(n.comparators) == 1
                and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value == old]
            if len(matches) != 1:
                raise ValueError('Requalify the single declared metadata comparison: ' + name + '/' + expression)
            change(matches[0].comparators[0], 'value', new, 'declared_source_stage_metadata')
        modified = copy.deepcopy(tree)
        for node, field, value in reversed(changes):
            setattr(node, field, value)
        if ast.dump(tree) != before:
            raise ValueError('Original code changed outside the declared source/stage metadata')
        exec(compile(ast.fix_missing_locations(modified), inspect.getsourcefile(function), 'exec'), module.__dict__)
        evidence.append(dict(function=name, declared_metadata_changes=counts, restored_AST_identical=True))
    return evidence
