"""Reuse the full retained-state/reward-transfer checks for source1601.

Only template source labels, expected update numbers and the old orientation
coefficient are adapted. The tested algorithm, saved tensors and native replay
comparisons stay identical to the prior qualified orientation-cost repair.
"""
import argparse
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import textwrap

import source1201_orientation40
import joint_pd_source1601_motion025_orientation80_runtime as repair
import qualify_source1201_orientation40 as template
from auto_train_mine import write_json
from joint_pd_continuation import sha

SCRIPT = Path(__file__).resolve()


def adapted():
    spec = importlib.util.spec_from_file_location('_qualified_source1601_orientation80', template.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.repair = repair
    evidence = []
    for name in ('qualify', 'frozen'):
        function = getattr(module, name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        original = ast.dump(tree)
        changes = []
        counts = {}

        def change(node, field, value, label):
            prior = copy.deepcopy(getattr(node, field))
            changes.append((node, field, prior))
            setattr(node, field, value)
            counts[label] = counts.get(label, 0) + 1

        class Adapter(ast.NodeVisitor):
            def visit_UnaryOp(self, node):
                if (isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant)
                        and node.operand.value == 20.):
                    change(node.operand, 'value', 40., 'old_negative_orientation')
                    return
                self.generic_visit(node)

            def visit_Constant(self, node):
                value = node.value
                if type(value) is int and value == 1400:
                    change(node, 'value', 1800, 'final_filename_iteration')
                elif type(value) is float and value == 20. and name == 'frozen':
                    change(node, 'value', 40., 'old_orientation_formula')
                elif isinstance(value, str):
                    revised = value.replace('source1201', 'source1601').replace('orientation40', 'motion025_orientation80')
                    revised = revised.replace('1401', '1801').replace('1400', '1800')
                    if revised != value:
                        change(node, 'value', revised, 'label_or_target_text')

            def visit_Call(self, node):
                expected = ast.parse('repair.previous.evaluator_tree()', mode='eval').body
                if ast.dump(node) == ast.dump(expected):
                    change(node, 'func', ast.parse('repair.evaluator_tree', mode='eval').body,
                           'same_original_evaluator_tree_entry')
                self.generic_visit(node)

        Adapter().visit(tree)
        if (counts.get('old_negative_orientation') != (1 if name == 'qualify' else 2)
                or (name == 'qualify' and (counts.get('final_filename_iteration') != 1
                    or counts.get('same_original_evaluator_tree_entry') != 1))
                or (name == 'frozen' and counts.get('old_orientation_formula') != 1)):
            raise ValueError('Requalify explicit adapter constant counts: ' + name + repr(counts))
        modified = copy.deepcopy(tree)
        for node, field, value in reversed(changes):
            setattr(node, field, value)
        if ast.dump(tree) != original:
            raise ValueError('Qualification changed beyond the declared labels/counters/reward source')
        exec(compile(ast.fix_missing_locations(modified), template.__file__, 'exec'), module.__dict__)
        evidence.append(dict(function=name, changes=counts, restored_AST_matches_original=True))
    return module, evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-previous', type=Path)
    args = parser.parse_args()
    repair.repair_ready()
    module, changes = adapted()
    if args.frozen_previous:
        result = module.frozen(args.frozen_previous, args.output)
        mode = 'frozen'
    else:
        if args.source is None:
            parser.error('--source is required for CPU qualification')
        result = module.qualify(args.source, args.output)
        mode = 'cpu'
    write_json(args.output / ('qualification_adapter_' + mode + '.json'), dict(
        adapter_sha256=sha(SCRIPT), original_template_sha256=sha(template.__file__),
        runtime_sha256=sha(repair.SCRIPT), adaptations=changes,
        qualified=bool(result['qualified']), source_sha256=repair.ANCHOR_SHA,
        actual_PPO_updates_executed=0, original_thresholds_changed=False,
        policy_accepted=False))
    print(json.dumps({key: value for key, value in result.items()
                     if key != 'source_data_sha256'}, ensure_ascii=False))
