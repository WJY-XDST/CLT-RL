"""One reviewed angular-width repair through the intact200-update controller."""
import argparse
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path

import source1201_orientation40
import train_source2001_quality50_200 as template
import joint_pd_source2201_yaw025_runtime as repair
import qualify_source2201_yaw025 as qualifier
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
QUALIFICATION = ROOT / 'mine_model/results/joint_pd_source2201_yaw025_qualification_20261018'


def adapted():
    spec = importlib.util.spec_from_file_location('_source2201_yaw025_controller', template.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCRIPT, module.QUALIFICATION = SCRIPT, QUALIFICATION
    module.repair, module.qualifier = repair, qualifier
    evidence = []
    for name in ('qualifications', 'prepare', 'review', 'run'):
        tree = ast.parse(inspect.getsource(getattr(module, name)))
        original = ast.dump(tree)
        changes, counts = [], {}

        def change(node, field, value, label):
            changes.append((node, field, copy.deepcopy(getattr(node, field))))
            setattr(node, field, value)
            counts[label] = counts.get(label, 0) + 1

        class Adapter(ast.NodeVisitor):
            def visit_List(self, node):
                if (len(node.elts) == 2 and isinstance(node.elts[0], ast.Constant)
                        and node.elts[0].value == 1.
                        and isinstance(node.elts[1], ast.Attribute)
                        and node.elts[1].attr == 'QUALITY_FLOOR'):
                    change(node.elts[0], 'value', .25, 'old_reward_metadata')
                self.generic_visit(node)

            def visit_Attribute(self, node):
                if isinstance(node.value, ast.Name) and node.value.id == 'repair' and node.attr == 'QUALITY_FLOOR':
                    change(node, 'attr', 'REWARD_VALUE', 'new_reward_value_attribute')
                self.generic_visit(node)

            def visit_Constant(self, node):
                if isinstance(node.value, str):
                    value = node.value.replace('plain_quality50_', 'plain_yaw025_').replace(
                        'motion025_height3_quality50', 'motion025_height3_quality50_yaw025')
                    if value != node.value:
                        change(node, 'value', value, 'run_label_only')

        Adapter().visit(tree)
        if name in ('qualifications', 'prepare', 'review') and counts.get('old_reward_metadata') != 1:
            raise ValueError('Requalify the exact controller reward metadata adaptation')
        modified = copy.deepcopy(tree)
        for node, field, value in reversed(changes):
            setattr(node, field, value)
        if ast.dump(tree) != original:
            raise ValueError('Source selection, training, metrics or acceptance gates changed')
        exec(compile(ast.fix_missing_locations(modified), template.__file__, 'exec'), module.__dict__)
        evidence.append(dict(function=name, adaptations=counts, restored_AST_identical=True))
    module.adapter_evidence = dict(script_sha256=sha(SCRIPT), original_template_sha256=sha(template.__file__),
        changes=evidence, completed200_full27_full17_source_guards_preserved=True,
        original_acceptance_and_review_predicates_preserved=True)
    original_prepare = module.prepare

    def prepare_with_exploration_evidence(output, previous):
        addendum = ROOT / 'mine_model/results/joint_pd_source2201_yaw025_exploration_addendum_20261018.json'
        evidence = json.loads(addendum.read_text())
        if (evidence['source_checkpoint_sha256'] != repair.ANCHOR_SHA
                or not evidence['reward_and_exploration_changes_not_confounded']
                or evidence['acceptance_changed']
                or any(sha(p) != h for p, h in evidence['evidence'].items())):
            raise ValueError('Preserve the measured effective-exploration review')
        plan, cfg = original_prepare(output, previous)
        plan['source_files'].update(evidence['evidence'])
        plan['source_files'][str(addendum)] = sha(addendum)
        plan['effective_exploration_review'] = str(addendum)
        plan['effective_exploration_review_required_per_block'] = True
        module.write_json(Path(output) / 'experiment_plan.json', plan)
        return plan, cfg

    module.prepare = prepare_with_exploration_evidence
    return module


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    module = adapted()
    if args.prepare_only:
        module.prepare(args.output, args.previous)
    else:
        module.run(args.output, args.previous, args.unit)
