"""Original full51/static readers bound to the single angular-weight trial."""
import argparse
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import re

import review_velocity010_full_height_and_retention as template
import review_unchanged_yaw025_full_height as waiter
import joint_pd_source3001_yaw150_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
STATIC_QUALIFICATION = ROOT / 'mine_model/results/joint_pd_yaw150_static_retention_qualification_20261019_v2.json'


def family():
    spec = importlib.util.spec_from_file_location('_yaw150_original_full51_and_static_readers', template.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCRIPT, module.STATIC_QUALIFICATION, module.runtime = SCRIPT, STATIC_QUALIFICATION, runtime
    tree = ast.parse(inspect.getsource(module.adapted))
    before, edits, counts = ast.dump(tree), [], {}

    def change(node, value, label):
        edits.append((node, copy.deepcopy(node.value)))
        node.value = value
        counts[label] = counts.get(label, 0) + 1

    class Adapter(ast.NodeVisitor):
        def visit_Constant(self, node):
            clocks = {2801: 3001, 3001: 3201, 60020: 64020}
            if type(node.value) is int and node.value in clocks:
                change(node, clocks[node.value], 'source_or_target_clock_metadata')
            elif type(node.value) is float and node.value in (.025, .01):
                change(node, 1. if node.value == .025 else 1.5, 'single_reward_coefficient_metadata')
            elif isinstance(node.value, str):
                replacements = {'source2801': 'source3001', 'source3001': 'source3201',
                    'candidate3001': 'candidate3201',
                    'velocity010': 'yaw150', 'tracking_sigma': 'tracking_ang_vel',
                    'one-width': 'one-coefficient', 'single-width': 'single-coefficient'}
                value = re.sub('|'.join(re.escape(key) for key in replacements),
                    lambda match: replacements[match.group()], node.value)
                if value != node.value:
                    change(node, value, 'source_reward_or_evidence_label_metadata')

    Adapter().visit(tree)
    if counts.get('single_reward_coefficient_metadata') != 4:
        raise ValueError('Requalify the exact four reward-value metadata constants')
    changed = copy.deepcopy(tree)
    for node, value in reversed(edits):
        node.value = value
    if ast.dump(tree) != before:
        raise ValueError('Original native selectors, metric readers or acceptance predicates changed')
    exec(compile(ast.fix_missing_locations(changed), template.__file__, 'exec'), module.__dict__)
    module.outer_adapter_evidence = dict(function='adapted', metadata_changes=counts,
        restored_AST_identical=True, original_template_sha256=sha(template.__file__))
    return module


def adapted(launch, qualification):
    retained = family()
    module, launch, evidence = retained.adapted(launch, qualification)
    return module, launch, [*evidence, retained.outer_adapter_evidence]


def retention():
    return family().retention()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-launch', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--qualify-checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--static-retention-output', type=Path)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    module, launch, evidence = adapted(args.training_launch, args.qualification)
    adapter = args.qualification.with_name(args.qualification.stem + '_adapter.json')
    if args.qualify_checkpoint:
        result = module.qualify(args.qualify_checkpoint, args.qualification)
        static = retention().qualify(args.qualify_checkpoint, module.EXPECTED_TRAINING, STATIC_QUALIFICATION)
        write_json(adapter, dict(qualified=result['qualified'] and static['qualified'],
            entry_sha256=sha(SCRIPT), launch_sha256=sha(args.training_launch),
            runtime_sha256=sha(runtime.SCRIPT), template_sha256=sha(template.__file__),
            adaptations=evidence, original_thresholds_changed=False,
            native_simulations_started=0, PPO_updates_performed=0, policy_accepted=False))
    else:
        proof = json.loads(adapter.read_text())
        if (not proof['qualified'] or proof['entry_sha256'] != sha(SCRIPT)
                or proof['launch_sha256'] != sha(args.training_launch)
                or proof['runtime_sha256'] != sha(runtime.SCRIPT)
                or proof['template_sha256'] != sha(template.__file__)
                or proof['original_thresholds_changed']):
            raise ValueError('Keep the exact qualified reader, launch, profile and original gates')
        if args.output is None or args.static_retention_output is None:
            parser.error('Require the full51 and independent static output paths')
        if args.wait:
            waiter.wait_completed(module, args.output.with_name(args.output.name + '_waiter.json'))
        module.run(args.output)
        result = retention().run(module.EXPECTED_TRAINING, args.output, args.static_retention_output)
    print(json.dumps(dict(qualified=result.get('qualified'),
        static_retention_verified=result.get('static_retention_verified'), policy_accepted=False)))
