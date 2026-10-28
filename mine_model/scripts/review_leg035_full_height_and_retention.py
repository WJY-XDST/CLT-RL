"""Unchanged27/51/static readers for the single leg035 exploration trial."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path

import review_unchanged_yaw025_full_height as template
import joint_pd_source2601_leg035_runtime as runtime
import validate_yaw025_static_retention10 as retention_template
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
STATIC_QUALIFICATION = ROOT / 'mine_model/results/joint_pd_leg035_static_retention_qualification_20261019.json'


def adapted(launch_path, qualification):
    tree = ast.parse(inspect.getsource(template.adapted))
    before, edits, labels = ast.dump(tree), [], 0
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == 'exploration' and isinstance(node.value, ast.Dict) and not node.value.keys:
            edits.append((node, 'value', node.value))
            node.value = ast.parse("{'minimum_leg_std':[.05,.035]}", mode='eval').body
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and 'yaw025_unchanged' in node.value:
            edits.append((node, 'value', node.value))
            node.value = node.value.replace('yaw025_unchanged', 'leg035')
            labels += 1
    if sum(field == 'value' and isinstance(value, ast.Dict) for _,field,value in edits) != 1 or labels < 1:
        raise ValueError('Requalify the launch-delta guard and output labels')
    changed = copy.deepcopy(tree)
    for node, field, value in reversed(edits):
        setattr(node, field, value)
    if ast.dump(tree) != before:
        raise ValueError('Original metric, continuation-source or acceptance guards changed')
    scope = dict(template.__dict__, SCRIPT=SCRIPT, runtime=runtime)
    exec(compile(ast.fix_missing_locations(changed), str(template.SCRIPT), 'exec'), scope)
    module, launch, original_evidence = scope['adapted'](launch_path, qualification)
    return module, launch, dict(original_evidence=original_evidence,
        parent_reader_sha256=sha(template.SCRIPT),
        launch_exploration_delta_guard={'minimum_leg_std':[.05,.035]},
        restored_parent_AST_identical=True, labels_adapted=labels,
        original_numeric_metrics_and_acceptance_unchanged=True)


def retention():
    module = retention_template.adapted()
    module.runtime, module.SCRIPT, module.QUALIFICATION = runtime, SCRIPT, STATIC_QUALIFICATION
    return module


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
        retained = retention().qualify(args.qualify_checkpoint, module.EXPECTED_TRAINING, STATIC_QUALIFICATION)
        write_json(adapter, dict(qualified=result['qualified'] and retained['qualified'],
            entry_sha256=sha(SCRIPT), launch_sha256=sha(args.training_launch),
            runtime_sha256=sha(runtime.SCRIPT), template_sha256=sha(template.SCRIPT),
            adaptations=evidence, PPO_updates_performed=0, native_simulations_started=0,
            original_thresholds_changed=False, policy_accepted=False))
    else:
        proof = json.loads(adapter.read_text())
        if (not proof['qualified'] or proof['entry_sha256'] != sha(SCRIPT)
                or proof['launch_sha256'] != sha(args.training_launch)
                or proof['runtime_sha256'] != sha(runtime.SCRIPT)
                or proof['template_sha256'] != sha(template.SCRIPT)
                or proof['original_thresholds_changed']):
            raise ValueError('Preserve the exact qualified reader, native launch and unchanged gates')
        if args.output is None or args.static_retention_output is None:
            parser.error('Require full51 and independent static output paths')
        if args.wait:
            template.wait_completed(module, args.output.with_name(args.output.name + '_waiter.json'))
        result = module.run(args.output)
        result = retention().run(module.EXPECTED_TRAINING, args.output, args.static_retention_output)
    print(json.dumps(dict(qualified=result.get('qualified'),
        static_retention_verified=result.get('static_retention_verified'),
        policy_accepted=False)), flush=True)
