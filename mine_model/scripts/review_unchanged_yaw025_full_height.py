"""Review one unchanged yaw025200-update continuation and static retention.

Reuse the complete original27/51/static readers. Bind source/target clocks to
the actual training launch; never repeat PPO or a retained native baseline.
"""
import argparse
import ast
import copy
from datetime import datetime
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import time

import source1201_orientation40
import joint_pd_source2201_yaw025_runtime as runtime
import review_motion025_orientation80_full_height as template
import validate_yaw025_static_retention10 as retention
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'


def adapted(launch_path, qualification):
    launch_path, qualification = Path(launch_path).resolve(), Path(qualification).resolve()
    launch = json.loads(launch_path.read_text())
    decision = json.loads(Path(launch['decision']).read_text())
    if (launch['target_actual_updates'] != launch['source_actual_updates'] + 200
            or launch['target_all17_Adam_step'] != 20 * launch['target_actual_updates']
            or launch['next_configuration_changes'] != dict(rewards={}, commands={}, exploration={}, controller={}, physics={})
            or launch['controller_sha256'] != sha(launch['controller'])
            or launch['decision_sha256'] != sha(launch['decision'])):
        raise ValueError('Use the exact unchanged-recipe200-update continuation launch')
    baseline = Path(decision['full51_output']) / 'candidate'
    spec = importlib.util.spec_from_file_location('_unchanged_yaw025_full51_reader', template.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SCRIPT, module.QUALIFICATION, module.BASELINE = SCRIPT, qualification, baseline
    module.EXPECTED_TRAINING, module.EXPECTED_UNIT = Path(launch['output']), launch['unit']

    def dependencies():
        result = list(template.dependencies())
        result[2] = runtime
        return tuple(result)

    def baseline_readonly(api):
        original, corrected, _, _, _, digest, _ = api
        path = baseline / 'motion025_full_height_diagnosis.json'
        report = json.loads(path.read_text())
        source = Path(report['checkpoint'])
        if report['checkpoint_sha256'] != decision['source_checkpoint_sha256'] or digest(source) != report['checkpoint_sha256']:
            raise ValueError('Baseline does not belong to the reviewed continuation source')
        _, cfg = runtime.require_source(source)
        pins = {**report['source_files'], str(path): digest(path),
            str(original.SCRIPT): digest(original.SCRIPT), str(corrected.SCRIPT): digest(corrected.SCRIPT)}
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('Retained native baseline changed')
        reader, writes = corrected.corrected_reader(), []
        reader.__globals__['write_json'] = lambda p, value: writes.append((Path(p), value))
        rebuilt = reader(baseline, source)
        if len(writes) != 1 or writes[0][0] != path or writes[0][1] != rebuilt or rebuilt != report:
            raise ValueError('Original baseline reader no longer reproduces every retained trial')
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError('Read-only baseline reconstruction changed evidence')
        return source, cfg, report, pins

    module.dependencies, module.baseline_readonly = dependencies, baseline_readonly
    evidence = []
    for name in ('qualify', 'run'):
        tree = ast.parse(inspect.getsource(getattr(module, name)))
        before, changes, counts = ast.dump(tree), [], {}

        def change(node, field, value, label):
            changes.append((node, field, copy.deepcopy(getattr(node, field))))
            setattr(node, field, value)
            counts[label] = counts.get(label, 0) + 1

        class Adapter(ast.NodeVisitor):
            def visit_UnaryOp(self, node):
                if (name == 'run' and isinstance(node.op, ast.USub)
                        and isinstance(node.operand, ast.Constant)
                        and type(node.operand.value) is float and node.operand.value in (40., 80.)):
                    change(node, 'op', ast.UAdd(), 'unchanged_reward_sign_metadata')
                    change(node.operand, 'value', .025, 'unchanged_reward_value_metadata')
                    return
                self.generic_visit(node)

            def visit_keyword(self, node):
                self.generic_visit(node)
                if name == 'run' and node.arg == 'only_saved_configuration_change':
                    # Retain this AST object so reversing the parent replacement
                    # reconnects its original children before reversing them.
                    changes.append((node, 'value', node.value))
                    node.value = ast.Dict(keys=[], values=[])
                    label = 'empty_configuration_delta_metadata'
                    counts[label] = counts.get(label, 0) + 1

            def visit_Constant(self, node):
                if type(node.value) is int and node.value in (1801, 36020):
                    change(node, 'value', launch['target_actual_updates'] if node.value == 1801
                        else launch['target_all17_Adam_step'], 'target_clock_metadata')
                elif isinstance(node.value, str):
                    value = node.value.replace('orientation80', 'yaw025_unchanged').replace(
                        'source1801', 'source' + str(launch['target_actual_updates'])).replace(
                        'source1601', 'source' + str(launch['source_actual_updates'])).replace(
                        'candidate1801', 'candidate' + str(launch['target_actual_updates']))
                    if value in ('orientation', 'rewards.orientation'):
                        value = value.replace('orientation', 'tracking_sigma_ang')
                    if value != node.value:
                        change(node, 'value', value, 'reward_key_or_source_label_metadata')

        Adapter().visit(tree)
        if name == 'run' and (counts.get('unchanged_reward_sign_metadata') != 6
                or counts.get('unchanged_reward_value_metadata') != 6
                or counts.get('empty_configuration_delta_metadata') != 1):
            raise ValueError('Requalify exact unchanged-recipe comparison metadata')
        modified = copy.deepcopy(tree)
        for node, field, value in reversed(changes):
            setattr(node, field, value)
        if ast.dump(tree) != before:
            raise ValueError('Original source guards or metric/acceptance predicates changed')
        exec(compile(ast.fix_missing_locations(modified), template.__file__, 'exec'), module.__dict__)
        evidence.append(dict(function=name, metadata_changes=counts, restored_AST_identical=True))
    return module, launch, evidence


def wait_completed(module, status_path):
    while True:
        try:
            state = json.loads((module.EXPECTED_TRAINING / 'status.json').read_text())
            ready = (state['state'] == 'agent_review_required' and not state['training_active']
                and all((module.EXPECTED_TRAINING / p).is_file() for p in (
                    'stationary_review.json', 'native_learning_signal/completed_external.json',
                    'replay/assessment.json', 'replay/completed.json')))
        except (FileNotFoundError, json.JSONDecodeError):
            ready = False
        if ready:
            return
        raw = subprocess.check_output(['systemctl', '--user', 'show', module.EXPECTED_UNIT,
            '--property=LoadState,MainPID,ActiveState,SubState,ExecMainStatus'], text=True)
        owner = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
        report = dict(observed_at=datetime.now().astimezone().isoformat(),
            state='waiting_completed200_full27', exact_unit=module.EXPECTED_UNIT,
            output=str(module.EXPECTED_TRAINING), owner=owner,
            training_restarted=False, policy_accepted=False)
        write_json(status_path, report)
        if (owner['ActiveState'] not in ('active', 'activating', 'deactivating', 'reloading')
                or (owner['SubState'] == 'exited' and owner['MainPID'] == '0')):
            raise ValueError('Exact owner ended without complete evidence; inspect without repeating PPO')
        time.sleep(30.)


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
        write_json(adapter, dict(qualified=result['qualified'], entry_sha256=sha(SCRIPT),
            launch_sha256=sha(args.training_launch), template_sha256=sha(template.__file__),
            runtime_sha256=sha(runtime.SCRIPT), metadata_adaptations=evidence,
            original_source_guards_thresholds_and_metrics_preserved=True,
            qualification_is_not_future_model_acceptance=True, PPO_updates_performed=0))
    else:
        proof = json.loads(adapter.read_text())
        if (not proof['qualified'] or proof['entry_sha256'] != sha(SCRIPT)
                or proof['launch_sha256'] != sha(args.training_launch)
                or proof['template_sha256'] != sha(template.__file__)
                or proof['runtime_sha256'] != sha(runtime.SCRIPT)):
            raise ValueError('Requalify changed reader or launch bindings')
        if args.output is None:
            parser.error('--output is required')
        if args.wait:
            wait_completed(module, args.output.with_name(args.output.name + '_waiter.json'))
        result = module.run(args.output)
        if args.static_retention_output:
            result = retention.adapted().run(module.EXPECTED_TRAINING, args.output,
                                              args.static_retention_output)
    print(json.dumps(dict(qualified=result.get('qualified'), output=str(args.output),
        static_retention_verified=result.get('static_retention_verified'), policy_accepted=False), ensure_ascii=False))
