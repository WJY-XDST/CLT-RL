"""Qualify a declared exploration-bound transfer with original retained-state checks.

The original load/save, all17 Adam, registered projections and inference
roundtrip operations remain intact. Only delta metadata and the one-time
profile flag are adapted; no optimizer step or native simulation is executed.
"""
import argparse
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path

import qualify_retained_plain27_source as template
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()


def qualify(source, runtime_path, decision_path, output):
    source, runtime_path, decision_path, output = [Path(p).resolve() for p in
        (source, runtime_path, decision_path, output)]
    if runtime_path.parent != ROOT / 'mine_model/experiments':
        raise ValueError('Require an owned source-bound native runtime')
    spec = importlib.util.spec_from_file_location('_declared_plain27_exploration_transfer', runtime_path)
    repair = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair)
    repair.repair_ready()
    if (decision_path != repair.DECISION or sha(source) != repair.ANCHOR_SHA
            or not repair.DELTA['exploration']
            or any(repair.DELTA[k] for k in ('commands', 'rewards', 'controller', 'physics'))):
        raise ValueError('Qualify one exact source-bound exploration-only revision')
    tree = ast.parse(inspect.getsource(template.qualify))
    before, edits = ast.dump(tree), []
    original_flag = ast.dump(ast.parse("any(arg.startswith('--repair_') for arg in argv)", mode='eval').body)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'empty'):
            if ast.dump(node.value) != ast.dump(ast.parse(
                    'dict(rewards={}, commands={}, exploration={}, controller={}, physics={})', mode='eval').body):
                raise ValueError('Recheck the original empty-delta declaration')
            edits.append((node, 'value', node.value))
            node.value = ast.parse('repair.DELTA', mode='eval').body
        elif isinstance(node, ast.Call) and ast.dump(node) == original_flag:
            edits.extend([(node, 'func', node.func), (node, 'args', node.args)])
            node.func = ast.Name(id='bool', ctx=ast.Load())
            node.args = [ast.parse("[arg for arg in argv if arg.startswith('--repair_')] != [runtime.FLAG]", mode='eval').body]
    if len(edits) != 3:
        raise ValueError('Requalify the exact delta and one-time profile declarations')
    modified = copy.deepcopy(tree)
    for node, field, old in reversed(edits):
        setattr(node, field, old)
    if ast.dump(tree) != before:
        raise ValueError('Original state, projection or roundtrip operations changed')
    scope = dict(template.__dict__, SCRIPT=SCRIPT, repair=repair)
    exec(compile(ast.fix_missing_locations(modified), template.__file__, 'exec'), scope)
    result = scope['qualify'](source, runtime_path, decision_path, output)
    offline = Path(result['untrained_roundtrip_checkpoint'])
    _, cfg = repair.require_source(offline)
    argv, final, _ = repair.command(offline, cfg, 'offline_descendant_exploration_transfer', 200,
        cfg['commands'], output / 'not_a_native_stop_request')
    actual, profile = result['source_actual_updates'], repair.profile()
    if (repair.FLAG in argv or final != actual + 199
            or argv[argv.index('--target_iteration') + 1] != str(actual + 200)
            or float(argv[argv.index('--min_exploration_std') + 1]) != profile['global_minimum_std']
            or float(argv[argv.index('--min_wheel_exploration_std') + 1]) != profile['minimum_wheel_std']):
        raise ValueError('Ordinary descendants must preserve clocks and avoid repeated transfer')
    try:
        repair.require_source(repair.base.SOURCE)
    except ValueError:
        pass
    else:
        raise ValueError('Originalmodel200 must remain rejected')
    result.update(original_source200_rejected=True, first_profile_flag_exactly_once=True,
        ordinary_descendant_command_has_no_transfer_flag=True,
        original_CPU_state_operations_preserved=True, metadata_adapter_restored_AST_identical=True,
        scope='CPU retained-state/projection/CLI qualification; no future candidate performance is implied')
    for path in (SCRIPT, Path(template.__file__), runtime_path):
        result['source_files'][str(path)] = sha(path)
    write_json(output / 'qualification.json', result)
    write_json(output / 'cpu_adapter_qualification.json', dict(qualified=True,
        script_sha256=sha(SCRIPT), template_sha256=sha(template.__file__),
        original_state_roundtrip_and_projection_operations_preserved=True,
        expected_delta=repair.DELTA, first_profile_flag_checked=True,
        ordinary_descendant_flag_checked=True, restored_AST_identical=True,
        actual_PPO_updates_executed=0, optimizer_steps_executed=0, policy_accepted=False))
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'runtime', 'decision', 'output'):
        cli.add_argument('--' + name, type=Path, required=True)
    args = cli.parse_args()
    result = qualify(args.source, args.runtime, args.decision, args.output)
    print(json.dumps(dict(qualified=result['qualified'], source_actual_updates=result['source_actual_updates'],
        expected_projected_fixture=result['expected_projected_fixture'],
        actual_PPO_updates=0, optimizer_steps=0, simulator_started=False)))
