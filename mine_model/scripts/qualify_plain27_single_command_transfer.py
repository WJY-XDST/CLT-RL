"""Qualify one sampler-field transfer while retaining the original CPU checks."""
import argparse
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import yaml

import qualify_retained_plain27_source as template
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, validate_continuation_config
from qualify_source801_drift50_transfer import differences

SCRIPT = Path(__file__).resolve()


def load(runtime_path):
    runtime_path = Path(runtime_path).resolve()
    if runtime_path.parent != ROOT / 'mine_model/experiments':
        raise ValueError('Require an owned runtime')
    spec = importlib.util.spec_from_file_location('_qualified_plain27_single_command', runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.repair_ready()
    if (runtime.DELTA['commands'] != {runtime.COMMAND_KEY: [runtime.OLD_VALUE, runtime.COMMAND_VALUE]}
            or any(runtime.DELTA[k] for k in ('rewards', 'exploration', 'controller', 'physics'))):
        raise ValueError('Exactly one existing sampler field may change')
    return runtime


def qualify(source, runtime_path, decision_path, output):
    source, runtime_path, decision_path, output = map(lambda p: Path(p).resolve(),
        (source, runtime_path, decision_path, output))
    repair = load(runtime_path)
    if decision_path != repair.DECISION or sha(source) != repair.ANCHOR_SHA:
        raise ValueError('Require the exact reviewed source and decision')
    original_configs = {str(p): sha(p) for p in (source.parent / 'params/env.yaml',
        source.parent / 'params/agent.yaml')}

    def configure_roundtrip(roundtrip, cfg):
        expected = copy.deepcopy(cfg)
        expected['commands'][repair.COMMAND_KEY] = repair.COMMAND_VALUE
        validate_continuation_config(cfg, expected, expected['commands'])
        if differences(cfg, expected) != ['/commands/' + repair.COMMAND_KEY]:
            raise ValueError('Roundtrip configuration changed outside the one sampler field')
        (Path(roundtrip) / 'params/env.yaml').write_text(yaml.safe_dump(expected, sort_keys=False))

    tree = ast.parse(inspect.getsource(template.qualify))
    before, edits = ast.dump(tree), []

    def change(node, field, value):
        edits.append((node, field, getattr(node, field)))
        setattr(node, field, value)

    original_flag = ast.dump(ast.parse("any(arg.startswith('--repair_') for arg in argv)", mode='eval').body)
    copy_call = ast.dump(ast.parse("shutil.copytree(source.parent / 'params', roundtrip / 'params')").body[0])
    copied = False
    for node in list(ast.walk(tree)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'empty':
            expected_empty = ast.parse('dict(rewards={}, commands={}, exploration={}, controller={}, physics={})', mode='eval').body
            if ast.dump(node.value) != ast.dump(expected_empty):
                raise ValueError('Recheck the original delta declaration')
            change(node, 'value', ast.parse('repair.DELTA', mode='eval').body)
        elif isinstance(node, ast.Call) and ast.dump(node) == original_flag:
            change(node, 'func', ast.Name(id='bool', ctx=ast.Load()))
            change(node, 'args', [ast.parse("[a for a in argv if a.startswith('--repair_')] != [runtime.FLAG]", mode='eval').body])
        elif isinstance(node, ast.keyword) and node.arg == 'saved_env_and_agent_configurations_unchanged':
            change(node, 'arg', 'saved_source_env_and_agent_configurations_unchanged')
        for field, value in list(ast.iter_fields(node)):
            if isinstance(value, list):
                indices = [i for i, item in enumerate(value) if isinstance(item, ast.Expr) and ast.dump(item) == copy_call]
                if indices:
                    if copied or len(indices) != 1:
                        raise ValueError('Recheck the original roundtrip copy operation')
                    revised = list(value)
                    revised.insert(indices[0] + 1, ast.parse('configure_roundtrip(roundtrip, cfg)').body[0])
                    change(node, field, revised)
                    copied = True
    if len(edits) != 5 or not copied:
        raise ValueError('Requalify delta, flag, source-config label and explicit roundtrip configuration insertion')
    changed = copy.deepcopy(tree)
    for node, field, old in reversed(edits):
        setattr(node, field, old)
    if ast.dump(tree) != before:
        raise ValueError('Original CPU state, Adam, projection or inference operations changed')
    scope = dict(template.__dict__, SCRIPT=SCRIPT, repair=repair, configure_roundtrip=configure_roundtrip)
    exec(compile(ast.fix_missing_locations(changed), template.__file__, 'exec'), scope)
    result = scope['qualify'](source, runtime_path, decision_path, output)
    offline = Path(result['untrained_roundtrip_checkpoint'])
    _, cfg = repair.require_source(offline)
    # Native IsaacLab YAML contains tagged tuples. Use the same validated,
    # normalized checkpoint reader as native continuation, without loading
    # arbitrary YAML constructors or changing the saved source file.
    _, original_cfg = repair.require_source(source)
    agent_unchanged = sha(source.parent / 'params/agent.yaml') == sha(offline.parent / 'params/agent.yaml')
    if (differences(original_cfg, cfg) != ['/commands/' + repair.COMMAND_KEY]
            or not agent_unchanged or any(sha(p) != h for p, h in original_configs.items())):
        raise ValueError('Saved source configs or the roundtrip agent changed unexpectedly')
    argv, final, _ = repair.command(offline, cfg, 'offline_sampler_descendant', 200,
        cfg['commands'], output / 'not_a_native_stop_request')
    if repair.FLAG in argv or final != result['source_actual_updates'] + 199:
        raise ValueError('The descendant repeated the transfer or altered update clocks')
    try:
        repair.require_source(repair.base.SOURCE)
    except ValueError:
        pass
    else:
        raise ValueError('Originalmodel200 must remain rejected')
    result.update(saved_source_env_and_agent_configurations_unchanged=True,
        saved_agent_configuration_unchanged=True, saved_roundtrip_env_diff=differences(original_cfg, cfg),
        only_declared_roundtrip_sampler_changed=True, first_profile_flag_exactly_once=True,
        ordinary_descendant_command_has_no_transfer_flag=True, original_source200_rejected=True,
        original_CPU_state_operations_preserved=True, metadata_adapter_restored_AST_identical=True,
        scope='CPU state/projection/CLI qualification with one explicit offline sampler-field change; no PPO or simulation.')
    for p in (SCRIPT, Path(template.__file__), runtime_path):
        result['source_files'][str(p)] = sha(p)
    write_json(output / 'qualification.json', result)
    write_json(output / 'cpu_adapter_qualification.json', dict(qualified=True,
        script_sha256=sha(SCRIPT), template_sha256=sha(template.__file__),
        original_state_roundtrip_and_projection_operations_preserved=True,
        expected_delta=repair.DELTA, restored_AST_identical=True,
        declared_roundtrip_configuration_insertion=True,
        actual_PPO_updates_executed=0, optimizer_steps_executed=0, policy_accepted=False))
    return result


def frozen(previous, output, runtime_path):
    repair = load(runtime_path)
    import qualify_source3601_motion0375 as original
    import plain27_sampler_metadata_adapter as adapter
    module = adapter.private_module(original.__file__, '_motion075_original_frozen_comparator')
    module.runtime, module.SCRIPT = repair, SCRIPT
    evidence = adapter.retarget(module, ('frozen',), integers={'*': {15: 0}},
        strings={'source15': 'source0'})
    result = module.frozen(previous, output)
    result.update(metadata_adapter_evidence=evidence,
        original_frozen_all_rewards_and_physics_bitwise_comparator_preserved=True,
        qualifier_sha256=sha(SCRIPT))
    result['source_data_sha256'].update({str(p): sha(p) for p in
        (SCRIPT, Path(original.__file__), adapter.SCRIPT)})
    write_json(Path(output) / 'native_command_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path)
    cli.add_argument('--decision', type=Path)
    cli.add_argument('--runtime', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--frozen-previous', type=Path)
    args = cli.parse_args()
    result = (frozen(args.frozen_previous, args.output, args.runtime) if args.frozen_previous
        else qualify(args.source, args.runtime, args.decision, args.output))
    print(json.dumps(dict(qualified=result['qualified'], PPO_updates=0, policy_accepted=False)))
