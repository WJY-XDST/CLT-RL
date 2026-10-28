"""Qualify one reward-field transfer while retaining the original CPU checks."""
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
    spec = importlib.util.spec_from_file_location('_qualified_plain27_single_reward', runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    runtime.repair_ready()
    if (runtime.DELTA['rewards'] != {runtime.REWARD_KEY: [runtime.OLD_VALUE, runtime.REWARD_VALUE]}
            or any(runtime.DELTA[k] for k in ('commands', 'exploration', 'controller', 'physics'))):
        raise ValueError('Exactly one existing reward field may change')
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
        expected['rewards'][repair.REWARD_KEY] = repair.REWARD_VALUE
        validate_continuation_config(cfg, expected, expected['commands'],
            reward_overrides={'rewards.' + repair.REWARD_KEY: repair.REWARD_VALUE})
        if differences(cfg, expected) != ['/rewards/' + repair.REWARD_KEY]:
            raise ValueError('Roundtrip configuration changed outside the one reward field')
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
    if (differences(original_cfg, cfg) != ['/rewards/' + repair.REWARD_KEY]
            or not agent_unchanged or any(sha(p) != h for p, h in original_configs.items())):
        raise ValueError('Saved source configs or the roundtrip agent changed unexpectedly')
    argv, final, _ = repair.command(offline, cfg, 'offline_reward_descendant', 200,
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
        only_declared_roundtrip_reward_changed=True, first_profile_flag_exactly_once=True,
        ordinary_descendant_command_has_no_transfer_flag=True, original_source200_rejected=True,
        original_CPU_state_operations_preserved=True, metadata_adapter_restored_AST_identical=True,
        scope='CPU state/projection/CLI qualification with one explicit offline reward-field change; no PPO or simulation.')
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
    before, after = Path(previous) / 'replay', Path(output) / 'native_reward_transfer'
    read = lambda root, name: json.loads((root / name).read_text())
    a, b = read(before, 'manifest.json'), read(after, 'manifest.json')
    da, db = read(before, 'completed.json'), read(after, 'completed.json')
    ca, cb = read(before, 'resolved_env.json'), read(after, 'resolved_env.json')
    if (a['checkpoint_sha256'] != repair.ANCHOR_SHA or b['checkpoint_sha256'] != repair.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not da['completed'] or not db['completed']
            or da['samples'] != 27000 or db['samples'] != 27000
            or da['resets'] != db['resets']
            or differences(ca, cb) != ['/rewards/' + repair.REWARD_KEY]
            or ca['rewards'][repair.REWARD_KEY] != repair.OLD_VALUE
            or cb['rewards'][repair.REWARD_KEY] != repair.REWARD_VALUE):
        raise ValueError('Require identical frozen27 cases, source and single reward-field change')
    old, new = [np.genfromtxt(root / 'trace.csv', delimiter=',', names=True) for root in (before, after)]
    changed = {'reward_total', 'reward_track_ang_vel'}
    if (old.dtype.names != new.dtype.names or len(new) != 27000
            or any(not np.isfinite(new[k]).all() for k in new.dtype.names)
            or any(not np.array_equal(old[k], new[k]) for k in old.dtype.names if k not in changed)):
        raise ValueError('Frozen physics, actions or another reward changed')
    for rows, completion in ((old, da), (new, db)):
        if np.count_nonzero((rows['terminated'] != 0) | (rows['timeout'] != 0)) != completion['resets']:
            raise ValueError('Raw OR reset count differs')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as x, np.load(after / 'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[k], y[k]) for k in x.files):
            raise ValueError('Frozen raw observations or deterministic means changed')
    criteria = read(repair.ROOT / 'mine_model/config', 'tracking_acceptance.json')
    h = abs(new['height_m'] - new['height_cmd_m']) / (criteria['relative_error'] * abs(new['height_cmd_m']))
    angle = np.maximum(abs(new['pitch_rad']), abs(new['roll_rad'])) / np.deg2rad(criteria['body_p95_deg'])
    floor = cb['rewards']['tracking_task_quality_gate_floor']
    quality = floor + (1 - floor) / ((1 + np.maximum(h - .5, 0)**2) * (1 + np.maximum(angle - .5, 0)**2))
    expected = cb['rewards']['tracking_ang_vel'] * quality * np.exp(
        -(new['yaw_rate_rad_s'] - new['yaw_cmd_rad_s'])**2 / cb['rewards']['tracking_sigma_ang']) * b['policy_dt']
    formula_error = float(abs(expected - new['reward_track_ang_vel']).max())
    residual = abs(new['reward_total'] - old['reward_total'] - new['reward_track_ang_vel'] + old['reward_track_ang_vel'])
    tolerance = 8 * np.finfo(np.float32).eps * np.maximum.reduce(
        (np.ones(len(new)), abs(old['reward_total']), abs(new['reward_total'])))
    evidence = read(after, 'yaw_sigma010_runtime_evidence.json')
    if (formula_error > 2.e-7 or np.any(residual > tolerance)
            or evidence['profile'] != repair.profile() or evidence['authoritative_current_delta'] != repair.DELTA):
        raise ValueError('Independent angular formula, total reward delta or actual runtime differs')
    result = dict(qualified=True, source_sha256=repair.ANCHOR_SHA,
        only_reward_change=repair.DELTA['rewards'], full27_samples=27000,
        identical_source_resets_including_failures=da['resets'],
        physical_outputs_observations_actions_and_other_rewards_bitwise_equal=True,
        independent_angular_formula_max_error=formula_error, total_reward_delta_max_error=float(residual.max()),
        original_dense_yaw_quadratic_penalty_unchanged=True,
        height_posture_quality_rewards_exploration_and_execution_unchanged=True,
        source_data_sha256={str(root / name): sha(root / name) for root in (before, after)
            for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz')},
        implementation_sha256=sha(repair.SCRIPT), qualifier_sha256=sha(SCRIPT),
        PPO_updates_performed=0, policy_accepted=False,
        limitation='Frozen reward equivalence only; learning benefit remains unverified.')
    write_json(Path(output) / 'native_reward_transfer_qualification.json', result)
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
