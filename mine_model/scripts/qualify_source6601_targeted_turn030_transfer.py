"""Retained-state and frozen-replay qualification for one sampler field."""
import argparse
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, validate_continuation_config
import qualify_retained_plain27_source as template
import joint_pd_source6601_targeted_turn030_runtime as runtime

SCRIPT = Path(__file__).resolve()
OUTPUT = ROOT / 'mine_model/results/joint_pd_source6601_targeted_turn030_qualification_v3_20261021'


def single_added_field(before, after):
    key = runtime.sampler.CONFIG_KEY
    expected = copy.deepcopy(before)
    if key in expected['commands']:
        raise ValueError('The source must predate the new sampling field')
    expected['commands'][key] = runtime.FRACTION
    if expected != after:
        raise ValueError('Configuration differs outside the declared added field')
    return ['/commands/' + key]


def qualify():
    runtime.repair_ready()
    source, _, original_cfg = runtime.anchor()
    key = runtime.sampler.CONFIG_KEY

    def configure_roundtrip(roundtrip, cfg):
        expected = copy.deepcopy(cfg)
        expected['commands'][key] = runtime.FRACTION
        validate_continuation_config(cfg, expected, expected['commands'])
        if single_added_field(cfg, expected) != ['/commands/' + key]:
            raise ValueError('Only the declared training sampler field may change')
        (Path(roundtrip) / 'params/env.yaml').write_text(yaml.safe_dump(expected, sort_keys=False))

    # Keep the original real retained-runner load, all17 Adam comparisons,
    # registered projection fixture, save/reload and inference operations.
    tree = ast.parse(inspect.getsource(template.qualify))
    before, edits = ast.dump(tree), []

    def change(node, field, value):
        edits.append((node, field, getattr(node, field)))
        setattr(node, field, value)

    flag = ast.dump(ast.parse("any(arg.startswith('--repair_') for arg in argv)", mode='eval').body)
    copy_call = ast.dump(ast.parse("shutil.copytree(source.parent / 'params', roundtrip / 'params')").body[0])
    copied = False
    for node in list(ast.walk(tree)):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'empty':
            if ast.dump(node.value) != ast.dump(ast.parse(
                    'dict(rewards={}, commands={}, exploration={}, controller={}, physics={})', mode='eval').body):
                raise ValueError('Requalify the original delta declaration')
            change(node, 'value', ast.parse('repair.DELTA', mode='eval').body)
        elif isinstance(node, ast.Call) and ast.dump(node) == flag:
            change(node, 'func', ast.Name(id='bool', ctx=ast.Load()))
            change(node, 'args', [ast.parse(
                "[a for a in argv if a.startswith('--repair_')] != [runtime.FLAG]", mode='eval').body])
        elif isinstance(node, ast.keyword) and node.arg == 'saved_env_and_agent_configurations_unchanged':
            change(node, 'arg', 'saved_source_env_and_agent_configurations_unchanged')
        for field, value in list(ast.iter_fields(node)):
            if isinstance(value, list):
                indices = [i for i, item in enumerate(value)
                           if isinstance(item, ast.Expr) and ast.dump(item) == copy_call]
                if indices:
                    if copied or len(indices) != 1:
                        raise ValueError('Requalify the roundtrip config insertion')
                    revised = list(value)
                    revised.insert(indices[0] + 1, ast.parse('configure_roundtrip(roundtrip, cfg)').body[0])
                    change(node, field, revised)
                    copied = True
    if len(edits) != 5 or not copied:
        raise ValueError('Original numeric CPU operations must remain intact')
    modified = copy.deepcopy(tree)
    for node, field, value in reversed(edits):
        setattr(node, field, value)
    if ast.dump(tree) != before:
        raise ValueError('Unexpected change to original retained-state qualification')
    scope = dict(template.__dict__, SCRIPT=SCRIPT, repair=runtime, configure_roundtrip=configure_roundtrip)
    exec(compile(ast.fix_missing_locations(modified), str(template.SCRIPT), 'exec'), scope)
    result = scope['qualify'](source, runtime.SCRIPT, runtime.DECISION, OUTPUT)
    offline = Path(result['untrained_roundtrip_checkpoint'])
    _, cfg = runtime.require_source(offline)
    diff = single_added_field(original_cfg, cfg)
    if diff != ['/commands/' + key] or sha(source.parent / 'params/agent.yaml') != sha(offline.parent / 'params/agent.yaml'):
        raise ValueError('Offline descendant changed outside the one command field')
    argv, final, _ = runtime.command(offline, cfg, 'offline_sampler_descendant', 200,
                                    cfg['commands'], OUTPUT / 'not_a_native_stop_request')
    if runtime.FLAG in argv or final != 6800:
        raise ValueError('Descendant must not repeat transfer or reset clocks')

    # Saved unknown fields remain strict. Only this declared training field
    # may be restored before construction; this also covers descendant replay.
    runtime.install_sampling_field_restore()
    from wheel_legged_gym_isaaclab.config_overrides import apply_config_overrides
    config = SimpleNamespace(commands=SimpleNamespace())
    apply_config_overrides(config, {'commands.' + key: .3})
    if getattr(config.commands, key) != .3:
        raise ValueError('Known saved field did not restore')
    rejected = 0
    for overrides in ({'commands.' + key: .4}, {'commands.unknown_sampler_field': .3}):
        try:
            apply_config_overrides(config, overrides)
        except ValueError:
            rejected += 1
    if rejected != 2:
        raise ValueError('Unknown saved fields or unqualified fractions were accepted')
    result.update(saved_source_env_and_agent_configurations_unchanged=True,
        saved_agent_configuration_unchanged=True, saved_roundtrip_env_diff=diff,
        only_declared_roundtrip_command_field_changed=True,
        original_CPU_numeric_state_projection_and_inference_operations_preserved=True,
        metadata_adapter_restored_AST_identical=True, ordinary_descendant_has_no_transfer_flag=True,
        strict_known_training_field_restore_qualified=True,
        original_model200_reload=False, native_transfer_qualified=False,
        scope='CPU retained-state qualification only; no PPO or simulation')
    for path in (SCRIPT, template.SCRIPT, runtime.SCRIPT, Path(runtime.sampler.__file__)):
        result['source_files'][str(path)] = sha(path)
    write_json(OUTPUT / 'qualification.json', result)
    return result


def frozen():
    runtime.repair_ready()
    before, after = runtime.TRAINING / 'replay', OUTPUT / 'native_sampler_transfer'
    read = lambda root, name: json.loads((root / name).read_text())
    a, b = [read(root, 'manifest.json') for root in (before, after)]
    da, db = [read(root, 'completed.json') for root in (before, after)]
    ca, cb = [read(root, 'resolved_env.json') for root in (before, after)]
    if (a['checkpoint_sha256'] != runtime.ANCHOR_SHA or b['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not da['completed'] or not db['completed']
            or da['samples'] != 27000 or db['samples'] != 27000 or da['resets'] != db['resets']
            or single_added_field(ca, cb) != ['/commands/' + runtime.sampler.CONFIG_KEY]
            or cb['commands'][runtime.sampler.CONFIG_KEY] != .3):
        raise ValueError('Frozen source27 must differ in the one declared sampler field only')
    old, new = [np.genfromtxt(root / 'trace.csv', delimiter=',', names=True) for root in (before, after)]
    if (old.dtype.names != new.dtype.names or len(old) != 27000 or len(new) != 27000
            or any(not np.isfinite(new[key]).all() for key in new.dtype.names)
            or any(not np.array_equal(old[key], new[key]) for key in old.dtype.names)):
        raise ValueError('Frozen physical, reward or action rows changed')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as x, np.load(after / 'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[key], y[key]) for key in x.files):
            raise ValueError('Frozen policy inputs or deterministic outputs changed')
    for rows, complete in ((old, da), (new, db)):
        if np.count_nonzero((rows['terminated'] != 0) | (rows['timeout'] != 0)) != complete['resets']:
            raise ValueError('Raw OR reset count changed')
    evidence = read(after, 'targeted_turn_runtime_evidence.json')
    if evidence['profile'] != runtime.profile() or evidence['authoritative_current_delta'] != runtime.DELTA:
        raise ValueError('Bound runtime evidence differs')
    result = dict(qualified=True, source_sha256=runtime.ANCHOR_SHA,
        only_reward_change={}, only_command_change=runtime.DELTA['commands'],
        full27_samples=27000, identical_source_resets_including_failures=da['resets'],
        physical_outputs_observations_actions_and_all_rewards_bitwise_equal=True,
        physical_outputs_observations_actions_and_other_rewards_bitwise_equal=True,
        fixed_replay_sampler_bypassed_without_extra_RNG_draws=True,
        source_data_sha256={str(root / name): sha(root / name) for root in (before, after)
            for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz')},
        implementation_sha256=sha(runtime.SCRIPT), qualifier_sha256=sha(SCRIPT),
        PPO_updates_performed=0, optimizer_steps_performed=0, policy_accepted=False,
        limitation='Fixed-replay parity only; the learning benefit of targeted sampling is unverified')
    result['source_data_sha256'].update({str(p): sha(p) for p in
        (SCRIPT, runtime.SCRIPT, Path(runtime.sampler.__file__), after / 'targeted_turn_runtime_evidence.json')})
    write_json(OUTPUT / 'native_sampler_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('mode', choices=('qualify', 'frozen'))
    args = cli.parse_args()
    result = qualify() if args.mode == 'qualify' else frozen()
    print(json.dumps({k: result[k] for k in ('qualified', 'source_actual_updates',
        'full27_samples', 'all_model_and_Adam_tensors_bitwise_preserved') if k in result}))
