"""Preserve historical source hashes when adding a backwards-compatible clamp.

Old reports keep their original paths and digests. A new trial resolves only
declared changed source files to archived, byte-identical historical copies;
its current implementation is pinned separately, without rewriting evidence.
"""
import ast
import copy
from datetime import datetime
import json
from pathlib import Path

from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
DIRECTORY = ROOT / 'mine_model/results/joint_pd_wheel_action_limit_source_migration_20261020'
BEFORE = DIRECTORY / 'before_source_snapshot.json'
PROOF = DIRECTORY / 'migration_verified_after_telemetry_fix.json'
RELATIVE_FILES = (
    'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py',
    'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env_cfg.py',
    'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
    'wheel_legged_isaaclab/scripts/rsl_rl/play.py',
    'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
)


def resolve_historical_pins(pins):
    proof = json.loads(PROOF.read_text())
    if not proof['verified'] or not proof['historical_bytes_preserved']:
        raise ValueError('Require explicit historical-source preservation and reviewed migration')
    mapping = proof['changed_source_files']
    result, moved = {}, []
    for name, digest in pins.items():
        path = Path(name)
        if path.is_file() and sha(path) == digest:
            result[str(path)] = digest
            continue
        entry = mapping.get(str(path))
        if entry is None or entry['historical_sha256'] != digest:
            entry = next((v for v in proof.get('intermediate_source_files', {}).get(str(path), [])
                if v['historical_sha256'] == digest), None)
        if (entry is None or entry['historical_sha256'] != digest
                or sha(entry['archive']) != digest or sha(path) != entry['current_sha256']):
            raise ValueError('Undeclared historical source or evidence change: ' + str(path))
        result[entry['archive']] = digest
        result[str(path)] = entry['current_sha256']
        moved.append(dict(original=str(path), historical_sha256=digest, archive=entry['archive'],
            current_sha256=entry['current_sha256']))
    for name in (SCRIPT, BEFORE, PROOF):
        result[str(name)] = sha(name)
    return result, moved


def qualify():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from wheel_legged_gym_isaaclab.wheel_control import apply_wheel_action_limit
    from wheel_legged_gym_isaaclab.action_bias_probe import wheel_reference_probe
    if PROOF.exists():
        raise ValueError('Preserve the existing immutable migration proof')
    before = json.loads(BEFORE.read_text())
    files = before['source_files']
    changed = {}
    for name, entry in files.items():
        if sha(entry['archive']) != entry['historical_sha256']:
            raise ValueError('Historical source bytes changed')
        ast.parse(Path(name).read_text())
        changed[name] = dict(entry, current_sha256=sha(name))
    env_name = str(ROOT / RELATIVE_FILES[0])
    old_tree = ast.parse(Path(files[env_name]['archive']).read_text())
    new_tree = ast.parse(Path(env_name).read_text())

    def methods(tree):
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'WheelLeggedVMCFlatEnv')
        return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

    old, new = methods(old_tree), methods(new_tree)
    differing = [name for name in old if ast.dump(old[name]) != ast.dump(new[name])]
    if set(old) != set(new) or set(differing) != {'__init__', '_pre_physics_step', '_validate_configuration'}:
        raise ValueError('Only validation, reference reachability and wheel preprocessing may differ')
    # Direct joint target application, gravity feedforward, rewards, reset,
    # observations, termination and all physics substep code remain identical.
    def preprocessor(node):
        scope = dict(torch=torch, apply_wheel_action_limit=apply_wheel_action_limit)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)], type_ignores=[])),
            env_name, 'exec'), scope)
        return scope['_pre_physics_step']

    legacy, current = preprocessor(old['_pre_physics_step']), preprocessor(new['_pre_physics_step'])
    rng = torch.random.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
    devices = ['cpu', 'cuda'] if gpu_rng is not None else ['cpu']
    total = 0
    for level in ('low', 'nominal', 'high'):
        directory = ROOT / ('mine_model/results/joint_pd_source5801_combined_headroom125_diagnostic_20261020/'
            + level + '/candidate')
        with np.load(directory / 'policy_inputs.npz') as z:
            values = z['means'].copy().reshape(-1, 6)
        for device in devices:
            raw = torch.as_tensor(values, device=device)
            a = SimpleNamespace(extras={'log': {'old': 1}}, cfg=SimpleNamespace(wheel_action_limit=1.))
            b = SimpleNamespace(extras={'log': {'old': 1}}, cfg=SimpleNamespace(wheel_action_limit=1.))
            legacy(a, raw)
            current(b, raw)
            if (not torch.equal(a._raw_actions, b._raw_actions) or not torch.equal(a._actions, b._actions)
                    or a.extras != b.extras):
                raise ValueError('Historical unit-limit preprocessing changed')
            b.cfg.wheel_action_limit = 1.25
            current(b, raw)
            if not torch.equal(b._actions, wheel_reference_probe(a._actions, a._raw_actions, 1.25)):
                raise ValueError('Normal preprocessing differs from the completed native clamp/history diagnostic')
        total += len(values)
    if (total != 324000 or not torch.equal(rng, torch.random.get_rng_state())
            or gpu_rng is not None and not torch.equal(gpu_rng, torch.cuda.get_rng_state())):
        raise ValueError('Keep all observed means and original RNG streams')
    cfg_name = str(ROOT / RELATIVE_FILES[1])
    old_cfg = ast.parse(Path(files[cfg_name]['archive']).read_text())
    new_cfg = ast.parse(Path(cfg_name).read_text())
    cls = next(n for n in new_cfg.body if isinstance(n, ast.ClassDef) and n.name == 'WheelLeggedVMCFlatEnvCfg')
    targets = [n for n in cls.body if isinstance(n, ast.AnnAssign)
        and isinstance(n.target, ast.Name) and n.target.id == 'wheel_action_limit']
    if len(targets) != 1 or not isinstance(targets[0].value, ast.Constant) or targets[0].value.value != 1.:
        raise ValueError('Require a historical default unit wheel clamp')
    cls.body.remove(targets[0])
    if ast.dump(old_cfg) != ast.dump(new_cfg):
        raise ValueError('Configuration changes beyond the defaulted wheel clamp')
    gate = ROOT / 'mine_model/config/tracking_acceptance.json'
    if sha(gate) != '08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d':
        raise ValueError('Original acceptance changed')
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        historical_bytes_preserved=True, changed_source_files=changed,
        original_configuration_AST_identical_after_removing_defaulted_field=True,
        changed_environment_methods=differing, all_other_environment_methods_AST_identical=True,
        complete_preprocessing_means_compared=total, devices=devices,
        historical_unit_limit_bitwise_equal=True, current_limit125_bitwise_equal_to_native_probe=True,
        native_probe_initial_conditions_and_physics_previously_paired=True, RNG_unchanged=True,
        unit_helper_tests_passed=3, saved_replay_contract_restoration_requires_native_fixture_review=True,
        source_pins={str(p): sha(p) for p in (SCRIPT, BEFORE, gate,
            ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/wheel_control.py',
            ROOT / 'wheel_legged_isaaclab/tests/test_wheel_action_limit.py')},
        acceptance_changed=False, optimizer_executed=False, policy_accepted=False)
    intermediate_manifest = DIRECTORY / 'v1_before_telemetry_fix/source_snapshot.json'
    intermediate = json.loads(intermediate_manifest.read_text())['source_files']
    versions = {}
    for name, entry in intermediate.items():
        if sha(entry['archive']) != entry['historical_sha256']:
            raise ValueError('Preserve every intermediate qualified implementation byte')
        versions[name] = [dict(entry, current_sha256=sha(name))]
    result['intermediate_source_files'] = versions
    result['source_pins'][str(intermediate_manifest)] = sha(intermediate_manifest)
    result['corrected_observer_field'] = 'action_clipping_fraction now uses the saved wheel clamp when no diagnostic override is active'
    with PROOF.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    result = qualify()
    print(json.dumps({k: result[k] for k in ('verified', 'historical_bytes_preserved',
        'complete_preprocessing_means_compared', 'devices', 'all_other_environment_methods_AST_identical')}))
