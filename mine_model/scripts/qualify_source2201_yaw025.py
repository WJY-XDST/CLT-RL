"""Qualify full retained PPO state and frozen27 angular-reward equivalence."""
import argparse
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import source1201_orientation40
import joint_pd_source2201_yaw025_runtime as repair
import qualify_source2001_quality50 as template
from auto_train_mine import write_json
from joint_pd_continuation import sha
from qualify_source801_drift50_transfer import differences

SCRIPT = Path(__file__).resolve()


def adapted_cpu():
    spec = importlib.util.spec_from_file_location('_source2201_yaw025_cpu_qualifier', template.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.repair, module.SCRIPT = repair, SCRIPT
    tree = ast.parse(inspect.getsource(module.cpu))
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
                change(node.elts[0], 'value', .25, 'old_reward_value_metadata')
            self.generic_visit(node)

        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'repair' and node.attr == 'QUALITY_FLOOR':
                change(node, 'attr', 'REWARD_VALUE', 'new_reward_value_attribute')
            self.generic_visit(node)

        def visit_Constant(self, node):
            if type(node.value) is int and node.value == 2200:
                change(node, 'value', 2400, 'target_checkpoint_number')
            elif isinstance(node.value, str):
                value = ('2401' if node.value == '2201' else
                    node.value.replace('source2001_quality50', 'source2201_yaw025'))
                if value != node.value:
                    change(node, 'value', value, 'source_label_or_target_clock')

    Adapter().visit(tree)
    if counts.get('old_reward_value_metadata') != 1 or counts.get('target_checkpoint_number') != 1:
        raise ValueError('Requalify the exact CPU metadata adaptation')
    modified = copy.deepcopy(tree)
    for node, field, value in reversed(changes):
        setattr(node, field, value)
    if ast.dump(tree) != original:
        raise ValueError('Full-state CPU checks changed beyond declared reward/clock metadata')
    exec(compile(ast.fix_missing_locations(modified), template.__file__, 'exec'), module.__dict__)
    return module, counts


def frozen(previous, output):
    before, after = Path(previous) / 'replay', Path(output) / 'native_reward_transfer'
    read = lambda root, name: json.loads((root / name).read_text())
    a, b = read(before, 'manifest.json'), read(after, 'manifest.json')
    da, db = read(before, 'completed.json'), read(after, 'completed.json')
    ca, cb = read(before, 'resolved_env.json'), read(after, 'resolved_env.json')
    if (a['checkpoint_sha256'] != repair.ANCHOR_SHA or b['checkpoint_sha256'] != repair.ANCHOR_SHA
            or a['cases'] != b['cases'] or a['seeds'] != b['seeds']
            or not da['completed'] or not db['completed']
            or da['samples'] != 27000 or db['samples'] != 27000
            or da['resets'] != 24 or db['resets'] != da['resets']
            or differences(ca, cb) != ['/rewards/' + repair.REWARD_KEY]
            or ca['rewards'][repair.REWARD_KEY] != .25
            or cb['rewards'][repair.REWARD_KEY] != repair.REWARD_VALUE):
        raise ValueError('Require identical full27 source physics, including all24 failed resets')
    old, new = [np.genfromtxt(root / 'trace.csv', delimiter=',', names=True) for root in (before, after)]
    changed = {'reward_total', 'reward_track_ang_vel'}
    if (old.dtype.names != new.dtype.names or len(new) != 27000
            or any(not np.array_equal(old[key], new[key]) for key in old.dtype.names if key not in changed)):
        raise ValueError('Frozen actions, physical states or another reward changed')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as x, np.load(after / 'policy_inputs.npz', allow_pickle=False) as y:
        if set(x.files) != set(y.files) or any(not np.array_equal(x[key], y[key]) for key in x.files):
            raise ValueError('Frozen raw observations or actor means changed')
    criteria = json.loads((repair.ROOT / 'mine_model/config/tracking_acceptance.json').read_text())
    height = abs(new['height_m'] - new['height_cmd_m']) / (criteria['relative_error'] * abs(new['height_cmd_m']))
    body = np.maximum(abs(new['pitch_rad']), abs(new['roll_rad'])) / np.deg2rad(criteria['body_p95_deg'])
    quality = .5 + .5 / ((1. + np.maximum(height - .5, 0.)**2) * (1. + np.maximum(body - .5, 0.)**2))
    expected = cb['rewards']['tracking_ang_vel'] * quality * np.exp(
        -(new['yaw_rate_rad_s'] - new['yaw_cmd_rad_s'])**2 / repair.REWARD_VALUE) * b['policy_dt']
    formula = float(abs(expected - new['reward_track_ang_vel']).max())
    residual = abs(new['reward_total'] - old['reward_total']
        - new['reward_track_ang_vel'] + old['reward_track_ang_vel'])
    roundoff = 8. * np.finfo(np.float32).eps * np.maximum.reduce((
        np.ones(len(new)), abs(old['reward_total']), abs(new['reward_total'])))
    if formula > 2.e-7 or np.any(residual > roundoff):
        raise ValueError('Independent angular formula or summed reward delta differs')
    patch = read(after, 'quality_criteria_runtime_patch.json')
    if (patch['implementation_sha256'] != sha(repair.criteria_fix.SCRIPT)
            or not patch['both_violation_penalties_zero']
            or patch['loaded_criteria_evidence']['sha256'] != sha(repair.ROOT / 'mine_model/config/tracking_acceptance.json')):
        raise ValueError('Require independently loaded original10% criteria')
    result = dict(qualified=True, source_sha256=repair.ANCHOR_SHA,
        only_reward_change={repair.REWARD_KEY: [.25, repair.REWARD_VALUE]},
        full27_samples=27000, identical_source_resets_including_failures=24,
        physical_outputs_observations_actions_and_other_rewards_bitwise_equal=True,
        independent_angular_formula_max_error=formula, total_reward_delta_max_error=float(residual.max()),
        both_violation_penalties_zero=True, criteria_loaded_by_quality_reward_alone=True,
        criteria_fix_sha256=sha(repair.criteria_fix.SCRIPT),
        source_data_sha256={str(root / name): sha(root / name) for root in (before, after)
            for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz')},
        implementation_sha256=sha(repair.SCRIPT), qualifier_sha256=sha(SCRIPT),
        PPO_updates_performed=0, policy_accepted=False,
        limitation='Frozen angular reward equivalence, not learned convergence or acceptance.')
    write_json(Path(output) / 'native_reward_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frozen-previous', type=Path)
    args = parser.parse_args()
    repair.repair_ready()
    if args.frozen_previous:
        result = frozen(args.frozen_previous, args.output)
    else:
        if args.source is None:
            parser.error('--source is required')
        module, changes = adapted_cpu()
        result = module.cpu(args.source, args.output)
        write_json(args.output / 'qualification_adapter_cpu.json', dict(qualified=True,
            adapter_sha256=sha(SCRIPT), original_template_sha256=sha(template.__file__),
            adaptations=changes, restored_AST_identical=True,
            original_full_state_and_optimizer_checks_preserved=True, PPO_updates_performed=0))
    print(json.dumps(dict(qualified=result['qualified'], profile=result.get('profile'),
        only_reward_change=result['only_reward_change'], policy_accepted=False), ensure_ascii=False))
