"""Original fixed-input comparison for a single reward trial with saved wheels."""
import argparse
import ast
import json
from pathlib import Path
from joint_pd_continuation import sha
from audit_saved_action_limits import transform
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
import compare_plain27_recorded_joint_targets as original
import joint_pd_source6201_lin300_runtime as runtime
SCRIPT = Path(__file__).resolve()

def compare(directory, through):
    directory = Path(directory).resolve()
    plan = json.loads((directory / 'experiment_plan.json').read_text())
    health = json.loads((directory / f'policy_update_health_prefix_{through:04d}_readonly.json').read_text())
    baseline = Path(plan['source_checkpoint'])
    candidate = Path(health['snapshots'][-1]['checkpoint_after'])
    (_, before) = runtime.require_source(baseline)
    (_, after) = runtime.require_source(candidate)
    if plan['next_configuration_changes'] != runtime.DELTA or plan['only_controller_change'] or sha(baseline) != runtime.ANCHOR_SHA or any((checkpoint_wheel_action_limit(cfg) != 1.25 or cfg['action_scale_vel'] != 15.0 for cfg in (before, after))):
        raise ValueError('Require the exact6001 single reward trial with unchanged saved wheel125')
    tree = transform(original.compare, [('np.clip(r[..., (2, 5)], -1., 1.)', 'np.clip(r[..., (2, 5)], -saved_limit(cfg), saved_limit(cfg))')])
    scope = dict(original.__dict__, SCRIPT=SCRIPT, saved_limit=checkpoint_wheel_action_limit)
    exec(compile(ast.fix_missing_locations(tree), str(original.SCRIPT), 'exec'), scope)
    (path, report) = scope['compare'](directory, through)
    report.update(only_reward_change=runtime.DELTA['rewards'], only_controller_change={}, source_and_candidate_saved_wheel_action_limits=[1.25, 1.25], original_four_leg_mapping_all17_Adam_RNG_and_configuration_checks_preserved=True, original_AST_restored_identical=True, wheel_target_differences_from_policy_changes_on_fixed_source_inputs=True)
    report['source_pins'].update({str(p): sha(p) for p in (SCRIPT, original.SCRIPT, Path(transform.__code__.co_filename), runtime.SCRIPT)})
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return (path, report)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--through', type=int, required=True)
    args = parser.parse_args()
    (path, report) = compare(args.directory, args.through)
    print(json.dumps(dict(report=str(path), fixed_input_only=True, original_AST_restored_identical=report['original_AST_restored_identical'])))
