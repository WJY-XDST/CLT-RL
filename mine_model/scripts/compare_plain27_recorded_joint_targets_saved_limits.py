"""The original fixed-input leg comparison with a declared wheel-clamp change."""
import argparse
import ast
import json
from pathlib import Path

from joint_pd_continuation import sha
from audit_saved_action_limits import transform
from legacy_wheel_control_schema import differences_with_wheel_default
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
import compare_plain27_recorded_joint_targets as original
import joint_pd_source5801_wheel125_runtime as runtime

SCRIPT = Path(__file__).resolve()


def compare(directory, through):
    directory = Path(directory).resolve()
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    health = json.loads((directory / f'policy_update_health_prefix_{through:04d}_readonly.json').read_text())
    baseline = Path(plan['source_checkpoint'])
    candidate = Path(health['snapshots'][-1]['checkpoint_after'])
    _, before = runtime.require_source(baseline)
    _, after = runtime.require_source(candidate)
    if (plan.get('only_controller_change') != runtime.DELTA['controller']
            or plan['next_configuration_changes'] != runtime.DELTA
            or sha(baseline) != runtime.ANCHOR_SHA
            or checkpoint_wheel_action_limit(before) != 1.
            or checkpoint_wheel_action_limit(after) != 1.25
            or before['action_scale_vel'] != 15. or after['action_scale_vel'] != 15.):
        raise ValueError('Require exactly the retained5801 unit-to1.25 clamp transfer')
    tree = transform(original.compare, [
        ("{'/log_dir'}", "{'/log_dir', '/wheel_action_limit'}"),
        ('np.clip(r[..., (2, 5)], -1., 1.)',
            'np.clip(r[..., (2, 5)], -saved_limit(cfg), saved_limit(cfg))'),
    ])
    scope = dict(original.__dict__, SCRIPT=SCRIPT, saved_limit=checkpoint_wheel_action_limit,
        differences=differences_with_wheel_default)
    exec(compile(ast.fix_missing_locations(tree), str(original.SCRIPT), 'exec'), scope)
    path, report = scope['compare'](directory, through)
    report.update(only_controller_change=runtime.DELTA['controller'],
        source_and_candidate_saved_wheel_action_limits=[1., 1.25],
        original_four_leg_mapping_all17_Adam_RNG_and_fixed_input_checks_preserved=True,
        original_AST_restored_identical=True,
        historical_missing_wheel_default_applied_to_schema_comparison_only=True,
        wheel_target_difference_includes_declared_controller_and_policy_changes=True)
    report['source_pins'].update({str(p): sha(p) for p in (SCRIPT, original.SCRIPT,
        Path(transform.__code__.co_filename), runtime.SCRIPT,
        Path(differences_with_wheel_default.__code__.co_filename))})
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return path, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--through', type=int, required=True)
    args = parser.parse_args()
    path, report = compare(args.directory, args.through)
    print(json.dumps(dict(report=str(path), fixed_input_only=True,
        source_FK_parity=report['source_recorded_FK_length_parity_max_error_m'],
        original_AST_restored_identical=report['original_AST_restored_identical'])))
