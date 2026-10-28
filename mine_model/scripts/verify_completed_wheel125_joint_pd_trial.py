"""Keep every original200 verification while admitting one declared wheel cap."""
import argparse
import ast
import json
from pathlib import Path

from joint_pd_continuation import sha
import verify_completed_plain27_joint_pd_trial as original
from audit_saved_action_limits import transform
import joint_pd_source5801_wheel125_runtime as runtime

SCRIPT = Path(__file__).resolve()


def verify(directory, destination=None):
    directory = Path(directory).resolve()
    plan = json.loads((directory / 'experiment_plan.json').read_text())
    if (plan['source_actual_ppo_updates'] != 5801 or plan['target_actual_ppo_updates'] != 6001
            or plan['source_sha256'] != runtime.ANCHOR_SHA
            or plan.get('only_controller_change') != runtime.DELTA['controller']
            or plan['next_configuration_changes'] != runtime.DELTA
            or not plan['action_map_or_controller_changed']
            or plan['training_config']['wheel_action_limit'] != 1.25
            or plan['training_config']['action_scale_vel'] != 15.
            or plan['only_reward_change'] or plan['only_command_change'] or plan['only_exploration_change']):
        raise ValueError('Require exactly the qualified scale15/unit-to1.25 wheel-clamp200 trial')
    path = directory / 'native_learning_signal/normal_wheel_limit_observation.json'
    mapping = json.loads(path.read_text())
    if (mapping['actual_policy_steps'] != 200 * 48 or mapping['native_environments'] != 4096
            or mapping['all_environment_actions_checked'] != 200 * 48 * 4096
            or mapping['wheel_action_limit'] != 1.25 or mapping['wheel_reference_max_rad_s'] != 18.75
            or mapping['wheel_mapping_max_error'] or mapping['leg_premap_max_error']
            or mapping['observer_changes_actions_or_rewards'] or mapping['added_loss_or_backward']):
        raise ValueError('Require every native environment action to obey the declared normal wheel/leg map')
    tree = transform(original.verify, [
        ("plan['action_map_or_controller_changed']", 'not DECLARED_SINGLE_CAP_VERIFIED'),
        ("{'/log_dir'}", "{'/log_dir', '/wheel_action_limit'}"),
    ])
    scope = dict(original.__dict__, DECLARED_SINGLE_CAP_VERIFIED=True, SCRIPT=SCRIPT)
    exec(compile(ast.fix_missing_locations(tree), str(original.__file__), 'exec'), scope)
    result = scope['verify'](directory, destination)
    target = Path(destination).resolve() if destination is not None else directory / 'final_checkpoint_verification_readonly_20261019.json'
    result.update(only_controller_change=runtime.DELTA['controller'],
        original_other_checkpoint_Adam_RNG_alignment_and_actor_checks_preserved=True,
        original_verifier_AST_restored_identical=True,
        all39321600_native_wheel_and_leg_premaps_verified=True,
        normal_controller_mapping_evidence=str(path))
    result['source_files'].update({str(p): sha(p) for p in (SCRIPT, Path(original.__file__), path)})
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    report = verify(args.directory, args.destination)
    print(json.dumps({k: report[k] for k in ('verified', 'actual_ppo_updates', 'all17_Adam_step',
        'all39321600_native_wheel_and_leg_premaps_verified')}))
