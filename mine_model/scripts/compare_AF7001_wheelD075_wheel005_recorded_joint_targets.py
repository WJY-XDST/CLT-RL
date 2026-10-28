"""Keep the original fixed-state reader with the explicitly admitted drive delta."""
import argparse
import ast
import json
from pathlib import Path

from audit_saved_action_limits import transform
from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
import compare_plain27_recorded_joint_targets as original
import joint_pd_AF7001_wheelD075_wheel005_runtime as runtime


SCRIPT = Path(__file__).resolve()


def expected_controller_and_population_difference(configs, plan):
    before, after = configs
    assert plan['source_sha256'] == runtime.ANCHOR_SHA
    assert plan['source_actual_ppo_updates'] == 7001
    assert plan['next_configuration_changes'] == runtime.DELTA
    assert not plan['only_reward_change'] and not plan['only_command_change']
    assert before['scene']['num_envs'] == before['terrain']['num_envs'] == 4096
    assert after['scene']['num_envs'] == after['terrain']['num_envs'] == 8192
    assert before['wheel_damping'] == before['robot']['actuators']['wheels']['damping'] == .5
    assert after['wheel_damping'] == after['robot']['actuators']['wheels']['damping'] == .75
    expected = runtime.expected_configuration(before, 8192)
    expected['log_dir'] = after['log_dir']
    assert after == expected, 'No other physical, reward, observation or motor contract may change'
    return {'/log_dir', '/scene/num_envs', '/terrain/num_envs',
            '/wheel_damping', '/robot/actuators/wheels/damping'}


def compare(directory, through):
    # Transform the original uncompiled function exactly once. The transform
    # reverses both expressions and verifies the rest of the numerical AST.
    tree = transform(original.compare, [
        ("{'/log_dir'}", 'expected_controller_and_population_difference(cfgs, plan)'),
        ('np.clip(r[..., (2, 5)], -1., 1.)',
         'np.clip(r[..., (2, 5)], -saved_limit(cfg), saved_limit(cfg))')])
    scope = dict(original.__dict__, SCRIPT=SCRIPT,
        expected_controller_and_population_difference=expected_controller_and_population_difference,
        saved_limit=checkpoint_wheel_action_limit)
    exec(compile(ast.fix_missing_locations(tree), str(original.SCRIPT), 'exec'), scope)
    assert scope['compare'].__globals__ is scope
    path, report = scope['compare'](Path(directory), through)
    receipt = Path(directory) / f'wheelD075_fixed_source_comparison_prefix_{through:04d}_qualification.json'
    assert not receipt.exists()
    files = (SCRIPT, original.SCRIPT, runtime.SCRIPT, Path(transform.__code__.co_filename),
             Path(checkpoint_wheel_action_limit.__code__.co_filename), path)
    pins = {str(p): sha(p) for p in files}
    assert all(sha(p) == digest for p, digest in {**report['source_pins'], **pins}.items())
    with receipt.open('x') as stream:
        json.dump(dict(qualified=True, actual_native_population=8192,
            declared_source_population=4096, controller_delta=runtime.DELTA['controller'],
            original_compare_AST_restored_identical_except_declared_contract_and_saved_wheel_bound=True,
            complete_saved_configuration_equality_to_declared_trial_required=True,
            original_all17_Adam_RNG_input_identity_motor_mapping_FK_and_windows_preserved=True,
            source_report=str(path), source_pins=pins, training_modified=False,
            simulator_started=False, optimizer_executed=False, policy_accepted=False),
            stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return path, report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    path, report = compare(args.directory, args.through)
    print(json.dumps(dict(report=str(path),
        source_actor_parity=report['source_actor_recorded_mean_parity_max_error'],
        source_FK_parity=report['source_recorded_FK_length_parity_max_error_m'],
        training_modified=False, closed_loop_acceptance=False)), flush=True)
