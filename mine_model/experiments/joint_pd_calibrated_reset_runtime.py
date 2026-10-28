"""One globally uniform nominal reset amplitude with standard full-state PPO.

The matched common-floor control uses +/-0.5 root velocity perturbations during
training and +/-0.05 during nominal replay. This trial changes only the existing
training reset_velocity_final scalar to0.05. Evaluation, command sampling,
physics, reward equations and PPO remain unchanged. No operating condition
selects a reset rule, and no saved model/std/Adam/LR tensor is reset.
"""
import argparse
import copy
from pathlib import Path
import sys

import joint_pd_calibrated_reward_runtime as implementation


SCRIPT = Path(__file__).resolve()
reward_namespace = implementation.recipe.__globals__
wheel_namespace = reward_namespace['parent_namespace']
saved_namespace = wheel_namespace['saved_namespace']
read_recipe = reward_namespace['base_recipe']
parent_profile = implementation.profile_for_count
parent_train = reward_namespace['train']
parent_qualify = reward_namespace['qualify']
parent_compare = reward_namespace['parent_compare']
parent_expected = implementation.expected_configuration
parent_command = implementation.command
parent_private = implementation.private_for_count


def recipe():
    value = read_recipe()
    assert value['saved_bundle_ordinary_sampler_transfer']
    assert not value['case_specific_training_logic']
    assert not value['command_overrides'] and not value['command_changes']
    physics = value['physics_changes']
    assert set(physics) == {'robot.spawn.usd_path', 'feedforward_force'}
    target = value['physics_target']
    assert physics['robot.spawn.usd_path'][1] == target['asset']
    assert implementation.sha(target['asset']) == target['asset_sha256']
    assert physics['feedforward_force'][1] == target['vehicle_mass_kg'] * 9.81 / 2.
    controls = value['control_changes']
    assert set(controls) == {'wheel_damping', 'robot.actuators.wheels.damping'}
    assert controls['wheel_damping'] == controls['robot.actuators.wheels.damping']
    assert controls['wheel_damping'][1] == value['control_overrides']['wheel_damping']
    rewards = value['reward_changes']
    assert set(rewards) == set(value['reward_overrides']) == {'tracking_sigma_ang'}
    before_sigma, after_sigma = rewards['tracking_sigma_ang']
    assert 0. < after_sigma < before_sigma
    assert after_sigma == value['reward_overrides']['tracking_sigma_ang']
    before, after = value['source_exploration'], value['exploration']
    keys = {'global_minimum_std', 'minimum_leg_std', 'minimum_wheel_std'}
    assert set(before) == set(after) == keys
    assert 0. < after['minimum_leg_std'] < before['minimum_leg_std']
    assert 0. < after['minimum_wheel_std'] < before['minimum_wheel_std']
    assert after['global_minimum_std'] == after['minimum_leg_std'] == after['minimum_wheel_std']
    assert max(after['global_minimum_std'], after['minimum_leg_std']) < max(
        before['global_minimum_std'], before['minimum_leg_std'])
    changes = {k: [before[k], after[k]] for k in keys if before[k] != after[k]}
    assert set(changes) == keys
    assert value['exploration_changes'] == changes and value['initial_std_reset'] is False
    reset = value['training_environment_changes']
    assert reset == {'reset_velocity_final': [0.5, 0.05]}
    assert value['training_environment_overrides'] == {'reset_velocity_final': 0.05}
    assert value['nominal_replay_reset_amplitude_unchanged'] == 0.05
    assert not value.get('algorithm_overrides', {})
    assert value['effective_change_relative_to_source'] == dict(
        physics=physics, controller=controls, rewards=rewards, exploration=changes,
        training_environment=reset)
    return value


def expected_configuration(cfg, count):
    result = parent_expected(cfg, count)
    assert result['reset_velocity_initial'] == 0.05
    before, after = recipe()['training_environment_changes']['reset_velocity_final']
    assert result['reset_velocity_final'] in (before, after)
    result['reset_velocity_final'] = after
    return result


def profile_for_count(count):
    result = parent_profile(count)
    result.update(profile='plain27_calibrated_common_noise_nominal_reset_v1',
        native_runtime_sha256=implementation.sha(SCRIPT),
        precision_noise_runtime_sha256=implementation.sha(SCRIPT),
        uniform_exploration_changes=recipe()['exploration_changes'],
        changed_noise_heads=list(range(6)), saved_initial_std_reset=False,
        all_six_std_entries_trainable=True, leg_exploration_floor_unchanged=False,
        saved_leg_PD_action_maps_and_exploration_retained=False,
        saved_leg_PD_and_action_maps_retained=True, saved_initial_distribution_retained=True,
        same_common_future_floor_for_every_command_and_head=True,
        same_future_wheel_floor_for_every_command=True,
        global_training_environment_changes=recipe()['training_environment_changes'],
        all_environment_reset_velocity_amplitude=0.05,
        nominal_replay_reset_amplitude_unchanged=0.05,
        ordinary_PPO_algorithm_and_adaptive_LR_unchanged=True)
    return result


def command(*args, **kwargs):
    # A descendant may already carry the explicit physics authorization flag.
    original = wheel_namespace['old_command']

    def unique_flag(*aa, **kk):
        argv, final, saved = original(*aa, **kk)
        assert argv.count('--allow_physics_contract_change') <= 1
        return [arg for arg in argv if arg != '--allow_physics_contract_change'], final, saved

    wheel_namespace['old_command'] = unique_flag
    try:
        argv, final, saved = parent_command(*args, **kwargs)
    finally:
        wheel_namespace['old_command'] = original
    for key, value in recipe()['training_environment_overrides'].items():
        prefix = 'env.' + key + '='
        indices = [i for i, arg in enumerate(argv) if arg.startswith(prefix)]
        assert len(indices) <= 1
        if indices:
            argv[indices[0]] = prefix + str(value)
        else:
            argv.append(prefix + str(value))
    assert argv.count('--allow_physics_contract_change') == 1
    return argv, final, saved


def private_for_count(count, selected_profile=None, *, is_replay=False):
    module = parent_private(count, selected_profile, is_replay=is_replay)
    original = module.observe_environment

    def observe(env, output):
        result = original(env, output)
        if not is_replay:
            raw = env.unwrapped
            assert raw.cfg.reset_velocity_initial == raw.cfg.reset_velocity_final == 0.05
            from auto_train_mine import write_json
            write_json(Path(output) / 'uniform_nominal_reset_native_contract.json', dict(
                verified=True, actual_num_envs=raw.num_envs,
                reset_velocity_initial=raw.cfg.reset_velocity_initial,
                reset_velocity_final=raw.cfg.reset_velocity_final,
                global_training_environment_changes=recipe()['training_environment_changes'],
                same_reset_for_every_environment_and_command=True,
                command_sampler_and_reset_function_unchanged=True,
                acceptance_reset_amplitude_unchanged=True,
                extra_observation_callback_calls=0, policy_accepted=False))
        return result

    module.observe_environment = observe
    return module


for namespace in (reward_namespace, wheel_namespace, saved_namespace):
    namespace['recipe'] = recipe
    namespace['SCRIPT'] = SCRIPT
    namespace['expected_configuration'] = expected_configuration
reward_namespace['profile_for_count'] = profile_for_count
wheel_namespace['profile_for_count'] = profile_for_count
implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.expected_configuration = expected_configuration
implementation.profile_for_count = profile_for_count
implementation.command = command
implementation.private_for_count = private_for_count
sys.modules['joint_pd_calibrated_reset_runtime'] = implementation


def qualify(recipe_path, output):
    import json
    result = parent_qualify(recipe_path, output)
    before, after = recipe()['source_exploration'], recipe()['exploration']
    legacy = result['fifty_update_native_command']
    source, _, cfg = implementation.baseline()
    native_argv, final, _ = command(source, cfg,
        'calibrated_wheel_CPU_boundary', 50, cfg['commands'], output / 'NO_STOP_REQUEST')
    prefixes = tuple('env.' + key + '=' for key in recipe()['training_environment_overrides'])
    strip_reset = lambda argv: [arg for arg in argv if not arg.startswith(prefixes)]
    assert strip_reset(native_argv) == strip_reset(legacy)
    assert final == implementation.saved_updates(source) + 49
    assert native_argv[native_argv.index('--min_exploration_std') + 1] == str(after['global_minimum_std'])
    assert native_argv[native_argv.index('--min_wheel_exploration_std') + 1] == str(after['minimum_wheel_std'])
    assert native_argv.count('env.reset_velocity_final=0.05') == 1
    expected = expected_configuration(cfg, 8192)
    parent = parent_expected(cfg, 8192)
    assert {k: [parent[k], v] for k, v in expected.items() if parent[k] != v} == recipe()['training_environment_changes']
    result.update(saved_leg_PD_action_maps_and_exploration_retained=False,
        saved_leg_PD_and_action_maps_retained=True,
        source_exploration=before, future_projection_floors=after,
        source_saved_sixstd_not_reset=True,
        effective_leg_floor_before_after=[max(before['global_minimum_std'], before['minimum_leg_std']),
                                         max(after['global_minimum_std'], after['minimum_leg_std'])],
        effective_wheel_floor_before_after=[max(before['global_minimum_std'], before['minimum_wheel_std']),
                                           max(after['global_minimum_std'], after['minimum_wheel_std'])],
        future_projection_change_is_not_a_saved_std_reset=True,
        global_training_environment_changes=recipe()['training_environment_changes'],
        nominal_evaluation_and_acceptance_reset_amplitude_unchanged=0.05,
        exact_incremental_native_environment_change_verified=True,
        fifty_update_native_command=native_argv)
    (output / 'qualification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def train(request):
    import train_saved_joint_pd_review50 as workflow
    previous = workflow.write_json

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            changes = recipe()['exploration_changes']
            value['next_configuration_changes']['exploration'] = changes
            value['next_configuration_changes']['training_environment'] = recipe()['training_environment_changes']
            value.update(only_exploration_change=changes,
                only_training_environment_change=recipe()['training_environment_changes'],
                global_reset_velocity_amplitude=0.05,
                nominal_replay_reset_amplitude_unchanged=0.05,
                standard_PPO_algorithm_unchanged=True,
                source_exploration=recipe()['source_exploration'],
                actual_future_projection_floors=recipe()['exploration'],
                saved_initial_std_reset=False, leg_exploration_floor_unchanged=False,
                saved_leg_PD_action_maps_exploration_and_reward_structure_retained=False,
                saved_leg_PD_action_maps_and_reward_structure_retained=True,
                same_common_future_floor_for_every_command_and_head=True,
                no_condition_specific_noise_rule=True)
        return previous(path, value)

    workflow.write_json = write
    try:
        return parent_train(request)
    finally:
        workflow.write_json = previous


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        at = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:at])
        implementation.configure(args.recipe)
        implementation.native(args.observation_output, sys.argv[at + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:])
        implementation.configure(args.recipe)
        implementation.replay(remaining)
    elif mode == 'compare':
        parser.add_argument('directory', type=Path)
        parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        parent_compare(args.directory.resolve(), args.through)
    elif mode == 'qualify':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        qualify(args.recipe.resolve(), args.output.resolve())
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        train(args.request.resolve())
    else:
        raise ValueError(mode)
else:
    sys.modules[__name__] = implementation
