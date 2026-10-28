"""One existing global yaw-kernel scalar, with ordinary full-state PPO.

The calibrated physics and common wheel gain are declared independently. No
command, height, speed, yaw sign or evaluation case selects a reward/control
branch. The Gaussian action distribution is loaded unchanged; reward sigma
is the denominator in the existing tracking kernel, not action exploration.
"""
import argparse
import ast
import copy
from pathlib import Path
import sys

import joint_pd_calibrated_wheel_runtime as implementation


SCRIPT = Path(__file__).resolve()
parent_namespace = implementation.expected_configuration.__globals__
parent_script = parent_namespace['SCRIPT']
base_recipe = parent_namespace['read_recipe']
parent_expected = implementation.expected_configuration
parent_profile = implementation.profile_for_count
parent_evaluator = implementation.evaluator_tree
parent_train = parent_namespace['train']
parent_compare = parent_namespace['compare']
parent_qualify = parent_namespace['qualify']


def recipe():
    value = base_recipe()
    assert value['saved_bundle_ordinary_sampler_transfer']
    assert not value['case_specific_training_logic']
    assert not value['command_overrides'] and not value['command_changes']
    assert value['source_exploration'] == value['exploration']
    changes = value['physics_changes']
    assert set(changes) == {'robot.spawn.usd_path', 'feedforward_force'}
    target = value['physics_target']
    assert changes['robot.spawn.usd_path'][1] == target['asset']
    assert implementation.sha(target['asset']) == target['asset_sha256']
    assert changes['feedforward_force'][1] == target['vehicle_mass_kg'] * 9.81 / 2.
    controls = value['control_changes']
    assert set(controls) == {'wheel_damping', 'robot.actuators.wheels.damping'}
    assert controls['wheel_damping'] == controls['robot.actuators.wheels.damping']
    assert controls['wheel_damping'][1] == value['control_overrides']['wheel_damping']
    rewards = value['reward_changes']
    assert set(rewards) == set(value['reward_overrides']) == {'tracking_sigma_ang'}
    before, after = rewards['tracking_sigma_ang']
    assert 0. < after < before and after == value['reward_overrides']['tracking_sigma_ang']
    assert value['effective_change_relative_to_source'] == dict(
        physics=changes, controller=controls, rewards=rewards)
    return value


def expected_configuration(cfg, count):
    result = parent_expected(cfg, count)
    for key, pair in recipe()['reward_changes'].items():
        assert result['rewards'][key] in pair
        result['rewards'][key] = pair[1]
    return result


def profile_for_count(count):
    result = parent_profile(count)
    result.update(profile='plain27_calibrated_global_yaw_kernel_v1',
        native_runtime_sha256=implementation.sha(SCRIPT),
        calibrated_reward_runtime_sha256=implementation.sha(SCRIPT),
        retained_calibrated_control_runtime_sha256=implementation.sha(parent_script),
        uniform_reward_overrides=recipe()['reward_overrides'],
        global_reward_changes=recipe()['reward_changes'],
        saved_leg_PD_rewards_action_maps_and_exploration_retained=False,
        saved_leg_PD_action_maps_and_exploration_retained=True,
        existing_reward_equations_retained=True,
        positive_yaw_bonus_maximum_and_negative_yaw_cost_retained=True,
        reward_sigma_is_not_action_exploration_std=True)
    return result


def evaluator_tree():
    tree, path, proof = parent_evaluator()
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    positions = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'height' for t in n.targets)]
    assert len(positions) == 1
    original = ast.dump(tree)
    inserted = ast.parse('\n'.join('cfg.rewards.' + key + ' = ' + repr(value)
        for key, value in recipe()['reward_overrides'].items())).body
    at = positions[0]
    main.body[at:at] = inserted
    changed = copy.deepcopy(tree)
    del main.body[at:at + len(inserted)]
    assert ast.dump(tree) == original
    proof.update(declared_global_reward_overrides=recipe()['reward_overrides'],
                 source_and_descendant_replays_use_same_declared_reward=True,
                 original_metric_and_acceptance_math_unchanged=True)
    return ast.fix_missing_locations(changed), path, proof


parent_namespace['SCRIPT'] = SCRIPT
parent_namespace['recipe'] = recipe
parent_namespace['expected_configuration'] = expected_configuration
parent_namespace['profile_for_count'] = profile_for_count
parent_namespace['saved_namespace']['recipe'] = recipe
parent_namespace['saved_namespace']['expected_configuration'] = expected_configuration
implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.expected_configuration = expected_configuration
implementation.profile_for_count = profile_for_count
implementation.evaluator_tree = evaluator_tree
sys.modules['joint_pd_calibrated_reward_runtime'] = implementation


def qualify(recipe_path, output):
    import json
    result = parent_qualify(recipe_path, output)
    command = result['fifty_update_native_command']
    value = recipe()['reward_overrides']['tracking_sigma_ang']
    assert command.count('env.rewards.tracking_sigma_ang=' + str(value)) == 1
    source, _, cfg = implementation.baseline()
    expected = expected_configuration(cfg, 8192)
    changed = {k: [v, expected['rewards'][k]] for k, v in cfg['rewards'].items()
               if v != expected['rewards'][k]}
    assert changed == recipe()['reward_changes']
    _, _, replay_proof = evaluator_tree()
    result.update(saved_leg_PD_rewards_action_maps_and_exploration_retained=False,
        saved_leg_PD_action_maps_and_exploration_retained=True,
        exact_global_reward_changes_verified=changed,
        independent_replay_reward_configuration_proof=replay_proof,
        action_std_and_all17Adam_still_bitwise_preserved=True,
        reward_sigma_is_not_action_exploration_std=True)
    (output / 'qualification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def train(request):
    import train_saved_joint_pd_review50 as workflow
    previous = workflow.write_json

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            value['next_configuration_changes']['rewards'] = recipe()['reward_changes']
            value.update(only_reward_change=recipe()['reward_changes'],
                saved_leg_PD_action_maps_exploration_and_reward_structure_retained=True,
                positive_yaw_bonus_maximum_and_negative_yaw_cost_retained=True,
                reward_sigma_is_not_action_exploration_std=True)
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
