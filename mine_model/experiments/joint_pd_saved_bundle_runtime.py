"""Continue a validated saved controller with the ordinary class sampler.

The source supplies gains, action bounds, rewards and commands. Only the
explicitly retired targeted-sampler extension is removed from the in-memory
configuration; checkpoint tensors and paired source files remain immutable.
There are no operating-condition branches or case lists in this runtime.
"""
import argparse
import copy
from pathlib import Path
import sys

import joint_pd_uniform_control_runtime_v3 as implementation


SCRIPT = Path(__file__).resolve()
BASE_SCRIPT = Path(implementation.__file__).resolve()
old_recipe = implementation.recipe
old_baseline = implementation.baseline
old_command = implementation.command
old_evaluator_tree = implementation.evaluator_tree
RETIRED_COMMAND_FIELD = 'targeted_turn_command_fraction'


def normalize_configuration(cfg):
    value = copy.deepcopy(cfg)
    commands = value['commands']
    retired = commands.pop(RETIRED_COMMAND_FIELD, None)
    if retired is not None:
        assert type(retired) in (int, float) and 0. <= retired <= 1.
        assert commands['grouped_training'] and commands['fixed_standing_groups']
    return value


def recipe():
    value = old_recipe()
    assert value['saved_bundle_ordinary_sampler_transfer'] is True
    assert value['removed_training_extension'] == RETIRED_COMMAND_FIELD
    assert not value['reward_changes'] and not value['control_changes']
    assert not value['effective_change_relative_to_source']
    assert value['case_specific_training_logic'] is False
    return value


def baseline():
    source, saved, cfg = old_baseline()
    return source, saved, normalize_configuration(cfg)


def expected_configuration(cfg, count):
    assert count == 8192
    value = normalize_configuration(cfg)
    assert value['scene']['num_envs'] == value['terrain']['num_envs'] == count
    from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
    value.setdefault('wheel_action_limit', checkpoint_wheel_action_limit(cfg))
    assert value['wheel_damping'] == value['robot']['actuators']['wheels']['damping']
    assert value['wheel_damping'] == recipe()['control_overrides']['wheel_damping']
    return value


def profile_for_count(count):
    assert count == 8192
    cfg = expected_configuration(baseline()[2], count)
    return dict(profile='plain27_saved_bundle_ordinary_sampler_v1',
        native_runtime_sha256=implementation.sha(SCRIPT),
        retained_runtime_sha256=implementation.sha(BASE_SCRIPT),
        uniform_recipe_sha256=implementation.sha(implementation.RECIPE_PATH),
        continuation_source_sha256=recipe()['source_checkpoint_sha256'],
        native_training_environment_count=count,
        fixed_standing_environment_count=int(count * cfg['commands']['standing_env_fraction']),
        internal_observations=27, actions=6, policy_class='ActorCritic',
        ordinary_PPO=True, auxiliary_losses=False,
        command_sampler='original_continuous_with_existing_roles',
        reward='uniform_existing_reward_coefficients',
        uniform_reward_overrides={}, global_control_overrides=recipe()['control_overrides'],
        global_control_changes={}, condition_specific_training_logic=False,
        changed_noise_heads=[], retired_targeted_sampler_enabled=False,
        saved_wheel_action_limit=cfg['wheel_action_limit'],
        saved_controller_and_reward_values_retained=True, **recipe()['exploration'])


def command(source, cfg, label, updates, commands, stop_file):
    argv, final, saved = old_command(source, cfg, label, updates, commands, stop_file)
    assert not any(RETIRED_COMMAND_FIELD in item for item in argv)
    # The native task's current default may differ from a saved controller.
    # Restore the checkpoint's bound explicitly, for every wheel and command.
    assert not any(item.startswith('env.wheel_action_limit=') for item in argv)
    argv.append('env.wheel_action_limit=' + str(expected_configuration(cfg, 8192)['wheel_action_limit']))
    return argv, final, saved


def evaluator_tree():
    """Normalize only the retired extension before applying saved commands."""
    import ast
    tree, path, proof = old_evaluator_tree()
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    points = [i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'saved_config' for t in node.targets)]
    assert len(points) == 1
    at = points[0] + 1
    original = ast.dump(tree)
    inserted = ast.parse('from joint_pd_saved_bundle_runtime import normalize_configuration\n'
                         'saved_config = normalize_configuration(saved_config)').body
    main.body[at:at] = inserted
    changed = copy.deepcopy(tree)
    del main.body[at:at + len(inserted)]
    assert ast.dump(tree) == original
    proof.update(retired_sampler_metadata_normalized_before_overrides=True,
                 original_evaluator_other_statements_identical=True,
                 numerical_evaluation_rules_changed=False)
    return ast.fix_missing_locations(changed), path, proof


implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.baseline = baseline
implementation.expected_configuration = expected_configuration
implementation.profile_for_count = profile_for_count
implementation.command = command
implementation.evaluator_tree = evaluator_tree
implementation.normalize_saved_bundle = normalize_configuration
implementation.normalize_configuration = normalize_configuration
sys.modules['joint_pd_saved_bundle_runtime'] = implementation


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        boundary = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:boundary])
        implementation.configure(args.recipe)
        implementation.native(args.observation_output, sys.argv[boundary + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:])
        implementation.configure(args.recipe)
        implementation.replay(remaining)
    else:
        raise ValueError(mode)
else:
    sys.modules[__name__] = implementation
