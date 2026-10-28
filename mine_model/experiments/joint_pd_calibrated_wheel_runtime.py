"""Ordinary full-state PPO with a declared mass and common wheel-gain revision.

The recipe supplies one wheel velocity gain for both sides and every command.
Physics, gains and their provenance are explicit; no operating condition or
evaluation case changes training, rewards, actions, sampling or acceptance.
The original runner, all Adam state and numerical review readers are retained.
"""
import argparse
import ast
import copy
import inspect
from pathlib import Path
import sys

import joint_pd_saved_bundle_review50_runtime as implementation


SCRIPT = Path(__file__).resolve()
saved_namespace = implementation.recipe.__globals__
read_recipe = saved_namespace['old_recipe']
normalize_configuration = saved_namespace['normalize_configuration']
old_profile = implementation.profile_for_count
old_command = implementation.command
old_private = implementation.private_for_count
old_evaluator = implementation.evaluator_tree


def recipe():
    value = read_recipe()
    assert value['saved_bundle_ordinary_sampler_transfer']
    assert not value['reward_changes'] and not value['reward_overrides']
    assert not value['command_changes'] and not value['command_overrides']
    assert not value['case_specific_training_logic']
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
    assert value['effective_change_relative_to_source'] == {
        'physics': changes, 'controller': controls}
    return value


def expected_configuration(cfg, count):
    assert count == 8192
    result = normalize_configuration(cfg)
    assert result['scene']['num_envs'] == result['terrain']['num_envs'] == count
    from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit
    result.setdefault('wheel_action_limit', checkpoint_wheel_action_limit(cfg))
    changes = recipe()['physics_changes']
    assert result['robot']['spawn']['usd_path'] in changes['robot.spawn.usd_path']
    assert result['feedforward_force'] in changes['feedforward_force']
    result['robot']['spawn']['usd_path'] = changes['robot.spawn.usd_path'][1]
    result['feedforward_force'] = changes['feedforward_force'][1]
    controls = recipe()['control_changes']
    assert result['wheel_damping'] == result['robot']['actuators']['wheels']['damping']
    assert result['wheel_damping'] in controls['wheel_damping']
    result['wheel_damping'] = controls['wheel_damping'][1]
    result['robot']['actuators']['wheels']['damping'] = controls['wheel_damping'][1]
    return result


def profile_for_count(count):
    result = old_profile(count)
    result.update(profile='plain27_calibrated_mass_common_wheel_gain_v1',
        native_runtime_sha256=implementation.sha(SCRIPT),
        calibrated_control_runtime_sha256=implementation.sha(SCRIPT),
        declared_physics_changes=recipe()['physics_changes'],
        nominal_vehicle_mass_kg=recipe()['physics_target']['vehicle_mass_kg'],
        global_control_changes=recipe()['control_changes'],
        saved_controller_and_reward_values_retained=False,
        saved_leg_PD_rewards_action_maps_and_exploration_retained=True,
        same_physics_and_wheel_gain_for_every_command_and_environment=True)
    return result


def command(source, cfg, label, updates, commands, stop_file):
    import json
    argv, final, saved = old_command(source, cfg, label, updates, commands, stop_file)
    for key, (_, value) in recipe()['physics_changes'].items():
        prefix = 'env.' + key + '='
        indices = [i for i, item in enumerate(argv) if item.startswith(prefix)]
        assert len(indices) <= 1
        argument = prefix + json.dumps(value)
        if indices:
            argv[indices[0]] = argument
        else:
            argv.append(argument)
    assert '--allow_physics_contract_change' not in argv
    argv.append('--allow_physics_contract_change')
    return argv, final, saved


def private_for_count(count, selected_profile=None, *, is_replay=False):
    # Only adapt the observer's expected mass and USD. The native runner,
    # environment, transition observer and optimizer code are unchanged.
    tree = ast.parse(inspect.getsource(old_private))
    original = ast.dump(tree)
    replacements = []
    for node in ast.walk(tree):
        for field, value in ast.iter_fields(node):
            entries = list(enumerate(value)) if isinstance(value, list) else [(field, value)]
            owner = value if isinstance(value, list) else node
            for key, entry in entries:
                expression = None
                if isinstance(entry, ast.Constant) and entry.value == 12.34:
                    expression = "recipe()['physics_target']['vehicle_mass_kg']"
                elif isinstance(entry, ast.Attribute) and ast.unparse(entry) == 'base.tires.ASSET':
                    expression = "Path(recipe()['physics_target']['asset'])"
                if expression:
                    replacements.append((owner, key, entry))
                    replacement = ast.parse(expression, mode='eval').body
                    if isinstance(owner, list):
                        owner[key] = replacement
                    else:
                        setattr(owner, key, replacement)
    assert len(replacements) == 2
    changed = copy.deepcopy(tree)
    for owner, key, value in replacements:
        if isinstance(owner, list):
            owner[key] = value
        else:
            setattr(owner, key, value)
    assert ast.dump(tree) == original
    scope = dict(old_private.__globals__, recipe=recipe,
                 expected_configuration=expected_configuration,
                 profile_for_count=profile_for_count, SCRIPT=SCRIPT)
    exec(compile(ast.fix_missing_locations(changed), str(SCRIPT), 'exec'), scope)
    return scope['private_for_count'](count, selected_profile, is_replay=is_replay)


def evaluator_tree():
    tree, path, proof = old_evaluator()
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    positions = [i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'height' for t in n.targets)]
    assert len(positions) == 1
    original = ast.dump(tree)
    changes = recipe()['physics_changes']
    gain = recipe()['control_overrides']['wheel_damping']
    inserted = ast.parse('cfg.robot.spawn.usd_path = ' + repr(changes['robot.spawn.usd_path'][1])
        + '\ncfg.feedforward_force = ' + repr(changes['feedforward_force'][1])
        + '\ncfg.wheel_damping = ' + repr(gain)
        + "\ncfg.robot.actuators['wheels'].damping = " + repr(gain)).body
    at = positions[0]
    main.body[at:at] = inserted
    changed = copy.deepcopy(tree)
    del main.body[at:at + len(inserted)]
    assert ast.dump(tree) == original
    proof.update(declared_physics_changes=changes,
                 declared_common_wheel_gain=recipe()['control_changes'],
                 no_condition_specific_control=True, original_replay_math_unchanged=True)
    return ast.fix_missing_locations(changed), path, proof


saved_namespace['recipe'] = recipe
saved_namespace['expected_configuration'] = expected_configuration
implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.expected_configuration = expected_configuration
implementation.profile_for_count = profile_for_count
implementation.command = command
implementation.private_for_count = private_for_count
implementation.evaluator_tree = evaluator_tree
sys.modules['joint_pd_calibrated_wheel_runtime'] = implementation


def compare(output, through):
    import review_saved_joint_pd_segment as segment
    old_normalize = implementation.normalize_saved_bundle
    implementation.normalize_saved_bundle = lambda cfg: expected_configuration(old_normalize(cfg), 8192)
    try:
        segment.compare(output, through)
    finally:
        implementation.normalize_saved_bundle = old_normalize


def validate_declared_contract(original, run, reward_profile=None, *,
                               fresh_weights=True, source_checkpoint=None):
    import yaml
    # Retain original per-run direct-joint, effort-envelope and full-policy
    # precision checks. Admit only explicit FF and common-gain differences.
    original(run, reward_profile, fresh_weights=fresh_weights, source_checkpoint=None)
    if source_checkpoint:
        before = yaml.safe_load((Path(source_checkpoint).parent / 'params/control_contract.yaml').read_text())
        after = yaml.safe_load((Path(run) / 'params/control_contract.yaml').read_text())
        targets = dict(feedforward_force=recipe()['physics_changes']['feedforward_force'][1],
                       wheel_damping=recipe()['control_overrides']['wheel_damping'])
        allowed = {k: [before[k], v] for k, v in targets.items() if before[k] != v}
        actual = {k: [v, after.get(k)] for k, v in before.items()
                  if k != 'fresh_weights' and after.get(k) != v}
        assert actual == allowed, (actual, allowed)
        assert all(after[k] == v for k, v in targets.items())


def train(request):
    import review_saved_joint_pd_segment as segment
    import train_joint_pd_standing_first as boundary
    api = segment.original
    old_write, old_health, old_processes = api.write_json, api.health50, segment.processes
    old_contract, old_verify = boundary.validate_contract, boundary.verify_update

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            value['next_configuration_changes'].update(
                physics=recipe()['physics_changes'], controller=recipe()['control_changes'])
            value.update(only_physics_change=recipe()['physics_changes'],
                only_controller_change=recipe()['control_changes'],
                only_runtime_sampling_change={},
                original_observation_and_action_maps_and_saved_controller_retained=False,
                original_observation_action_maps_leg_PD_rewards_and_exploration_retained=True)
        return old_write(path, value)

    def processes(delegate):
        base = old_processes(delegate)

        class DeclaredControlReaders(type(base)):
            def routed(self, argv):
                argv = super().routed(argv)
                if (isinstance(argv, (list, tuple)) and len(argv) > 2
                        and Path(argv[1]).name == segment.SCRIPT.name and argv[2] == 'compare'):
                    return [argv[0], str(SCRIPT), 'compare', '--recipe',
                            str(implementation.RECIPE_PATH), *argv[3:]]
                return argv
        return DeclaredControlReaders(delegate)

    def verify(*args, **kwargs):
        result = old_verify(*args, **kwargs)
        result.update(calibrated_asset_and_full_height_unchanged=False,
            physical_control_contract_unchanged=False,
            declared_mass_gravity_and_common_wheel_gain_transition_verified=True,
            declared_physics_changes=recipe()['physics_changes'],
            declared_control_changes=recipe()['control_changes'],
            all_other_physical_contract_fields_equal=True)
        return result

    api.write_json, api.health50, segment.processes = write, segment.health, processes
    boundary.validate_contract = lambda *a, **k: validate_declared_contract(old_contract, *a, **k)
    boundary.verify_update = verify
    try:
        return api.run(request)
    finally:
        api.write_json, api.health50, segment.processes = old_write, old_health, old_processes
        boundary.validate_contract, boundary.verify_update = old_contract, old_verify


def qualify(recipe_path, output):
    import json
    import shutil
    import yaml
    import train_joint_pd_standing_first as boundary
    from continue_saved_joint_pd_bundle import qualify as qualify_saved_state

    result = qualify_saved_state(recipe_path, output)
    source, _, cfg = implementation.baseline()
    expected = expected_configuration(cfg, 8192)
    argv, final, _ = command(source, cfg, 'calibrated_wheel_CPU_boundary', 50,
                              cfg['commands'], output / 'NO_STOP_REQUEST')
    assert final == implementation.saved_updates(source) + 49
    assert argv[argv.index('--num_envs') + 1] == '8192'
    assert argv[argv.index('--target_iteration') + 1] == str(final + 1)
    gain = recipe()['control_overrides']['wheel_damping']
    assert argv.count('env.wheel_damping=' + str(gain)) == 1
    assert argv.count('env.robot.actuators.wheels.damping=' + str(gain)) == 1
    assert argv.count('--allow_physics_contract_change') == 1
    fixture = output / 'declared_contract_fixture'
    (fixture / 'params').mkdir(parents=True)
    old = yaml.safe_load((source.parent / 'params/control_contract.yaml').read_text())
    current = dict(old, fresh_weights=False, wheel_damping=gain,
                   feedforward_force=expected['feedforward_force'])
    contract_path = fixture / 'params/control_contract.yaml'
    contract_path.write_text(yaml.safe_dump(current))
    shutil.copy2(source.parent / 'params/precision_mode.yaml', fixture / 'params/precision_mode.yaml')
    validate_declared_contract(boundary.validate_contract, fixture,
                              fresh_weights=False, source_checkpoint=source)
    # This previously failed only after the entire mass-adaptation segment.
    # Prove the exact declared transition works before starting another PPO run.
    wrong = dict(current, leg_effort_limit=current['leg_effort_limit'] + 1.)
    contract_path.write_text(yaml.safe_dump(wrong))
    try:
        validate_declared_contract(boundary.validate_contract, fixture,
                                  fresh_weights=False, source_checkpoint=source)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('Unexpected effort-limit change escaped original contract checks')
    contract_path.write_text(yaml.safe_dump(current))
    result.update(saved_controller_and_reward_values_retained=False,
        saved_leg_PD_rewards_action_maps_and_exploration_retained=True,
        declared_physics_changes=recipe()['physics_changes'],
        declared_control_changes=recipe()['control_changes'],
        fifty_update_native_command=argv,
        exact_declared_contract_transition_prechecked=True,
        undeclared_effort_limit_change_rejected=True)
    (output / 'qualification.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        boundary_index = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:boundary_index])
        implementation.configure(args.recipe)
        implementation.native(args.observation_output, sys.argv[boundary_index + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:])
        implementation.configure(args.recipe)
        implementation.replay(remaining)
    elif mode == 'qualify':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        qualify(args.recipe.resolve(), args.output.resolve())
    elif mode == 'compare':
        parser.add_argument('directory', type=Path)
        parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        compare(args.directory.resolve(), args.through)
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        train(args.request.resolve())
    else:
        raise ValueError(mode)
else:
    sys.modules[__name__] = implementation
