"""Declared chassis-mass transfer with unchanged ordinary direct-joint PPO.

Only the versioned USD and half-vehicle gravity force change. Every command
uses the same physics, PD gains, rewards and exploration. Initial model,
six std values, all17 Adam states and actual adaptive LR remain intact.
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
base_recipe = saved_namespace['old_recipe']
old_expected = implementation.expected_configuration
old_profile = implementation.profile_for_count
old_command = implementation.command
old_private = implementation.private_for_count
old_evaluator = implementation.evaluator_tree


def recipe():
    value = base_recipe()
    assert value['saved_bundle_ordinary_sampler_transfer']
    assert not value['reward_changes'] and not value['control_changes']
    assert not value['case_specific_training_logic']
    assert value['source_exploration'] == value['exploration']
    changes = value['physics_changes']
    assert set(changes) == {'robot.spawn.usd_path', 'feedforward_force'}
    target = value['physics_target']
    assert changes['robot.spawn.usd_path'][1] == target['asset']
    assert implementation.sha(target['asset']) == target['asset_sha256']
    assert changes['feedforward_force'][1] == target['vehicle_mass_kg'] * 9.81 / 2.
    assert value['effective_change_relative_to_source'] == {'physics': changes}
    return value


def expected_configuration(cfg, count):
    result = old_expected(cfg, count)
    changes = recipe()['physics_changes']
    assert cfg['robot']['spawn']['usd_path'] == changes['robot.spawn.usd_path'][0]
    assert cfg['feedforward_force'] == changes['feedforward_force'][0]
    result['robot']['spawn']['usd_path'] = changes['robot.spawn.usd_path'][1]
    result['feedforward_force'] = changes['feedforward_force'][1]
    return result


def profile_for_count(count):
    result = old_profile(count)
    result.update(profile='plain27_saved_calibrated_body_mass_v1',
        calibrated_mass_runtime_sha256=implementation.sha(SCRIPT),
        declared_physics_changes=recipe()['physics_changes'],
        nominal_vehicle_mass_kg=recipe()['physics_target']['vehicle_mass_kg'],
        saved_controller_and_reward_values_retained=False,
        saved_PD_gains_rewards_and_action_maps_retained=True,
        owner_corrected_mass_and_nominal_gravity_force_updated=True,
        same_physics_for_every_command_and_environment=True)
    return result


def command(source, cfg, label, updates, commands, stop_file):
    argv, final, saved = old_command(source, cfg, label, updates, commands, stop_file)
    import json
    for key, (_, value) in recipe()['physics_changes'].items():
        prefix = 'env.' + key + '='
        matches = [i for i, item in enumerate(argv) if item.startswith(prefix)]
        assert len(matches) <= 1, key
        argument = prefix + json.dumps(value)
        if matches:
            argv[matches[0]] = argument
        else:
            argv.append(argument)
    assert '--allow_physics_contract_change' not in argv
    argv.append('--allow_physics_contract_change')
    return argv, final, saved


def private_for_count(count, selected_profile=None, *, is_replay=False):
    # Adapt only the observer's expected USD and total-mass checks. Preserve
    # every native action, reward, loader and optimizer implementation.
    tree = ast.parse(inspect.getsource(old_private))
    original = ast.dump(tree)
    replaced = []
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
                    replaced.append((owner, key, entry))
                    replacement = ast.parse(expression, mode='eval').body
                    if isinstance(owner, list):
                        owner[key] = replacement
                    else:
                        setattr(owner, key, replacement)
    assert len(replaced) == 2
    modified = copy.deepcopy(tree)
    for owner, key, value in replaced:
        if isinstance(owner, list):
            owner[key] = value
        else:
            setattr(owner, key, value)
    assert ast.dump(tree) == original
    scope = dict(old_private.__globals__, recipe=recipe,
                 expected_configuration=expected_configuration,
                 profile_for_count=profile_for_count, SCRIPT=SCRIPT)
    exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), 'exec'), scope)
    return scope['private_for_count'](count, selected_profile, is_replay=is_replay)


def evaluator_tree():
    tree, path, proof = old_evaluator()
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    positions = [i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'height' for t in node.targets)]
    assert len(positions) == 1
    original = ast.dump(tree)
    spec = recipe()['physics_changes']
    import json
    inserted = ast.parse('cfg.robot.spawn.usd_path = ' + repr(spec['robot.spawn.usd_path'][1])
                         + '\ncfg.feedforward_force = ' + repr(spec['feedforward_force'][1])).body
    at = positions[0]
    main.body[at:at] = inserted
    modified = copy.deepcopy(tree)
    del main.body[at:at + len(inserted)]
    assert ast.dump(tree) == original
    proof.update(declared_mass_and_gravity_only=spec,
                 no_case_specific_mass_or_feedback=True,
                 original_replay_math_unchanged=True)
    return ast.fix_missing_locations(modified), path, proof


saved_namespace['recipe'] = recipe
implementation.SCRIPT = SCRIPT
implementation.recipe = recipe
implementation.expected_configuration = expected_configuration
implementation.profile_for_count = profile_for_count
implementation.command = command
implementation.private_for_count = private_for_count
implementation.evaluator_tree = evaluator_tree
sys.modules['joint_pd_saved_mass_runtime'] = implementation


def compare(output, through):
    import review_saved_joint_pd_segment as segment
    old_normalize = implementation.normalize_saved_bundle

    def expected(cfg):
        result = old_normalize(cfg)
        changes = recipe()['physics_changes']
        assert result['robot']['spawn']['usd_path'] == changes['robot.spawn.usd_path'][0]
        assert result['feedforward_force'] == changes['feedforward_force'][0]
        result['robot']['spawn']['usd_path'] = changes['robot.spawn.usd_path'][1]
        result['feedforward_force'] = changes['feedforward_force'][1]
        return result

    implementation.normalize_saved_bundle = expected
    try:
        segment.compare(output, through)
    finally:
        implementation.normalize_saved_bundle = old_normalize


def train(request):
    import review_saved_joint_pd_segment as segment
    api = segment.original
    old_write, old_health, old_processes = api.write_json, api.health50, segment.processes

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            value['next_configuration_changes']['physics'] = recipe()['physics_changes']
            value['only_physics_change'] = recipe()['physics_changes']
            value['only_runtime_sampling_change'] = {}
            value['owner_corrected_mass_adaptation'] = True
        return old_write(path, value)

    def processes(delegate):
        base = old_processes(delegate)

        class PhysicsReaders(type(base)):
            def routed(self, argv):
                argv = super().routed(argv)
                if (isinstance(argv, (list, tuple)) and len(argv) > 2
                        and Path(argv[1]).name == segment.SCRIPT.name and argv[2] == 'compare'):
                    return [argv[0], str(SCRIPT), 'compare', '--recipe',
                            str(implementation.RECIPE_PATH), *argv[3:]]
                return argv
        return PhysicsReaders(delegate)

    api.write_json, api.health50, segment.processes = write, segment.health, processes
    try:
        return api.run(request)
    finally:
        api.write_json, api.health50, segment.processes = old_write, old_health, old_processes


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
    elif mode == 'qualify':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        from continue_saved_joint_pd_bundle import qualify
        qualify(args.recipe.resolve(), args.output.resolve())
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        train(args.request.resolve())
    elif mode == 'compare':
        parser.add_argument('directory', type=Path)
        parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:])
        implementation.configure(args.recipe)
        compare(args.directory.resolve(), args.through)
    else:
        raise ValueError(mode)
else:
    sys.modules[__name__] = implementation
