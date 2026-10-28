"""One declared global positive linear-bonus width over the pinned source runtime.

The source runtime retains ordinary PPO, complete state restoration, global
controller/floor declarations, and original replay readers. This adapter only
sets the existing tracking_sigma variance. No command or case is inspected.
"""
import argparse
import ast
import copy
import json
import math
from pathlib import Path
import sys

import joint_pd_uniform_source_refinement_runtime as base

SCRIPT = Path(__file__).resolve()
ORIGINAL_CONFIGURE = base.configure
ORIGINAL_EXPECTED = base.expected_configuration
ORIGINAL_PROFILE = base.profile_for_count
ORIGINAL_IMMEDIATE = base.immediate_changes
ORIGINAL_COMMAND = base.command
ORIGINAL_EVALUATOR = base.evaluator_tree
ORIGINAL_QUALIFY = base.qualify
BASE_SCRIPT = base.SCRIPT
_path = None
_changes = None


def __getattr__(name):
    return getattr(base, name)


def configure(path):
    global _path, _changes
    path = Path(path).resolve()
    if _path is not None:
        assert path == _path, 'One immutable recipe per process'
        return base.recipe()
    specification = json.loads(path.read_text())['uniform_source_refinement']
    changes = specification['global_existing_reward_changes']
    assert set(changes) == {'tracking_sigma'}
    pair = changes['tracking_sigma']
    assert len(pair) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in pair)
    assert 0. < pair[1] < pair[0]
    ORIGINAL_CONFIGURE(path)
    assert base.baseline()[2]['rewards']['tracking_sigma'] == pair[0]
    _path, _changes = path, changes
    namespaces = list({id(v): v for v in [base.__dict__, *base.NAMESPACES]}.values())
    for namespace in namespaces:
        namespace.update(SCRIPT=SCRIPT, expected_configuration=expected_configuration,
                         profile_for_count=profile_for_count)
    base.configure = configure
    base.immediate_changes = immediate_changes
    base.command = command
    base.evaluator_tree = evaluator_tree
    base.parent.configure = configure
    base.parent.command = command
    base.parent.evaluator_tree = evaluator_tree
    return base.recipe()


def expected_configuration(cfg, count):
    normalized = copy.deepcopy(cfg)
    for name, pair in _changes.items():
        assert normalized['rewards'][name] in pair
        normalized['rewards'][name] = pair[0]
    result = ORIGINAL_EXPECTED(normalized, count)
    for name, pair in _changes.items():
        assert result['rewards'][name] == pair[0]
        result['rewards'][name] = pair[1]
    return result


def immediate_changes(cfg, source_floors=None):
    result = ORIGINAL_IMMEDIATE(cfg, source_floors)
    target = expected_configuration(cfg, 8192)
    result['rewards'] = {name: [cfg['rewards'][name], target['rewards'][name]]
                         for name in _changes if cfg['rewards'][name] != target['rewards'][name]}
    return result


def profile_for_count(count):
    result = ORIGINAL_PROFILE(count)
    result.update(profile='plain27_uniform_saved_source_linear_bonus_width_refinement_v1',
        global_reward_changes=copy.deepcopy(immediate_changes(base.baseline()[2])['rewards']),
        uniform_reward_overrides={**result['uniform_reward_overrides'],
                                  **{name: pair[1] for name, pair in _changes.items()}},
        source_physics_reward_commands_leg_PD_and_action_maps_retained=False,
        source_physics_commands_leg_PD_and_action_maps_retained=True,
        same_global_linear_bonus_width_for_every_command_and_group=True,
        global_linear_bonus_width_base_runtime_sha256=base.sha(BASE_SCRIPT))
    return result


def command(source, cfg, label, updates, commands, stop_file):
    argv, final, saved = ORIGINAL_COMMAND(source, cfg, label, updates, commands, stop_file)
    for name, pair in _changes.items():
        prefix = 'env.rewards.' + name + '='
        matches = [i for i, value in enumerate(argv) if value.startswith(prefix)]
        assert len(matches) == 1
        argv[matches[0]] = prefix + str(pair[1])
    return argv, final, saved


def evaluator_tree():
    tree, source, proof = ORIGINAL_EVALUATOR()
    original = ast.dump(tree)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    indices = [i for i, node in enumerate(main.body) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == 'height' for target in node.targets)]
    assert len(indices) == 1
    inserted = ast.parse('\n'.join('cfg.rewards.' + name + ' = ' + repr(pair[1])
                                   for name, pair in _changes.items())).body
    at = indices[0]
    main.body[at:at] = inserted
    modified = copy.deepcopy(tree)
    del main.body[at:at + len(inserted)]
    assert ast.dump(tree) == original
    proof.update(global_existing_linear_bonus_width_changes=copy.deepcopy(_changes),
                 only_added_replay_config_assignments=True,
                 original_numeric_readers_and_thresholds_unchanged=True)
    return ast.fix_missing_locations(modified), source, proof


def qualify(path, output):
    configure(path)
    result = ORIGINAL_QUALIFY(path, output)
    result.update(global_existing_linear_bonus_width_changes=copy.deepcopy(_changes),
                  same_global_positive_width_for_all_commands=True,
                  base_runtime_sha256=base.sha(BASE_SCRIPT),
                  parameter_change_selected=False, training_launched=False)
    result['source_pins'][str(BASE_SCRIPT)] = base.sha(BASE_SCRIPT)
    (output / 'qualification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return result


def train(request_path):
    # The original source runtime requires an explicit decision bound to its
    # completed735 review, true native source proof, and frozen controller test.
    request = json.loads(Path(request_path).read_text())
    configure(request['recipe'])
    return base.train(request_path)


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    if mode == 'native':
        parser.add_argument('--observation-output', type=Path, required=True)
        at = sys.argv.index('--')
        args = parser.parse_args(sys.argv[1:at]); configure(args.recipe)
        base.native(args.observation_output, sys.argv[at + 1:])
    elif mode == 'replay':
        args, remaining = parser.parse_known_args(sys.argv[1:]); configure(args.recipe)
        base.replay(remaining)
    elif mode == 'qualify':
        parser.add_argument('--output', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:]); result = qualify(args.recipe.resolve(), args.output.resolve())
        print(json.dumps(dict(qualified=result['qualified'], checkpoint_sha256=result['checkpoint_sha256'], actual_ppo_updates=result['actual_ppo_updates'], training_launched=False)))
    elif mode == 'compare':
        parser.add_argument('directory', type=Path); parser.add_argument('--through', type=int, required=True)
        args = parser.parse_args(sys.argv[1:]); configure(args.recipe); base.compare(args.directory.resolve(), args.through)
    elif mode == 'train':
        parser.add_argument('--request', type=Path, required=True)
        args = parser.parse_args(sys.argv[1:]); configure(args.recipe); train(args.request.resolve())
    else:
        raise ValueError(mode)
