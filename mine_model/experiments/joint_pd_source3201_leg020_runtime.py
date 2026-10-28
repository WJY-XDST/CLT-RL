"""Prospective source3201 leg-floor trial, gated by its completed diagnosis.

Saved initial std, ordinary plain27 PPO, all17 Adam states, actual adaptive LR,
rewards, commands, physical execution and the complete height range are retained.
"""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import joint_pd_source3001_yaw150_runtime as previous
import joint_pd_source2201_yaw025_runtime as shared_guards
import joint_pd_source200_leg05_runtime as guard_template
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = '17d80a11ee66ce76e20487387bb5dd92e8f01de05ad0a2b0feeb47ea49ad5cb9'
LEG_STD_FLOOR = .02
FLAG = '--repair_motion025_leg020'
KEY, base, criteria_fix = previous.KEY, previous.base, previous.criteria_fix
REWARD_KEY, REWARD_VALUE = previous.REWARD_KEY, previous.REWARD_VALUE
DECISION = ROOT / 'mine_model/results/joint_pd_source3201_leg020_trial_decision_20261019.json'


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source3201_motion025_yaw150_leg020_v1',
        minimum_leg_std=LEG_STD_FLOOR, minimum_wheel_std=.03,
        native_runtime_sha256=sha(SCRIPT), parent_yaw150_runtime_sha256=sha(previous.SCRIPT),
        exploration_trial_anchor_sha256=ANCHOR_SHA,
        authority='completed200_full27_full51_own_static_and_matched_frozen_noise_diagnosis')
    return result


def recipe(rewards):
    return previous.recipe(rewards)


def repair_ready():
    decision = json.loads(DECISION.read_text())
    static = json.loads((Path(decision['source_static_retention_output']) / 'static_retention.json').read_text())
    pair = json.loads((Path(decision['matched_sampling_output']) / 'paired_leg_sampling_diagnosis.json').read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 3201 or decision['source_all17_Adam_step'] != 64020
            or decision['planned_actual_updates'] != 200
            or decision['next_configuration_changes'] != dict(rewards={}, commands={},
                exploration={'minimum_leg_std': [.035, LEG_STD_FLOOR]}, controller={}, physics={})
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != ANCHOR_SHA
            or pair['checkpoint_sha256'] != ANCHOR_SHA or not pair['exact_same_Gaussian_draws']
            or not pair['bitwise_equal_initial_observations_and_means']
            or not decision['frozen_probe_is_not_learning_or_acceptance']
            or not decision['source_is_unaccepted_repair_branch']
            or decision['optimizer_reset'] or decision['std_reset']
            or decision['original_model200_reload'] or decision['faster_curriculum_authorized']
            or any(sha(path) != digest for path, digest in decision['source_pins'].items())):
        raise ValueError('Require the completed source3201 diagnosis and one measured exploration-floor decision')
    return decision


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_leg020_private_original_plain27_guards', shared_guards.__file__)
    family = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(family)
    family.profile, family.recipe = profile, recipe
    module = family.private_runtime()
    original_require, original_observe = module.require_source, module.observe_environment
    transition = source is not None
    if transition:
        previous.require_source(source)
        if sha(source) != ANCHOR_SHA or saved_updates(source) != 3201:
            raise ValueError('Only the exact complete3201 state starts this floor experiment')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload originalmodel200')
        if transition and sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        return original_require(path, initial_transfer=False)

    def observe(env, output):
        result = original_observe(env, output)
        from auto_train_mine import write_json
        write_json(Path(output) / 'leg020_runtime_evidence.json', dict(profile=profile(),
            only_exploration_floor_change={'minimum_leg_std': [.035, LEG_STD_FLOOR]},
            inherited_angular_width_patch_is_lineage_metadata=True,
            initial_saved_std_not_reset=True, all_six_std_entries_remain_trainable=True,
            reward_command_controller_physics_and_reset_configurations_unchanged=True,
            relative_error_target=.1, added_loss_or_backward=False, policy_accepted=False))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    transfer = transition or sha(source) == ANCHOR_SHA
    return private_runtime(source if transfer else None).require_source(source)


def install_leg_exploration(output):
    spec = importlib.util.spec_from_file_location('_leg020_original_projection', guard_template.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LEG_STD_FLOOR, module.profile = LEG_STD_FLOOR, profile
    tree = ast.parse(inspect.getsource(module.install_leg_exploration))
    before, edits = ast.dump(tree), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(
                '[INFO]: User-requested leg-only exploration floor .05;'):
            edits.append((node, node.value))
            node.value = node.value.replace('floor .05;', 'floor .02;')
    if len(edits) != 1:
        raise ValueError('Requalify the original projection logging adapter')
    changed = copy.deepcopy(tree)
    for node, value in edits:
        node.value = value
    if ast.dump(tree) != before:
        raise ValueError('Original exploration projection implementation changed')
    exec(compile(ast.fix_missing_locations(changed), guard_template.__file__, 'exec'), module.__dict__)
    return module.install_leg_exploration(output)


def native(output, arguments):
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    if (sha(source) == ANCHOR_SHA) != (FLAG in arguments):
        raise ValueError('Declare the exact first-profile floor transfer')
    scope = dict(shared_guards.__dict__, SCRIPT=SCRIPT, FLAG=FLAG, repair_ready=repair_ready,
        require_source=require_source, private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)
    function = types.FunctionType(shared_guards.native.__code__, scope,
        shared_guards.native.__name__, shared_guards.native.__defaults__, shared_guards.native.__closure__)
    return function(output, arguments)


def evaluator_tree():
    return previous.evaluator_tree()


def replay(arguments):
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source)
    if FLAG in arguments or '--overrides_json' in arguments:
        raise ValueError('Replay the exact saved configuration')
    tree, path = evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
        _install_quality_criteria=criteria_fix.install,
        _observe_source200=private_runtime().observe_environment))


def command(source, *args, **kwargs):
    argv, final, saved = previous.command(source, *args, **kwargs)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv, final, saved


def replay_command(source, output):
    argv = previous.replay_command(source, output)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    return argv


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'replay':
        replay(sys.argv[1:])
    elif mode == 'native':
        boundary = sys.argv.index('--')
        if sys.argv[1:boundary] != ['--observation-output', sys.argv[2]]:
            raise ValueError('Keep native observation CLI')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only native ordinary PPO and frozen replay')
