"""Prospective single exploration-floor trial; no reward/control changes.

Native use requires the completed matched diagnosis and explicit source-bound
decision. Initial saved std and all17 Adam states remain bitwise unchanged.
"""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import joint_pd_source2201_yaw025_runtime as previous
import joint_pd_source200_leg05_runtime as guard_template
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = 'f0a15a94bd7772f34ed0f0acc202c9af5f3355642be946102ce7a5174d49faba'
LEG_STD_FLOOR = .035
FLAG = '--repair_motion025_leg035'
KEY, base, criteria_fix = previous.KEY, previous.base, previous.criteria_fix
REWARD_KEY, REWARD_VALUE = previous.REWARD_KEY, previous.REWARD_VALUE
DECISION = ROOT / 'mine_model/results/joint_pd_source2601_leg035_trial_decision_20261019.json'


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source2601_motion025_yaw025_leg035_v1',
        minimum_leg_std=LEG_STD_FLOOR, minimum_wheel_std=.03,
        native_runtime_sha256=sha(SCRIPT), parent_yaw025_runtime_sha256=sha(previous.SCRIPT),
        exploration_trial_anchor_sha256=ANCHOR_SHA,
        authority='matched_frozen_noise_diagnosis_and_one200_update_trial_only')
    return result


def recipe(rewards):
    return previous.recipe(rewards)


def repair_ready():
    decision = json.loads(DECISION.read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 2601 or decision['source_all17_Adam_step'] != 52020
            or decision['planned_actual_updates'] != 200
            or decision['next_configuration_changes'] != dict(rewards={}, commands={},
                exploration={'minimum_leg_std': [.05, LEG_STD_FLOOR]}, controller={}, physics={})
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or not decision['current_checkpoint_static_retention_verified']
            or not decision['frozen_probe_is_not_learning_or_acceptance']
            or decision['optimizer_reset'] or decision['std_reset']
            or decision['original_model200_reload'] or decision['faster_curriculum_authorized']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require the source2601 matched-diagnosis200-update decision')
    return decision


def private_runtime(source=None):
    module = previous.private_runtime()
    original_require = module.require_source

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Do not reload original model200')
        if sha(path) == ANCHOR_SHA:
            saved, cfg = previous.require_source(path)
            if saved_updates(path) != 2601:
                raise ValueError('Exploration transfer source clock differs')
            return saved, cfg
        return original_require(path, initial_transfer=False)

    module.profile, module.require_source = profile, require
    return module


def require_source(source, transition=False):
    return private_runtime().require_source(source)


def install_leg_exploration(output):
    spec = importlib.util.spec_from_file_location('_leg035_same_projection_guard', guard_template.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.LEG_STD_FLOOR, module.profile = LEG_STD_FLOOR, profile
    # Only correct the inherited human-readable line. The projection, retained
    # parameter identity, and before/after full optimizer checks are identical.
    tree = ast.parse(inspect.getsource(module.install_leg_exploration))
    before, edits = ast.dump(tree), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('[INFO]: User-requested leg-only exploration floor .05;'):
            edits.append((node, node.value))
            node.value = node.value.replace('floor .05;', 'floor .035;')
    if len(edits) != 1:
        raise ValueError('Requalify logging-only exploration guard adaptation')
    changed = copy.deepcopy(tree)
    for node, value in edits:
        node.value = value
    if ast.dump(tree) != before:
        raise ValueError('Exploration projection implementation changed')
    exec(compile(ast.fix_missing_locations(changed), guard_template.__file__, 'exec'), module.__dict__)
    return module.install_leg_exploration(output)


def native(output, arguments):
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    if (sha(source) == ANCHOR_SHA) != (FLAG in arguments):
        raise ValueError('Declare the exact first-profile transfer at native load')
    if sha(source) == ANCHOR_SHA:
        repair_ready()
    scope = dict(previous.__dict__, SCRIPT=SCRIPT, FLAG=FLAG, repair_ready=repair_ready,
        require_source=require_source, private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)
    function = types.FunctionType(previous.native.__code__, scope,
        previous.native.__name__, previous.native.__defaults__, previous.native.__closure__)
    return function(output, arguments)


def evaluator_tree():
    return previous.evaluator_tree()


def replay(arguments):
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source)
    if FLAG in arguments or '--overrides_json' in arguments:
        raise ValueError('Frozen replay uses the exact saved configuration')
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
            raise ValueError('Keep native observer CLI')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only native PPO and frozen replay')
