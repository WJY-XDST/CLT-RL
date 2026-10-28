"""One reviewed sampler expansion; retain the complete source3601 plain PPO state."""
import ast
import copy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import joint_pd_source3201_leg020_runtime as previous
import joint_pd_source2201_yaw025_runtime as shared
import source1201_orientation40
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = '9829e9eed94c8d47f1086c0028023efac135c8540094783d18168159232716e1'
MOTION_SCALE = .375
FLAG = '--begin_verified_motion0375'
KEY, base, criteria_fix = previous.KEY, previous.base, previous.criteria_fix
DECISION = ROOT / 'mine_model/results/joint_pd_source3601_motion0375_decision_20261019.json'
DELTA = dict(rewards={}, commands={'training_motion_scale': [.25, MOTION_SCALE]},
    exploration={}, controller={}, physics={})


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source3601_motion0375_yaw150_leg020_v1',
        training_motion_scale=MOTION_SCALE, native_runtime_sha256=sha(SCRIPT),
        parent_leg020_runtime_sha256=sha(previous.SCRIPT), motion_expansion_anchor_sha256=ANCHOR_SHA,
        authority='own_complete_static_retention_and_frozen_wider_51_physical_review')
    return result


def recipe(rewards):
    return previous.recipe(rewards)


def repair_ready():
    decision = json.loads(DECISION.read_text())
    static = json.loads(Path(decision['source_static_retention_report']).read_text())
    scout = json.loads(Path(decision['wider_frozen_report']).read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 3601 or decision['source_all17_Adam_step'] != 72020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not decision['source_is_unaccepted_repair_branch']
            or not decision['gradual_command_expansion_selected']
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != ANCHOR_SHA
            or scout['checkpoint_sha256'] != ANCHOR_SHA or scout['tested_command_scale'] != MOTION_SCALE
            or scout['total_trials'] != 51 or scout['survivors'] != 51 or scout['resets']
            or scout['training_modified'] or scout['optimizer_updated']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require exact source3601 static/wider/learning review; expansion is not acceptance')
    return decision


def metadata_guards(module, name, declarations):
    function = getattr(module, name)
    tree = ast.parse(inspect.getsource(function))
    original = copy.deepcopy(tree)
    edits = []
    for expression, old, new in declarations:
        expected = ast.dump(ast.parse(expression, mode='eval').body)
        matches = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
            and len(n.ops) == 1 and isinstance(n.ops[0], ast.NotEq)
            and ast.dump(n.left) == expected and len(n.comparators) == 1
            and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value == old]
        if len(matches) != 1:
            raise ValueError('Requalify the declared metadata guard: ' + name + '/' + expression)
        node = matches[0].comparators[0]
        edits.append((node, old))
        node.value = new
    modified = copy.deepcopy(tree)
    for node, old in edits:
        node.value = old
    if ast.dump(tree) != ast.dump(original):
        raise ValueError('Code changed outside one metadata comparison')
    exec(compile(ast.fix_missing_locations(modified), inspect.getsourcefile(function), 'exec'), module.__dict__)


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_motion0375_private_plain27_guards', base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata_guards(module, 'require_source', [("cfg['commands']['training_motion_scale']", 0., MOTION_SCALE)])
    metadata_guards(module, 'observe_environment', [
        ('cfg.commands.training_motion_scale', 0., MOTION_SCALE), ('cfg.rewards.base_height', 2., 3.)])
    original_require, original_observe = module.require_source, module.observe_environment
    transition = source is not None
    if transition:
        _, cfg = previous.require_source(source)
        if (sha(source) != ANCHOR_SHA or saved_updates(source) != 3601
                or cfg['commands']['training_motion_scale'] != .25):
            raise ValueError('Only the exact complete3601 state starts this sampler expansion')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never repeat the original model200 rollback')
        if transition and sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        saved, cfg = original_require(path, initial_transfer=False)
        if not recipe(cfg['rewards']):
            raise ValueError('Retain the source3601 reward recipe')
        return saved, cfg

    def observe(env, output):
        raw = env.unwrapped
        if (not recipe(raw.cfg.rewards.to_dict()) or raw._tracking_goal_criteria is None
                or raw._tracking_goal_criteria['relative_error'] != .1):
            raise ValueError('Actual reward recipe or original10% criteria differ')
        result = original_observe(env, output)
        source1201_orientation40.previous.install_failure_capture(raw, output)
        from auto_train_mine import write_json
        write_json(Path(output) / 'motion0375_runtime_evidence.json', dict(profile=profile(),
            only_command_change=DELTA['commands'], reward_exploration_physics_reset_and_control_unchanged=True,
            saved_std_not_reset=True, all_six_std_entries_remain_trainable=True,
            quality_criteria_runtime_proof=type(raw)._quality_criteria_runtime_proof,
            relative_error_target=.1, added_loss_or_backward=False, policy_accepted=False))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    function = previous.install_leg_exploration
    retained = types.FunctionType(function.__code__, dict(previous.__dict__, profile=profile),
        function.__name__, function.__defaults__, function.__closure__)
    return retained(output)


def native(output, arguments):
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    if (sha(source) == ANCHOR_SHA) != (FLAG in arguments):
        raise ValueError('Declare exactly the reviewed first-profile source3601 transfer')
    scope = dict(shared.__dict__, SCRIPT=SCRIPT, FLAG=FLAG, repair_ready=repair_ready,
        require_source=require_source, private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)
    retained = types.FunctionType(shared.native.__code__, scope, shared.native.__name__,
        shared.native.__defaults__, shared.native.__closure__)
    return retained(output, arguments)


def evaluator_tree():
    return previous.evaluator_tree()


def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source, transition=transition)
    if transition:
        if ('--overrides_json' not in arguments or json.loads(Path(
                arguments[arguments.index('--overrides_json') + 1]).read_text())
                != {'commands.training_motion_scale': MOTION_SCALE}):
            raise ValueError('Frozen transfer changes only the training command amplitude')
    elif '--overrides_json' in arguments:
        raise ValueError('Descendants replay their saved configuration')
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
            raise ValueError('Keep the native observation output contract')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only ordinary PPO and frozen replay')
