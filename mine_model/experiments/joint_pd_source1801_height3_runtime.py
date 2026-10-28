"""One height reward coefficient change from the reviewed full source1801.

Preserve direct motor-angle actions, plain27 PPO, full17 Adam/std/LR,
orientation40, velocity025, full height range and physical execution.
"""
import ast
import inspect
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
import source1201_orientation40
import joint_pd_stationary_stage as stage
import joint_pd_source1601_velocity025_runtime as previous
from evaluate_source200_stationary_height import evaluator_tree

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = '58567814cfa677c823a868a8e6fb8078dacf07b77b92bbf62638f3a63b19c5e0'
HEIGHT_WEIGHT = 3.
VELOCITY_SIGMA = .025
FLAG = '--repair_motion025_height3'
KEY, base = previous.KEY, previous.base
DECISION = ROOT/'mine_model/results/joint_pd_source1801_height3_priority_decision_20261018.json'
SOURCE_OUTPUT = ROOT/'mine_model/results/joint_pd_optimization_20261018_144942_170607_source1601_velocity025_200'


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source1801_motion025_height3_v1',
        native_runtime_sha256=sha(SCRIPT), previous_runtime_sha256=sha(previous.SCRIPT),
        height_reward_coefficient=HEIGHT_WEIGHT, height_repair_anchor_sha256=ANCHOR_SHA,
        authority='completed200_full27_full51_and_midpoint100_height_regression_review')
    return result


def recipe(rewards):
    return (previous.recipe(dict(rewards, base_height=2.))
        and rewards['base_height'] == HEIGHT_WEIGHT)


def repair_ready():
    decision = json.loads(DECISION.read_text())
    report = json.loads((SOURCE_OUTPUT/'replay/assessment.json').read_text())
    proof = json.loads((SOURCE_OUTPUT/'final_checkpoint_all200_saved_clocks_readonly.json').read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 1801 or decision['source_all17_Adam_step'] != 36020
            or decision['planned_only_reward_change_from_selected_source'] != {'base_height':[2.,HEIGHT_WEIGHT]}
            or not decision['source_is_unaccepted_repair_branch']
            or not decision['source_full200_full27_full51_verified']
            or decision['source_canonical_survivors'] != 27 or decision['source_canonical_resets'] != 0
            or decision['source_paired_passed_trials'] != 9
            or decision['midpoint_not_selected']['canonical_survivors'] != 24
            or decision['midpoint_not_selected']['canonical_resets'] != 36
            or report['checkpoint_sha256'] != ANCHOR_SHA or report['survivors'] != 27 or report['resets']
            or proof['checkpoint_sha256'] != ANCHOR_SHA or proof['all17_Adam_step'] != 36020
            or not proof['all200_saved_checkpoints_finite_and_clocked']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or any(sha(p) != digest for p,digest in decision['source_pins'].items())):
        raise ValueError('Require the full reviewed end-state; this source is for repair, never promotion')
    return decision


def _height_environment_guard(module):
    tree = ast.parse(inspect.getsource(module.observe_environment))
    original = ast.dump(tree)
    changes = []
    # Compile once: inspect.getsource cannot recover the original function
    # after a preceding AST compilation resets its source line numbers.
    for expression, before, after in (
            ('cfg.commands.training_motion_scale', 0., .25),
            ('cfg.rewards.base_height', 2., HEIGHT_WEIGHT)):
        expected = ast.dump(ast.parse(expression, mode='eval').body)
        nodes = [n for n in ast.walk(tree) if isinstance(n, ast.Compare)
            and ast.dump(n.left) == expected and len(n.ops) == 1 and isinstance(n.ops[0], ast.NotEq)
            and len(n.comparators) == 1 and isinstance(n.comparators[0], ast.Constant)
            and n.comparators[0].value == before]
        if len(nodes) != 1:
            raise ValueError('Requalify the exact environment metadata guard: ' + expression)
        changes.append((nodes[0].comparators[0], before))
        nodes[0].comparators[0].value = after
    modified = copy.deepcopy(tree)
    for node, value in changes:
        node.value = value
    if ast.dump(tree) != original:
        raise ValueError('Physical observer changed beyond height reward metadata')
    exec(compile(ast.fix_missing_locations(modified), module.__file__, 'exec'), module.__dict__)


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_source1801_height3_private',base.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Only these metadata guards change; base height cost stays exactly zero.
    previous.previous._motion_guard(module,'require_source',"cfg['commands']['training_motion_scale']")
    _height_environment_guard(module)
    require_original, observe_original = module.require_source, module.observe_environment
    transition = False
    if source is not None:
        _, cfg = previous.require_source(source)
        if (sha(source) != ANCHOR_SHA or saved_updates(source) != 1801
                or cfg['rewards']['base_height'] != 2.
                or not recipe(dict(cfg['rewards'],base_height=HEIGHT_WEIGHT))):
            raise ValueError('Only exact source1801 starts this height-weight repair')
        transition = True

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload the original model200 for this repair')
        if transition and sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        saved, cfg = require_original(path,initial_transfer=False)
        if not recipe(cfg['rewards']):
            raise ValueError('Keep the exact velocity025/height3/orientation40 recipe')
        return saved, cfg

    def observe(env, output):
        if not recipe(env.unwrapped.cfg.rewards.to_dict()):
            raise ValueError('Actual rewards differ from the declared velocity-width repair')
        result = observe_original(env,output)
        previous.previous.previous.previous.install_failure_capture(env.unwrapped,output)
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    function = previous.install_leg_exploration
    retained = types.FunctionType(function.__code__,dict(function.__globals__,profile=profile),
        function.__name__,function.__defaults__,function.__closure__)
    return retained(output)


def native(output, arguments):
    arguments = list(arguments); transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG); repair_ready()
    source = Path(arguments[arguments.index('--checkpoint_path')+1])
    require_source(source,transition=transition)
    if [a for a in arguments if a.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=ActorCritic']:
        raise ValueError('Keep plain27 ActorCritic and ordinary PPO')
    private_runtime(source if transition else None).install_runner(output)
    private_runtime().install_environment_observer(output)
    install_leg_exploration(output)
    import train_joint_pd_observed as observed
    from joint_pd_review_interval200 import install_observer
    from auto_train_mine import write_json
    write_json(Path(output).parent/'review_interval_runtime.json',install_observer(observed))
    observed.run(output,base.tires.arguments(arguments))


def replay(arguments):
    arguments = list(arguments); transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
    source = Path(arguments[arguments.index('--checkpoint')+1])
    require_source(source,transition=transition)
    if transition:
        if '--overrides_json' not in arguments:
            raise ValueError('Frozen transfer must declare the one height-weight override')
        delta = json.loads(Path(arguments[arguments.index('--overrides_json')+1]).read_text())
        if delta != {'rewards.base_height':HEIGHT_WEIGHT}:
            raise ValueError('Only the positive height coefficient changes')
    elif '--overrides_json' in arguments:
        raise ValueError('Descendants replay their exact saved configuration')
    sys.argv = [str(SCRIPT),*base.tires.arguments(arguments)]
    tree, original = evaluator_tree()
    exec(compile(tree,str(original),'exec'),dict(__name__='__main__',__file__=str(original),
        _observe_source200=private_runtime().observe_environment))


def command(source, *args, **kwargs):
    argv, final, saved = previous.command(source,*args,**kwargs)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv, final, saved


def replay_command(source, output):
    argv = previous.replay_command(source,output)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    return argv


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'replay':
        replay(sys.argv[1:])
    elif mode == 'native':
        boundary = sys.argv.index('--')
        if sys.argv[1:boundary] != ['--observation-output',sys.argv[2]]:
            raise ValueError('Keep the exact native observer output contract')
        native(Path(sys.argv[2]),sys.argv[boundary+1:])
    else:
        raise ValueError('Only native learning or frozen replay')
