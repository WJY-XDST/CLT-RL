"""Single orientation-cost repair of the completed motion025/source1601.

Retain plain27 ActorCritic, four direct joint-angle targets, all17 Adam states,
saved adaptive LR/std, physics, height reward, reset distribution and motion
sampling. Only the existing orientation coefficient changes -40 to -80.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys

from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates
import source1201_orientation40
import joint_pd_source1401_motion025_runtime as previous
from evaluate_source200_stationary_height import evaluator_tree

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = 'fa7544e120906edec848df37765c46aaf92aa267492b3e172c570aacfdfb3180'
ORIENTATION = -80.
MOTION_SCALE = .25
FLAG = '--repair_motion025_orientation80'
KEY, base = previous.KEY, previous.base
DECISION = ROOT / 'mine_model/results/joint_pd_source1601_motion025_priority_repair_decision_20261017.json'


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source1601_motion025_orientation80_v1',
        native_runtime_sha256=sha(SCRIPT), previous_runtime_sha256=sha(previous.SCRIPT),
        orientation=ORIENTATION, repair_anchor_sha256=ANCHOR_SHA,
        authority='user_stability_first_repair_of_measured_low_speed_regression')
    return result


def recipe(rewards):
    return (rewards['orientation'] == ORIENTATION
            and previous.previous.recipe(dict(rewards, orientation=-40.)))


def repair_ready():
    evidence = json.loads(DECISION.read_text())
    if (evidence['source_checkpoint_sha256'] != ANCHOR_SHA
            or evidence['source_actual_updates'] != 1601
            or evidence['source_all17_Adam_step'] != 32020
            or evidence['planned_only_reward_change'] != {'orientation': [-40., ORIENTATION]}
            or not evidence['static_all_original_metrics_and_actual_legs_retained']
            or evidence['paired_low_speed_survivors'] != [51, 51]
            or evidence['paired_low_speed_resets'] != [0, 0]
            or evidence['low_speed_reverse_low_bodyP95_deg'] <= 2.5
            or any(sha(path) != digest for path, digest in evidence['source_pins'].items())):
        raise ValueError('Require the intact completed200/full27/paired51 priority diagnosis')
    return evidence


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_source1601_orientation80_private', base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Same qualified metadata-only sampler guards as the motion025 parent.
    previous._motion_guard(module, 'require_source', "cfg['commands']['training_motion_scale']")
    previous._motion_guard(module, 'observe_environment', 'cfg.commands.training_motion_scale')
    original_require, original_observe = module.require_source, module.observe_environment
    transition_sha = None
    if source is not None:
        _, cfg = previous.require_source(source)
        if (sha(source) != ANCHOR_SHA or saved_updates(source) != 1601
                or cfg['commands']['training_motion_scale'] != MOTION_SCALE
                or cfg['rewards']['orientation'] != -40.):
            raise ValueError('Only the exact reviewed motion025/source1601 starts this repair')
        transition_sha = ANCHOR_SHA

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload original model200 for this repair')
        if transition_sha is not None and sha(path) == transition_sha:
            return previous.require_source(path)
        saved, cfg = original_require(path, initial_transfer=False)
        if not recipe(cfg['rewards']):
            raise ValueError('Keep the exact orientation80/smooth10/standing50 recipe')
        return saved, cfg

    def observe(env, output):
        if not recipe(env.unwrapped.cfg.rewards.to_dict()):
            raise ValueError('Native rewards differ from the declared single coefficient repair')
        result = original_observe(env, output)
        previous.previous.previous.install_failure_capture(env.unwrapped, output)
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    path = previous.previous.previous.previous.__file__
    spec = importlib.util.spec_from_file_location('_source1601_unchanged_leg05_exploration', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.profile = profile
    return module.install_leg_exploration(output)


def native(output, arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    require_source(source, transition=transition)
    if [a for a in arguments if a.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=ActorCritic']:
        raise ValueError('Keep the original plain27 ActorCritic')
    private_runtime(source if transition else None).install_runner(output)
    private_runtime().install_environment_observer(output)
    install_leg_exploration(output)
    import train_joint_pd_observed as observed
    from joint_pd_review_interval200 import install_observer
    from auto_train_mine import write_json
    write_json(Path(output).parent / 'review_interval_runtime.json', install_observer(observed))
    observed.run(output, base.tires.arguments(arguments))


def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source, transition=transition)
    if transition:
        if '--overrides_json' not in arguments:
            raise ValueError('Frozen reward transfer must explicitly declare its one override')
        overrides = json.loads(Path(arguments[arguments.index('--overrides_json') + 1]).read_text())
        if overrides != {'rewards.orientation': ORIENTATION}:
            raise ValueError('Only the existing orientation coefficient changes')
    elif '--overrides_json' in arguments:
        raise ValueError('Native descendants replay their exact saved contract')
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    tree, original = evaluator_tree()
    exec(compile(tree, str(original), 'exec'), dict(__name__='__main__', __file__=str(original),
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
            raise ValueError('Keep the exact native observation output contract')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only declared native learning or frozen replay')
