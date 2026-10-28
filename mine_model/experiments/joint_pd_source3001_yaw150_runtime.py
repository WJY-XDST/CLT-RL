"""One existing angular-reward coefficient change after source3001 review.

Retain plain27 PPO, all17 Adam states, actual adaptive LR, saved Gaussian
std, direct-joint execution and the complete height range. No new reward term.
"""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import joint_pd_source2801_velocity010_runtime as previous
import joint_pd_source2201_yaw025_runtime as shared_guards
import joint_pd_source2601_leg035_runtime as floor_runtime
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = 'f0fa511ebf2ef31b35f4ad311ee67468acc50e6fc0fd216e337ae90c1f7a8a01'
REWARD_KEY, OLD_VALUE, REWARD_VALUE = 'tracking_ang_vel', 1., 1.5
FLAG = '--repair_motion025_yaw150'
KEY, base, criteria_fix = previous.KEY, previous.base, previous.criteria_fix
DECISION = ROOT / 'mine_model/results/joint_pd_source3001_yaw150_trial_decision_20261019.json'


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source3001_motion025_leg035_velocity010_yaw150_v1',
        tracking_ang_vel=REWARD_VALUE, native_runtime_sha256=sha(SCRIPT),
        parent_velocity010_runtime_sha256=sha(previous.SCRIPT),
        yaw_weight_trial_anchor_sha256=ANCHOR_SHA,
        authority='completed200_full27_full51_static_retention_and_signed_yaw_error_budget')
    return result


def recipe(rewards):
    return (previous.recipe(dict(rewards, tracking_ang_vel=OLD_VALUE))
            and rewards[REWARD_KEY] == REWARD_VALUE)


def repair_ready():
    decision = json.loads(DECISION.read_text())
    static = json.loads((Path(decision['static_retention_output']) / 'static_retention.json').read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 3001
            or decision['source_all17_Adam_step'] != 60020
            or decision['planned_actual_updates'] != 200
            or decision['next_configuration_changes'] != dict(rewards={REWARD_KEY: [OLD_VALUE, REWARD_VALUE]},
                commands={}, exploration={}, controller={}, physics={})
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != ANCHOR_SHA
            or not decision['source_full27_stability_regression_preserved']
            or not decision['source_is_unaccepted_repair_branch']
            or decision['optimizer_reset'] or decision['std_reset']
            or decision['original_model200_reload'] or decision['faster_curriculum_authorized']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require source3001 own complete review and one angular-weight trial decision')
    return decision


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_yaw150_private_plain27_guards', shared_guards.__file__)
    family = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(family)
    family.profile, family.recipe = profile, recipe
    module = family.private_runtime()
    original_require, original_observe = module.require_source, module.observe_environment
    transition = source is not None
    if transition:
        _, cfg = previous.require_source(source)
        if (sha(source) != ANCHOR_SHA or saved_updates(source) != 3001
                or cfg['rewards'][REWARD_KEY] != OLD_VALUE):
            raise ValueError('Only the exact completed source3001 crosses the first-profile boundary')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload originalmodel200')
        if transition and sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        return original_require(path, initial_transfer=False)

    def observe(env, output):
        if not recipe(env.unwrapped.cfg.rewards.to_dict()):
            raise ValueError('Actual rewards differ beyond the declared angular coefficient')
        result = original_observe(env, output)
        from auto_train_mine import write_json
        write_json(Path(output) / 'angular_weight_runtime_evidence.json', dict(
            profile=profile(), only_reward_configuration_change={REWARD_KEY: [OLD_VALUE, REWARD_VALUE]},
            inherited_angular_width_patch_is_lineage_metadata=True,
            unchanged_tracking_sigma=.01, unchanged_tracking_sigma_ang=.025,
            relative_error_target=.1, exploration_floors_unchanged=True,
            physical_execution_unchanged=True, added_loss_or_backward=False, policy_accepted=False))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    function = floor_runtime.install_leg_exploration
    retained = types.FunctionType(function.__code__, dict(floor_runtime.__dict__, profile=profile),
        function.__name__, function.__defaults__, function.__closure__)
    return retained(output)


def native(output, arguments):
    function = previous.native
    scope = dict(previous.__dict__, SCRIPT=SCRIPT, ANCHOR_SHA=ANCHOR_SHA, FLAG=FLAG,
        repair_ready=repair_ready, require_source=require_source, private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)
    retained = types.FunctionType(function.__code__, scope, function.__name__,
        function.__defaults__, function.__closure__)
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
        if ('--overrides_json' not in arguments
                or json.loads(Path(arguments[arguments.index('--overrides_json') + 1]).read_text())
                != {'rewards.' + REWARD_KEY: REWARD_VALUE}):
            raise ValueError('Declare exactly the single angular-coefficient frozen override')
    elif '--overrides_json' in arguments:
        raise ValueError('Descendants replay their exact saved configuration')
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
        raise ValueError('Only native PPO and frozen replay')
