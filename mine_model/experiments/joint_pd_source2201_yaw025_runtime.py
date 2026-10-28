"""One angular reward width change; preserve the reviewed source2201 state."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types

import source1201_orientation40
import joint_pd_source2001_quality50_runtime as previous
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = 'df97d135ff953bc0d9e8fc5283ea4bc1473bc046cad58746c9b97fc299620912'
REWARD_KEY = 'tracking_sigma_ang'
REWARD_VALUE = .025
FLAG = '--repair_motion025_yaw025'
KEY, base = previous.KEY, previous.base
criteria_fix = previous.criteria_fix
SOURCE_OUTPUT = ROOT / 'mine_model/results/joint_pd_optimization_20261018_203302_683086_source2001_quality50_200'
FULL51 = ROOT / 'mine_model/results/joint_pd_motion025_quality50_full_height_20261018_source2201'
DECISION = ROOT / 'mine_model/results/joint_pd_source2201_yaw025_priority_decision_20261018.json'


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source2201_motion025_quality50_yaw025_v1',
        native_runtime_sha256=sha(SCRIPT), previous_runtime_sha256=sha(previous.SCRIPT),
        tracking_sigma_ang=REWARD_VALUE, yaw_repair_anchor_sha256=ANCHOR_SHA,
        authority='completed200_full27_full51_and_measured_static_yaw_reward_precision')
    return result


def recipe(rewards):
    return (previous.recipe(dict(rewards, tracking_sigma_ang=.25))
        and rewards[REWARD_KEY] == REWARD_VALUE)


def repair_ready():
    decision = json.loads(DECISION.read_text())
    report = json.loads((SOURCE_OUTPUT / 'replay/assessment.json').read_text())
    full51 = json.loads((FULL51 / 'candidate/motion025_full_height_diagnosis.json').read_text())
    proof = json.loads((SOURCE_OUTPUT / 'final_checkpoint_verification_readonly.json').read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 2201 or decision['source_all17_Adam_step'] != 44020
            or decision['planned_only_reward_change'] != {REWARD_KEY: [.25, REWARD_VALUE]}
            or not decision['source_is_unaccepted_repair_branch']
            or not decision['source_full200_full27_full51_verified']
            or report['checkpoint_sha256'] != ANCHOR_SHA
            or report['survivors'] != 24 or report['resets'] != 24
            or full51['checkpoint_sha256'] != ANCHOR_SHA
            or full51['survivors'] != 51 or full51['resets']
            or proof['actual_ppo_updates'] != 2201 or proof['all17_Adam_step'] != 44020
            or not proof['all200_full4096_alignment_errors_zero']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require the complete reviewed source2201; repair is not acceptance')
    return decision


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_source2201_yaw025_private', base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    height = previous.previous
    height.previous.previous._motion_guard(module, 'require_source', "cfg['commands']['training_motion_scale']")
    height._height_environment_guard(module)
    original_require, original_observe = module.require_source, module.observe_environment
    transition = source is not None
    if transition:
        _, cfg = previous.require_source(source)
        if (sha(source) != ANCHOR_SHA or saved_updates(source) != 2201
                or cfg['rewards'][REWARD_KEY] != .25
                or not recipe(dict(cfg['rewards'], tracking_sigma_ang=REWARD_VALUE))):
            raise ValueError('Only the exact reviewed2201 starts this one-width yaw repair')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload original model200 or reset retained optimizer')
        if transition and sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        saved, cfg = original_require(path, initial_transfer=False)
        if not recipe(cfg['rewards']):
            raise ValueError('Keep the exact quality50/height3/orientation40/yaw025 recipe')
        return saved, cfg

    def observe(env, output):
        raw = env.unwrapped
        if (not recipe(raw.cfg.rewards.to_dict()) or raw._tracking_goal_criteria is None
                or raw._tracking_goal_criteria['relative_error'] != .1):
            raise ValueError('Actual yaw reward or original criteria differ')
        result = original_observe(env, output)
        source1201_orientation40.previous.install_failure_capture(raw, output)
        from auto_train_mine import write_json
        write_json(Path(output) / 'quality_criteria_runtime_patch.json', dict(
            type(raw)._quality_criteria_runtime_proof,
            loaded_criteria_evidence=raw._tracking_goal_evidence,
            only_reward_configuration_change={REWARD_KEY: [.25, REWARD_VALUE]},
            both_violation_penalties_zero=True))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    function = previous.install_leg_exploration
    retained = types.FunctionType(function.__code__, dict(function.__globals__, profile=profile),
        function.__name__, function.__defaults__, function.__closure__)
    return retained(output)


def evaluator_tree():
    return previous.evaluator_tree()


def native(output, arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    require_source(source, transition=transition)
    if [a for a in arguments if a.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=ActorCritic']:
        raise ValueError('Keep the original plain27 ActorCritic and ordinary PPO')
    private_runtime(source if transition else None).install_runner(output)
    private_runtime().install_environment_observer(output)
    previous.install_native_criteria_fix()
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
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source, transition=transition)
    if transition:
        if '--overrides_json' not in arguments:
            raise ValueError('Declare the single frozen angular-width override')
        delta = json.loads(Path(arguments[arguments.index('--overrides_json') + 1]).read_text())
        if delta != {'rewards.' + REWARD_KEY: REWARD_VALUE}:
            raise ValueError('Only the existing angular reward width changes')
    elif '--overrides_json' in arguments:
        raise ValueError('Descendants replay their exact saved configuration')
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    tree, path = evaluator_tree()
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
        raise ValueError('Only native training and frozen replay')
