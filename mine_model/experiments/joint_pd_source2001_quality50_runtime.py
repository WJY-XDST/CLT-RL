"""Repair posture/height priority through the existing smooth motion bonus.

One saved reward setting changes: quality floor1 -> .5. Keep direct joint
angle actions, plain27 PPO, all17 Adam/LR/std, measured physics, full commands,
height3, orientation40 and velocity025. No optional violation cost is enabled.
"""
import ast
import copy
import json
from pathlib import Path
import sys
import types

import source1201_orientation40
import joint_pd_source1801_height3_runtime as previous
import joint_pd_quality_criteria_runtime as criteria_fix
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = '5b8c4063396e8f85f696d2f9d701df4aa1c8e5f97c0068d0af511cea79abfb0c'
QUALITY_FLOOR = .5
REWARD_KEY = 'tracking_task_quality_gate_floor'
FLAG = '--repair_motion025_quality50'
KEY, base = previous.KEY, previous.base
SOURCE_OUTPUT = ROOT / 'mine_model/results/joint_pd_optimization_20261018_recovered_height3_200_source1801'
DECISION = ROOT / 'mine_model/results/joint_pd_source2001_quality50_priority_decision_20261018.json'


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source2001_motion025_height3_quality50_v1',
        native_runtime_sha256=sha(SCRIPT), previous_runtime_sha256=sha(previous.SCRIPT),
        quality_floor=QUALITY_FLOOR, quality_criteria_fix_sha256=sha(criteria_fix.SCRIPT),
        quality_repair_anchor_sha256=ANCHOR_SHA,
        authority='reviewed_full200_full27_and51_posture_height_priority_repair')
    return result


def recipe(rewards):
    return (previous.recipe(rewards) and rewards[REWARD_KEY] == QUALITY_FLOOR
        and rewards['tracking_goal_violation'] == 0.
        and rewards['tracking_goal_component_violation'] == 0.)


def repair_ready():
    evidence = json.loads(DECISION.read_text())
    report = json.loads((SOURCE_OUTPUT / 'replay/assessment.json').read_text())
    if (evidence['source_checkpoint_sha256'] != ANCHOR_SHA
            or evidence['source_actual_updates'] != 2001 or evidence['source_all17_Adam_step'] != 40020
            or evidence['planned_only_reward_change'] != {REWARD_KEY: [1., QUALITY_FLOOR]}
            or not evidence['source_is_unaccepted_repair_branch']
            or not evidence['source_full200_full27_full51_verified']
            or report['checkpoint_sha256'] != ANCHOR_SHA
            or report['survivors'] != 24 or report['resets'] != 27
            or evidence['relative_error_target'] != .1 or evidence['acceptance_changed']
            or any(sha(p) != h for p, h in evidence['source_pins'].items())):
        raise ValueError('Require the exact complete reviewed height3 repair source, never promotion')
    return evidence


def private_runtime(source=None):
    module = previous.private_runtime()
    original_require, original_observe = module.require_source, module.observe_environment
    transition = source is not None
    if transition:
        _, cfg = previous.require_source(source)
        if sha(source) != ANCHOR_SHA or saved_updates(source) != 2001 or cfg['rewards'][REWARD_KEY] != 1.:
            raise ValueError('Only the exact reviewed2001 starts this priority repair')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload original model200 or reset retained optimizer')
        if transition and sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        saved, cfg = original_require(path, initial_transfer=False)
        if not recipe(cfg['rewards']):
            raise ValueError('Keep only the declared smooth bonus quality setting')
        return saved, cfg

    def observe(env, output):
        raw = env.unwrapped
        if (not recipe(raw.cfg.rewards.to_dict()) or raw._tracking_goal_criteria is None
                or raw._tracking_goal_criteria['relative_error'] != .1):
            raise ValueError('Actual quality reward or independently loaded criteria differ')
        result = original_observe(env, output)
        from auto_train_mine import write_json
        proof = type(raw)._quality_criteria_runtime_proof
        write_json(Path(output) / 'quality_criteria_runtime_patch.json', dict(proof,
            loaded_criteria_evidence=raw._tracking_goal_evidence,
            only_reward_configuration_change={REWARD_KEY: [1., QUALITY_FLOOR]},
            both_violation_penalties_zero=True))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_native_criteria_fix():
    import gymnasium as gym
    original = gym.make

    def make(*args, **kwargs):
        task = args[0] if args else kwargs.get('id')
        if task == 'WheelLeggedJointPD-Flat-v0':
            from wheel_legged_gym_isaaclab.tasks.direct.wheel_legged_vmc_flat.wheel_legged_vmc_flat_env import WheelLeggedVMCFlatEnv
            criteria_fix.install(WheelLeggedVMCFlatEnv)
        return original(*args, **kwargs)

    gym.make = make


def evaluator_tree():
    tree, path = previous.evaluator_tree()
    original = ast.dump(tree)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    imports = [i for i, n in enumerate(main.body) if isinstance(n, ast.ImportFrom)
        and any(alias.name == 'WheelLeggedVMCFlatEnv' for alias in n.names)]
    if len(imports) != 1:
        raise ValueError('Requalify the original evaluation class import')
    main.body.insert(imports[0] + 1, ast.parse('_install_quality_criteria(WheelLeggedVMCFlatEnv)').body[0])
    changed = copy.deepcopy(tree)
    main.body.pop(imports[0] + 1)
    if ast.dump(tree) != original:
        raise ValueError('Evaluation changed beyond installing the same criteria dependency fix')
    return ast.fix_missing_locations(changed), path


def install_leg_exploration(output):
    function = previous.install_leg_exploration
    retained = types.FunctionType(function.__code__, dict(function.__globals__, profile=profile),
        function.__name__, function.__defaults__, function.__closure__)
    return retained(output)


def native(output, arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    require_source(source, transition=transition)
    if [a for a in arguments if a.startswith('agent.policy.class_name=')] != ['agent.policy.class_name=ActorCritic']:
        raise ValueError('Retain plain27 ActorCritic and ordinary full PPO')
    private_runtime(source if transition else None).install_runner(output)
    private_runtime().install_environment_observer(output)
    install_native_criteria_fix()
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
            raise ValueError('Declare the single frozen quality reward override')
        delta = json.loads(Path(arguments[arguments.index('--overrides_json') + 1]).read_text())
        if delta != {'rewards.' + REWARD_KEY: QUALITY_FLOOR}:
            raise ValueError('Only the existing smooth motion bonus quality floor may change')
    elif '--overrides_json' in arguments:
        raise ValueError('Replay descendants with their exact saved reward configuration')
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
