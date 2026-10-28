"""One linear-speed reward width change from the retained source1601.

Preserve direct motor-angle actions, plain27 PPO, full17 Adam/std/LR,
orientation40, height rewards, full height range and physical execution.
"""
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
import joint_pd_source1401_motion025_runtime as previous
from evaluate_source200_stationary_height import evaluator_tree

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = 'fa7544e120906edec848df37765c46aaf92aa267492b3e172c570aacfdfb3180'
VELOCITY_SIGMA = .025
FLAG = '--repair_motion025_velocity025'
KEY, base = previous.KEY, previous.base
DECISION = ROOT/'mine_model/results/joint_pd_source1601_velocity025_priority_decision_20261018.json'
SOURCE_OUTPUT = ROOT/'mine_model/results/joint_pd_optimization_20261017_173938_589374_source1401_motion025_200'


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source1601_motion025_velocity025_v1',
        native_runtime_sha256=sha(SCRIPT), previous_runtime_sha256=sha(previous.SCRIPT),
        tracking_sigma=VELOCITY_SIGMA, velocity_repair_anchor_sha256=ANCHOR_SHA,
        authority='completed_height100_rejection_and_measured_low_speed_kernel_sensitivity')
    return result


def recipe(rewards):
    return (previous.previous.recipe(rewards)
        and rewards['tracking_sigma'] == VELOCITY_SIGMA
        and rewards['tracking_lin_vel'] == 2. and rewards['lin_vel_error_sq'] == -2.
        and rewards['base_height_error_sq'] == 0. and rewards['base_height'] == 2.
        and rewards['base_height_sigma'] == .0025
        and rewards['joint_target_leg_length_error_sq'] == 0.
        and rewards['tracking_sigma_ang'] == .25 and rewards['tracking_ang_vel'] == 1.
        and rewards['yaw_rate_error_sq'] == -1.)


def repair_ready():
    decision = json.loads(DECISION.read_text())
    report = json.loads((SOURCE_OUTPUT/'replay/assessment.json').read_text())
    legs = json.loads((SOURCE_OUTPUT/'replay/height_leg_command_assessment.json').read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 1601 or decision['source_all17_Adam_step'] != 32020
            or decision['planned_only_reward_change_from_selected_source'] != {'tracking_sigma':[.25,VELOCITY_SIGMA]}
            or decision['canonical_survivors'] != [27,24] or decision['canonical_resets'] != [0,60]
            or decision['paired_passed_trials'] != [15,3]
            or not decision['source_static_all_original_metrics_and_actual_legs_passed']
            or decision['rejected_static_all_original_metrics_passed']
            or report['checkpoint_sha256'] != ANCHOR_SHA
            or not stage.summary(report)['stationary_all_original_metrics_passed']
            or not legs['static_stage_candidate_passed']
            or any(sha(p) != digest for p,digest in decision['source_pins'].items())):
        raise ValueError('Require completed27+51 rejection and the intact full-state static-qualified source1601')
    return decision


def private_runtime(source=None):
    spec = importlib.util.spec_from_file_location('_source1601_velocity025_private',base.__file__)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    # Only these metadata guards change; base height cost stays exactly zero.
    previous._motion_guard(module,'require_source',"cfg['commands']['training_motion_scale']")
    previous._motion_guard(module,'observe_environment','cfg.commands.training_motion_scale')
    require_original, observe_original = module.require_source, module.observe_environment
    transition = False
    if source is not None:
        _, cfg = previous.require_source(source)
        if (sha(source) != ANCHOR_SHA or saved_updates(source) != 1601
                or cfg['rewards']['tracking_sigma'] != .25
                or not recipe(dict(cfg['rewards'],tracking_sigma=VELOCITY_SIGMA))):
            raise ValueError('Only exact source1601 starts this single velocity-width repair')
        transition = True

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload the original model200 for this repair')
        if transition and sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        saved, cfg = require_original(path,initial_transfer=False)
        if not recipe(cfg['rewards']):
            raise ValueError('Keep the exact velocity025/height0/orientation40 reward recipe')
        return saved, cfg

    def observe(env, output):
        if not recipe(env.unwrapped.cfg.rewards.to_dict()):
            raise ValueError('Actual rewards differ from the declared velocity-width repair')
        result = observe_original(env,output)
        previous.previous.previous.install_failure_capture(env.unwrapped,output)
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
            raise ValueError('Frozen transfer must declare the one velocity-width override')
        delta = json.loads(Path(arguments[arguments.index('--overrides_json')+1]).read_text())
        if delta != {'rewards.tracking_sigma':VELOCITY_SIGMA}:
            raise ValueError('Only the positive forward-speed kernel width changes')
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
