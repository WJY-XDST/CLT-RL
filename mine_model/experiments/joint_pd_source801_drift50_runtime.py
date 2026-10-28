"""Retain source801 and strengthen only its existing translation-drift cost."""
import copy
import importlib.util

from joint_pd_continuation import sha
import joint_pd_source200_stationary_runtime as base
import joint_pd_source601_wheel_tracking_runtime as previous

KEY = base.KEY
LEG_STD_FLOOR = .05
GLOBAL_STD_FLOOR = .02
WHEEL_STD_FLOOR = .03
HEIGHT_SIGMA = .0025
STANDING_VELOCITY_SCALE = -50.
WHEEL_TRACKING_SCALE = -.1
tires = base.tires
prior = base


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source801_stationary_leg05_drift50_wheel01_v1',
        native_runtime_sha256=sha(__file__), previous_runtime_sha256=sha(previous.__file__),
        standing_velocity=STANDING_VELOCITY_SCALE,
        authority='user_preserve_height_gains_repair_zero_speed_drift_before_motion')
    return result


def private_runtime(transition_source=None):
    spec = importlib.util.spec_from_file_location('_source801_drift50_private', base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_require, original_observe = module.require_source, module.observe_environment
    transition_sha = None
    if transition_source is not None:
        _, cfg = previous.require_source(transition_source)
        if (base.saved_updates(transition_source) != 801
                or cfg['rewards']['standing_velocity'] != -25.
                or cfg['rewards']['standing_wheel_tracking'] != WHEEL_TRACKING_SCALE):
            raise ValueError('Only the independently diagnosed source801 begins this drift-cost repair')
        transition_sha = sha(transition_source)

    def require(source, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Do not repeat the original source200 transfer')
        if transition_sha is not None and sha(source) == transition_sha:
            return previous.require_source(source)
        saved, cfg = original_require(source, initial_transfer=False)
        if (cfg['rewards']['standing_velocity'] != STANDING_VELOCITY_SCALE
                or cfg['rewards']['standing_wheel_tracking'] != WHEEL_TRACKING_SCALE):
            raise ValueError('Keep the qualified drift50/wheel01 recipe')
        return saved, cfg

    def observe(env, output):
        rewards = env.unwrapped.cfg.rewards
        if (rewards.standing_velocity != STANDING_VELOCITY_SCALE
                or rewards.standing_wheel_tracking != WHEEL_TRACKING_SCALE):
            raise ValueError('Actual native rewards differ from the single drift-cost revision')
        return original_observe(env, output)

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, *, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_runner(output, *, transition_source=None):
    return private_runtime(transition_source).install_runner(output)


def observe_environment(env, output):
    return private_runtime().observe_environment(env, output)


def install_environment_observer(output):
    return private_runtime().install_environment_observer(output)


def install_leg_exploration(output):
    spec = importlib.util.spec_from_file_location('_unchanged_source801_exploration', previous.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.profile = profile
    return module.install_leg_exploration(output)
