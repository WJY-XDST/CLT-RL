"""Retain the reviewed smooth10 branch and align static reset noise with its stage.

Runtime support only. A controller must still require a completed200/full27
review before choosing a real source and starting another block.
"""
import copy
import importlib.util
from pathlib import Path

from joint_pd_continuation import sha
from joint_pd_reset_stage import prepare_stationary_reset, validate_stationary_reset_revision
import source1001_reference_smooth10 as parent

SCRIPT = Path(__file__).resolve()
KEY = parent.KEY
RESET_INITIAL = .05
FULL_MOTION_RESET_FINAL = .5
FLAG = '--stationary_reset_stage_transfer'


def profile():
    result = copy.deepcopy(parent.profile())
    result.update(profile='plain_stationary_leg05_drift50_wheel01_smooth10_reset05_v1',
        native_runtime_sha256=sha(SCRIPT), previous_runtime_sha256=sha(parent.SCRIPT),
        reset_velocity_initial=RESET_INITIAL, reset_velocity_final=RESET_INITIAL,
        full_motion_reset_velocity_final_retained=FULL_MOTION_RESET_FINAL,
        reset_stage_helper_sha256=sha(Path(prepare_stationary_reset.__code__.co_filename)),
        native_entry_sha256=sha(SCRIPT.with_name('stationary_reset_observed.py')),
        reset_sampling_change_explicit=True, PPO_clock_reset=False)
    return result


def private_runtime(source=None, *, evaluation=False):
    spec = importlib.util.spec_from_file_location('_stationary_reset05_private', parent.base.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_require, original_observe = module.require_source, module.observe_environment
    source_sha = None
    if source is not None:
        _, cfg = parent.require_source(source)
        if (cfg['commands']['training_motion_scale'] != 0.
                or cfg['reset_velocity_initial'] != RESET_INITIAL
                or cfg['reset_velocity_final'] != FULL_MOTION_RESET_FINAL):
            raise ValueError('Only the diagnosed full-noise stationary parent begins this sampling repair')
        source_sha = sha(source)

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload the original model200 for a reset-stage revision')
        if source_sha is not None and sha(path) == source_sha:
            return parent.require_source(path)
        saved, cfg = original_require(path, initial_transfer=False)
        if (cfg['reset_velocity_initial'] != RESET_INITIAL or cfg['reset_velocity_final'] != RESET_INITIAL
                or cfg['rewards']['standing_velocity'] != -50.
                or cfg['rewards']['standing_wheel_tracking'] != -.1
                or cfg['rewards']['action_smooth'] != -.1
                or cfg['rewards']['action_smooth_wheel_multiplier'] != 5.):
            raise ValueError('Keep the declared static reset05 and unchanged reward recipe')
        return saved, cfg

    def observe(env, output):
        raw = env.unwrapped
        # Canonical replay draws its declared .05 velocities separately after
        # the base reset, and therefore sets these two internal fields to zero.
        expected = 0. if evaluation else RESET_INITIAL
        if raw.cfg.reset_velocity_initial != expected or raw.cfg.reset_velocity_final != expected:
            raise ValueError('Actual reset-noise fields differ from this native/replay stage')
        rewards = raw.cfg.rewards
        if (rewards.standing_velocity != -50. or rewards.standing_wheel_tracking != -.1
                or rewards.action_smooth != -.1 or rewards.action_smooth_wheel_multiplier != 5.):
            raise ValueError('A reward changed during the reset-only revision')
        result = original_observe(env, output)
        parent.install_failure_capture(raw, output)
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, *, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    spec = importlib.util.spec_from_file_location('_stationary_reset05_exploration', parent.previous.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.profile = profile
    return module.install_leg_exploration(output)


def expected_config(source, *, transition):
    _, cfg = require_source(source, transition=transition)
    expected = prepare_stationary_reset(cfg, full_motion_final=FULL_MOTION_RESET_FINAL)
    validate_stationary_reset_revision(cfg, expected, full_motion_final=FULL_MOTION_RESET_FINAL)
    return expected
