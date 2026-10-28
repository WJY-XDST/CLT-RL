"""Gated one-field angular-width repair from the complete6801 reference.

The runtime is provisional until a data-based decision admits it. It retains
the original ordinary27-input PPO, all17 Adam states, actual LR, six learned
Gaussian heads, targeted sampler, complete command ranges and physical control.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys

import torch

from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
import plain27_sampler_metadata_adapter as adapter
import joint_pd_source6601_targeted_turn030_runtime as previous

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261021_032504_524539_source6601_targeted_turn030_200'
DECISION = RESULTS / 'joint_pd_source6801_targeted030_yaw0075_trial_decision_20261021.json'
ANCHOR_SHA = '82d733cd85618ca78365be77a490eb3e85a196ee87862a8821fe60caef0108e0'
REWARD_KEY, OLD_VALUE, REWARD_VALUE = 'tracking_sigma_ang', .01, .0075
FLAG = '--repair_targeted030_yaw_sigma0075_from6801'
KEY, base, criteria_fix, bind = previous.KEY, previous.base, previous.criteria_fix, previous.bind
DELTA = dict(rewards={REWARD_KEY: [OLD_VALUE, REWARD_VALUE]}, commands={},
             exploration={}, controller={}, physics={})


def read(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=1)
def anchor():
    final = read(TRAINING / 'final_checkpoint_verification_readonly_20261019.json')
    source = Path(final['checkpoint'])
    saved, cfg = previous.require_source(source)
    if (not final['verified'] or final['actual_ppo_updates'] != 6801
            or final['all17_Adam_step'] != 136020 or sha(source) != ANCHOR_SHA
            or cfg['rewards'][REWARD_KEY] != OLD_VALUE
            or cfg['rewards']['tracking_ang_vel'] != 2.
            or cfg['rewards']['tracking_lin_vel'] != 3.
            or cfg['rewards']['orientation'] != -60.
            or cfg['commands'][previous.sampler.CONFIG_KEY] != .3
            or cfg['wheel_action_limit'] != 1.25 or cfg['action_scale_vel'] != 15.):
        raise ValueError('Require exact6801 with the existing full-envelope sampler and controls')
    return source, saved, cfg


def profile():
    value = copy.deepcopy(previous.profile())
    value.update(profile='plain_source6801_targeted030_yaw_sigma0075_v1',
        native_runtime_sha256=sha(SCRIPT),
        parent_targeted030_runtime_sha256=sha(previous.SCRIPT),
        tracking_sigma_ang=REWARD_VALUE, angular_width_trial_anchor_sha256=ANCHOR_SHA,
        authority='complete6801_reference_and7001_regression_review_with_remaining_diagnoses')
    return value


def recipe(rewards):
    return (previous.recipe(dict(rewards, tracking_sigma_ang=OLD_VALUE))
            and rewards[REWARD_KEY] == REWARD_VALUE)


def require_source(path, transition=False, *, initial_transfer=False):
    if initial_transfer:
        raise ValueError('Never reload originalmodel200 or reset the optimizer')
    _, before, cfg = anchor()
    path = Path(path).resolve()
    if sha(path) == ANCHOR_SHA:
        return previous.require_source(path)
    if transition:
        raise ValueError('Only exact6801 may start the one-field reward-width transfer')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = copy.deepcopy(cfg)
    expected['rewards'][REWARD_KEY] = REWARD_VALUE
    expected['log_dir'] = current.get('log_dir')
    states, infos = saved['optimizer_state_dict']['state'], saved.get('infos')
    actual = saved_updates(path)
    if (not tensor_finite(saved) or current != expected or not isinstance(infos, dict)
            or infos.get(KEY) != profile() or actual < 6801 or len(states) != 17
            or any(float(v['step']) != 20 * actual for v in states.values())
            or saved['model_state_dict'].keys() != before['model_state_dict'].keys()
            or any(saved['model_state_dict'][k].shape != v.shape
                   for k, v in before['model_state_dict'].items())):
        raise ValueError('Require the exact reward-width descendant and continuous17 Adam clocks')
    return saved, current


def repair_ready():
    if not DECISION.exists():
        raise ValueError('Finish remaining reviews and select this provisional one-field trial first')
    d = read(DECISION)
    final, full, static, long = [read(d[k]) for k in (
        'source_final_verification', 'source_full_review', 'source_static_raw_manifest',
        'source_original_sustained120')]
    compared = read(d['compared7001_full_review'])
    signed = read(d['compared7001_signed_height_review'])
    headroom = read(d['paired_reset_headroom_diagnosis'])
    if (not d['selected_single_yaw_width_trial'] or d['source_checkpoint_sha256'] != ANCHOR_SHA
            or d['source_actual_updates'] != 6801 or d['source_all17_Adam_step'] != 136020
            or d['planned_actual_updates'] != 200 or d['target_actual_updates'] != 7001
            or d['next_configuration_changes'] != DELTA
            or not final['verified'] or final['checkpoint_sha256'] != ANCHOR_SHA
            or not full['all_original_and_retained_reviews_complete'] or full['actual_updates'] != 6801
            or not static['raw_integrity_verified'] or not static['static_retention_verified']
            or not long['raw_integrity_verified'] or long['original_gates_and_nonzero_relative10_both_passed'] != 54
            or not compared['all_original_and_retained_reviews_complete'] or compared['actual_updates'] != 7001
            or not signed['completed'] or not headroom['verified']
            or d['relative_error_target'] != .1 or d['acceptance_changed']
            or d['optimizer_reset'] or d['std_reset'] or d['original_model200_reload']
            or d['faster_curriculum_authorized'] or not d['all_failed_and_regressed_trials_preserved']
            or not d['historical_sigma005_posture_regressions_reviewed']
            or any(sha(p) != digest for p, digest in d['source_pins'].items())):
        raise ValueError('Require complete paired evidence and one selected reward field without resets')
    return d


# Reuse the current sampler and ordinary runtime. Retarget only source clocks
# and evidence labels; the new recipe guard admits exactly the one reward field.
_api = adapter.private_module(previous.SCRIPT, '_source6801_yaw0075_targeted030_api')
_provider = adapter.private_module(Path(previous._provider.__file__), '_source6801_yaw0075_provider')
FACTORY_METADATA = adapter.retarget(_provider, ('private_runtime',),
    integers={'*': {6201: 6801}},
    strings={'lin300_runtime_evidence.json': 'yaw0075_base_invariants.json'},
    keyword_names={'source_actual_updates': 'trial_anchor_actual_updates',
        'height_yaw_command_exploration_physics_reset_and_direct_joint_execution_unchanged':
        'inherited_height_exploration_physics_reset_and_direct_joint_execution_retained'})
SAMPLER_METADATA = adapter.retarget(_api, ('private_runtime',),
    integers={'*': {6601: 6801}},
    strings={'targeted_turn_runtime_evidence.json': 'yaw0075_targeted030_runtime_evidence.json'},
    keyword_names={'reward_exploration_physics_reset_and_control_unchanged':
        'inherited_sampler_exploration_physics_reset_and_control_retained',
        'sampler_RNG_draws_intentionally_added_only_for_mature_nonfixed_training':
        'retained_sampler_draws_only_on_mature_nonfixed_training'})
for _module in (_api, _provider):
    for _name in ('SCRIPT', 'ANCHOR_SHA', 'FLAG', 'DELTA', 'profile', 'recipe',
                  'require_source', 'repair_ready', 'anchor', 'TRAINING', 'DECISION'):
        setattr(_module, _name, globals()[_name])
_api._provider = _provider
private_runtime = _api.private_runtime
install_leg_exploration = _provider.install_leg_exploration
_api.install_leg_exploration = install_leg_exploration
install_sampling_field_restore = _api.install_sampling_field_restore
evaluator_tree = _api.evaluator_tree


def native(output, arguments):
    return _api.native(output, arguments)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    source = Path(source).resolve()
    saved, actual_cfg = require_source(source)
    anchor_path, _, anchor_cfg = anchor()
    if updates != 200 or cfg != actual_cfg or commands != actual_cfg['commands']:
        raise ValueError('Keep exact200, the saved source and the complete unchanged command contract')
    overrides = dict(kwargs.pop('reward_overrides', {}))
    if overrides.pop('rewards.' + REWARD_KEY, REWARD_VALUE) != REWARD_VALUE:
        raise ValueError('Only the provisional angular-width value is permitted')
    if any(not key.startswith('rewards.') or anchor_cfg['rewards'].get(key[8:]) != value
           for key, value in overrides.items()):
        raise ValueError('No second reward field may change')
    # Build the successful current CLI on its unchanged accepted configuration.
    # Then replace only the declared reward option and source/iteration metadata.
    argv, _, _ = previous.command(anchor_path, anchor_cfg, label, updates,
        anchor_cfg['commands'], stop_file, reward_overrides=overrides, **kwargs)
    old_option = f'env.rewards.{REWARD_KEY}={OLD_VALUE}'
    if argv.count(old_option) != 1 or any(a.startswith('--repair_') for a in argv):
        raise ValueError('Requalify the original CLI prototype before executing anything')
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    argv[argv.index(old_option)] = f'env.rewards.{REWARD_KEY}={REWARD_VALUE}'
    actual = saved_updates(source)
    argv[argv.index('--checkpoint_path') + 1] = str(source)
    argv[argv.index('--target_iteration') + 1] = str(actual + 200)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    if any(x in argv for x in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
        raise ValueError('Do not reset or overwrite retained source state')
    return argv, actual + 199, saved


def replay_command(source, output):
    source = Path(source).resolve()
    require_source(source)
    anchor_path, _, _ = anchor()
    argv = previous.replay_command(anchor_path, output)
    if any(a.startswith('--repair_') for a in argv):
        raise ValueError('The reviewed6801 prototype must not repeat its old sampler transfer')
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    argv[argv.index('--checkpoint') + 1] = str(source)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv


def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source, transition=transition)
    if transition:
        if ('--overrides_json' not in arguments or read(arguments[
                arguments.index('--overrides_json') + 1]) != {'rewards.' + REWARD_KEY: REWARD_VALUE}):
            raise ValueError('Frozen qualification changes exactly one reward field')
    elif sha(source) == ANCHOR_SHA or '--overrides_json' in arguments:
        raise ValueError('Normal replay needs the exact saved descendant configuration')
    install_sampling_field_restore()
    import gymnasium as gym
    original_make = gym.make
    gym.make = _api._wrap_make(original_make)
    tree, path = evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    try:
        exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
            _install_quality_criteria=criteria_fix.install,
            _configure_sampling=_api._configure,
            _observe_source200=private_runtime().observe_environment))
    finally:
        gym.make = original_make


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'replay':
        replay(sys.argv[1:])
    elif mode == 'native':
        boundary = sys.argv.index('--')
        if sys.argv[1:boundary] != ['--observation-output', sys.argv[2]]:
            raise ValueError('Keep the original native observation-output contract')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only ordinary PPO and frozen replay are supported')
