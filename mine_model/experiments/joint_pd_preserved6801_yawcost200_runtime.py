"""Prepared single dense-yaw cost trial from the preserved6801 checkpoint.

This module changes one existing reward coefficient. It cannot admit a native
transfer until a source-selection decision pins the complete7001 review,
signed height transitions and fresh-seed small-yaw120 evidence. Keep the
source's27 inputs, joint-angle PD, all17 Adam states, LR, six std, sampler,
physical model and all original gates. Preparation is not policy acceptance.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys

import torch

from joint_pd_continuation import ROOT, sha, tensor_finite, validate_continuation_config
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
import plain27_sampler_metadata_adapter as adapter
import joint_pd_source6601_targeted_turn030_runtime as previous

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261021_032504_524539_source6601_targeted_turn030_200'
DECISION = RESULTS / 'joint_pd_preserved6801_yawcost200_trial_decision_20261022.json'
QUALIFICATION = RESULTS / 'joint_pd_preserved6801_yawcost200_qualification_20261022'
ANCHOR_SHA = '82d733cd85618ca78365be77a490eb3e85a196ee87862a8821fe60caef0108e0'
REJECTED_SHA = 'a78ab531f3ffc812f15cbf9581a4ec378f6c30a31d6b9930c827282862c0d54c'
REWARD_KEY, OLD_VALUE, REWARD_VALUE = 'yaw_rate_error_sq', -1., -200.
FLAG = '--repair_preserved6801_yawcost200'
KEY, base, criteria_fix, bind = previous.KEY, previous.base, previous.criteria_fix, previous.bind
MOTION_SCALE = 1.
DELTA = dict(rewards={REWARD_KEY: [OLD_VALUE, REWARD_VALUE]}, commands={},
    exploration={}, controller={}, physics={})


@lru_cache(maxsize=1)
def anchor():
    final = json.loads((TRAINING / 'final_checkpoint_verification_readonly_20261019.json').read_text())
    source = Path(final['checkpoint'])
    saved, cfg = previous.require_source(source)
    if (not final['verified'] or final['actual_ppo_updates'] != 6801
            or final['all17_Adam_step'] != 136020 or sha(source) != ANCHOR_SHA
            or cfg['rewards'][REWARD_KEY] != OLD_VALUE
            or cfg['rewards']['tracking_lin_vel'] != 3.
            or cfg['rewards']['tracking_ang_vel'] != 2.
            or cfg['rewards']['orientation'] != -60.
            or cfg['wheel_action_limit'] != 1.25 or cfg['action_scale_vel'] != 15.
            or cfg['commands']['training_motion_scale'] != 1.
            or cfg['commands']['targeted_turn_command_fraction'] != .3):
        raise ValueError('Require the exact preserved6801 and its complete saved recipe')
    return source, saved, cfg


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_preserved6801_targeted030_yawcost200_v1',
        native_runtime_sha256=sha(SCRIPT), parent_targeted030_runtime_sha256=sha(previous.SCRIPT),
        yaw_rate_error_sq=REWARD_VALUE, yaw_penalty_trial_anchor_sha256=ANCHOR_SHA,
        authority='completed735_static810_original120_signed_and_small_yaw120_source_selection')
    return result


def recipe(rewards):
    return previous.recipe(dict(rewards, yaw_rate_error_sq=OLD_VALUE)) and rewards[REWARD_KEY] == REWARD_VALUE


def require_source(path, transition=False, *, initial_transfer=False):
    if initial_transfer:
        raise ValueError('Originalmodel200 and optimizer resets remain forbidden')
    _, before, cfg = anchor()
    path = Path(path).resolve()
    if sha(path) == ANCHOR_SHA:
        return previous.require_source(path)
    if transition:
        raise ValueError('Only exact preserved6801 may begin this one-field transfer')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = copy.deepcopy(cfg)
    expected['rewards'][REWARD_KEY] = REWARD_VALUE
    expected['log_dir'] = current.get('log_dir')
    states, actual = saved['optimizer_state_dict']['state'], saved_updates(path)
    if (not tensor_finite(saved) or current != expected
            or saved.get('infos', {}).get(KEY) != profile() or actual < 6801
            or len(states) != 17 or any(float(v['step']) != actual * 20 for v in states.values())
            or saved['model_state_dict'].keys() != before['model_state_dict'].keys()
            or any(saved['model_state_dict'][k].shape != v.shape
                   for k, v in before['model_state_dict'].items())):
        raise ValueError('Require the exact reward descendant and all retained Adam clocks')
    return saved, current


def repair_ready():
    d = json.loads(DECISION.read_text())
    read = lambda key: json.loads(Path(d[key]).read_text())
    final, full, static, sustained, signed = [read(k) for k in (
        'source_final_verification', 'source_full_review', 'source_static_raw_manifest',
        'source_original_sustained120', 'source_fullspeed_height_transition_review')]
    rejected, rejected_signed, small, fixed, long = [read(k) for k in (
        'previous_trial_full_review', 'previous_trial_signed_height_review',
        'previous_trial_small_yaw_sustained120', 'all735_source_selection_comparison',
        'previous_trial_independent120_raw')]
    if (not d['selected_single_dense_yaw_penalty_trial']
            or d['source_checkpoint_sha256'] != ANCHOR_SHA or d['source_actual_updates'] != 6801
            or d['source_all17_Adam_step'] != 136020 or d['planned_actual_updates'] != 200
            or d['target_actual_updates'] != 7001 or d['next_configuration_changes'] != DELTA
            or not final['verified'] or final['checkpoint_sha256'] != ANCHOR_SHA
            or not full['all_original_and_retained_reviews_complete'] or full['actual_updates'] != 6801
            or not static['raw_integrity_verified'] or not static['static_retention_verified']
            or static['checkpoint_sha256'] != ANCHOR_SHA
            or not sustained['raw_integrity_verified'] or sustained['survivors'] != 54
            or sustained['original_gates_and_nonzero_relative10_both_passed'] != 54
            or sustained['raw_OR_resets'] or not signed['completed']
            or not rejected['all_original_and_retained_reviews_complete']
            or rejected['actual_updates'] != 7001 or rejected['checkpoint_sha256'] != REJECTED_SHA
            or not rejected_signed['completed'] or not small['completed']
            or small['results']['source6801']['checkpoint_sha256'] != ANCHOR_SHA
            or small['results']['candidate7001']['checkpoint_sha256'] != REJECTED_SHA
            or not fixed['verified'] or not fixed['original735_all_cases_seeds_commands_and_initial_observations_bitwise_paired']
            or fixed['summaries']['source6801']['original_AND_literal10_passed']
                <= fixed['summaries']['candidate7001']['original_AND_literal10_passed']
            or not long['raw_integrity_verified'] or long['checkpoint_sha256'] != REJECTED_SHA
            or d['relative_error_target'] != .1 or d['acceptance_changed']
            or d['optimizer_reset'] or d['std_reset'] or d['actual_LR_reset']
            or d['original_model200_reload'] or not d['all_failures_and_regressions_retained']
            or any(sha(p) != h for p, h in d['source_pins'].items())):
        raise ValueError('Complete every owned review and select this single existing reward coefficient first')
    return d


# Reuse the qualified plant and retained sampler. Only clocks and truthful
# evidence labels are adapted; no simulation or reward operations are added.
_plant = adapter.private_module(previous.previous.previous.SCRIPT, '_yawcost200_preserved6801_plant')
PLANT_METADATA = adapter.retarget(_plant, ('private_runtime',),
    integers={'*': {6201: 6801}},
    strings={'lin300_runtime_evidence.json': 'preserved6801_yawcost200_base_execution_evidence.json'},
    keyword_names={'height_yaw_command_exploration_physics_reset_and_direct_joint_execution_unchanged':
                   'height_commands_exploration_physics_reset_and_direct_joint_execution_unchanged'})
for _name in ('SCRIPT', 'ANCHOR_SHA', 'FLAG', 'DELTA', 'profile', 'recipe', 'require_source', 'repair_ready'):
    setattr(_plant, _name, globals()[_name])
_sampler_api = adapter.private_module(previous.SCRIPT, '_yawcost200_preserved6801_retained_sampler')
for _name in ('SCRIPT', 'ANCHOR_SHA', 'FLAG', 'DELTA', 'DECISION', 'TRAINING',
              'profile', 'recipe', 'require_source', 'repair_ready'):
    setattr(_sampler_api, _name, globals()[_name])
_sampler_api._provider = _plant
SAMPLER_METADATA = adapter.retarget(_sampler_api, ('private_runtime',),
    integers={'*': {6601: 6801}},
    strings={'targeted_turn_runtime_evidence.json': 'retained_targeted030_yawcost200_runtime_evidence.json'},
    keyword_names={'reward_exploration_physics_reset_and_control_unchanged':
                   'exploration_physics_reset_and_control_unchanged',
                   'sampler_RNG_draws_intentionally_added_only_for_mature_nonfixed_training':
                   'retained_sampler_RNG_draws_only_for_mature_nonfixed_training'})
_sampler_api.FACTORY_METADATA = PLANT_METADATA + SAMPLER_METADATA
_sampler_factory = _sampler_api.private_runtime


def private_runtime(source=None):
    _sampler_api.install_sampling_field_restore()
    return _sampler_factory(source)


_sampler_api.private_runtime = private_runtime


def install_leg_exploration(output):
    return bind(previous.install_leg_exploration, profile=profile)(output)


_sampler_api.install_leg_exploration = install_leg_exploration


def native(output, arguments):
    return _sampler_api.native(output, arguments)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    source, stop_file = Path(source).resolve(), Path(stop_file).resolve()
    saved, actual_cfg = require_source(source)
    if updates != 200 or cfg != actual_cfg or commands != cfg['commands']:
        raise ValueError('Keep exactly200 ordinary updates and the entire saved command contract')
    overrides = kwargs.pop('reward_overrides', {})
    if any(k != 'rewards.' + REWARD_KEY or v != REWARD_VALUE for k, v in overrides.items()):
        raise ValueError('Only the declared existing yaw cost may change')
    if (kwargs.pop('minimum_wheel_std', .01) not in (.01, .03)
            or kwargs.pop('joint_mean_guard_coef', 0.) != 0.
            or kwargs.pop('joint_height_feedback_coef', 0.) != 0. or kwargs):
        raise ValueError('Keep saved exploration and ordinary PPO without auxiliary losses')
    expected = copy.deepcopy(cfg)
    expected['rewards'][REWARD_KEY] = REWARD_VALUE
    validate_continuation_config(cfg, expected, commands,
        reward_overrides={'rewards.' + REWARD_KEY: REWARD_VALUE})
    launch = json.loads((TRAINING / 'launch.json').read_text())
    argv = list(launch['command'])
    option = f'env.rewards.{REWARD_KEY}={OLD_VALUE}'
    if (launch['source_sha256'] != previous.ANCHOR_SHA or argv[1] != str(previous.SCRIPT)
            or argv.count(previous.FLAG) != 1 or argv.count(option) != 1
            or argv.count('agent.save_interval=1') != 1
            or argv.count('env.rewards.tracking_lin_vel=3.0') != 1
            or argv.count('env.rewards.tracking_ang_vel=2.0') != 1
            or argv[argv.index('--min_exploration_std') + 1] != '0.01'
            or argv[argv.index('--min_wheel_exploration_std') + 1] != '0.01'):
        raise ValueError('Requalify the exact successful6801 CLI prototype')
    argv[1] = str(SCRIPT)
    argv.remove(previous.FLAG)
    argv[argv.index(option)] = f'env.rewards.{REWARD_KEY}={REWARD_VALUE}'
    actual = saved_updates(source)
    for flag, value in {'--observation-output': stop_file.parent / 'native_learning_signal',
            '--checkpoint_path': source, '--target_iteration': actual + 200,
            '--run_name': label, '--stop_file': stop_file}.items():
        argv[argv.index(flag) + 1] = str(value)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    if any(x in argv for x in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
        raise ValueError('Never reset the selected saved source')
    return argv, actual + 199, saved


evaluator_tree = _sampler_api.evaluator_tree


def replay_command(source, output):
    require_source(source)
    source_anchor, _, _ = anchor()
    argv = previous.replay_command(source_anchor, output)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    argv[argv.index('--checkpoint') + 1] = str(Path(source).resolve())
    if sha(source) == ANCHOR_SHA:
        argv.extend([FLAG, '--overrides_json', str(QUALIFICATION / 'native_reward_transfer_overrides.json')])
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
        if ('--overrides_json' not in arguments or json.loads(Path(
                arguments[arguments.index('--overrides_json') + 1]).read_text())
                != {'rewards.' + REWARD_KEY: REWARD_VALUE}):
            raise ValueError('The frozen transfer may change exactly this yaw coefficient')
    elif sha(source) == ANCHOR_SHA or '--overrides_json' in arguments:
        raise ValueError('Normal replay requires the exact saved descendant configuration')
    _sampler_api.install_sampling_field_restore()
    import gymnasium as gym
    original_make = gym.make
    gym.make = _sampler_api._wrap_make(original_make)
    tree, path = evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    try:
        exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
            _install_quality_criteria=criteria_fix.install,
            _configure_sampling=_sampler_api._configure,
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
            raise ValueError('Keep the existing native observation-output contract')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only ordinary retained-state PPO and frozen replay are supported')
