"""One leg-floor trial from the better fully reviewed6801 branch.

Retain that checkpoint's27-input policy,17 Adam states, adaptive LR, six
initial std, existing targeted.3 sampler and complete physical recipe. Only
the future four-leg floor changes from.01 to.005; the wheel floor stays.01.
The rejected6601 branch and its complete evidence remain preserved.
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
import joint_pd_source200_leg05_runtime as projection_template

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261021_032504_524539_source6601_targeted_turn030_200'
DECISION = RESULTS / 'joint_pd_preserved6801_leg005_trial_decision_20261021.json'
ANCHOR_SHA = '82d733cd85618ca78365be77a490eb3e85a196ee87862a8821fe60caef0108e0'
FLAG = '--repair_preserved6801_leg005'
LEG_STD_FLOOR = GLOBAL_STD_FLOOR = .005
WHEEL_STD_FLOOR = .01
MOTION_SCALE = 1.
KEY, base, criteria_fix, bind, recipe = previous.KEY, previous.base, previous.criteria_fix, previous.bind, previous.recipe
DELTA = dict(rewards={}, commands={}, exploration={
    'minimum_leg_std': [.01, .005], 'global_minimum_std': [.01, .005]}, controller={}, physics={})


@lru_cache(maxsize=1)
def anchor():
    final = json.loads((TRAINING / 'final_checkpoint_verification_readonly_20261019.json').read_text())
    source = Path(final['checkpoint'])
    saved, cfg = previous.require_source(source)
    if (not final['verified'] or final['actual_ppo_updates'] != 6801
            or final['all17_Adam_step'] != 136020 or sha(source) != ANCHOR_SHA
            or cfg['rewards']['yaw_rate_error_sq'] != -1.
            or cfg['rewards']['tracking_lin_vel'] != 3. or cfg['rewards']['tracking_ang_vel'] != 2.
            or cfg['rewards']['orientation'] != -60.
            or cfg['wheel_action_limit'] != 1.25 or cfg['action_scale_vel'] != 15.
            or cfg['commands']['training_motion_scale'] != 1.
            or cfg['commands']['targeted_turn_command_fraction'] != .3):
        raise ValueError('Require exact preserved6801, its sampler and the whole saved recipe')
    return source, saved, cfg


def profile():
    value = copy.deepcopy(previous.profile())
    value.update(profile='plain_preserved6801_targeted030_leg005_v1',
        native_runtime_sha256=sha(SCRIPT), parent_targeted030_runtime_sha256=sha(previous.SCRIPT),
        minimum_leg_std=LEG_STD_FLOOR, minimum_wheel_std=WHEEL_STD_FLOOR,
        global_minimum_std=GLOBAL_STD_FLOOR, changed_noise_heads=[0, 1, 3, 4],
        exploration_trial_anchor_sha256=ANCHOR_SHA,
        authority='complete6801_and6601_fixed735_static810_120s_signed_height_comparison')
    return value


def require_source(path, transition=False, *, initial_transfer=False):
    if initial_transfer:
        raise ValueError('Originalmodel200 and optimizer resets remain forbidden')
    _, before, cfg = anchor()
    path = Path(path).resolve()
    if sha(path) == ANCHOR_SHA:
        return previous.require_source(path)
    if transition:
        raise ValueError('Only exact preserved6801 may begin this floor trial')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = copy.deepcopy(cfg)
    expected['log_dir'] = current.get('log_dir')
    actual = saved_updates(path)
    states = saved['optimizer_state_dict']['state']
    if (not tensor_finite(saved) or current != expected
            or saved.get('infos', {}).get(KEY) != profile() or actual < 6801
            or len(states) != 17 or any(float(v['step']) != actual * 20 for v in states.values())
            or saved['model_state_dict'].keys() != before['model_state_dict'].keys()
            or any(saved['model_state_dict'][key].shape != value.shape
                   for key, value in before['model_state_dict'].items())):
        raise ValueError('Require this exact floor descendant and every retained Adam clock')
    return saved, current


def repair_ready():
    d = json.loads(DECISION.read_text())
    read = lambda key: json.loads(Path(d[key]).read_text())
    final, full, static, sustained, signed = [read(k) for k in (
        'source_final_verification', 'source_full_review', 'source_static_retention_report',
        'source_original_sustained120', 'source_fullspeed_height_transition_review')]
    rejected, rejected_signed = [read(k) for k in ('previous_trial_full_review', 'previous_trial_signed_height_review')]
    fixed = read('all735_source_selection_comparison')['summaries']
    if (not d['selected_leg_only_exploration_floor_trial']
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
            or rejected['actual_updates'] != 6601 or not rejected_signed['completed']
            or fixed['preserved6801']['original_AND_literal10_passed'] != 609
            or fixed['candidate6601']['original_AND_literal10_passed'] != 561
            or fixed['preserved6801']['body_original_gate_failures']
            or fixed['candidate6601']['body_original_gate_failures']
            or d['relative_error_target'] != .1 or d['acceptance_changed']
            or d['optimizer_reset'] or d['std_reset'] or d['original_model200_reload']
            or not d['all_failures_and_regressions_retained']
            or any(sha(p) != h for p, h in d['source_pins'].items())):
        raise ValueError('Require every completed branch review and one selected floor-only200 trial')
    return d


# Reuse fresh source functions rather than compiled adapters with stale line
# locations. Their execution operations are unchanged after undoing metadata.
_plant = adapter.private_module(previous.previous.previous.SCRIPT, '_preserved6801_leg005_plant_guard')
PLANT_METADATA = adapter.retarget(_plant, ('private_runtime',),
    integers={'*': {6201: 6801}},
    strings={'lin300_runtime_evidence.json': 'preserved6801_leg005_base_execution_evidence.json'},
    keyword_names={'height_yaw_command_exploration_physics_reset_and_direct_joint_execution_unchanged':
                   'height_yaw_command_physics_reset_and_direct_joint_execution_unchanged'})
for _name in ('SCRIPT', 'ANCHOR_SHA', 'FLAG', 'DELTA', 'profile', 'recipe', 'require_source', 'repair_ready'):
    setattr(_plant, _name, globals()[_name])
_sampler_api = adapter.private_module(previous.SCRIPT, '_preserved6801_leg005_retained_sampler_guard')
for _name in ('SCRIPT', 'ANCHOR_SHA', 'FLAG', 'DELTA', 'DECISION', 'TRAINING',
              'profile', 'recipe', 'require_source', 'repair_ready'):
    setattr(_sampler_api, _name, globals()[_name])
_sampler_api._provider = _plant
SAMPLER_METADATA = adapter.retarget(_sampler_api, ('private_runtime',),
    integers={'*': {6601: 6801}},
    strings={'targeted_turn_runtime_evidence.json': 'retained_targeted030_leg005_runtime_evidence.json'},
    keyword_names={'reward_exploration_physics_reset_and_control_unchanged':
                   'reward_physics_reset_and_control_unchanged',
                   'sampler_RNG_draws_intentionally_added_only_for_mature_nonfixed_training':
                   'retained_sampler_RNG_draws_only_for_mature_nonfixed_training'})
_sampler_api.FACTORY_METADATA = PLANT_METADATA + SAMPLER_METADATA
_sampler_factory = _sampler_api.private_runtime


def private_runtime(source=None):
    _sampler_api.install_sampling_field_restore()
    return _sampler_factory(source)


_sampler_api.private_runtime = private_runtime


def install_leg_exploration(output):
    def log(*values, **kwargs):
        old = '[INFO]: User-requested leg-only exploration floor .05; wheel floor .03 and all Adam states retained.'
        if values and values[0] == old:
            values = ('[INFO]: Preserved6801: future leg floor .005; wheel floor .01; saved std and17 Adam states retained.', *values[1:])
        print(*values, **kwargs)
    return bind(projection_template.install_leg_exploration,
        LEG_STD_FLOOR=LEG_STD_FLOOR, GLOBAL_STD_FLOOR=GLOBAL_STD_FLOOR,
        WHEEL_STD_FLOOR=WHEEL_STD_FLOOR, profile=profile, print=log)(output)


_sampler_api.install_leg_exploration = install_leg_exploration


def native(output, arguments):
    return _sampler_api.native(output, arguments)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    source, stop_file = Path(source).resolve(), Path(stop_file).resolve()
    saved, actual_cfg = require_source(source)
    if updates != 200 or cfg != actual_cfg or commands != actual_cfg['commands']:
        raise ValueError('Keep exactly200 ordinary updates and every saved environment field')
    overrides = kwargs.pop('reward_overrides', {})
    if any(not k.startswith('rewards.') or actual_cfg['rewards'].get(k[8:]) != v
           for k, v in overrides.items()):
        raise ValueError('This trial cannot change any source6801 reward')
    if (kwargs.pop('minimum_wheel_std', .01) not in (.01, .03)
            or kwargs.pop('joint_mean_guard_coef', 0.) != 0.
            or kwargs.pop('joint_height_feedback_coef', 0.) != 0. or kwargs):
        raise ValueError('Keep the six trainable heads, wheel floor and ordinary PPO')
    launch = json.loads((TRAINING / 'launch.json').read_text())
    argv = list(launch['command'])
    if (launch['source_sha256'] != previous.ANCHOR_SHA or argv[1] != str(previous.SCRIPT)
            or argv.count(previous.FLAG) != 1 or argv.count('agent.save_interval=1') != 1
            or argv[argv.index('--min_exploration_std') + 1] != '0.01'
            or argv[argv.index('--min_wheel_exploration_std') + 1] != '0.01'
            or argv.count('env.rewards.yaw_rate_error_sq=-1.0') != 1
            or argv.count('env.rewards.tracking_lin_vel=3.0') != 1
            or argv.count('env.rewards.tracking_ang_vel=2.0') != 1):
        raise ValueError('Requalify the exact completed6801 CLI prototype')
    argv[1] = str(SCRIPT)
    argv.remove(previous.FLAG)
    actual = saved_updates(source)
    for flag, value in {'--observation-output': stop_file.parent / 'native_learning_signal',
            '--checkpoint_path': source, '--target_iteration': actual + 200,
            '--run_name': label, '--stop_file': stop_file,
            '--min_exploration_std': GLOBAL_STD_FLOOR}.items():
        argv[argv.index(flag) + 1] = str(value)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    if any(x in argv for x in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
        raise ValueError('Never reset the saved policy, std or optimizer')
    return argv, actual + 199, saved


evaluator_tree = _sampler_api.evaluator_tree


def replay_command(source, output):
    require_source(source)
    source_anchor, _, _ = anchor()
    argv = previous.replay_command(source_anchor, output)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    argv[argv.index('--checkpoint') + 1] = str(Path(source).resolve())
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv


def replay(arguments):
    return _sampler_api.replay(arguments)


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
        raise ValueError('Only original ordinary PPO or frozen replay')
