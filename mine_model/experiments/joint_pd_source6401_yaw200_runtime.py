"""One provisional yaw-weight1.5 to2.0 repair from the fully reviewed6401.

The entry refuses execution until its source/retained/120s/fullspeed-height
reviews and explicit evidence-based trial decision are complete. Keep all17
Adam states, actual LR, all six std entries, physics and direct joint targets.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path

import torch

from joint_pd_continuation import ROOT, sha, tensor_finite, validate_continuation_config
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
import plain27_sampler_metadata_adapter as adapter
import joint_pd_source6201_lin300_runtime as previous

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261020_recovered_source6201_lin300_200'
DECISION = RESULTS / 'joint_pd_source6401_yaw200_trial_decision_20261021.json'
ANCHOR_SHA = '2e5d10432cfc7575cde6cd56abf439a4e784161e130a20fb80940be685dc5120'
REWARD_KEY, OLD_VALUE, REWARD_VALUE = 'tracking_ang_vel', 1.5, 2.0
FLAG = '--repair_fullspeed_yaw200'
LIMIT, MOTION_SCALE = 1.25, 1.0
KEY, base, criteria_fix, bind = previous.KEY, previous.base, previous.criteria_fix, previous.bind
DELTA = dict(rewards={REWARD_KEY: [OLD_VALUE, REWARD_VALUE]}, commands={},
    exploration={}, controller={}, physics={})
PROTOTYPE = RESULTS / 'joint_pd_optimization_20261020_195952_083563_source6201_lin300_200/launch.json'
PROTOTYPE_PROOF = RESULTS / 'joint_pd_lin300_recovery_qualification_20261020/qualification.json'


@lru_cache(maxsize=1)
def anchor():
    final = json.loads((TRAINING / 'final_checkpoint_verification_readonly_20261019.json').read_text())
    source = Path(final['checkpoint'])
    if (not final['verified'] or final['actual_ppo_updates'] != 6401
            or final['all17_Adam_step'] != 128020 or sha(source) != ANCHOR_SHA):
        raise ValueError('Require exact6401, its full167+33 ledger and every retained Adam clock')
    saved, cfg = previous.require_source(source)
    if (cfg['rewards'][REWARD_KEY] != OLD_VALUE or cfg['rewards']['tracking_lin_vel'] != 3.
            or cfg['rewards']['orientation'] != -60. or cfg['wheel_action_limit'] != LIMIT
            or cfg['commands']['training_motion_scale'] != 1.):
        raise ValueError('Keep the exact reviewed linear3/orientation60 source')
    return source, saved, cfg


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source6401_fullspeed_lin300_yaw200_v1',
        native_runtime_sha256=sha(SCRIPT), parent_linear_runtime_sha256=sha(previous.SCRIPT),
        tracking_ang_vel=REWARD_VALUE, angular_weight_trial_anchor_sha256=ANCHOR_SHA,
        authority='own6401_complete657_retained78_static_moving_steering120s_and_signed_fullspeed_height_reviews')
    return result


def recipe(rewards):
    old = dict(rewards)
    old[REWARD_KEY] = OLD_VALUE
    return previous.recipe(old) and rewards[REWARD_KEY] == REWARD_VALUE


def require_source(path, transition=False, *, initial_transfer=False):
    if initial_transfer:
        raise ValueError('Originalmodel200 may not be reloaded')
    _, before, cfg = anchor()
    path = Path(path).resolve()
    if sha(path) == ANCHOR_SHA:
        return previous.require_source(path)
    if transition:
        raise ValueError('Only the exact reviewed6401 starts this one-field transfer')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = copy.deepcopy(cfg)
    expected['rewards'][REWARD_KEY] = REWARD_VALUE
    expected['log_dir'] = current.get('log_dir')
    states = saved['optimizer_state_dict']['state']
    infos = saved.get('infos')
    declared = infos.get(KEY) if isinstance(infos, dict) else None
    if (not tensor_finite(saved) or declared != profile() or current != expected
            or saved_updates(path) < 6401 or len(states) != 17
            or any(float(x['step']) != 20 * saved_updates(path) for x in states.values())
            or saved['model_state_dict'].keys() != before['model_state_dict'].keys()
            or any(saved['model_state_dict'][k].shape != value.shape
                for k, value in before['model_state_dict'].items())):
        raise ValueError('Require this exact reward descendant and complete finite17 Adam states')
    return saved, current


def repair_ready():
    decision = json.loads(DECISION.read_text())
    read = lambda key: json.loads(Path(decision[key]).read_text())
    final, full = read('source_final_verification'), read('source_full_review')
    static, long = read('source_static_raw_manifest'), read('source_original_sustained120')
    extra = read('source_fullspeed_height_transition_review')
    modes = extra['results']['candidate6401']['modes']
    if (not decision['selected_single_angular_weight_trial']
            or decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 6401 or decision['source_all17_Adam_step'] != 128020
            or decision['planned_actual_updates'] != 200 or decision['target_actual_updates'] != 6601
            or decision['next_configuration_changes'] != DELTA
            or not final['verified'] or final['checkpoint_sha256'] != ANCHOR_SHA
            or not full['all_original_and_retained_reviews_complete'] or full['actual_updates'] != 6401
            or not static['raw_integrity_verified'] or not static['static_retention_verified']
            or not long['raw_integrity_verified'] or long['survivors'] != 54 or long['raw_OR_resets']
            or not extra['completed'] or len(modes) != 2
            or any(m['survivors'] != 27 or m['resets'] for m in modes)
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or decision['faster_curriculum_authorized'] or not decision['all_failures_retained']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require every own6401 review and one explicitly selected angular-weight repair')
    return decision


# Retain the qualified ordinary runtime implementation and its original base
# provider chain. Retarget only the source clock and owned evidence basename.
_provider = adapter.private_module(previous.SCRIPT, '_source6401_yaw200_private_runtime_provider')
PRIVATE_ADAPTER = adapter.retarget(_provider, ('private_runtime',),
    integers={'*': {6201: 6401}}, strings={'lin300': 'yaw200', 'reviewed6001': 'reviewed6401'})
for _name in ('SCRIPT', 'ANCHOR_SHA', 'FLAG', 'REWARD_KEY', 'OLD_VALUE', 'REWARD_VALUE', 'DELTA',
        'profile', 'recipe', 'require_source', 'repair_ready'):
    setattr(_provider, _name, globals()[_name])
private_runtime = _provider.private_runtime
install_leg_exploration = _provider.install_leg_exploration
evaluator_tree = _provider.evaluator_tree


def native(output, arguments):
    return _provider.native(output, arguments)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    source, stop_file = Path(source).resolve(), Path(stop_file).resolve()
    saved, actual_cfg = require_source(source)
    if updates != 200 or cfg != actual_cfg or commands != cfg['commands']:
        raise ValueError('Keep an exact200 ordinary continuation and the entire saved command envelope')
    overrides = kwargs.pop('reward_overrides', {})
    # The preserved full200 helper passes its historical nominal floor. The
    # existing runtime already replaces it with the saved .01 profile. Keep
    # that compatibility explicit; the resulting native CLI remains .01.
    nominal_floor = kwargs.pop('minimum_wheel_std', .01)
    guard_coef = kwargs.pop('joint_mean_guard_coef', 0.)
    feedback_coef = kwargs.pop('joint_height_feedback_coef', 0.)
    if nominal_floor not in (.01, .03) or guard_coef != 0. or feedback_coef != 0.:
        raise ValueError('Keep the saved exploration profile and both auxiliary losses disabled')
    if kwargs or any(k != 'rewards.' + REWARD_KEY or value != REWARD_VALUE for k, value in overrides.items()):
        raise ValueError('Only the selected existing angular weight may change')
    expected = copy.deepcopy(cfg)
    expected['rewards'][REWARD_KEY] = REWARD_VALUE
    validate_continuation_config(cfg, expected, commands,
        reward_overrides={'rewards.' + REWARD_KEY: REWARD_VALUE})
    proof = json.loads(PROTOTYPE_PROOF.read_text())
    prototype = json.loads(PROTOTYPE.read_text())
    if (not proof['qualified'] or sha(PROTOTYPE) != proof['source_pins'][str(PROTOTYPE)]
            or prototype['source_sha256'] != previous.ANCHOR_SHA):
        raise ValueError('Preserve the successful ordinary200 native CLI prototype')
    argv = list(prototype['command'])
    if argv[1] != str(previous.SCRIPT) or argv.count(previous.FLAG) != 1:
        raise ValueError('Unexpected prototype runtime or first-transfer flag')
    argv[1] = str(SCRIPT)
    argv.remove(previous.FLAG)
    source_updates = saved_updates(source)
    for flag, value in {
            '--observation-output': stop_file.parent / 'native_learning_signal',
            '--checkpoint_path': source, '--target_iteration': source_updates + updates,
            '--run_name': label, '--stop_file': stop_file}.items():
        argv[argv.index(flag) + 1] = str(value)
    option = f'env.rewards.{REWARD_KEY}={OLD_VALUE}'
    if (argv.count(option) != 1 or argv.count('env.rewards.tracking_lin_vel=3.0') != 1
            or argv.count('agent.save_interval=1') != 1):
        raise ValueError('Prototype reward or per-update checkpoint retention changed')
    argv[argv.index(option)] = f'env.rewards.{REWARD_KEY}={REWARD_VALUE}'
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    if any(flag in argv for flag in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
        raise ValueError('Never reset the retained source')
    return argv, source_updates + updates - 1, saved


def replay_command(source, output):
    return _provider.replay_command(source, output)


def replay(arguments):
    return _provider.replay(arguments)


if __name__ == '__main__':
    import sys
    mode = sys.argv.pop(1)
    if mode == 'replay':
        replay(sys.argv[1:])
    elif mode == 'native':
        boundary = sys.argv.index('--')
        if sys.argv[1:boundary] != ['--observation-output', sys.argv[2]]:
            raise ValueError('Keep the original native output contract')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only frozen replay or ordinary retained-state native PPO')
