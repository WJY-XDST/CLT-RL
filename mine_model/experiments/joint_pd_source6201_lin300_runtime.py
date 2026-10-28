"""One orientation reward-weight trial from fully reviewed6001, with fixed wheel125.

Keep plain27 direct joint-position PPO, the saved six Gaussian heads, all17
Adam states, actual adaptive LR, physical control and the complete envelope.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys
import torch
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
from wheel_legged_gym_isaaclab.wheel_control import validated_wheel_action_limit
import joint_pd_source6001_orientation060_runtime as previous
SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261020_175930_562396_source6001_orientation060_200'
DECISION = RESULTS / 'joint_pd_source6201_lin300_trial_decision_20261020.json'
ANCHOR_SHA = '7c02af3901d2f444568a07867e7d212e3326d4339080844c5fb5b01d660958b2'
(REWARD_KEY, OLD_VALUE, REWARD_VALUE) = ('tracking_lin_vel', 2.0, 3.0)
FLAG = '--repair_fullspeed_lin300'
(LIMIT, MOTION_SCALE) = (1.25, 1.0)
(KEY, base, criteria_fix) = (previous.KEY, previous.base, previous.criteria_fix)
bind = previous.bind
DELTA = dict(rewards={REWARD_KEY: [OLD_VALUE, REWARD_VALUE]}, commands={}, exploration={}, controller={}, physics={})

@lru_cache(maxsize=1)
def anchor():
    final = json.loads((TRAINING / 'final_checkpoint_verification_readonly_20261019.json').read_text())
    source = Path(final['checkpoint'])
    if not final['verified'] or final['actual_ppo_updates'] != 6201 or final['all17_Adam_step'] != 124020 or (sha(source) != ANCHOR_SHA):
        raise ValueError('Require the exact completed6001 and continuous all17 Adam states')
    (saved, cfg) = previous.require_source(source)
    if cfg['rewards'][REWARD_KEY] != OLD_VALUE or cfg['wheel_action_limit'] != LIMIT or cfg['action_scale_vel'] != 15.0 or (cfg['commands']['training_motion_scale'] != 1.0):
        raise ValueError('Preserve the reviewed full-speed wheel125 source')
    return (source, saved, cfg)

def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source6201_fullspeed_orientation060_lin300_v1', native_runtime_sha256=sha(SCRIPT), parent_orientation060_runtime_sha256=sha(previous.SCRIPT), tracking_lin_vel=REWARD_VALUE, linear_precision_trial_anchor_sha256=ANCHOR_SHA, authority='own6001_complete657_static_moving_steering_and120s_reviews')
    return result

def recipe(rewards):
    return previous.recipe(dict(rewards, tracking_lin_vel=OLD_VALUE)) and rewards[REWARD_KEY] == REWARD_VALUE

def require_source(path, transition=False, *, initial_transfer=False):
    if initial_transfer:
        raise ValueError('Never repeat originalmodel200 or reset the optimizer')
    (_, before, cfg) = anchor()
    path = Path(path).resolve()
    if sha(path) == ANCHOR_SHA:
        return previous.require_source(path)
    if transition:
        raise ValueError('Only the declared6001 may cross the single reward-weight boundary')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = copy.deepcopy(cfg)
    expected['rewards'][REWARD_KEY] = REWARD_VALUE
    expected['log_dir'] = current.get('log_dir')
    states = saved['optimizer_state_dict']['state']
    infos = saved.get('infos')
    declared = infos.get(KEY) if isinstance(infos, dict) else None
    if not tensor_finite(saved) or declared != profile() or current != expected or (saved_updates(path) < 6201) or (len(states) != 17) or any((float(v['step']) != 20 * saved_updates(path) for v in states.values())) or (saved['model_state_dict'].keys() != before['model_state_dict'].keys()) or any((saved['model_state_dict'][k].shape != v.shape for (k, v) in before['model_state_dict'].items())):
        raise ValueError('Require this exact reward-weight descendant and every continuous Adam clock')
    return (saved, current)

def repair_ready():
    decision = json.loads(DECISION.read_text())
    read = lambda key: json.loads(Path(decision[key]).read_text())
    (final, full) = (read('source_final_verification'), read('source_full_review'))
    (static, long) = (read('source_static_raw_manifest'), read('source_original_sustained120'))
    if not decision['selected_single_linear_weight_trial'] or decision['source_checkpoint_sha256'] != ANCHOR_SHA or decision['source_actual_updates'] != 6201 or (decision['source_all17_Adam_step'] != 124020) or (decision['planned_actual_updates'] != 200) or (decision['target_actual_updates'] != 6401) or (decision['next_configuration_changes'] != DELTA) or (not final['verified']) or (final['checkpoint_sha256'] != ANCHOR_SHA) or (not full['all_required_review_data_complete']) or (full['actual_updates'] != 6201) or (not static['raw_integrity_verified']) or (not static['static_retention_verified']) or (not long['raw_integrity_verified']) or (long['survivors'] != 54) or long['raw_OR_resets'] or (decision['relative_error_target'] != 0.1) or decision['acceptance_changed'] or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload'] or decision['faster_curriculum_authorized'] or (not decision['all_failures_retained']) or any((sha(p) != h for (p, h) in decision['source_pins'].items())):
        raise ValueError('Require complete own6001 evidence and exactly one selected reward field')
    return decision

def private_runtime(source=None):
    if source is not None and (sha(source) != ANCHOR_SHA or saved_updates(source) != 6201):
        raise ValueError('Only the completely reviewed6001 starts this transfer')
    parent = previous.previous.previous
    module = bind(parent.ordinary.private_runtime, profile=profile, recipe=recipe, MOTION_SCALE=1.0)()
    original_observe = module.observe_environment

    def observe(env, output):
        result = original_observe(env, output)
        raw = env.unwrapped
        if validated_wheel_action_limit(raw.cfg.wheel_action_limit) != LIMIT or raw.cfg.action_scale_vel != 15.0 or (not recipe(raw.cfg.rewards.to_dict())):
            raise ValueError('Actual reward or retained wheel/leg controller differs')
        original = raw._pre_physics_step
        count = 0

        def pre_step(actions):
            nonlocal count
            original(actions)
            if not torch.equal(raw._actions[:, (2, 5)], raw._raw_actions[:, (2, 5)].clamp(-LIMIT, LIMIT)) or not torch.equal(raw._actions[:, (0, 1, 3, 4)], raw._raw_actions[:, (0, 1, 3, 4)].clamp(-1.0, 1.0)):
                raise ValueError('Normal wheel/leg action mapping differs on a live environment')
            count += 1
            if count % 48 == 0:
                write_json(Path(output) / 'normal_wheel_limit_observation.json', dict(actual_policy_steps=count, all_environment_actions_checked=count * raw.num_envs, native_environments=raw.num_envs, wheel_action_limit=LIMIT, action_scale_vel=15.0, wheel_reference_max_rad_s=18.75, wheel_mapping_max_error=0.0, leg_premap_max_error=0.0, observer_changes_actions_or_rewards=False, added_loss_or_backward=False))
        raw._pre_physics_step = pre_step
        write_json(Path(output) / 'lin300_runtime_evidence.json', dict(profile=profile(), authoritative_current_delta=DELTA, source_actual_updates=6201, saved_std_not_reset=True, all_six_std_entries_remain_trainable=True, wheel_action_limit=1.25, action_scale_vel=15.0, wheel_reference_max_rad_s=18.75, height_yaw_command_exploration_physics_reset_and_direct_joint_execution_unchanged=True, original_relative_error_target=0.1, added_loss_or_backward=False, policy_accepted=False))
        return result
    (module.profile, module.require_source, module.observe_environment) = (profile, require_source, observe)
    return module

def install_leg_exploration(output):
    return bind(previous.install_leg_exploration, profile=profile)(output)

def native(output, arguments):
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    if (sha(source) == ANCHOR_SHA) != (FLAG in arguments):
        raise ValueError('Declare the first6001 reward-weight transfer exactly once')
    return bind(previous.previous.native_template.native, SCRIPT=SCRIPT, FLAG=FLAG, repair_ready=repair_ready, require_source=require_source, private_runtime=private_runtime, install_leg_exploration=install_leg_exploration)(output, arguments)

def command(source, *args, **kwargs):
    kwargs = dict(kwargs)
    overrides = dict(kwargs.get('reward_overrides', {}))
    requested = overrides.get('rewards.' + REWARD_KEY, REWARD_VALUE)
    if requested != REWARD_VALUE:
        raise ValueError('Only the selected orientation reward weight is allowed')
    overrides['rewards.' + REWARD_KEY] = OLD_VALUE
    kwargs['reward_overrides'] = overrides
    (argv, final, saved) = bind(previous.command, SCRIPT=SCRIPT, ANCHOR_SHA=ANCHOR_SHA, FLAG=FLAG, require_source=require_source)(source, *args, **kwargs)
    options = [a for a in argv if a.startswith('env.rewards.' + REWARD_KEY + '=')]
    if options != [f'env.rewards.{REWARD_KEY}={OLD_VALUE}']:
        raise ValueError('Requalify the unchanged historical command builder')
    argv[argv.index(options[0])] = f'env.rewards.{REWARD_KEY}={REWARD_VALUE}'
    return (argv, final, saved)
evaluator_tree = previous.evaluator_tree

def replay_command(source, output):
    return bind(previous.replay_command, SCRIPT=SCRIPT, require_source=require_source)(source, output)

def replay(arguments):
    arguments = list(arguments)
    transition = FLAG in arguments
    if transition:
        arguments.remove(FLAG)
        repair_ready()
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source, transition=transition)
    if transition:
        if '--overrides_json' not in arguments or json.loads(Path(arguments[arguments.index('--overrides_json') + 1]).read_text()) != {'rewards.' + REWARD_KEY: REWARD_VALUE}:
            raise ValueError('The frozen transfer changes exactly one orientation reward field')
    elif sha(source) == ANCHOR_SHA or '--overrides_json' in arguments:
        raise ValueError('Normal replay requires a saved descendant and its exact configuration')
    (tree, path) = evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path), _install_quality_criteria=criteria_fix.install, _observe_source200=private_runtime().observe_environment))
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
        raise ValueError('Only ordinary PPO and frozen replay')
