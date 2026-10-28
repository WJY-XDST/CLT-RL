"""One wheel-clamp transfer, retaining plain27 PPO and every saved Adam tensor.

The scale remains 15 rad/s. Only wheel action bounds become +/-1.25;
the four leg position heads and the complete reward/physics recipe are intact.
"""
import copy
from functools import lru_cache
import json
from pathlib import Path
import sys
import types

import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, tensor_finite, optimizer_step
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
from wheel_legged_gym_isaaclab.wheel_control import validated_wheel_action_limit
import joint_pd_source5401_motion100_runtime as previous
import joint_pd_source2201_yaw025_runtime as native_template

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261020_114137_813058_source5601_consolidation200'
DECISION = RESULTS / 'joint_pd_source5801_wheel125_trial_decision_v2_20261020.json'
ANCHOR_SHA = '7533651683efe6c6a2c251688141d01f8fd890e34c0a4f63a138c009b63c0fe9'
FLAG = '--begin_verified_wheel125'
LIMIT = 1.25
KEY, base, criteria_fix, recipe = previous.KEY, previous.base, previous.criteria_fix, previous.recipe
DELTA = dict(rewards={}, commands={}, exploration={}, controller={'wheel_action_limit': [1., LIMIT]}, physics={})


@lru_cache(maxsize=1)
def anchor():
    final = json.loads((TRAINING / 'final_checkpoint_verification_readonly_20261019.json').read_text())
    source = Path(final['checkpoint'])
    if (not final['verified'] or final['actual_ppo_updates'] != 5801
            or final['all17_Adam_step'] != 116020 or sha(source) != ANCHOR_SHA):
        raise ValueError('Require the exact fully retained5801 anchor')
    saved, cfg = previous.require_source(source)
    if cfg.get('wheel_action_limit', 1.) != 1. or cfg['action_scale_vel'] != 15.:
        raise ValueError('The historical controller must retain its unit clamp and scale15')
    return source, saved, cfg


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source5801_fullspeed_wheel125_v1',
        native_runtime_sha256=sha(SCRIPT), parent_fullspeed_runtime_sha256=sha(previous.SCRIPT),
        wheel_clamp_transfer_anchor_sha256=ANCHOR_SHA, wheel_action_limit=LIMIT,
        action_scale_vel=15., wheel_reference_max_rad_s=18.75,
        wheel_controller_sha256=sha(ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/wheel_control.py'),
        authority='paired_current324_and120s_wheel_clamp_diagnosis; not_hardware_rating_or_acceptance')
    return result


def require_source(path, transition=False, *, initial_transfer=False):
    if initial_transfer:
        raise ValueError('Never reload originalmodel200 or reset the optimizer')
    source, before, cfg = anchor()
    path = Path(path).resolve()
    if sha(path) == ANCHOR_SHA:
        return previous.require_source(path)
    if transition:
        raise ValueError('Only the declared5801 anchor crosses the wheel-clamp boundary')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    expected = copy.deepcopy(cfg)
    expected['wheel_action_limit'] = LIMIT
    expected['log_dir'] = current.get('log_dir')
    states = saved['optimizer_state_dict']['state']
    if (not tensor_finite(saved) or saved.get('infos', {}).get(KEY) != profile()
            or current != expected or saved_updates(path) < 5801
            or len(states) != 17 or any(float(v['step']) != 20 * saved_updates(path) for v in states.values())
            or saved['model_state_dict'].keys() != before['model_state_dict'].keys()
            or any(saved['model_state_dict'][k].shape != v.shape for k, v in before['model_state_dict'].items())):
        raise ValueError('Require the exact wheel125 plain27 descendant and all17 continuous Adam clocks')
    return saved, current


def repair_ready():
    decision = json.loads(DECISION.read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 5801 or decision['source_all17_Adam_step'] != 116020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not decision['selected_single_controller_clamp_trial'] or not decision['all_failures_retained']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or any(sha(p) != digest for p, digest in decision['source_pins'].items())):
        raise ValueError('Require the reviewed single-cap trial and preserved historical evidence')
    return decision


def private_runtime(source=None):
    if source is not None and sha(source) != ANCHOR_SHA:
        raise ValueError('Only the reviewed5801 state starts this transfer')
    module = previous.private_runtime()
    original_observe = module.observe_environment

    def observe(env, output):
        result = original_observe(env, output)
        raw = env.unwrapped
        limit = validated_wheel_action_limit(raw.cfg.wheel_action_limit)
        if limit != LIMIT or raw.cfg.action_scale_vel != 15.:
            raise ValueError('Actual wheel reference bounds differ from the qualified single-cap change')
        original = raw._pre_physics_step
        count = 0

        def pre_step(actions):
            nonlocal count
            original(actions)
            if (not torch.equal(raw._actions[:, (2, 5)], raw._raw_actions[:, (2, 5)].clamp(-LIMIT, LIMIT))
                    or not torch.equal(raw._actions[:, (0, 1, 3, 4)],
                        raw._raw_actions[:, (0, 1, 3, 4)].clamp(-1., 1.))):
                raise ValueError('Normal wheel/leg action mapping differs on a live environment')
            count += 1
            if count % 48 == 0:
                write_json(Path(output) / 'normal_wheel_limit_observation.json', dict(
                    actual_policy_steps=count, all_environment_actions_checked=count * raw.num_envs,
                    native_environments=raw.num_envs, wheel_action_limit=LIMIT,
                    action_scale_vel=15., wheel_reference_max_rad_s=18.75,
                    wheel_mapping_max_error=0., leg_premap_max_error=0.,
                    observer_changes_actions_or_rewards=False, added_loss_or_backward=False))

        raw._pre_physics_step = pre_step
        write_json(Path(output) / 'wheel125_runtime_evidence.json', dict(profile=profile(),
            only_controller_change=DELTA['controller'], direct_joint_target_heads=[0, 1, 3, 4],
            wheel_velocity_heads=[2, 5], wheel_action_limit=LIMIT, action_scale_vel=15.,
            wheel_reference_max_rad_s=18.75, separate_motor_speed_limit_unverified=True,
            physics_rewards_commands_reset_and_saved_std_preserved=True,
            original_relative_error_target=.1, acceptance_changed=False, policy_accepted=False))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require_source, observe
    return module


def install_leg_exploration(output):
    return previous.bind(previous.install_leg_exploration, profile=profile)(output)


def native(output, arguments):
    source = Path(arguments[arguments.index('--checkpoint_path') + 1])
    if (sha(source) == ANCHOR_SHA) != (FLAG in arguments):
        raise ValueError('Declare the first5801 wheel-clamp transfer exactly once')
    function = previous.bind(native_template.native, SCRIPT=SCRIPT, FLAG=FLAG,
        repair_ready=repair_ready, require_source=require_source, private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)
    return function(output, arguments)


def command(source, *args, **kwargs):
    require_source(source)
    argv, final, saved = previous.command(source, *args, **kwargs)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    argv.append('env.wheel_action_limit=1.25')
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv, final, saved


def replay_command(source, output):
    require_source(source)
    argv = previous.replay_command(source, output)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    return argv


def replay(arguments):
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source)
    if sha(source) == ANCHOR_SHA or FLAG in arguments or '--overrides_json' in arguments:
        raise ValueError('Normal replay requires a saved wheel125 configuration without interventions')
    tree, path = previous.evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
        _install_quality_criteria=criteria_fix.install, _observe_source200=private_runtime().observe_environment))


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
        raise ValueError('Only retained-state ordinary PPO and normal frozen replay')
