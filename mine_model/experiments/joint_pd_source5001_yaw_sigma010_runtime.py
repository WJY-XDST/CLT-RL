"""Single angular reward width trial from the completely reviewed5001 state.

Keep the saved Gaussian, all17 Adam states, adaptive LR, plain27 direct PD,
physical execution and command envelope. This is an unaccepted repair branch.
"""
import copy
import json
from pathlib import Path
import sys

import joint_pd_source4801_precision010_runtime as previous
import joint_pd_source4001_motion0625_runtime as ordinary
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = '5d6af47009b69a17b63469a12be5cff939380dede7f38d0f6fa5f48860d13176'
FLAG = '--repair_motion0625_yaw_sigma010'
DECISION = ROOT / 'mine_model/results/joint_pd_source5001_yaw_sigma010_trial_decision_v2_20261019.json'
REWARD_KEY, OLD_VALUE, REWARD_VALUE = 'tracking_sigma_ang', .025, .01
MOTION_SCALE = .625
LEG_STD_FLOOR = WHEEL_STD_FLOOR = GLOBAL_STD_FLOOR = .01
DELTA = dict(rewards={REWARD_KEY: [OLD_VALUE, REWARD_VALUE]}, commands={},
    exploration={}, controller={}, physics={})
KEY, base, criteria_fix = previous.KEY, previous.base, previous.criteria_fix
bind = previous.previous.bind


def profile():
    value = copy.deepcopy(previous.profile())
    value.update(profile='plain_source5001_motion0625_yaw_sigma010_v1',
        tracking_sigma_ang=REWARD_VALUE, native_runtime_sha256=sha(SCRIPT),
        parent_precision010_runtime_sha256=sha(previous.SCRIPT),
        angular_precision_trial_anchor_sha256=ANCHOR_SHA,
        authority='own5001_complete_static_fullheight_moving_and_wider_frozen_review')
    return value


def recipe(rewards):
    return (previous.recipe(dict(rewards, tracking_sigma_ang=OLD_VALUE))
        and rewards[REWARD_KEY] == REWARD_VALUE)


def repair_ready():
    decision = json.loads(DECISION.read_text())
    read = lambda name: json.loads(Path(decision[name]).read_text())
    final = read('source_final_verification')
    static = read('source_static_retention_report')
    static_raw = read('source_static_raw_manifest')
    moving = read('source_moving_height_summary')
    moving_raw = read('source_moving_height_raw_integrity')
    noise = read('source_fixed_model_moving_noise_review')
    full = read('source_full_height_summary')
    wider = read('source_wider_frozen_review')
    if (not decision['selected_single_yaw_width_trial']
            or decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 5001 or decision['source_all17_Adam_step'] != 100020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not final['verified'] or final['checkpoint_sha256'] != ANCHOR_SHA
            or final['actual_ppo_updates'] != 5001 or final['all17_Adam_step'] != 100020
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != ANCHOR_SHA
            or not static_raw['raw_integrity_verified'] or static_raw['checkpoint_sha256'] != ANCHOR_SHA
            or moving['checkpoint_sha256'] != ANCHOR_SHA or moving['supplemental_steps_ready'] is not False
            or moving['results']['deterministic']['all_numeric_phase_gates_passed'] is not False
            or any(moving['results'][k]['survivors'] != 27 or moving['results'][k]['resets']
                for k in ('deterministic', 'saved_Gaussian'))
            or not moving_raw['verified'] or moving_raw['checkpoint_sha256'] != ANCHOR_SHA
            or noise['checkpoint_sha256'] != ANCHOR_SHA
            or not noise['same_frozen_checkpoint_case_seed_and_phase_plan']
            or not noise['all54_frozen_trials_survived_without_reset']
            or full['checkpoint_sha256'] != ANCHOR_SHA
            or sum(v['passed'] for v in full['suites'].values()) != 192
            or any(v['survivors'] != 51 or v['resets'] for v in full['suites'].values())
            or wider['actual_updates'] != 5001 or wider['probes']['1.0']['survivors'] != 42
            or wider['probes']['1.0']['resets'] != 72
            or not decision['all_failures_retained_including_wider_stability']
            or not decision['source_is_unaccepted_repair_branch']
            or not decision['reward_trial_benefit_unproven']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset']
            or decision['original_model200_reload'] or decision['faster_curriculum_authorized']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require own5001 reviews, retained failures and the single yaw-width decision')
    return decision


def private_runtime(source=None):
    module = bind(ordinary.private_runtime, profile=profile, recipe=recipe)()
    original_require, original_observe = module.require_source, module.observe_environment
    if source is not None and (sha(source) != ANCHOR_SHA or saved_updates(source) != 5001):
        raise ValueError('Only reviewed5001 may cross this profile boundary')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload originalmodel200 or reset retained Adam')
        if sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        state, cfg = original_require(path, initial_transfer=False)
        if state.get('infos', {}).get(KEY) != profile() or saved_updates(path) < 5001:
            raise ValueError('Require this exact angular-width descendant')
        return state, cfg

    def observe(env, output):
        if not recipe(env.unwrapped.cfg.rewards.to_dict()):
            raise ValueError('Actual rewards differ beyond the declared angular width')
        result = original_observe(env, output)
        from auto_train_mine import write_json
        write_json(Path(output) / 'yaw_sigma010_runtime_evidence.json', dict(profile=profile(),
            authoritative_current_delta=DELTA, source_actual_updates=5001,
            initial_saved_std_not_reset=True, all_six_std_entries_remain_trainable=True,
            inherited_sampler_migration_labels_are_lineage_only=True,
            unchanged_tracking_sigma=.01, unchanged_angular_reward_weight=1.5,
            height_pose_command_exploration_physics_reset_and_direct_joint_execution_unchanged=True,
            added_loss_or_backward=False, relative_error_target=.1, policy_accepted=False))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    if transition and sha(source) != ANCHOR_SHA:
        raise ValueError('Only the declared source can request first transfer')
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    return bind(previous.install_leg_exploration, profile=profile)(output)


def native(output, arguments):
    return bind(ordinary.native, SCRIPT=SCRIPT, ANCHOR_SHA=ANCHOR_SHA, FLAG=FLAG,
        repair_ready=repair_ready, require_source=require_source,
        private_runtime=private_runtime, install_leg_exploration=install_leg_exploration)(output, arguments)


def command(source, *args, **kwargs):
    argv, final, saved = previous.command(source, *args, **kwargs)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    options = [a for a in argv if a.startswith('env.rewards.' + REWARD_KEY + '=')]
    if options not in ([f'env.rewards.{REWARD_KEY}={OLD_VALUE}'],
                       [f'env.rewards.{REWARD_KEY}={REWARD_VALUE}']):
        raise ValueError('Unexpected angular width in the original command')
    argv[argv.index(options[0])] = f'env.rewards.{REWARD_KEY}={REWARD_VALUE}'
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv, final, saved


evaluator_tree = previous.evaluator_tree


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
            raise ValueError('Frozen transfer must declare exactly the angular-width change')
    elif '--overrides_json' in arguments:
        raise ValueError('Descendants replay their exact saved configuration')
    tree, path = evaluator_tree()
    sys.argv = [str(SCRIPT), *base.tires.arguments(arguments)]
    exec(compile(tree, str(path), 'exec'), dict(__name__='__main__', __file__=str(path),
        _install_quality_criteria=criteria_fix.install,
        _observe_source200=private_runtime().observe_environment))


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
            raise ValueError('Keep the original native observation-output contract')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    else:
        raise ValueError('Only ordinary PPO and frozen replay')
