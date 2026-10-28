"""One reviewed sampler-only transfer to the user's full speed envelope."""
import copy
import json
from pathlib import Path
import sys

import joint_pd_source5201_motion075_runtime as previous
import joint_pd_source4001_motion0625_runtime as ordinary
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = '874773e92d5da143c4c9f30cd31f763b02652c38aa86feee7f2f14d94db6ddc4'
FLAG = '--repair_verified_motion100'
DECISION = ROOT / 'mine_model/results/joint_pd_source5401_motion100_decision_20261019.json'
COMMAND_KEY, OLD_VALUE, COMMAND_VALUE = 'training_motion_scale', .75, 1.
MOTION_SCALE = COMMAND_VALUE
DELTA = dict(rewards={}, commands={COMMAND_KEY: [OLD_VALUE, COMMAND_VALUE]},
    exploration={}, controller={}, physics={})
KEY, base, criteria_fix, bind = previous.KEY, previous.base, previous.criteria_fix, previous.bind
recipe, evaluator_tree = previous.recipe, previous.evaluator_tree


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source5401_motion100_yaw_sigma010_v1',
        training_motion_scale=MOTION_SCALE, native_runtime_sha256=sha(SCRIPT),
        parent_motion075_runtime_sha256=sha(previous.SCRIPT),
        motion_expansion_anchor_sha256=ANCHOR_SHA,
        authority='user_full_speed_request_and_own5401_static_moving_fullheight_combined_reviews')
    return result


def repair_ready():
    decision = json.loads(DECISION.read_text())
    read = lambda key: json.loads(Path(decision[key]).read_text())
    final, raw = read('source_final_verification'), read('source_static_raw_manifest')
    moving, moving_raw = read('source_moving_height_summary'), read('source_moving_height_raw_integrity')
    all_errors, paired = read('source_all_errors_including_full_speed'), read('source_combined_turns_summary')
    steering, full = read('source_forward_steering_steps_summary'), read('wider_frozen_report')
    steering_spec = read('source_forward_steering_steps_qualification')
    user_request = read('user_full_speed_sampling_request')
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 5401 or decision['source_all17_Adam_step'] != 108020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not decision['source_is_unaccepted_repair_branch']
            or not decision['user_requested_full_actual_sampling_selected']
            or not final['verified'] or final['checkpoint_sha256'] != ANCHOR_SHA
            or final['actual_ppo_updates'] != 5401 or final['all17_Adam_step'] != 108020
            or not raw['raw_integrity_verified'] or not raw['static_retention_verified']
            or raw['checkpoint_sha256'] != ANCHOR_SHA or raw['all17_Adam_step'] != 108020
            or moving['checkpoint_sha256'] != ANCHOR_SHA or not moving_raw['verified']
            or moving_raw['checkpoint_sha256'] != ANCHOR_SHA
            or not all_errors['verified'] or all_errors['checkpoint_sha256'] != ANCHOR_SHA
            or all_errors['total_trials'] != 657 or all_errors['all7_metrics_and_limits_raw_reproduction_max_error']
            or paired['checkpoint_sha256'] != ANCHOR_SHA or paired['actual_updates'] != 5401
            or any(paired['results'][name]['total_trials'] != 324 for name in ('baseline', 'candidate'))
            or not paired['same_reset_seeds_initial_observations_and_physical_configuration_verified']
            or steering['checkpoint_sha256'] != ANCHOR_SHA or steering['total_trials'] != 27
            or not steering_spec['qualified'] or not steering_spec['full_height_coverage_verified']
            or steering_spec['checkpoint_sha256'] != ANCHOR_SHA
            or len(set(steering_spec['actual_tested_heights_m'])) != 3
            or Path(steering['original_report']).parent.parent != Path(steering_spec['output'])
            or full['checkpoint_sha256'] != ANCHOR_SHA or full['tested_command_scale'] != 1.
            or full['total_trials'] != 51 or full['training_modified'] or full['optimizer_updated']
            or user_request['user_requested_speed_range_m_s'] != [-.8, .8]
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or not decision['all_wider_combined_and_transition_failures_preserved']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require complete own5401 reviews and retain every failure before full-speed learning')
    return decision


def private_runtime(source=None):
    module = bind(ordinary.private_runtime, profile=profile, recipe=recipe, MOTION_SCALE=MOTION_SCALE)()
    original_require, original_observe = module.require_source, module.observe_environment
    if source is not None and (sha(source) != ANCHOR_SHA or saved_updates(source) != 5401):
        raise ValueError('Only the completely reviewed5401 checkpoint crosses this sampler profile')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload originalmodel200 or reset retained Adam')
        if sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        state, cfg = original_require(path, initial_transfer=False)
        if (state.get('infos', {}).get(KEY) != profile() or saved_updates(path) < 5401
                or cfg['commands'][COMMAND_KEY] != MOTION_SCALE):
            raise ValueError('Require the exact full-speed descendant profile')
        return state, cfg

    def observe(env, output):
        result = original_observe(env, output)
        from auto_train_mine import write_json
        write_json(Path(output) / 'motion100_runtime_evidence.json', dict(profile=profile(),
            authoritative_current_delta=DELTA, only_command_change=DELTA['commands'],
            reward_exploration_physics_reset_and_direct_joint_execution_unchanged=True,
            actual_command_speed_range_m_s=[-.8, .8], actual_command_yaw_range_rad_s=[-.3, .3],
            saved_std_not_reset=True, all_six_std_entries_remain_trainable=True,
            inherited_motion0625_evidence_filename_is_lineage_only=True,
            relative_error_target=.1, added_loss_or_backward=False, policy_accepted=False))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    if transition and sha(source) != ANCHOR_SHA:
        raise ValueError('Only the declared source can request the first full-speed transfer')
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
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv, final, saved


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
                != {'commands.' + COMMAND_KEY: COMMAND_VALUE}):
            raise ValueError('Frozen transfer changes only the declared full-speed sampler field')
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
