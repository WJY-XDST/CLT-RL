"""One sampler-only expansion from the completely reviewed5201 checkpoint."""
import copy
import json
from pathlib import Path
import sys

import joint_pd_source5001_yaw_sigma010_runtime as previous
import joint_pd_source4001_motion0625_runtime as ordinary
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = '3579c3c905781a6347374da6d542be85eea4f0231ec9f0bb1e80d1cec2589c89'
FLAG = '--repair_verified_motion075'
DECISION = ROOT / 'mine_model/results/joint_pd_source5201_motion075_decision_20261019.json'
COMMAND_KEY, OLD_VALUE, COMMAND_VALUE = 'training_motion_scale', .625, .75
MOTION_SCALE = COMMAND_VALUE
DELTA = dict(rewards={}, commands={COMMAND_KEY: [OLD_VALUE, COMMAND_VALUE]},
    exploration={}, controller={}, physics={})
KEY, base, criteria_fix, bind = previous.KEY, previous.base, previous.criteria_fix, previous.bind
recipe, evaluator_tree = previous.recipe, previous.evaluator_tree


def profile():
    result = copy.deepcopy(previous.profile())
    result.update(profile='plain_source5201_motion075_yaw_sigma010_v1',
        training_motion_scale=MOTION_SCALE, native_runtime_sha256=sha(SCRIPT),
        parent_yaw_sigma010_runtime_sha256=sha(previous.SCRIPT),
        motion_expansion_anchor_sha256=ANCHOR_SHA,
        authority='own5201_complete_static_moving_fullheight_and_frozen075_physical_review')
    return result


def repair_ready():
    decision = json.loads(DECISION.read_text())
    read = lambda key: json.loads(Path(decision[key]).read_text())
    final = read('source_final_verification')
    static = read('source_static_retention_report')
    raw = read('source_static_raw_manifest')
    moving = read('source_moving_height_summary')
    moving_raw = read('source_moving_height_raw_integrity')
    scout = read('wider_frozen_report')
    noise = read('source_saved_Gaussian51_comparison')
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 5201 or decision['source_all17_Adam_step'] != 104020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not decision['source_is_unaccepted_repair_branch']
            or not decision['gradual_command_expansion_selected']
            or not final['verified'] or final['checkpoint_sha256'] != ANCHOR_SHA
            or final['actual_ppo_updates'] != 5201 or final['all17_Adam_step'] != 104020
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != ANCHOR_SHA
            or not raw['raw_integrity_verified'] or raw['checkpoint_sha256'] != ANCHOR_SHA
            or moving['checkpoint_sha256'] != ANCHOR_SHA or not moving['supplemental_steps_ready']
            or not moving['results']['deterministic']['all_numeric_phase_gates_passed']
            or any(moving['results'][k]['survivors'] != 27 or moving['results'][k]['resets']
                for k in ('deterministic', 'saved_Gaussian'))
            or not moving_raw['verified'] or moving_raw['checkpoint_sha256'] != ANCHOR_SHA
            or scout['checkpoint_sha256'] != ANCHOR_SHA or scout['tested_command_scale'] != MOTION_SCALE
            or scout['total_trials'] != 51 or scout['survivors'] != 51 or scout['resets']
            or scout['training_modified'] or scout['optimizer_updated']
            or any(not t['survived'] or t['failed_prerequisites']
                or set(t['failed_metrics']) - {'yaw_mae_rad_s'} for t in scout['trials'])
            or noise['checkpoint_sha256'] != ANCHOR_SHA
            or not noise['bitwise_equal_initial_observations_and_means']
            or noise['saved_Gaussian_survivors'] != 51 or noise['saved_Gaussian_resets']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or not decision['all_failures_retained_including_full_speed_height_and_stability']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require intact own5201 reviews; sampler expansion is not final acceptance')
    return decision


def private_runtime(source=None):
    module = bind(ordinary.private_runtime, profile=profile, recipe=recipe, MOTION_SCALE=MOTION_SCALE)()
    original_require, original_observe = module.require_source, module.observe_environment
    if source is not None and (sha(source) != ANCHOR_SHA or saved_updates(source) != 5201):
        raise ValueError('Only the reviewed5201 checkpoint crosses this sampler profile')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload originalmodel200 or reset retained Adam')
        if sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        state, cfg = original_require(path, initial_transfer=False)
        if (state.get('infos', {}).get(KEY) != profile() or saved_updates(path) < 5201
                or cfg['commands'][COMMAND_KEY] != MOTION_SCALE):
            raise ValueError('Require the exact motion075 descendant profile')
        return state, cfg

    def observe(env, output):
        result = original_observe(env, output)
        from auto_train_mine import write_json
        write_json(Path(output) / 'motion075_runtime_evidence.json', dict(profile=profile(),
            authoritative_current_delta=DELTA, only_command_change=DELTA['commands'],
            reward_exploration_physics_reset_and_direct_joint_execution_unchanged=True,
            saved_std_not_reset=True, all_six_std_entries_remain_trainable=True,
            inherited_motion0625_evidence_filename_is_lineage_only=True,
            relative_error_target=.1, added_loss_or_backward=False, policy_accepted=False))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    if transition and sha(source) != ANCHOR_SHA:
        raise ValueError('Only the declared source can request the first sampler transfer')
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
            raise ValueError('Frozen transfer changes only the declared training sampler field')
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
