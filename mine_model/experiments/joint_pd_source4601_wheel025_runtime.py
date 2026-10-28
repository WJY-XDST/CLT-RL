"""One wheel exploration-floor trial; preserve the complete source4601 PPO.

The saved initial std stays .03 on wheels. Only its future lower bound becomes
.025. Leg exploration, commands, rewards, physics and the direct PD stay intact.
"""
import copy
from pathlib import Path
import sys
import types

import joint_pd_source4001_motion0625_runtime as previous
import joint_pd_source200_leg05_runtime as projection_template
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = 'c12c6118a2459c51d04c895802dddcbae17e4794fa24c42eabb07222f4524bc1'
FLAG = '--repair_motion0625_wheel025'
DECISION = ROOT / 'mine_model/results/joint_pd_source4601_wheel025_trial_decision_20261019.json'
DELTA = dict(rewards={}, commands={}, exploration={'minimum_wheel_std': [.03, .025]},
    controller={}, physics={})
KEY, base, criteria_fix, recipe = previous.KEY, previous.base, previous.criteria_fix, previous.recipe


def bind(function, **values):
    result = types.FunctionType(function.__code__, dict(function.__globals__, **values),
        function.__name__, function.__defaults__, function.__closure__)
    result.__kwdefaults__ = function.__kwdefaults__
    return result


def profile():
    value = copy.deepcopy(previous.profile())
    value.update(profile='plain_source4601_motion0625_wheel025_leg020_v1',
        minimum_wheel_std=.025, changed_noise_heads=[2, 5],
        native_runtime_sha256=sha(SCRIPT), parent_motion0625_runtime_sha256=sha(previous.SCRIPT),
        wheel_floor_trial_anchor_sha256=ANCHOR_SHA,
        authority='completed200_four51_own_static_moving_height_and_matched_sampling_review')
    return value


def repair_ready():
    import json
    decision = json.loads(DECISION.read_text())
    static = json.loads(Path(decision['source_static_retention_report']).read_text())
    moving = json.loads(Path(decision['moving_height_raw_integrity_report']).read_text())
    pair = json.loads(Path(decision['matched_wheel_sampling_report']).read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 4601 or decision['source_all17_Adam_step'] != 92020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != ANCHOR_SHA
            or not moving['verified'] or moving['checkpoint_sha256'] != ANCHOR_SHA
            or not moving['modes']['deterministic']['all_steady_height_pose_and_support_phase_gates_passed']
            or any(moving['modes'][key]['survivors'] != 27 or moving['modes'][key]['reported_resets']
                for key in ('deterministic', 'saved_Gaussian'))
            or not pair['exact_same_Gaussian_draws'] or not pair['bitwise_equal_initial_observations_and_means']
            or pair['checkpoint_sha256'] != ANCHOR_SHA
            or any(result['survivors'] != 51 or result['resets'] for result in pair['results'].values())
            or not decision['source_is_unaccepted_repair_branch']
            or not decision['frozen_sampling_is_not_learning_or_acceptance']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or decision['faster_curriculum_authorized']
            or any(sha(path) != digest for path, digest in decision['source_pins'].items())):
        raise ValueError('Require the exact completed4601 diagnosis and isolated wheel-floor decision')
    return decision


def private_runtime(source=None):
    # Bind the actual defining globals: exported parent functions can belong to
    # private modules, so copying the public module namespace loses their imports.
    module = bind(previous.private_runtime, profile=profile)()
    original_require, original_observe = module.require_source, module.observe_environment
    if source is not None:
        previous.require_source(source)
        if sha(source) != ANCHOR_SHA or saved_updates(source) != 4601:
            raise ValueError('Only the exact complete4601 begins this floor trial')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never repeat the originalmodel200 rollback')
        if sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        return original_require(path, initial_transfer=False)

    def observe(env, output):
        result = original_observe(env, output)
        from auto_train_mine import write_json
        write_json(Path(output) / 'wheel025_runtime_evidence.json', dict(profile=profile(),
            authoritative_current_delta=DELTA, inherited_motion_expansion_metadata_is_lineage_only=True,
            saved_initial_std_not_reset=True, leg_std_floor_unchanged=.02,
            all_six_std_entries_remain_trainable=True, added_loss_or_backward=False,
            commands_rewards_physics_reset_and_direct_joint_execution_unchanged=True,
            relative_error_target=.1, policy_accepted=False))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    def log(*values, **kwargs):
        old = '[INFO]: User-requested leg-only exploration floor .05; wheel floor .03 and all Adam states retained.'
        if values and values[0] == old:
            values = ('[INFO]: Leg floor .02; wheel floor .025; saved initial std and all Adam states retained.', *values[1:])
        print(*values, **kwargs)

    # Original projection implementation and both original step hooks. Changing
    # this lower bound does not reset the saved distribution or Adam moments.
    return bind(projection_template.install_leg_exploration, LEG_STD_FLOOR=.02,
        WHEEL_STD_FLOOR=.025, GLOBAL_STD_FLOOR=.02, profile=profile, print=log)(output)


def native(output, arguments):
    return bind(previous.native, SCRIPT=SCRIPT, ANCHOR_SHA=ANCHOR_SHA, FLAG=FLAG,
        repair_ready=repair_ready, require_source=require_source, private_runtime=private_runtime,
        install_leg_exploration=install_leg_exploration)(output, arguments)


def command(source, *args, **kwargs):
    argv, final, saved = previous.command(source, *args, **kwargs)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    index = argv.index('--min_wheel_exploration_std') + 1
    if float(argv[index]) != .03:
        raise ValueError('Recheck the immutable parent wheel-floor command')
    argv[index] = '.025'
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv, final, saved


evaluator_tree = previous.evaluator_tree


def replay(arguments):
    arguments = list(arguments)
    if FLAG in arguments or '--overrides_json' in arguments:
        raise ValueError('Wheel-floor replay uses saved weights and physical configuration')
    source = Path(arguments[arguments.index('--checkpoint') + 1])
    require_source(source)
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
