"""Reviewed precision exploration trial; keep direct PD and the entire saved PPO.

The initial learned distribution is retained exactly. Only future exploration
lower bounds can become .01; no new loss, feedback controller or target map.
"""
import copy
import json
from pathlib import Path
import sys

import joint_pd_source4601_wheel025_runtime as previous
import joint_pd_source200_leg05_runtime as projection_template
from joint_pd_continuation import ROOT, sha
from train_joint_pd_standing_seed import saved_updates

SCRIPT = Path(__file__).resolve()
ANCHOR_SHA = '90c17dff2bedfe692e79fb5107f5129039cd3c61d46c2bfee18c811095373393'
FLAG = '--repair_motion0625_precision010'
DECISION = ROOT / 'mine_model/results/joint_pd_source4801_precision010_trial_decision_20261019.json'
MOTION_SCALE = .625
LEG_STD_FLOOR = WHEEL_STD_FLOOR = GLOBAL_STD_FLOOR = .01
DELTA = dict(rewards={}, commands={}, exploration={
    'minimum_leg_std': [.02, .01], 'minimum_wheel_std': [.025, .01],
    'global_minimum_std': [.02, .01]}, controller={}, physics={})
KEY, base, criteria_fix, recipe = previous.KEY, previous.base, previous.criteria_fix, previous.recipe


def profile():
    value = copy.deepcopy(previous.profile())
    value.update(profile='plain_source4801_motion0625_precision010_v1',
        minimum_leg_std=LEG_STD_FLOOR, minimum_wheel_std=WHEEL_STD_FLOOR,
        global_minimum_std=GLOBAL_STD_FLOOR, changed_noise_heads=list(range(6)),
        native_runtime_sha256=sha(SCRIPT), parent_wheel025_runtime_sha256=sha(previous.SCRIPT),
        precision_trial_anchor_sha256=ANCHOR_SHA,
        authority='completed4801_own_static_moving_and_same_noise_sampling_factorial_review')
    return value


def repair_ready():
    decision = json.loads(DECISION.read_text())
    if decision.get('selected_precision_floor_trial') is not True:
        raise ValueError('Precision trial is not selected until actual current-source reviews complete')
    static = json.loads(Path(decision['source_static_retention_report']).read_text())
    moving = json.loads(Path(decision['source_moving_height_summary']).read_text())
    raw = json.loads(Path(decision['source_moving_height_raw_integrity']).read_text())
    factorial = json.loads(Path(decision['source_sampling_factorial_review']).read_text())
    final = json.loads(Path(decision['source_final_verification']).read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 4801 or decision['source_all17_Adam_step'] != 96020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not final['verified'] or final['checkpoint_sha256'] != ANCHOR_SHA
            or final['actual_ppo_updates'] != 4801 or final['all17_Adam_step'] != 96020
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != ANCHOR_SHA
            or moving['checkpoint_sha256'] != ANCHOR_SHA or not moving['supplemental_steps_ready']
            or not moving['results']['deterministic']['all_numeric_phase_gates_passed']
            or any(moving['results'][k]['survivors'] != 27 or moving['results'][k]['resets']
                for k in ('deterministic', 'saved_Gaussian'))
            or not raw['verified'] or raw['checkpoint_sha256'] != ANCHOR_SHA
            or factorial['checkpoint_sha256'] != ANCHOR_SHA
            or not factorial['same_initial_states_and_Gaussian_draws']
            or not factorial['all_variants_full51_survived_without_reset']
            or not factorial['saved_model_std_and_optimizer_unchanged']
            or not decision['source_is_unaccepted_repair_branch']
            or not decision['frozen_sampling_does_not_prove_learning_benefit']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset']
            or decision['original_model200_reload'] or decision['faster_curriculum_authorized']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require complete4801, continuous all17 state and the isolated precision-floor review')
    return decision


def private_runtime(source=None):
    # Reuse the original ordinary-PPO runtime directly. The intermediate
    # wheel025 wrapper's literal leg-floor label does not describe this trial.
    module = previous.bind(previous.previous.private_runtime, profile=profile)()
    original_require, original_observe = module.require_source, module.observe_environment
    if source is not None:
        previous.require_source(source)
        if sha(source) != ANCHOR_SHA or saved_updates(source) != 4801:
            raise ValueError('Only the reviewed4801 begins this precision transfer')

    def require(path, *, initial_transfer=False):
        if initial_transfer:
            raise ValueError('Never reload originalmodel200 or reset retained Adam')
        if sha(path) == ANCHOR_SHA:
            return previous.require_source(path)
        state, cfg = original_require(path, initial_transfer=False)
        if state.get('infos', {}).get(KEY) != profile() or saved_updates(path) < 4801:
            raise ValueError('Require this exact precision descendant, not an older repair anchor')
        return state, cfg

    def observe(env, output):
        result = original_observe(env, output)
        from auto_train_mine import write_json
        write_json(Path(output) / 'precision010_runtime_evidence.json', dict(profile=profile(),
            authoritative_current_delta=DELTA, source_actual_updates=4801,
            initial_saved_std_not_reset=True, all_six_std_entries_remain_trainable=True,
            inherited_sampler_migration_labels_are_lineage_only=True,
            commands_rewards_physics_reset_and_direct_joint_execution_unchanged=True,
            added_loss_or_backward=False, relative_error_target=.1, policy_accepted=False))
        return result

    module.profile, module.require_source, module.observe_environment = profile, require, observe
    return module


def require_source(source, transition=False):
    return private_runtime(source if transition else None).require_source(source)


def install_leg_exploration(output):
    def log(*values, **kwargs):
        old = '[INFO]: User-requested leg-only exploration floor .05; wheel floor .03 and all Adam states retained.'
        if values and values[0] == old:
            values = ('[INFO]: Future leg/wheel lower bounds .01; initial saved std and all Adam states retained.', *values[1:])
        print(*values, **kwargs)

    return previous.bind(projection_template.install_leg_exploration,
        LEG_STD_FLOOR=LEG_STD_FLOOR, WHEEL_STD_FLOOR=WHEEL_STD_FLOOR,
        GLOBAL_STD_FLOOR=GLOBAL_STD_FLOOR, profile=profile, print=log)(output)


def native(output, arguments):
    return previous.bind(previous.previous.native, SCRIPT=SCRIPT, ANCHOR_SHA=ANCHOR_SHA,
        FLAG=FLAG, repair_ready=repair_ready, require_source=require_source,
        private_runtime=private_runtime, install_leg_exploration=install_leg_exploration)(output, arguments)


def command(source, *args, **kwargs):
    argv, final, saved = previous.command(source, *args, **kwargs)
    argv[argv.index(str(previous.SCRIPT))] = str(SCRIPT)
    for option, old, new in (('--min_exploration_std', .02, GLOBAL_STD_FLOOR),
                             ('--min_wheel_exploration_std', .025, WHEEL_STD_FLOOR)):
        index = argv.index(option) + 1
        if float(argv[index]) != old:
            raise ValueError('Recheck the immutable parent exploration command')
        argv[index] = str(new)
    if sha(source) == ANCHOR_SHA:
        argv.append(FLAG)
    return argv, final, saved


evaluator_tree = previous.evaluator_tree


def replay(arguments):
    arguments = list(arguments)
    if FLAG in arguments or '--overrides_json' in arguments:
        raise ValueError('Replay uses the exact saved physical and model configuration')
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
