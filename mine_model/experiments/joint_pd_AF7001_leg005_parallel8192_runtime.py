"""Prepared ordinary27-input joint-PD continuation with8192 environments.

Retain all saved source state and change only the future leg exploration
floor. Scene/terrain parallelism uses the separately verified count adapter. Native training remains gated on a complete source-selection
decision, its full tracking reviews and an exact-source capacity receipt.
Capacity-only checkpoints and the originalmodel200 are never admitted.
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
import joint_pd_preserved6801_yawcost200_runtime as family
import joint_pd_yawcost200_capacity_runtime as count_factory
import joint_pd_source200_leg05_runtime as projection_template

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
DECISION = RESULTS / 'joint_pd_AF7001_leg005_parallel8192_trial_decision_20261023.json'
ADMISSION = RESULTS / 'joint_pd_AF7001_leg005_parallel8192_admission_20261023.json'
ANCHOR_SHA = 'af47077587044230d5814f5e0e76a633366260698471da6162f3411494e64f49'
LEG_STD_FLOOR = GLOBAL_STD_FLOOR = .005
WHEEL_STD_FLOOR = .01
SOURCE_REPORT = RESULTS / 'joint_pd_optimization_20261022_135230_239804_preserved6801_yawcost200_200/final_checkpoint_verification_readonly_20261019.json'
KEY, base, criteria_fix, bind = family.KEY, family.base, family.criteria_fix, family.bind
NUM_ENVS, STANDING_ENVS = 8192, 2048
FLAG = '--increase_parallel_envs8192'
DELTA = dict(rewards={}, commands={}, exploration={'minimum_leg_std': [.01, .005], 'global_minimum_std': [.01, .005]}, controller={}, physics={})
PARALLELISM_CHANGE = {'scene.num_envs': [4096, 8192], 'terrain.num_envs': [4096, 8192]}


@lru_cache(maxsize=1)
def baseline():
    report = json.loads(SOURCE_REPORT.read_text())
    source = Path(report['checkpoint'])
    saved, cfg = family.require_source(source)
    if (not report['verified'] or report['actual_ppo_updates'] != 7001
            or report['all17_Adam_step'] != 140020 or report['checkpoint_sha256'] != sha(source)
            or cfg['scene']['num_envs'] != 4096 or cfg['terrain']['num_envs'] != 4096):
        raise ValueError('Require the exact completely reviewed7001 physical/controller recipe')
    return source, saved, cfg


def profile():
    result = copy.deepcopy(family.profile())
    result.update(profile='plain_AF7001_yawcost200_leg005_parallel8192_v1',
        minimum_leg_std=LEG_STD_FLOOR, global_minimum_std=GLOBAL_STD_FLOOR,
        minimum_wheel_std=WHEEL_STD_FLOOR, changed_noise_heads=[0, 1, 3, 4],
        native_runtime_sha256=sha(SCRIPT), parent_runtime_sha256=sha(family.SCRIPT),
        native_training_environment_count=NUM_ENVS,
        fixed_standing_environment_count=STANDING_ENVS,
        samples_per_original_PPO_update=NUM_ENVS * 48,
        original_Adam_minibatches_per_update=20,
        parallelism_only_change=False, capacity_only_probe=False,
        authority='complete_source_reviews_and_exact_source_per_block_capacity_admission')
    return result


def recipe(rewards):
    return family.recipe(rewards)


def require_source(path, transition=False, *, initial_transfer=False):
    if initial_transfer:
        raise ValueError('Never repeat the originalmodel200 transfer or reset Adam')
    path = Path(path).resolve()
    saved = torch.load(path, map_location='cpu', weights_only=True)
    current = validate_checkpoint_robot(path)
    declared = saved.get('infos', {}).get(KEY)
    actual = saved_updates(path)
    if not isinstance(declared, dict) or actual < 7001 or declared.get('capacity_only_probe', False):
        raise ValueError('Capacity weights and early/unreviewed historical branches are not production sources')
    if declared == family.profile():
        if actual != 7001 or sha(path) != ANCHOR_SHA:
            raise ValueError('Only the selected fully reviewed AF7001 may start this floor trial')
        return family.require_source(path)
    _, before, cfg = baseline()
    expected = copy.deepcopy(cfg)
    expected['log_dir'] = current.get('log_dir')
    for key in ('scene', 'terrain'):
        expected[key]['num_envs'] = NUM_ENVS
    states = saved['optimizer_state_dict']['state']
    if (transition or declared != profile() or current != expected or not tensor_finite(saved)
            or len(states) != 17 or any(float(v['step']) != actual * 20 for v in states.values())
            or saved['model_state_dict'].keys() != before['model_state_dict'].keys()
            or any(saved['model_state_dict'][k].shape != v.shape
                   for k, v in before['model_state_dict'].items())):
        raise ValueError('Require the unchanged8192 recipe and every continuous original Adam state')
    return saved, current


def private_runtime(source=None):
    if source is not None:
        require_source(source)
    # Reuse the actually tested8192 count adapter; its capacity-only save
    # tag is replaced by this separate ordinary production profile.
    sampler, evidence = bind(count_factory.sampler_factory,
        NUM_ENVS=NUM_ENVS, profile=profile)()
    sampler.profile, sampler.require_source = profile, require_source
    module = sampler.private_runtime()
    module.profile, module.require_source = profile, require_source
    module.parallel_population_assertion_evidence = evidence
    return module


def install_leg_exploration(output):
    def log(*values, **kwargs):
        old = '[INFO]: User-requested leg-only exploration floor .05; wheel floor .03 and all Adam states retained.'
        if values and values[0] == old:
            values = ('[INFO]: AF7001 leg floor .005; wheel floor .01; all saved std and17 Adam states retained.', *values[1:])
        print(*values, **kwargs)
    return bind(projection_template.install_leg_exploration, LEG_STD_FLOOR=LEG_STD_FLOOR,
        GLOBAL_STD_FLOOR=GLOBAL_STD_FLOOR, WHEEL_STD_FLOOR=WHEEL_STD_FLOOR,
        profile=profile, print=log)(output)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    source, stop_file = Path(source).resolve(), Path(stop_file).resolve()
    saved, actual_cfg = require_source(source)
    if updates != 200 or cfg != actual_cfg or commands != cfg['commands']:
        raise ValueError('Keep exactly200 full ordinary updates and the saved command envelope')
    overrides = kwargs.pop('reward_overrides', {})
    if any(not k.startswith('rewards.') or cfg['rewards'].get(k[8:]) != v
           for k, v in overrides.items()):
        raise ValueError('Changing the environment count cannot change a reward')
    if (kwargs.pop('minimum_wheel_std', .01) not in (.01, .03)
            or kwargs.pop('joint_mean_guard_coef', 0.) != 0.
            or kwargs.pop('joint_height_feedback_coef', 0.) != 0. or kwargs):
        raise ValueError('Retain the saved exploration and ordinary PPO without auxiliary losses')
    anchor, _, original_cfg = baseline()
    argv, _, _ = family.command(anchor, original_cfg, label, 200,
        original_cfg['commands'], stop_file)
    if argv[1] != str(family.SCRIPT) or argv[argv.index('--num_envs') + 1] != '4096':
        raise ValueError('Preserve the verified original full200 command prototype')
    argv[1] = str(SCRIPT)
    actual = saved_updates(source)
    for flag, value in {'--checkpoint_path': source, '--num_envs': NUM_ENVS,
            '--target_iteration': actual + 200, '--run_name': label,
            '--observation-output': stop_file.parent / 'native_learning_signal',
            '--stop_file': stop_file}.items():
        argv[argv.index(flag) + 1] = str(value)
    if argv[argv.index('--min_exploration_std') + 1] != '0.01':
        raise ValueError('Requalify the unchanged original global floor option')
    argv[argv.index('--min_exploration_std') + 1] = str(GLOBAL_STD_FLOOR)
    if cfg['scene']['num_envs'] == 4096:
        argv.append(FLAG)
    return argv, actual + 199, saved


def repair_ready():
    decision = json.loads(DECISION.read_text())
    if (not decision['selected_after_all_owned_tracking_reviews']
            or not decision['selected_exploration_only_trial']
            or decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 7001
            or decision['source_all17_Adam_step'] != 140020
            or decision['planned_actual_updates'] != 200
            or decision['target_actual_updates'] != 7201
            or decision['next_configuration_changes'] != DELTA
            or any(decision[key] for key in ['optimizer_reset','std_reset','actual_LR_reset',
                'original_model200_reload','acceptance_changed'])
            or decision['relative_error_target'] != .1
            or any(sha(path) != digest for path,digest in decision['source_pins'].items())):
        raise ValueError('Require the completely reviewed AF7001 and the single declared leg-floor trial')
    for item in decision['required_complete_review_receipts']:
        report = json.loads(Path(item['path']).read_text())
        if report.get(item['completion_key']) is not True:
            raise ValueError('Every owned review and noise diagnosis must complete before this trial')
    return decision


def review_ready(source):
    from check_plain27_leg005_environment_capacity_v7 import require_capacity_selection
    repair_ready()
    decision = json.loads(ADMISSION.read_text())
    if decision['proposal_decision_sha256'] != sha(DECISION):
        raise ValueError('Bind native admission to the immutable single-factor decision')
    saved, cfg = require_source(source)
    if (not decision['selected_after_all_owned_tracking_reviews']
            or decision['source_checkpoint_sha256'] != sha(source)
            or decision['source_actual_updates'] != saved_updates(source)
            or decision['source_all17_Adam_step'] != saved_updates(source) * 20
            or decision['target_actual_updates'] != saved_updates(source) + 200
            or decision['planned_actual_updates'] != 200
            or decision['next_configuration_changes'] != DELTA
            or decision['selected_num_envs'] != NUM_ENVS
            or decision['optimizer_reset'] or decision['std_reset']
            or decision['actual_LR_reset'] or decision['original_model200_reload']
            or decision['acceptance_changed'] or decision['relative_error_target'] != .1
            or not decision['all_failures_and_regressions_preserved']
            or any(sha(p) != digest for p, digest in decision['source_pins'].items())):
        raise ValueError('Complete all owned tracking reviews and choose the retained source before8192 PPO')
    for item in decision['required_complete_review_receipts']:
        report = json.loads(Path(item['path']).read_text())
        if report.get(item['completion_key']) is not True:
            raise ValueError('Every required tracking scope must complete before the next200')
    capacity = Path(decision['capacity_verified_summary'])
    report = json.loads(capacity.read_text())
    if any(sha(p) != digest for p, digest in report['source_pins'].items()):
        raise ValueError('Preserve the exact-source current capacity measurement')
    require_capacity_selection(capacity, block_id=decision['capacity_block_id'],
        source=source, num_envs=NUM_ENVS)
    return decision


def native(output, arguments):
    import joint_pd_source2201_yaw025_runtime as template
    source = Path(arguments[arguments.index('--checkpoint_path') + 1]).resolve()
    _, cfg = require_source(source)
    decision = review_ready(source)
    if (int(arguments[arguments.index('--num_envs') + 1]) != NUM_ENVS
            or int(arguments[arguments.index('--target_iteration') + 1]) != saved_updates(source) + 200
            or ((FLAG in arguments) != (cfg['scene']['num_envs'] == 4096))):
        raise ValueError('Declare the8192 parallelism boundary exactly once')
    family._sampler_api.install_sampling_field_restore()
    write_json(output.parent / 'parallel8192_native_admission.json', dict(
        native_num_envs=NUM_ENVS, standing_environments=STANDING_ENVS,
        motion_environments=NUM_ENVS - STANDING_ENVS, rollout_steps=48,
        original_Adam_minibatches_per_update=20, samples_per_update=NUM_ENVS * 48,
        source_checkpoint_sha256=sha(source), source_actual_updates=saved_updates(source),
        full_source_selection_decision=str(DECISION),
        capacity_verified_summary=decision['capacity_verified_summary'],
        rewards_commands_controller_and_physics_unchanged=True,
        only_exploration_change=DELTA['exploration'],
        source_model_all17_Adam_actual_LR_and_std_not_reset=True, policy_accepted=False))
    return bind(template.native, SCRIPT=SCRIPT, FLAG=FLAG,
        repair_ready=lambda: decision, require_source=require_source,
        private_runtime=private_runtime, install_leg_exploration=install_leg_exploration)(output, arguments)


evaluator_tree = family.evaluator_tree


def replay_command(source, output):
    require_source(source)
    anchor, _, _ = baseline()
    argv = family.replay_command(anchor, output)
    argv[argv.index(str(family.SCRIPT))] = str(SCRIPT)
    argv[argv.index('--checkpoint') + 1] = str(Path(source).resolve())
    return argv


def replay(arguments):
    if FLAG in arguments or '--overrides_json' in arguments:
        raise ValueError('Frozen replay keeps the exact saved recipe; no reward/physics transfer')
    return bind(family.replay, SCRIPT=SCRIPT, require_source=require_source,
        private_runtime=private_runtime)(arguments)


if __name__ == '__main__':
    mode = sys.argv.pop(1)
    if mode == 'native':
        boundary = sys.argv.index('--')
        if sys.argv[1:boundary] != ['--observation-output', sys.argv[2]]:
            raise ValueError('Keep the native observation-output contract')
        native(Path(sys.argv[2]), sys.argv[boundary + 1:])
    elif mode == 'replay':
        replay(sys.argv[1:])
    else:
        raise ValueError('Only ordinary joint-PD training and frozen policy replay are supported')
