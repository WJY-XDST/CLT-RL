"""Retain model200 once, then review each200 actual stationary PPO updates."""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import yaml

from auto_train_mine import write_json
import train_joint_pd_reward_block as block
import continue_joint_pd_observed100 as observed
import joint_pd_review_interval200 as interval
import joint_pd_acceptance_revision as acceptance
import joint_pd_stationary_stage as stage
from joint_pd_continuation import ROOT, RUNS, HEIGHT_DEFAULTS, continuation_command, sha, optimizer_step
from joint_pd_continuation import validate_continuation_config
from stationary_height_credit_decision import review
from train_joint_pd_standing_seed import nominal_standing_intact, saved_updates
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.joint_reset import NOMINAL_JOINT_RESET, nominal_reset_config
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
import joint_pd_source200_stationary_runtime as branch

RESULTS = ROOT / 'mine_model/results'
ENTRY = ROOT / 'mine_model/scripts/train_source200_stationary_observed.py'
EVALUATOR = ROOT / 'mine_model/scripts/evaluate_source200_stationary_height.py'
REFERENCE = RESULTS / 'joint_pd_optimization_20261017_053817_001119_stationary_twist100'
BASE_REPLAY = block.replay_command
QUALIFICATION = RESULTS / 'joint_pd_source200_stationary_broad_qualification_20261017/qualification.json'


def expected_initial_config():
    old = validate_checkpoint_robot(branch.SOURCE)
    current = validate_checkpoint_robot(Path(json.loads((REFERENCE / 'status.json').read_text())['checkpoint']))
    expected = nominal_reset_config(old)
    expected['robot']['spawn']['usd_path'] = str(branch.tires.ASSET)
    # Fill only modern default-zero fields missing in the historical YAML.
    rewards = copy.deepcopy(current['rewards'])
    rewards.update(old['rewards'])
    rewards['joint_target_leg_length_error_sq'] = 0.
    rewards['base_height_sigma'] = branch.HEIGHT_SIGMA
    expected['rewards'] = rewards
    commands = dict(HEIGHT_DEFAULTS, **old['commands'])
    commands.update(ranges_height=list(SAFE_BASE_HEIGHT_RANGE_M), full_height_range_from_start=True,
        training_motion_scale=0., standing_only_steps=0, yaw_start_steps=0, reverse_ramp_start_steps=0)
    expected['commands'] = commands
    baseline = nominal_reset_config(old)
    validate_continuation_config(baseline, copy.deepcopy(expected), commands,
        reward_overrides={'rewards.base_height_sigma': branch.HEIGHT_SIGMA}, robot_asset=str(branch.tires.ASSET))
    return expected


def environment():
    return dict(os.environ, PYTHONPATH=str(ROOT / 'wheel_legged_isaaclab') + ':' + str(ROOT / 'mine_model/scripts'),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')


def replay_command(checkpoint, replay, *, initial=False, overrides=None):
    argv = BASE_REPLAY(checkpoint, replay)
    original = str(ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py')
    argv[argv.index(original)] = str(EVALUATOR)
    if initial:
        argv.append('--initial_source200_transfer')
    if overrides is not None:
        argv.extend(['--overrides_json', str(overrides)])
    return branch.tires.arguments(argv)


def command(source, cfg, label, updates, commands, stop_file, **kwargs):
    if updates != interval.UPDATES:
        raise ValueError('Every review requires200 actual ordinary PPO updates')
    initial = sha(source) == branch.SOURCE_SHA
    kwargs.update(robot_asset=str(branch.tires.ASSET), minimum_wheel_std=branch.WHEEL_STD_FLOOR,
        joint_reset_profile=NOMINAL_JOINT_RESET, allow_reset_pose_change=initial,
        joint_mean_guard_coef=0., joint_height_feedback_coef=0.)
    argv, final, saved = continuation_command(source, cfg, label, updates, commands, stop_file, **kwargs)
    argv[argv.index('--min_exploration_std') + 1] = str(branch.LEG_STD_FLOOR)
    argv.extend(['agent.policy.class_name=ActorCritic', 'agent.algorithm.entropy_coef=.001'])
    if initial:
        argv.append('--initial_source200_transfer')
    argv = [argv[0], str(ENTRY), '--observation-output', str(Path(stop_file).parent / 'native_learning_signal'),
            '--', *argv[2:]]
    return branch.tires.arguments(argv), final, saved


def prepare(output, previous=None):
    output = Path(output).resolve()
    if (output.parent != RESULTS or not output.name.startswith('joint_pd_optimization_') or output.exists()):
        raise ValueError('Require a fresh output; never repeat an already-started200 block')
    qualification = json.loads(QUALIFICATION.read_text())
    if (not qualification['qualified']
            or not qualification['all17_model_and_Adam_states_bitwise_preserved']
            or qualification['source_sha256'] != branch.SOURCE_SHA
            or qualification['runtime_sha256'] != sha(branch.__file__)):
        raise ValueError('Require the qualified unchanged full-state source200 runtime')
    prior = Path(previous).resolve() if previous is not None else REFERENCE
    if previous is None:
        source, initial = branch.SOURCE, True
        saved, cfg = branch.require_source(source, initial_transfer=True)
        expected = expected_initial_config()
        source_replay = output / 'initial_replay'
    else:
        state = json.loads((prior / 'status.json').read_text())
        decision = json.loads((prior / 'stationary_review.json').read_text())
        if state['training_active'] or state['state'] != 'agent_review_required' or not decision['continue_from_candidate']:
            raise ValueError('Review regressions or stagnation before selecting another200-update source')
        source, initial = Path(state['checkpoint']), False
        saved, cfg = branch.require_source(source)
        expected = copy.deepcopy(cfg)
        source_replay = prior / 'replay'
        native = json.loads((prior / 'native_learning_signal/completed_external.json').read_text())
        report = json.loads((source_replay / 'assessment.json').read_text())
        done = json.loads((source_replay / 'completed.json').read_text())
        if (sha(source) != state['checkpoint_sha256'] or native['actual_observed_updates'] != 200
                or native['optimizer_minibatches'] != 4000 or native['PPO_updates_repeated'] != 0
                or native['checkpoint_sha256'] != sha(source) or report['checkpoint_sha256'] != sha(source)
                or report['total_trials'] != 27 or not done['completed'] or done['samples'] != 27000):
            raise ValueError('Continue the exact complete200/4000 native-update boundary')
    actual = saved_updates(source)
    if initial and any(saved['model_state_dict']['std'][i] < floor
                       for i, floor in enumerate([.02, .02, .03, .02, .02, .03])):
        raise ValueError('The first exploration guard must preserve every saved std at load')
    old_plan = json.loads((prior / 'experiment_plan.json').read_text())
    pins = dict(old_plan['source_files'])
    if any(sha(Path(name)) != digest for name, digest in pins.items()):
        raise ValueError('Preserve old branches and their frozen implementation')
    watched = [Path(__file__).resolve(), ENTRY, EVALUATOR, Path(branch.__file__).resolve(),
        Path(interval.__file__).resolve(), Path(acceptance.__file__).resolve(), acceptance.CURRENT,
        acceptance.POLICY, source, *sorted((source.parent / 'params').glob('*.yaml')),
        ROOT / 'mine_model/scripts/train_joint_pd_observed.py',
        ROOT / 'mine_model/scripts/continue_joint_pd_observed100.py',
        ROOT / 'mine_model/scripts/audit_stationary_height_prefix.py', QUALIFICATION]
    pins.update({str(p): sha(p) for p in watched})
    pins.update(branch.tires.profile()['asset_chain_sha256'])
    output.mkdir()
    overrides = {'robot.spawn.usd_path': str(branch.tires.ASSET), 'joint_reset_profile': NOMINAL_JOINT_RESET}
    overrides.update({'commands.' + k: v for k, v in expected['commands'].items()})
    overrides.update({'rewards.' + k: v for k, v in expected['rewards'].items()})
    write_json(output / 'baseline_overrides.json', overrides)
    pins[str(output / 'baseline_overrides.json')] = sha(output / 'baseline_overrides.json')
    plan = dict(created_at=datetime.now().astimezone().isoformat(), previous_output=str(prior),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=actual,
        source_Adam_step=optimizer_step(saved), source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(source_replay),
        protected_reference=str(source_replay / 'assessment.json'), source_files=pins,
        target_actual_ppo_updates=actual + 200, bounded_training_updates=200,
        expected_terminal_Adam_step=(actual + 200) * 20,
        run_name='plain_source200_stationary_' + output.name + '_u' + str(actual),
        reward_key='base_height_sigma', reward_value=branch.HEIGHT_SIGMA,
        only_reward_change={'base_height_sigma': [cfg['rewards']['base_height_sigma'], branch.HEIGHT_SIGMA]} if initial else {},
        exploration_guard=dict(minimum_leg_std=.02, minimum_wheel_std=.03, saved_initial_std_reset=False),
        training_motion_scale=0., native_num_envs=8192, full_height_range=list(SAFE_BASE_HEIGHT_RANGE_M),
        original_source200_loaded_only_at_first_block=initial, external_observation_size=27,
        policy_class='ActorCritic', architecture_migrated=False, ordinary_full_policy_PPO=True,
        auxiliary_losses_enabled=False, optimizer_reset=False, learned_std_reset=False,
        acceptance_revision=acceptance.profile(), profile=branch.profile(), training_config=expected,
        no_total_training_update_limit=True, next_block_requires_review=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    write_json(output / 'experiment_plan.json', plan)
    return plan, expected


def stationary_decision(after, before):
    after, before = (json.loads(json.dumps(x)) for x in (after, before))
    decision = review(after, before)
    a, b = stage.summary(after), stage.summary(before)
    baseline_still_failed = not b['stationary_pose_passed']
    recovery = (baseline_still_failed and a['stationary_survivors'] == 9 and not a['stationary_resets']
        and not decision['stationary_stability_regressions'] and not decision['stationary_height_regressions']
        and stage.quality(after) < stage.quality(before))
    decision.update(continue_from_candidate=bool(decision['promote'] or recovery),
        acceptance_revision=acceptance.profile(), policy_accepted=False,
        independent_sustained_verified=False, next_action_requires_review=True)
    return decision


def run(output, unit, previous=None):
    output = Path(output).resolve()
    plan, expected = prepare(output, previous)
    source = Path(plan['source_checkpoint'])
    initial = plan['original_source200_loaded_only_at_first_block']
    source_replay = Path(plan['source_replay'])
    env = environment()
    if initial:
        write_json(output / 'status.json', dict(state='source200_transfer_replay', training_active=False,
            actual_ppo_updates=plan['source_actual_ppo_updates'], live_ppo_updates=plan['source_actual_ppo_updates'],
            target_actual_ppo_updates=plan['target_actual_ppo_updates'], checkpoint=str(source), policy_accepted=False))
        argv = replay_command(source, source_replay, initial=True, overrides=output / 'baseline_overrides.json')
        write_json(output / 'initial_replay_launch.json', dict(command=argv, weights_updated=False))
        block.runtime.OUTPUT = output
        with (RUNS / '.cumulative_learning.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            block.runtime.run_child(argv, output / 'initial_replay.log', env)
        before = acceptance.assess_current(source_replay)
        before = json.loads((source_replay / 'assessment.json').read_text())
        if not nominal_standing_intact(before) or stage.summary(before)['stationary_survivors'] != 9:
            write_json(output / 'status.json', dict(state='source200_transfer_requires_diagnosis',
                training_active=False, actual_ppo_updates=plan['source_actual_ppo_updates'], policy_accepted=False))
            raise ValueError('Restore the source200 nominal standing before any PPO updates')
        transfer = dict(transfer_diagnostic_only=True, source_checkpoint_sha256=sha(source),
            nominal_standing_preserved=True, source200_weights_unchanged=True,
            profile=branch.profile(), original_mass_and_geometry_branch_preserved=True, policy_accepted=False)
        write_json(output / 'source200_transfer_verification.json', transfer)
        for name in ('assessment.json', 'manifest.json', 'completed.json', 'trace.csv', 'resolved_env.json', 'source200_environment.json'):
            plan['source_files'][str(source_replay / name)] = sha(source_replay / name)
        write_json(output / 'experiment_plan.json', plan)
    else:
        before = json.loads((source_replay / 'assessment.json').read_text())
    block.continuation_command = command
    block.replay_command = replay_command
    block.assess = acceptance.assess_current
    block.assess_directory = acceptance.assess_current_legs
    interval.install_finalizer(observed)

    def prepared(*unused):
        observed.record_resource_wait(output, plan)
        observed.drift.wait_memory(output.parent)
        return plan, expected

    pointer = dict(unit=unit, directory=str(output), source_checkpoint=str(source),
        controller_kind='reviewed_analytic_tire_physics100', actual_updates_per_review=200,
        training_motion_scale=0., optimizer_reset=False, weights_reinitialized=False,
        no_total_training_update_limit=True, policy_accepted=False)
    for name in ('joint_pd_active_joint_tracking.json', 'joint_pd_active_standing_seed.json'):
        write_json(RESULTS / name, pointer)
    block.run(SimpleNamespace(previous=Path(plan['previous_output']), output=output,
        reward_key='unused', value=0., updates=200, prepare_only=False), prepare_function=prepared)
    observed.finalize_observations(output)
    after = json.loads((output / 'replay/assessment.json').read_text())
    acceptance.assessor.assess(output / 'replay', 'assessment_historical10pct.json')
    write_json(output / 'stationary_review.json', stationary_decision(after, before))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output, args.previous)
    else:
        run(args.output, args.unit, args.previous)
