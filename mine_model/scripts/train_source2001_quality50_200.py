"""One reviewed smooth motion-bonus repair; retain full PPO state and commands."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

import source1201_orientation40
import joint_pd_source2001_quality50_runtime as repair
import qualify_source2001_quality50 as qualifier
import train_source1601_motion025_orientation80_200 as priority
import train_source200_stationary_height200_4096 as helper
import train_source200_drift25_measured_memory200 as memory
import joint_pd_acceptance_10pct_20261017 as acceptance
import joint_pd_stationary_stage as stage
import run_static_reset_pair_diagnosis as boundary
from joint_pd_priority import compare_priority
from joint_pd_continuation import ROOT, sha, optimizer_step, validate_continuation_config
from train_joint_pd_standing_seed import saved_updates
from auto_train_mine import write_json

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_source2001_quality50_qualification_20261018'


def qualifications():
    cpu = json.loads((QUALIFICATION / 'qualification.json').read_text())
    frozen = json.loads((QUALIFICATION / 'native_reward_transfer_qualification.json').read_text())
    expected = {repair.REWARD_KEY: [1., repair.QUALITY_FLOOR]}
    if (not cpu['qualified'] or cpu['profile'] != repair.profile()
            or cpu['implementation_sha256'] != sha(repair.SCRIPT)
            or cpu['qualifier_sha256'] != sha(qualifier.SCRIPT)
            or not cpu['all17_Adam_bitwise_preserved'] or not cpu['saved_std_bitwise_preserved']
            or not cpu['actual_saved_adaptive_LR_preserved'] or not cpu['one_configuration_field_changed']
            or not frozen['qualified'] or frozen['source_sha256'] != repair.ANCHOR_SHA
            or frozen['implementation_sha256'] != sha(repair.SCRIPT)
            or frozen['qualifier_sha256'] != sha(qualifier.SCRIPT)
            or frozen['only_reward_change'] != expected or cpu['only_reward_change'] != expected
            or not frozen['physical_outputs_observations_actions_and_other_rewards_bitwise_equal']
            or not frozen['both_violation_penalties_zero'] or not frozen['criteria_loaded_by_quality_reward_alone']
            or frozen['criteria_fix_sha256'] != sha(repair.criteria_fix.SCRIPT)
            or cpu['criteria_fix_sha256'] != sha(repair.criteria_fix.SCRIPT)
            or any(sha(p) != h for p, h in frozen['source_data_sha256'].items())):
        raise ValueError('Require intact full-state and frozen27 quality-reward equivalence')


def selected_source(previous):
    source = Path(json.loads((previous / 'status.json').read_text())['checkpoint']).resolve()
    transition = sha(source) == repair.ANCHOR_SHA
    module = repair.previous if transition else repair
    prior = boundary.smooth
    boundary.smooth = module
    try:
        source, entry = boundary.reviewed_source(previous)
    finally:
        boundary.smooth = prior
    if entry != module.SCRIPT:
        raise ValueError('Use the exact intact completed200/full27/full17 lineage')
    if transition:
        repair.repair_ready()
    elif not json.loads((previous / 'stationary_review.json').read_text())['continue_from_candidate']:
        raise ValueError('Review a finished repair before another200 updates')
    repair.require_source(source, transition=transition)
    return source, transition


def prepare(output, previous):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    if output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_optimization_'):
        raise ValueError('Use a fresh owned output and retain all earlier models/evidence')
    qualifications()
    source, transition = selected_source(previous)
    saved, cfg = repair.require_source(source, transition=transition)
    expected = copy.deepcopy(cfg)
    expected['rewards'][repair.REWARD_KEY] = repair.QUALITY_FLOOR
    validate_continuation_config(cfg, expected, cfg['commands'],
        reward_overrides={'rewards.' + repair.REWARD_KEY: repair.QUALITY_FLOOR})
    if cfg['scene']['num_envs'] != 4096 or cfg['terrain']['num_envs'] != 4096:
        raise ValueError('Keep4096 sampling and the same command distribution')
    old = json.loads((previous / 'experiment_plan.json').read_text())
    pins = dict(old['source_files'])
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('A pinned historical source or proof changed')
    watched = [SCRIPT, repair.SCRIPT, repair.criteria_fix.SCRIPT, repair.DECISION, qualifier.SCRIPT,
        source, *sorted((source.parent / 'params').glob('*.yaml')),
        QUALIFICATION / 'qualification.json', QUALIFICATION / 'native_reward_transfer_qualification.json',
        QUALIFICATION / 'native_reward_transfer/quality_criteria_runtime_patch.json',
        acceptance.CURRENT, acceptance.POLICY, Path(acceptance.__file__),
        *(previous / n for n in ('status.json', 'stationary_review.json', 'experiment_plan.json',
            'native_learning_signal/completed_external.json')),
        *(previous / 'replay' / n for n in ('assessment.json', 'height_leg_command_assessment.json',
            'manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz'))]
    pins.update({str(p.resolve()): sha(p) for p in watched})
    if transition:
        pins.update(repair.repair_ready()['source_pins'])
    actual = saved_updates(source)
    peak, requirement = memory.memory_requirement()
    plan = dict(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=actual,
        source_Adam_step=optimizer_step(saved), source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous / 'replay'),
        protected_reference=old['protected_reference'], verified_static_milestone=old['verified_static_milestone'],
        source_files=pins, target_actual_ppo_updates=actual + 200, bounded_training_updates=200,
        expected_terminal_Adam_step=(actual + 200) * 20,
        run_name='plain_quality50_' + output.name + '_u' + str(actual),
        reward_key=repair.REWARD_KEY, reward_value=repair.QUALITY_FLOOR,
        only_reward_change={repair.REWARD_KEY: [1., repair.QUALITY_FLOOR]} if transition else {},
        only_command_change={}, only_exploration_change={},
        criteria_loading_fix='Independent of the two disabled optional violation penalties',
        native_num_envs=4096, fixed_standing_environments=1024, training_motion_scale=.25,
        native_transition_batch=4096 * 48, optimizer_minibatches_per_actual_update=20,
        full_height_range=cfg['commands']['ranges_height'],
        exploration_guard=dict(minimum_leg_std=.05, minimum_wheel_std=.03,
            global_minimum_std=.02, saved_initial_std_reset=False),
        source_is_reviewed_unaccepted_repair_branch=transition,
        original_source200_loaded_only_at_first_block=False,
        source_model_and_optimizer_reloaded_from_original200=False,
        stationary_checkpoint_preserved=True, policy_class='ActorCritic', external_observation_size=27,
        architecture_migrated=False, ordinary_full_policy_PPO=True, auxiliary_losses_enabled=False,
        optimizer_reset=False, learned_std_reset=False, checkpoint_save_interval=1,
        both_optional_violation_penalties_zero=True, action_map_or_controller_changed=False,
        height3_orientation40_velocity025_unchanged=True,
        acceptance_revision=acceptance.profile(), profile=repair.profile(), training_config=expected,
        resource_reservation_method='Measured4096 native highwater plus4GiB',
        measured_native_RSS_highwater_bytes=peak, combined_memory_required_bytes=requirement,
        physical_available_memory_required_bytes=6 * 1024**3,
        next_block_requires_review=True, no_total_training_update_limit=True,
        post_block_full27_and_supplemental_low_speed_review_required=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    output.mkdir()
    write_json(output / 'experiment_plan.json', plan)
    return plan, expected


def command(*args, **kwargs):
    argv, final, saved = repair.command(*args, **kwargs)
    if argv.count('agent.save_interval=5') != 1:
        raise ValueError('Requalify the logging-only checkpoint cadence')
    argv[argv.index('agent.save_interval=5')] = 'agent.save_interval=1'
    return argv, final, saved


def review(output, after, before):
    after = json.loads((Path(output) / 'replay/assessment.json').read_text())
    legs = json.loads((Path(output) / 'replay/height_leg_command_assessment.json').read_text())
    result = stage.review(after, before)
    retained = bool(stage.summary(after)['stationary_all_original_metrics_passed']
        and legs['static_stage_candidate_passed'] and after['survivors'] == 27 and not after['resets'])
    comparison = compare_priority(after, before)
    eligible = bool(retained and not comparison['regressions']['stability']
        and priority.quality(after) < priority.quality(before))
    result.update(continue_from_candidate=eligible, promote=eligible,
        candidate_priority_quality=priority.quality(after), source_priority_quality=priority.quality(before),
        static_all_original_metrics_and_actual_leg_commands_retained=retained,
        full_scope_comparison=comparison, posture_then_height_then_velocity_priority=True,
        only_reward_repair={repair.REWARD_KEY: [1., repair.QUALITY_FLOOR]},
        both_optional_violation_penalties_zero=True, training_motion_scale=.25,
        original_acceptance_unchanged=True, acceptance_revision=acceptance.profile(),
        final_full27_gate_passed=bool(after['passed']), independent_sustained_verified=False,
        curriculum_selection_only=True, next_block_requires_review=True, policy_accepted=False)
    json.dumps(result)
    return result


def run(output, previous, unit):
    helper.prepare, helper.command = prepare, command
    helper.wait_memory = memory.wait_memory
    helper.base.acceptance = acceptance
    helper.base.replay_command = repair.replay_command
    helper.base.stationary_decision = lambda after, before: review(output, after, before)
    original = helper.write_json

    def pointer(path, value):
        path = Path(path)
        if path.parent == RESULTS and path.name in ('joint_pd_active_joint_tracking.json',
                'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
            value = dict(value, training_motion_scale=.25, fixed_standing_environments=1024,
                training_stage='motion025_height3_quality50')
        return original(path, value)

    helper.write_json = pointer
    try:
        helper.run(output, unit, previous)
    finally:
        helper.write_json = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output, args.previous)
    else:
        run(args.output, args.previous, args.unit)
