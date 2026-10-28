"""One200-update same-profile PPO block filling the verified low-yaw command gap."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, optimizer_step, validate_continuation_config
from train_joint_pd_standing_seed import saved_updates
import source4201_low_yaw_plan as selection
import qualify_source4201_low_yaw as qualifier
import train_source2001_quality50_200 as original_controller
import train_source200_stationary_height200_4096 as helper
import train_source200_drift25_measured_memory200 as memory
import joint_pd_acceptance_10pct_20261017 as acceptance

SCRIPT = Path(__file__).resolve()
RESULTS, QUALIFICATION = selection.RESULTS, selection.QUALIFICATION
runtime = selection.runtime


def qualifications():
    cpu = json.loads((QUALIFICATION / 'qualification.json').read_text())
    frozen = json.loads((QUALIFICATION / 'native_command_transfer_qualification.json').read_text())
    if (not cpu['qualified'] or cpu['profile'] != runtime.profile()
            or cpu['implementation_sha256'] != sha(runtime.SCRIPT)
            or cpu['qualifier_sha256'] != sha(qualifier.SCRIPT)
            or cpu['decision_sha256'] != sha(selection.DECISION)
            or cpu['source_sha256'] != selection.ANCHOR_SHA
            or cpu['next_configuration_changes'] != selection.DELTA
            or cpu['saved_env_diff'] != ['/commands/yaw_min_abs_rate']
            or not all(cpu[k] for k in ('all_model_and_Adam_tensors_bitwise_preserved',
                'initial_saved_std_bitwise_preserved', 'actual_saved_adaptive_LR_preserved',
                'exact_registered_projection_test_passed', 'untrained_metadata_roundtrip_inference_bitwise_equal',
                'saved_agent_configuration_unchanged', 'original200_rejected', 'existing_profile_transfer_flag_absent'))
            or cpu['actual_PPO_updates_executed'] or cpu['optimizer_steps_executed']
            or not frozen['qualified'] or frozen['source_sha256'] != selection.ANCHOR_SHA
            or frozen['only_command_change'] != selection.DELTA['commands']
            or not frozen['physics_actions_observations_means_and_all_rewards_bitwise_equal']
            or frozen['source_resets_preserved'] or frozen['PPO_updates_performed']
            or any(sha(p) != h for p, h in cpu['source_files'].items())
            or any(sha(p) != h for p, h in frozen['source_data_sha256'].items())):
        raise ValueError('Require intact CPU all-state and fixed27 native one-field proof')
    selection.ready()
    return cpu, frozen


def prepare(output, previous):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    decision = selection.ready()
    if (output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_optimization_')
            or previous != Path(decision['training_output'])):
        raise ValueError('Require a fresh owned block from this reviewed4201 output')
    cpu, frozen = qualifications()
    source = Path(decision['source_checkpoint'])
    saved, cfg = selection.require_source(source, transition=True)
    expected = copy.deepcopy(cfg)
    expected['commands']['yaw_min_abs_rate'] = 0.
    validate_continuation_config(cfg, expected, expected['commands'])
    if (saved_updates(source) != 4201 or optimizer_step(saved) != 84020
            or cfg['scene']['num_envs'] != 4096 or cfg['terrain']['num_envs'] != 4096
            or cfg['commands']['standing_env_fraction'] != .25
            or expected['commands']['training_motion_scale'] != .625):
        raise ValueError('Keep the source clocks,4096 envs,1024 standing and current maxima')
    old = json.loads((previous / 'experiment_plan.json').read_text())
    pins = {**old['source_files'], **decision['source_pins'], **cpu['source_files'], **frozen['source_data_sha256']}
    pins.update({str(p): sha(p) for p in (SCRIPT, selection.SCRIPT, qualifier.SCRIPT, runtime.SCRIPT,
        selection.DECISION, QUALIFICATION / 'qualification.json',
        QUALIFICATION / 'native_command_transfer_qualification.json', Path(original_controller.__file__),
        Path(helper.__file__), Path(memory.__file__))})
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Historical source or qualification changed')
    peak, requirement = memory.memory_requirement()
    plan = copy.deepcopy(old)
    plan.update(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=4201,
        source_Adam_step=84020, source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous / 'replay'),
        source_files=pins, target_actual_ppo_updates=4401, bounded_training_updates=200,
        expected_terminal_Adam_step=88020, run_name='plain_low_yaw_' + output.name + '_u4201',
        only_reward_change={}, only_command_change=selection.DELTA['commands'], only_exploration_change={},
        next_configuration_changes=selection.DELTA, unchanged_recipe_continuation=False,
        selected_for_another200_by_explicit_review=str(selection.DECISION),
        native_num_envs=4096, fixed_standing_environments=1024, training_motion_scale=.625,
        source_current_model_static_retention_verified=True,
        source_static_retention_report=decision['source_static_retention_report'],
        source_full27_passed=27, source_full_height_passed=201, source_full_height_trials=204,
        source_is_reviewed_unaccepted_repair_branch=True, source_full27_failures_preserved=True,
        original_source200_loaded_only_at_first_block=False,
        source_model_and_optimizer_reloaded_from_original200=False,
        optimizer_reset=False, learned_std_reset=False, checkpoint_save_interval=1,
        ordinary_full_policy_PPO=True, auxiliary_losses_enabled=False,
        reward_key='tracking_ang_vel', reward_value=expected['rewards']['tracking_ang_vel'],
        complete_original_block_entry_metadata=True, acceptance_revision=acceptance.profile(),
        profile=runtime.profile(), training_config=expected,
        existing_runtime_profile_retained=True, old_runtime_family_initial_delta_is_historical=True,
        authoritative_current_delta=selection.DELTA,
        measured_native_RSS_highwater_bytes=peak, combined_memory_required_bytes=requirement,
        current_exploration_review=str(previous / 'effective_exploration_prefix_0200_readonly.json'),
        current_policy_update_health_review=str(previous / 'policy_update_health_prefix_0200_readonly.json'),
        current_critic_and_command_coverage_review=str(previous / 'critic_and_command_coverage_prefix_0200_readonly.json'),
        current_wheel_error_budget=str(previous / 'candidate4201_canonical_wheel_tracking_budget_readonly_20261019.json'),
        current_height_error_budget=str(previous / 'terminal_joint_height_budget_prefix_0200_readonly.json'),
        current_geometric_exploration_review=str(previous / 'joint_target_geometric_exploration_prefix_0200_readonly.json'),
        current_gradient_allocation_review=str(previous / 'gradient_allocation_prefix_0200_readonly.json'),
        current_command_coverage_gap_report=decision['command_coverage_gap_report'],
        current_matched_sampling_signed_bias=decision['matched_signed_bias_report'],
        current_terminal_age_review=str(previous / 'terminal_event_ages_prefix_0200_readonly.json'),
        required_full_height_review_scales=[.25, .375, .5, .625],
        effective_exploration_and_optimizer_review_required_per_block=True,
        critic_and_command_coverage_review_required_per_block=True,
        additional_readonly_health_review_prefixes=[40, 100, 200],
        all_six_Gaussian_std_entries_remain_trainable=True, next_block_requires_review=True,
        faster_curriculum_authorized=False, own_static_retention_required=True,
        no_total_training_update_limit=True, policy_accepted=False, hardware_enabled=False, shutdown=False)
    output.mkdir()
    write_json(output / 'experiment_plan.json', plan)
    return plan, expected


def command(*args, **kwargs):
    argv, final, saved = runtime.command(*args, **kwargs)
    if argv.count('agent.save_interval=5') != 1 or runtime.FLAG in argv:
        raise ValueError('Keep same-profile load and logging-only checkpoint cadence')
    argv[argv.index('agent.save_interval=5')] = 'agent.save_interval=1'
    return argv, final, saved


def review(output, after, before):
    result = original_controller.review(output, after, before)
    result.update(only_reward_repair={}, only_exploration_repair={},
        only_command_change=selection.DELTA['commands'], training_motion_scale=.625,
        next_block_requires_review=True, own_static_retention_verified=False,
        retained_four_full_height_reviews_pending=True,
        original_numeric_readers_and_acceptance_predicates_unchanged=True)
    return result


def run(output, previous, unit):
    entry = json.loads((QUALIFICATION / 'full_original_block_entry_preflight.json').read_text())
    if (not entry['qualified'] or entry['controller_sha256'] != sha(SCRIPT)
            or entry['source_actual_updates'] != 4201 or entry['target_actual_updates'] != 4401
            or entry['only_command_change'] != selection.DELTA['commands']
            or entry['PPO_updates_performed'] or entry['simulator_started']):
        raise ValueError('Qualify the complete original block command before launching PPO')
    helper.prepare, helper.command = prepare, command
    helper.wait_memory = memory.wait_memory
    helper.base.acceptance = acceptance
    helper.base.replay_command = runtime.replay_command
    helper.base.stationary_decision = lambda after, before: review(output, after, before)
    original = helper.write_json

    def pointer(path, value):
        path = Path(path)
        if path.parent == RESULTS and path.name in ('joint_pd_active_joint_tracking.json',
                'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
            value = dict(value, training_motion_scale=.625, yaw_min_abs_rate=0.,
                fixed_standing_environments=1024, training_stage='low_yaw_coverage')
        return original(path, value)

    helper.write_json = pointer
    try:
        helper.run(output, unit, previous)
    finally:
        helper.write_json = original


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--previous', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--unit')
    cli.add_argument('--prepare-only', action='store_true')
    args = cli.parse_args()
    if args.prepare_only:
        prepare(args.output, args.previous)
    else:
        run(args.output, args.previous, args.unit)
