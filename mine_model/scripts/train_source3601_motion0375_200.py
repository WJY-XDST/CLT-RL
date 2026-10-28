"""One source-bound200-update ordinary PPO sampler expansion after complete review."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, optimizer_step, validate_continuation_config
from train_joint_pd_standing_seed import saved_updates
import joint_pd_source3601_motion0375_runtime as runtime
import qualify_source3601_motion0375 as qualifier
import train_source2001_quality50_200 as original_controller
import train_source200_stationary_height200_4096 as helper
import train_source200_drift25_measured_memory200 as memory
import joint_pd_acceptance_10pct_20261017 as acceptance

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_source3601_motion0375_qualification_20261019'


def qualifications():
    cpu = json.loads((QUALIFICATION / 'qualification.json').read_text())
    frozen = json.loads((QUALIFICATION / 'native_command_transfer_qualification.json').read_text())
    if (not cpu['qualified'] or cpu['profile'] != runtime.profile()
            or cpu['implementation_sha256'] != sha(runtime.SCRIPT)
            or cpu['qualifier_sha256'] != sha(qualifier.SCRIPT)
            or cpu['decision_sha256'] != sha(runtime.DECISION)
            or cpu['next_configuration_changes'] != runtime.DELTA
            or cpu['source_sha256'] != runtime.ANCHOR_SHA
            or not all(cpu[k] for k in ('all_model_and_Adam_tensors_bitwise_preserved',
                'initial_saved_std_bitwise_preserved', 'actual_saved_adaptive_LR_preserved',
                'exact_registered_projection_test_passed', 'untrained_metadata_roundtrip_inference_bitwise_equal',
                'saved_agent_configuration_unchanged', 'original200_and_undeclared_transfer_rejected'))
            or cpu['saved_env_diff'] != ['/commands/training_motion_scale']
            or cpu['actual_PPO_updates_executed'] or cpu['optimizer_steps_executed']
            or not frozen['qualified'] or frozen['source_sha256'] != runtime.ANCHOR_SHA
            or frozen['implementation_sha256'] != sha(runtime.SCRIPT)
            or frozen['qualifier_sha256'] != sha(qualifier.SCRIPT)
            or frozen['only_command_change'] != runtime.DELTA['commands']
            or frozen['source_resets_preserved'] != 15
            or not frozen['physics_actions_observations_means_and_all_rewards_bitwise_equal']
            or frozen['PPO_updates_performed']
            or any(sha(p) != h for p, h in cpu['source_files'].items())
            or any(sha(p) != h for p, h in frozen['source_data_sha256'].items())):
        raise ValueError('Require exact full-state CPU and original full27 fixed-command transfer proof')
    runtime.repair_ready()
    return cpu, frozen


def prepare(output, previous):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    if output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_optimization_'):
        raise ValueError('Use a fresh owned output and retain every earlier checkpoint')
    decision = runtime.repair_ready()
    if previous != Path(decision['training_output']):
        raise ValueError('This is one reviewed source3601 trial, not a repeated reset')
    cpu, frozen = qualifications()
    source = Path(decision['source_checkpoint'])
    saved, cfg = runtime.require_source(source, transition=True)
    expected = copy.deepcopy(cfg)
    expected['commands']['training_motion_scale'] = runtime.MOTION_SCALE
    validate_continuation_config(cfg, expected, expected['commands'])
    if (saved_updates(source) != 3601 or optimizer_step(saved) != 72020
            or cfg['scene']['num_envs'] != 4096 or cfg['terrain']['num_envs'] != 4096
            or expected['commands']['standing_env_fraction'] != .25):
        raise ValueError('Retain all17 source clocks,4096 envs and25pct dedicated standing')
    old = json.loads((previous / 'experiment_plan.json').read_text())
    pins = {**old['source_files'], **decision['source_pins'], **cpu['source_files'], **frozen['source_data_sha256']}
    pins.update({str(p): sha(p) for p in (SCRIPT, runtime.SCRIPT, qualifier.SCRIPT, runtime.DECISION,
        QUALIFICATION / 'qualification.json', QUALIFICATION / 'native_command_transfer_qualification.json',
        Path(original_controller.__file__), Path(helper.__file__), Path(memory.__file__))})
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('A pinned source, qualification or historical proof changed')
    peak, requirement = memory.memory_requirement()
    plan = dict(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=3601,
        source_Adam_step=72020, source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous / 'replay'),
        protected_reference=old['protected_reference'], verified_static_milestone=old['verified_static_milestone'],
        source_files=pins, target_actual_ppo_updates=3801, bounded_training_updates=200,
        expected_terminal_Adam_step=76020, run_name='plain_motion0375_' + output.name + '_u3601',
        only_reward_change={}, only_command_change=runtime.DELTA['commands'], only_exploration_change={},
        next_configuration_changes=runtime.DELTA, unchanged_recipe_continuation=False,
        selected_for_another200_by_explicit_review=str(runtime.DECISION),
        native_num_envs=4096, fixed_standing_environments=1024, training_motion_scale=runtime.MOTION_SCALE,
        native_transition_batch=4096 * 48, optimizer_minibatches_per_actual_update=20,
        full_height_range=cfg['commands']['ranges_height'],
        exploration_guard=dict(minimum_leg_std=.02, minimum_wheel_std=.03,
            global_minimum_std=.02, saved_initial_std_reset=False),
        source_is_reviewed_unaccepted_repair_branch=True, source_full27_failures_preserved=True,
        source_current_model_static_retention_verified=True,
        source_static_retention_report=decision['source_static_retention_report'],
        original_source200_loaded_only_at_first_block=False,
        source_model_and_optimizer_reloaded_from_original200=False, stationary_checkpoint_preserved=True,
        policy_class='ActorCritic', external_observation_size=27, architecture_migrated=False,
        ordinary_full_policy_PPO=True, auxiliary_losses_enabled=False, optimizer_reset=False,
        learned_std_reset=False, checkpoint_save_interval=1, both_optional_violation_penalties_zero=True,
        action_map_or_controller_changed=False, rewards_exploration_physics_and_reset_configurations_unchanged=True,
        acceptance_revision=acceptance.profile(), profile=runtime.profile(), training_config=expected,
        resource_reservation_method='Measured4096 native highwater plus4GiB',
        measured_native_RSS_highwater_bytes=peak, combined_memory_required_bytes=requirement,
        physical_available_memory_required_bytes=6 * 1024**3,
        current_exploration_review=str(previous / 'effective_exploration_prefix_0200_readonly.json'),
        current_policy_update_health_review=str(previous / 'policy_update_health_prefix_0200_readonly.json'),
        current_critic_and_command_coverage_review=str(previous / 'critic_and_command_coverage_prefix_0200_readonly.json'),
        current_wheel_error_budget=str(previous / 'wheel_tracking_budget_full51_source3601_readonly_20261019.json'),
        current_terminal_age_review=str(previous / 'terminal_event_ages_prefix_0200_readonly.json'),
        effective_exploration_and_optimizer_review_required_per_block=True,
        critic_and_command_coverage_review_required_per_block=True,
        additional_readonly_health_review_prefixes=[40, 100, 200],
        all_six_Gaussian_std_entries_remain_trainable=True, next_block_requires_review=True,
        no_total_training_update_limit=True, post_block_full27_and_supplemental_low_speed_review_required=True,
        original025_and_new0375_full_height_replay_required=True, own_static_retention_required=True,
        faster_curriculum_authorized=True, full_project_command_envelope_validation_unproven=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    output.mkdir()
    write_json(output / 'experiment_plan.json', plan)
    return plan, expected


def command(*args, **kwargs):
    argv, final, saved = runtime.command(*args, **kwargs)
    if argv.count('agent.save_interval=5') != 1:
        raise ValueError('Requalify the logging-only checkpoint cadence')
    argv[argv.index('agent.save_interval=5')] = 'agent.save_interval=1'
    return argv, final, saved


def review(output, after, before):
    # Reuse the original numeric readers and eligibility predicates unchanged.
    result = original_controller.review(output, after, before)
    result.update(only_reward_repair={}, only_exploration_repair={},
        only_command_change=runtime.DELTA['commands'], training_motion_scale=runtime.MOTION_SCALE,
        next_block_requires_review=True, own_static_retention_verified=False,
        original025_and_new0375_full_height_review_pending=True,
        original_numeric_readers_and_acceptance_predicates_unchanged=True)
    return result


def run(output, previous, unit):
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
            value = dict(value, training_motion_scale=runtime.MOTION_SCALE,
                fixed_standing_environments=1024, training_stage='motion0375')
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
