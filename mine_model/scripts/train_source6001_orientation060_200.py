"""One qualified200-update orientation-weight trial, with50/100 early inspection."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, optimizer_step, validate_continuation_config
from train_joint_pd_standing_seed import saved_updates
import joint_pd_source6001_orientation060_runtime as runtime
import qualify_source6001_orientation060_transfer as qualifier
import train_source5801_wheel125_200 as previous
import joint_pd_acceptance_10pct_20261017 as acceptance

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = qualifier.OUTPUT
helper = previous.helper


def prepare(output, previous_output):
    output, previous_output = Path(output).resolve(), Path(previous_output).resolve()
    decision = runtime.repair_ready()
    if (output.parent != RESULTS or output.exists() or previous_output != runtime.TRAINING
            or not output.name.startswith('joint_pd_optimization_')):
        raise ValueError('Use a fresh owned200 block from the completely reviewed6001')
    proof_path = QUALIFICATION / 'qualification.json'
    frozen_path = QUALIFICATION / 'native_reward_transfer_qualification.json'
    proof, frozen = [json.loads(p.read_text()) for p in (proof_path, frozen_path)]
    source, saved, cfg = runtime.anchor()
    if (not proof['qualified'] or proof['source_sha256'] != runtime.ANCHOR_SHA
            or proof['profile'] != runtime.profile() or proof['decision_sha256'] != sha(runtime.DECISION)
            or proof['implementation_sha256'] != sha(runtime.SCRIPT)
            or proof['qualifier_sha256'] != sha(qualifier.SCRIPT)
            or proof['next_configuration_changes'] != runtime.DELTA
            or not all(proof[k] for k in ('all_model_and_Adam_tensors_bitwise_preserved',
                'initial_saved_std_bitwise_preserved', 'actual_saved_adaptive_LR_preserved',
                'exact_registered_projection_test_passed', 'untrained_metadata_roundtrip_inference_bitwise_equal',
                'saved_source_env_and_agent_configurations_unchanged'))
            or not frozen['qualified'] or frozen['source_sha256'] != runtime.ANCHOR_SHA
            or frozen['only_reward_change'] != runtime.DELTA['rewards']
            or not frozen['physical_outputs_observations_actions_and_other_rewards_bitwise_equal']
            or frozen['PPO_updates_performed'] or proof['actual_PPO_updates_executed']
            or proof['optimizer_steps_executed'] or proof['simulator_started']
            or saved_updates(source) != 6001 or optimizer_step(saved) != 120020
            or any(sha(p) != h for p, h in proof['source_files'].items())
            or any(sha(p) != h for p, h in frozen['source_data_sha256'].items())):
        raise ValueError('Require complete retained-state CPU and frozen27 single-reward qualification')
    expected = copy.deepcopy(cfg)
    expected['rewards'][runtime.REWARD_KEY] = runtime.REWARD_VALUE
    validate_continuation_config(cfg, expected, expected['commands'],
        reward_overrides={'rewards.' + runtime.REWARD_KEY: runtime.REWARD_VALUE})
    plan = copy.deepcopy(json.loads((previous_output / 'experiment_plan.json').read_text()))
    pins = {**plan['source_files'], **decision['source_pins'],
            **proof['source_files'], **frozen['source_data_sha256']}
    for p in (SCRIPT, runtime.SCRIPT, runtime.DECISION, qualifier.SCRIPT, proof_path, frozen_path,
            previous.SCRIPT, previous_output / 'status.json', previous_output / 'stationary_review.json',
            previous_output / 'native_learning_signal/completed_external.json'):
        pins[str(p)] = sha(p)
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Preserve source, full review, controller, physics and every saved state')
    plan.update(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous_output),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=6001,
        source_Adam_step=120020, source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous_output / 'replay'),
        source_files=pins, target_actual_ppo_updates=6201, expected_terminal_Adam_step=124020,
        bounded_training_updates=200, early_learning_inspection_actual_updates=[50, 100],
        full_native_independent_review_after_actual_updates=200,
        early_inspection_is_not_independent_acceptance=True,
        run_name='plain_fullspeed_orientation060_' + output.name + '_u6001',
        reward_key=runtime.REWARD_KEY, reward_value=runtime.REWARD_VALUE,
        only_reward_change=runtime.DELTA['rewards'], only_command_change={}, only_exploration_change={},
        only_controller_change={}, action_map_or_controller_changed=False,
        only_wheel_action_clamp_and_history_changed=False,
        next_configuration_changes=runtime.DELTA, authoritative_current_delta=runtime.DELTA,
        unchanged_recipe_continuation=False, selected_single_orientation_weight_trial=str(runtime.DECISION),
        training_config=expected, profile=runtime.profile(), optimizer_reset=False, learned_std_reset=False,
        current_all_error_types_review=decision['source_all_errors'],
        current_literal_nonzero_relative_review=decision['source_literal_nonzero_relative_report'],
        current_posture_weakness_review=decision['source_posture_weakness_review'],
        source_current_model_static_retention_verified=True,
        source_static_retention_report=decision['source_static_raw_manifest'],
        source_all657_trials=657, source_all657_passed=606, source_all657_survivors=657,
        source_original_and_literal_nonzero10_both_passed=515,
        source_full27_passed=27, source_full_height_passed=297, source_full_height_trials=306,
        source_sustained120_survivors=54, source_sustained120_passed=45,
        source_original_sustained120=decision['source_original_sustained120'],
        source_supplemental_interior_posture_raw=decision['source_posture_weakness_review'],
        source_supplemental_interior54_passed=27,
        source_supplemental_interior54_body_failed=27,
        required_supplemental_interior54_manifest=decision['required_retained_interior54_cases'],
        required_supplemental_interior_seeds=decision['required_retained_interior_seeds'],
        required_supplemental_interior54_review_after_actual_updates=200,
        source_is_unaccepted_repair_branch=True,
        normal_saved_wheel_limit_enabled=True, wheel_action_scale=15., wheel_action_limit=1.25,
        diagnostic_headroom_not_enabled_in_training=True,
        inherited_sampler_and_controller_transfer_labels_are_lineage_only=True,
        actual_training_speed_range_m_s=[-.8, .8], actual_training_yaw_range_rad_s=[-.3, .3],
        training_motion_scale=1., fixed_standing_environments=1024, native_num_envs=4096,
        native_transition_batch=4096 * 48, optimizer_minibatches_per_actual_update=20,
        required_full_height_review_scales=[.25, .375, .5, .625, .75, 1.],
        required_combined_forward_and_reverse_turn_review=True,
        required_independent_fullspeed_sustained120_review=True,
        relative_error_target=.1, unchanged_full_height_ranges_and_max_speed=True,
        faster_curriculum_authorized=False, next_block_requires_review=True,
        no_total_training_update_limit=True, policy_accepted=False, hardware_enabled=False, shutdown=False)
    plan.pop('selected_single_controller_clamp_trial', None)
    output.mkdir()
    write_json(output / 'experiment_plan.json', plan)
    return plan, expected


def command(*args, **kwargs):
    argv, final, saved = runtime.command(*args, **kwargs)
    if argv.count('agent.save_interval=5') != 1:
        raise ValueError('Requalify logging-only per-update checkpoint retention')
    argv[argv.index('agent.save_interval=5')] = 'agent.save_interval=1'
    return argv, final, saved


def review(output, after, before):
    result = previous.parent.review(output, after, before)
    result.update(unchanged_recipe_consolidation=False, only_reward_repair=runtime.DELTA['rewards'],
        only_controller_change={}, wheel125_normal_saved_controller=True,
        current_controller_static_retention_verified=False,
        own_combined_turns_review_pending=True, own_sustained120_review_pending=True,
        early_learning_inspection_actual_updates=[50, 100], next_block_requires_review=True,
        policy_accepted=False)
    return result


def run(output, previous_output, unit):
    entry = json.loads((QUALIFICATION / 'full_original_block_entry_preflight.json').read_text())
    if (not entry['qualified'] or entry['controller_sha256'] != sha(SCRIPT)
            or entry['source_actual_updates'] != 6001 or entry['target_actual_updates'] != 6201
            or entry['only_reward_change'] != runtime.DELTA['rewards']
            or entry['PPO_updates_performed'] or entry['simulator_started']):
        raise ValueError('Require the complete original200 entry with exactly one reward delta')
    helper.prepare, helper.command = prepare, command
    helper.wait_memory = previous.parent.parent.previous_controller.memory.wait_memory
    helper.base.acceptance, helper.base.replay_command = acceptance, runtime.replay_command
    helper.base.stationary_decision = lambda after, before: review(output, after, before)
    original = helper.write_json

    def pointer(path, value):
        path = Path(path)
        if path.parent == RESULTS and path.name in ('joint_pd_active_joint_tracking.json',
                'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
            value = dict(value, training_motion_scale=1., yaw_min_abs_rate=0.,
                fixed_standing_environments=1024, training_stage='fullspeed_orientation060',
                wheel_action_limit=1.25, action_scale_vel=15.,
                early_learning_inspection_actual_updates=[50, 100])
        return original(path, value)

    helper.write_json = pointer
    try:
        return helper.run(output, unit, previous_output)
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
