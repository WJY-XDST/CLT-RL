"""One reviewed200-update wheel-clamp trial, with50/100 early inspection."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, optimizer_step
from train_joint_pd_standing_seed import saved_updates
import joint_pd_source5801_wheel125_runtime as runtime
import train_source5601_consolidation200 as parent
import qualify_source5801_wheel125_transfer as qualifier
import wheel_control_source_migration as migration
import joint_pd_acceptance_10pct_20261017 as acceptance

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = qualifier.OUTPUT
helper = parent.helper


def prepare(output, previous):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    decision = runtime.repair_ready()
    if (output.parent != RESULTS or output.exists() or previous != runtime.TRAINING
            or not output.name.startswith('joint_pd_optimization_')):
        raise ValueError('Use one fresh owned block from the reviewed5801 state')
    proof_path = QUALIFICATION / 'qualification.json'
    proof = json.loads(proof_path.read_text())
    fixture_path = RESULTS / 'joint_pd_source5801_wheel125_native_fixture_v2_20261020/native_fixture_verification.json'
    fixture = json.loads(fixture_path.read_text())
    if (not fixture['verified'] or not fixture['normal_saved_controller']
            or fixture['retained_source_sha256'] != runtime.ANCHOR_SHA
            or fixture['native_trials'] != 108 or fixture['raw_rows'] != 108000
            or not fixture['all_policy_inputs_outputs_and_physical_reward_rows_bitwise_equal']
            or not fixture['no_wheel_reference_probe_or_controller_override']
            or any(sha(p) != h for p, h in fixture['source_pins'].items())):
        raise ValueError('Require the normal saved-controller native equivalence before PPO')
    source, saved, cfg = runtime.anchor()
    if (not proof['qualified'] or proof['source_sha256'] != runtime.ANCHOR_SHA
            or proof['profile'] != runtime.profile() or proof['decision_sha256'] != sha(runtime.DECISION)
            or proof['implementation_sha256'] != sha(runtime.SCRIPT)
            or proof['qualifier_sha256'] != sha(qualifier.SCRIPT)
            or proof['next_configuration_changes'] != runtime.DELTA
            or not all(proof[k] for k in ('all_model_and_Adam_tensors_bitwise_preserved',
                'initial_saved_std_bitwise_preserved', 'actual_saved_adaptive_LR_preserved',
                'exact_registered_projection_test_passed', 'untrained_metadata_roundtrip_inference_bitwise_equal',
                'all324000_recorded_means_match_diagnostic_bitwise', 'historical_unit_clamp_and_leg_columns_preserved'))
            or proof['actual_PPO_updates_executed'] or proof['optimizer_steps_executed']
            or any(sha(p) != h for p, h in proof['source_files'].items())
            or saved_updates(source) != 5801 or optimizer_step(saved) != 116020):
        raise ValueError('Require actual retained-state CPU qualification and saved wheel-cap equivalence')
    old = json.loads((previous / 'experiment_plan.json').read_text())
    old_pins, resolved = migration.resolve_historical_pins(old['source_files'])
    plan = copy.deepcopy(old)
    pins = {**old_pins, **decision['source_pins'], **proof['source_files'], **fixture['source_pins']}
    for path in (SCRIPT, runtime.SCRIPT, runtime.DECISION, qualifier.SCRIPT, migration.SCRIPT,
            migration.PROOF, proof_path, parent.SCRIPT, previous / 'status.json',
            previous / 'stationary_review.json', previous / 'native_learning_signal/completed_external.json', fixture_path):
        pins[str(path)] = sha(path)
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Preserve every historical byte and the separately pinned current implementation')
    expected = copy.deepcopy(cfg)
    expected['wheel_action_limit'] = 1.25
    plan.update(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=5801,
        source_Adam_step=116020, source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous / 'replay'),
        source_files=pins, historical_source_relocations=resolved,
        target_actual_ppo_updates=6001, expected_terminal_Adam_step=120020,
        bounded_training_updates=200, early_learning_inspection_actual_updates=[50, 100],
        full_native_independent_review_after_actual_updates=200,
        early_inspection_is_not_independent_acceptance=True,
        run_name='plain_fullspeed_wheel125_' + output.name + '_u5801',
        reward_key='tracking_ang_vel', reward_value=cfg['rewards']['tracking_ang_vel'],
        only_reward_change={}, only_command_change={}, only_exploration_change={},
        only_controller_change=runtime.DELTA['controller'],
        action_map_or_controller_changed=True, only_wheel_action_clamp_and_history_changed=True,
        next_configuration_changes=runtime.DELTA, authoritative_current_delta=runtime.DELTA,
        unchanged_recipe_continuation=False, selected_single_controller_clamp_trial=str(runtime.DECISION),
        training_config=expected, profile=runtime.profile(), optimizer_reset=False, learned_std_reset=False,
        current_all_error_types_review=decision['source_all_errors'],
        source_current_model_static_retention_verified=True,
        source_static_retention_report=decision['source_static_raw_manifest'],
        source_all657_trials=657, source_all657_passed=586, source_all657_survivors=642,
        source_full27_passed=27, source_full_height_passed=297, source_full_height_trials=306,
        source_headroom_diagnostic=decision['source_headroom324_numeric'],
        source_headroom_sustained120_diagnostic=decision['source_headroom120'],
        source_original_sustained120=decision['source_original120'],
        source_is_unaccepted_repair_branch=True, wheel_action_scale=15., wheel_action_limit=1.25,
        current_normal_wheel_cap_requires_own_independent_review=True,
        diagnostic_headroom_not_enabled_in_training=True,
        normal_saved_wheel_limit_enabled=True,
        actual_training_speed_range_m_s=[-.8, .8], actual_training_yaw_range_rad_s=[-.3, .3],
        training_motion_scale=1., fixed_standing_environments=1024, native_num_envs=4096,
        native_transition_batch=4096 * 48, optimizer_minibatches_per_actual_update=20,
        required_full_height_review_scales=[.25, .375, .5, .625, .75, 1.],
        required_combined_forward_and_reverse_turn_review=True,
        required_independent_fullspeed_sustained120_review=True,
        source_body_limit_regressions_preserved=True, relative_error_target=.1,
        unchanged_full_height_ranges_and_max_speed=True, faster_curriculum_authorized=False,
        next_block_requires_review=True, no_total_training_update_limit=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    for key in ('metadata_adapter_evidence', 'selected_for_another200_by_explicit_review',
            'previous_frozen_sampling_diagnoses_apply_to_parent4201_only'):
        plan.pop(key, None)
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
    result = parent.review(output, after, before)
    result.update(unchanged_recipe_consolidation=False, only_controller_change=runtime.DELTA['controller'],
        wheel125_normal_saved_controller=True, current_controller_static_retention_verified=False,
        own_combined_turns_review_pending=True, own_sustained120_review_pending=True,
        early_learning_inspection_actual_updates=[50, 100], next_block_requires_review=True,
        policy_accepted=False)
    return result


def run(output, previous, unit):
    entry = json.loads((QUALIFICATION / 'full_original_block_entry_preflight.json').read_text())
    if (not entry['qualified'] or entry['controller_sha256'] != sha(SCRIPT)
            or entry['source_actual_updates'] != 5801 or entry['target_actual_updates'] != 6001
            or entry['only_controller_change'] != runtime.DELTA['controller']
            or entry['PPO_updates_performed'] or entry['simulator_started']):
        raise ValueError('Require the complete qualified ordinary200 entry with exactly one controller delta')
    helper.prepare, helper.command = prepare, command
    helper.wait_memory = parent.parent.previous_controller.memory.wait_memory
    helper.base.acceptance, helper.base.replay_command = acceptance, runtime.replay_command
    helper.base.stationary_decision = lambda after, before: review(output, after, before)
    original = helper.write_json

    def pointer(path, value):
        path = Path(path)
        if path.parent == RESULTS and path.name in ('joint_pd_active_joint_tracking.json',
                'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
            value = dict(value, training_motion_scale=1., yaw_min_abs_rate=0.,
                fixed_standing_environments=1024, training_stage='fullspeed_wheel125',
                wheel_action_limit=1.25, action_scale_vel=15.,
                early_learning_inspection_actual_updates=[50, 100])
        return original(path, value)

    helper.write_json = pointer
    try:
        return helper.run(output, unit, previous)
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
