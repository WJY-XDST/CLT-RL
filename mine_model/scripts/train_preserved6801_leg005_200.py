"""One200-update ordinary PPO trial with a lower leg-only exploration bound."""
import argparse
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import train_source4401_consolidation200 as original
import train_source6201_yawcost20_200 as lineage
import joint_pd_preserved6801_leg005_runtime as runtime
import qualify_preserved6801_leg005_transfer as qualifier
import joint_pd_acceptance_10pct_20261017 as acceptance

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = qualifier.OUTPUT
DECISION, ANCHOR_SHA, DELTA = runtime.DECISION, runtime.ANCHOR_SHA, runtime.DELTA
helper = lineage.helper
ready = runtime.repair_ready
_api = adapter.private_module(original.__file__, '_source6801_leg005_original200_plan')
for name in ('SCRIPT', 'RESULTS', 'QUALIFICATION', 'DECISION', 'ANCHOR_SHA', 'DELTA', 'runtime', 'qualifier', 'ready'):
    setattr(_api, name, globals()[name])
ADAPTER = adapter.retarget(_api, ('prepare',),
    integers={'*': {4401: 6801, 4601: 7001, 88020: 136020, 92020: 140020}},
    strings={'4401': '6801'},
    comparisons={'prepare': [("cfg['commands']['training_motion_scale']", .625, 1.)]})


def prepare(output, previous):
    frozen_path = QUALIFICATION / 'native_exploration_transfer_qualification.json'
    import json
    frozen = json.loads(frozen_path.read_text())
    if (not frozen['qualified'] or frozen['source_sha256'] != ANCHOR_SHA
            or frozen['only_exploration_change'] != DELTA['exploration']
            or not frozen['physical_outputs_observations_actions_and_all_rewards_bitwise_equal']
            or frozen['PPO_updates_performed']
            or any(sha(p) != h for p, h in frozen['source_data_sha256'].items())):
        raise ValueError('Require native frozen27 parity before ordinary PPO')
    plan, cfg = _api.prepare(output, previous)
    decision = ready()
    plan.update(run_name='plain_preserved6801_targeted030_leg005_' + Path(output).name + '_u6801',
        reward_key=None, reward_value=None,
        source_recipe_preserved=decision['source_recipe_preserved'],
        only_exploration_change=DELTA['exploration'], only_controller_change={},
        authoritative_current_delta=DELTA, unchanged_recipe_continuation=False,
        selected_leg_only_exploration_floor_trial=str(DECISION),
        exploration_guard=dict(minimum_leg_std=.005, minimum_wheel_std=.01,
            global_minimum_std=.005, saved_initial_std_reset=False),
        early_learning_inspection_actual_updates=[50, 100],
        full_native_independent_review_after_actual_updates=200,
        checkpoint_save_interval=1, source_is_reviewed_unaccepted_repair_branch=True,
        current_all_error_types_review=decision['source_all_errors'],
        current_literal_nonzero_relative_review=decision['source_literal_nonzero_relative_report'],
        source_full27_passed=27, source_full_height_passed=306, source_full_height_trials=306,
        source_all657_passed=642, source_all735_original_passed=720,
        source_all735_literal_both_passed=609, source_sustained120_passed=54,
        source_original_sustained120=decision['source_original_sustained120'],
        source_fullspeed_height_transition_review=decision['source_fullspeed_height_transition_review'],
        source_motion_and_steering_raw=decision['source_motion_and_steering_raw'],
        required_retained_interior54_cases=decision['required_retained_interior54_cases'],
        required_retained_interior_seeds=decision['required_retained_interior_seeds'],
        required_retained_left_interior24_cases=decision['required_retained_left_interior24_cases'],
        required_retained_left_interior_seeds=decision['required_retained_left_interior_seeds'],
        required_full_height_review_scales=[.25, .375, .5, .625, .75, 1.],
        required_combined_forward_and_reverse_turn_review=True,
        required_independent_fullspeed_sustained120_review=True,
        own_fullspeed_height_transition_review_required=True,
        actual_training_speed_range_m_s=[-.8, .8], actual_training_yaw_range_rad_s=[-.3, .3],
        training_motion_scale=1., fixed_standing_environments=1024, native_num_envs=4096,
        native_transition_batch=4096 * 48, optimizer_minibatches_per_actual_update=20,
        all_six_Gaussian_std_entries_remain_trainable=True,
        metadata_adapter_evidence=ADAPTER, next_block_requires_review=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    for key in ('selected_for_another200_by_explicit_review',
                'previous_frozen_sampling_diagnoses_apply_to_parent4201_only'):
        plan.pop(key, None)
    for p in (SCRIPT, runtime.SCRIPT, qualifier.SCRIPT, Path(original.__file__), lineage.SCRIPT, frozen_path):
        plan['source_files'][str(p)] = sha(p)
    plan['source_files'].update(frozen['source_data_sha256'])
    write_json(Path(output) / 'experiment_plan.json', plan)
    return plan, cfg


command = runtime.command


def review(output, after, before):
    result = original.review(output, after, before)
    result.update(only_reward_repair={}, only_exploration_repair=DELTA['exploration'],
        only_command_change={}, only_controller_change={}, unchanged_recipe_consolidation=False,
        next_block_requires_review=True, own_static_retention_verified=False, policy_accepted=False)
    return result


def run(output, previous, unit):
    import json
    entry = json.loads((QUALIFICATION / 'full_original_block_entry_preflight.json').read_text())
    if (not entry['qualified'] or entry['controller_sha256'] != sha(SCRIPT)
            or entry['source_actual_updates'] != 6801 or entry['target_actual_updates'] != 7001
            or entry['only_exploration_change'] != DELTA['exploration']
            or any(entry[k] for k in ('only_reward_change', 'only_command_change', 'only_controller_change'))
            or entry['PPO_updates_performed'] or entry['simulator_started']):
        raise ValueError('Require the full original200 entry preflight')
    helper.prepare, helper.command = prepare, command
    helper.wait_memory = original.previous_controller.memory.wait_memory
    helper.base.acceptance, helper.base.replay_command = acceptance, runtime.replay_command
    helper.base.stationary_decision = lambda after, before: review(output, after, before)
    original_write = helper.write_json

    def pointer(path, value):
        path = Path(path)
        if path.parent == RESULTS and path.name in ('joint_pd_active_joint_tracking.json',
                'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
            value = dict(value, training_motion_scale=1., fixed_standing_environments=1024,
                training_stage='preserved6801_targeted030_leg005', wheel_action_limit=1.25,
                action_scale_vel=15., early_learning_inspection_actual_updates=[50, 100])
        return original_write(path, value)
    helper.write_json = pointer
    try:
        return helper.run(output, unit, previous)
    finally:
        helper.write_json = original_write


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
