"""One reviewed unchanged200-update consolidation after the low-yaw coverage repair."""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, optimizer_step
from train_joint_pd_standing_seed import saved_updates
import joint_pd_source4001_motion0625_runtime as runtime
import qualify_retained_plain27_source as qualifier
import train_source4201_low_yaw_200 as previous_controller

SCRIPT = Path(__file__).resolve()
helper = previous_controller.helper
RESULTS = ROOT / 'mine_model/results'
DECISION = RESULTS / 'joint_pd_source4401_consolidation_decision_20261019.json'
QUALIFICATION = RESULTS / 'joint_pd_source4401_consolidation_qualification_20261019'
ANCHOR_SHA = 'e69faa8cc047416f27f2fe10ef9e4177bce3d61c78527ec3dd0f1309ae48309a'
DELTA = dict(rewards={}, commands={}, exploration={}, controller={}, physics={})


def ready():
    decision = json.loads(DECISION.read_text())
    static = json.loads(Path(decision['source_static_retention_report']).read_text())
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 4401 or decision['source_all17_Adam_step'] != 88020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not decision['reviewed_unchanged_consolidation_selected']
            or not static['static_retention_verified'] or static['checkpoint_sha256'] != ANCHOR_SHA
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or decision['faster_curriculum_authorized']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require the intact completed4401 independent/physical/learning review')
    return decision


def api_runtime():
    return SimpleNamespace(**dict(runtime.__dict__, ANCHOR_SHA=ANCHOR_SHA, DECISION=DECISION,
        DELTA=DELTA, repair_ready=ready))


def prepare(output, previous):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    decision = ready()
    if (output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_optimization_')
            or previous != Path(decision['training_output'])):
        raise ValueError('Use a fresh owned block from the reviewed4401 source')
    source = Path(decision['source_checkpoint'])
    saved, cfg = runtime.require_source(source)
    proof = json.loads((QUALIFICATION / 'qualification.json').read_text())
    if (not proof['qualified'] or proof['source_sha256'] != ANCHOR_SHA
            or proof['profile'] != runtime.profile() or proof['decision_sha256'] != sha(DECISION)
            or proof['implementation_sha256'] != sha(runtime.SCRIPT)
            or proof['qualifier_sha256'] != sha(qualifier.SCRIPT)
            or proof['next_configuration_changes'] != DELTA
            or not all(proof[k] for k in ('all_model_and_Adam_tensors_bitwise_preserved',
                'initial_saved_std_bitwise_preserved', 'actual_saved_adaptive_LR_preserved',
                'exact_registered_projection_test_passed', 'untrained_metadata_roundtrip_inference_bitwise_equal',
                'saved_env_and_agent_configurations_unchanged'))
            or proof['actual_PPO_updates_executed'] or proof['optimizer_steps_executed']
            or any(sha(p) != h for p, h in proof['source_files'].items())
            or saved_updates(source) != 4401 or optimizer_step(saved) != 88020
            or cfg['commands']['training_motion_scale'] != .625 or cfg['commands']['yaw_min_abs_rate'] != 0.):
        raise ValueError('Require exact unchanged-state/profile/configuration CPU qualification')
    old = json.loads((previous / 'experiment_plan.json').read_text())
    plan = copy.deepcopy(old)
    pins = {**old['source_files'], **decision['source_pins'], **proof['source_files']}
    pins.update({str(p): sha(p) for p in (SCRIPT, DECISION, runtime.SCRIPT, qualifier.SCRIPT,
        QUALIFICATION / 'qualification.json', Path(previous_controller.__file__),
        previous / 'status.json', previous / 'stationary_review.json',
        previous / 'native_learning_signal/completed_external.json')})
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('A pinned source or historical proof changed')
    peak, requirement = previous_controller.memory.memory_requirement()
    plan.update(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=4401,
        source_Adam_step=88020, source_Adam_states=17,
        source_optimizer_LR=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous / 'replay'),
        source_files=pins, target_actual_ppo_updates=4601, expected_terminal_Adam_step=92020,
        bounded_training_updates=200, run_name='plain_consolidation_' + output.name + '_u4401',
        only_reward_change={}, only_command_change={}, only_exploration_change={},
        next_configuration_changes=DELTA, authoritative_current_delta=DELTA,
        unchanged_recipe_continuation=True, selected_for_another200_by_explicit_review=str(DECISION),
        training_config=copy.deepcopy(cfg), profile=runtime.profile(), optimizer_reset=False,
        learned_std_reset=False, source_current_model_static_retention_verified=True,
        source_static_retention_report=decision['source_static_retention_report'],
        measured_native_RSS_highwater_bytes=peak, combined_memory_required_bytes=requirement,
        current_exploration_review=str(previous / 'effective_exploration_prefix_0200_readonly.json'),
        current_policy_update_health_review=str(previous / 'policy_update_health_prefix_0200_readonly.json'),
        current_critic_and_command_coverage_review=str(previous / 'critic_and_command_coverage_prefix_0200_readonly.json'),
        current_wheel_error_budget=str(previous / 'wheel_tracking_budget_candidate4401_0625_readonly_20261019.json'),
        current_height_error_budget=str(previous / 'native_motion_training_prefix_0200_readonly.json'),
        current_geometric_exploration_review=str(previous / 'joint_target_geometric_exploration_prefix_0200_readonly.json'),
        current_gradient_allocation_review=str(previous / 'gradient_allocation_prefix_0200_readonly.json'),
        current_command_coverage_gap_report=str(previous / 'low_yaw_command_coverage_prefix_0200_readonly.json'),
        current_signed_wheel_tracking_budget=str(previous / 'wheel_tracking_budget_candidate4401_0625_readonly_20261019.json'),
        current_terminal_age_review=str(previous / 'terminal_event_ages_prefix_0200_readonly.json'),
        source_full27_passed=27, source_full_height_passed=201, source_full_height_trials=204,
        unchanged_full_height_ranges_and_max_speed=True, faster_curriculum_authorized=False,
        next_block_requires_review=True, no_total_training_update_limit=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    plan.pop('current_matched_sampling_signed_bias', None)
    plan['previous_frozen_sampling_diagnoses_apply_to_parent4201_only'] = True
    output.mkdir()
    write_json(output / 'experiment_plan.json', plan)
    return plan, cfg


command = previous_controller.command


def review(output, after, before):
    result = previous_controller.original_controller.review(output, after, before)
    result.update(only_reward_repair={}, only_exploration_repair={}, only_command_change={},
        unchanged_recipe_consolidation=True, next_block_requires_review=True,
        own_static_retention_verified=False, retained_four_full_height_reviews_pending=True,
        original_numeric_readers_and_acceptance_predicates_unchanged=True)
    return result


def run(output, previous, unit):
    entry = json.loads((QUALIFICATION / 'full_original_block_entry_preflight.json').read_text())
    if (not entry['qualified'] or entry['controller_sha256'] != sha(SCRIPT)
            or entry['source_actual_updates'] != 4401 or entry['target_actual_updates'] != 4601
            or entry['only_command_change'] or entry['PPO_updates_performed'] or entry['simulator_started']):
        raise ValueError('Require the full original block-entry preflight')
    helper = previous_controller.helper
    helper.prepare, helper.command, helper.wait_memory = prepare, command, previous_controller.memory.wait_memory
    helper.base.acceptance = previous_controller.acceptance
    helper.base.replay_command = runtime.replay_command
    helper.base.stationary_decision = lambda after, before: review(output, after, before)
    original = helper.write_json

    def pointer(path, value):
        path = Path(path)
        if path.parent == RESULTS and path.name in ('joint_pd_active_joint_tracking.json',
                'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
            value = dict(value, training_motion_scale=.625, yaw_min_abs_rate=0.,
                fixed_standing_environments=1024, training_stage='low_yaw_consolidation')
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
