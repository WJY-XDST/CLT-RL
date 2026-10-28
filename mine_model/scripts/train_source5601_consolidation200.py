"""One explicitly reviewed unchanged200 block, retaining the full-speed5601 state."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import train_source4401_consolidation200 as template
import train_source5401_motion100_200 as parent
import joint_pd_source5401_motion100_runtime as runtime
import qualify_retained_plain27_source as qualifier

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
DECISION = RESULTS / 'joint_pd_source5601_consolidation_decision_20261020.json'
QUALIFICATION = RESULTS / 'joint_pd_source5601_consolidation_qualification_20261020'
ANCHOR_SHA = '9bf5a4932bc2706ebcd7d1031954d553925778e5f624e41b692d388b25fb5b74'
DELTA = dict(rewards={}, commands={}, exploration={}, controller={}, physics={})


def ready():
    decision = json.loads(DECISION.read_text())
    read = lambda key: json.loads(Path(decision[key]).read_text())
    final, static = read('source_final_verification'), read('source_static_raw_manifest')
    all_errors, moving = read('source_all_errors'), read('source_moving_height_raw_integrity')
    travel, headroom = read('source_complete_travel_review'), read('source_headroom_diagnostic')
    if (decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 5601 or decision['source_all17_Adam_step'] != 112020
            or decision['planned_actual_updates'] != 200 or decision['next_configuration_changes'] != DELTA
            or not decision['reviewed_unchanged_consolidation_selected']
            or not decision['source_is_unaccepted_repair_branch']
            or not final['verified'] or final['actual_ppo_updates'] != 5601
            or final['checkpoint_sha256'] != ANCHOR_SHA or final['all17_Adam_step'] != 112020
            or not static['raw_integrity_verified'] or not static['static_retention_verified']
            or static['checkpoint_sha256'] != ANCHOR_SHA
            or not all_errors['verified'] or all_errors['total_trials'] != 657
            or all_errors['checkpoint_sha256'] != ANCHOR_SHA
            or all_errors['all7_metrics_and_limits_raw_reproduction_max_error']
            or not moving['verified'] or moving['raw_rows'] != 135000
            or moving['checkpoint_sha256'] != ANCHOR_SHA
            or not travel['all_required_review_data_complete'] or travel['actual_updates'] != 5601
            or travel['checkpoint_sha256'] != ANCHOR_SHA
            or headroom['checkpoint_sha256'] != ANCHOR_SHA or headroom['total_trials'] != 51
            or headroom['counts_toward_original657_acceptance']
            or not decision['all_failed_and_regressed_trials_preserved']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or decision['faster_curriculum_authorized']
            or any(sha(p) != h for p, h in decision['source_pins'].items())):
        raise ValueError('Require completed own5601 full/static/travel/learning/physical reviews and explicit selection')
    return decision


_api = adapter.private_module(template.__file__, '_source5601_original_consolidation_controller')
_api.SCRIPT, _api.DECISION, _api.QUALIFICATION = SCRIPT, DECISION, QUALIFICATION
_api.runtime, _api.qualifier, _api.ANCHOR_SHA, _api.DELTA = runtime, qualifier, ANCHOR_SHA, DELTA
_api.ready = ready
ADAPTER_EVIDENCE = adapter.retarget(_api, ('prepare',),
    integers={'*': {4401: 5601, 4601: 5801, 88020: 112020, 92020: 116020}},
    strings={'4401': '5601', 'consolidation_': 'fullspeed_consolidation_'},
    comparisons={'prepare': [("cfg['commands']['training_motion_scale']", .625, 1.)]})
_original_prepare = _api.prepare


def api_runtime():
    return SimpleNamespace(**dict(runtime.__dict__, ANCHOR_SHA=ANCHOR_SHA, DECISION=DECISION,
        DELTA=DELTA, repair_ready=ready))


def prepare(output, previous):
    plan, cfg = _original_prepare(output, previous)
    decision = ready()
    current = Path(previous)
    plan.update(reward_key='tracking_ang_vel', reward_value=cfg['rewards']['tracking_ang_vel'],
        only_training_sampler_amplitude_changes=False,
        source_full_height_passed=decision['source_full_height_passed'], source_full_height_trials=306,
        source_all657_trials=657, source_all657_passed=decision['source_all657_passed'],
        source_all657_survivors=decision['source_all657_survivors'],
        current_all_error_types_review=decision['source_all_errors'],
        current_wheel_error_budget=decision['source_all_errors'],
        current_signed_wheel_tracking_budget=decision['source_all_errors'],
        current_wheel_reward_sensitivity_review=decision['source_wheel_reward_sensitivity'],
        current_signed_full_speed_command_coverage=decision['source_full_speed_command_coverage'],
        source_moving_height_summary=decision['source_moving_height_summary'],
        source_moving_height_raw_integrity=decision['source_moving_height_raw_integrity'],
        source_combined_turns_summary=decision['source_combined_turns_summary'],
        source_forward_steering_steps_summary=decision['source_forward_steering_steps_summary'],
        source_complete_travel_review=decision['source_complete_travel_review'],
        source_headroom_diagnostic=decision['source_headroom_diagnostic'],
        diagnostic_headroom_not_enabled_in_training=True,
        actual_training_speed_range_m_s=[-.8, .8], actual_training_yaw_range_rad_s=[-.3, .3],
        required_full_height_review_scales=[.25, .375, .5, .625, .75, 1.],
        required_combined_forward_and_reverse_turn_review=True,
        no_total_training_update_limit=True, metadata_adapter_evidence=ADAPTER_EVIDENCE)
    plan.pop('current_std_floor_pressure_review', None)
    plan.pop('current_command_coverage_gap_report', None)
    plan['source_wider_frozen_review'] = decision['source_full_speed_report']
    pins = [SCRIPT, Path(template.__file__), parent.SCRIPT, adapter.SCRIPT]
    pins.extend(Path(decision[k]) for k in ('source_all_errors', 'source_complete_travel_review',
        'source_moving_height_summary', 'source_moving_height_raw_integrity',
        'source_combined_turns_summary', 'source_forward_steering_steps_summary',
        'source_wheel_reward_sensitivity', 'source_headroom_diagnostic'))
    if any(not p.is_file() for p in pins):
        raise ValueError('Preserve every completed current-model review and new original-block adapter')
    plan['source_files'].update({str(p): sha(p) for p in pins})
    write_json(Path(output) / 'experiment_plan.json', plan)
    return plan, cfg


command, helper = parent.command, parent.helper


def review(output, after, before):
    result = _api.review(output, after, before)
    result.pop('retained_four_full_height_reviews_pending', None)
    result.update(retained_six_full_height_reviews_pending=True,
        combined_forward_reverse_turns_review_pending=True,
        own_forward_steering_fullheight_steps_pending=True)
    return result


def run(output, previous, unit):
    entry = json.loads((QUALIFICATION / 'full_original_block_entry_preflight.json').read_text())
    if (not entry['qualified'] or entry['controller_sha256'] != sha(SCRIPT)
            or entry['source_actual_updates'] != 5601 or entry['target_actual_updates'] != 5801
            or entry['only_command_change'] or entry['PPO_updates_performed'] or entry['simulator_started']):
        raise ValueError('Require the complete unchanged original-block entry preflight')
    helper.prepare, helper.command, helper.wait_memory = prepare, command, parent.previous_controller.memory.wait_memory
    helper.base.acceptance = parent._api.acceptance
    helper.base.replay_command = runtime.replay_command
    helper.base.stationary_decision = lambda after, before: review(output, after, before)
    original = helper.write_json

    def pointer(path, value):
        path = Path(path)
        if path.parent == RESULTS and path.name in ('joint_pd_active_joint_tracking.json',
                'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
            value = dict(value, training_motion_scale=1., yaw_min_abs_rate=0.,
                fixed_standing_environments=1024, training_stage='fullspeed_consolidation')
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
