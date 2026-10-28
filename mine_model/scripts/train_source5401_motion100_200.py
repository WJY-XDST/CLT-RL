"""One reviewed200-update full-speed sampler continuation from exact5401."""
import argparse
import json
from pathlib import Path

import train_source3601_motion0375_200 as template
import joint_pd_source5401_motion100_runtime as runtime
import qualify_plain27_single_command_transfer as qualifier
import plain27_sampler_metadata_adapter as adapter
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_source5401_motion100_qualification_20261019'
_api = adapter.private_module(template.__file__, '_source5401_original_sampler_controller')
_api.SCRIPT, _api.QUALIFICATION, _api.runtime, _api.qualifier = SCRIPT, QUALIFICATION, runtime, qualifier
ADAPTER_EVIDENCE = adapter.retarget(_api, ('qualifications', 'prepare', 'review', 'run'),
    integers={'*': {3601: 5401, 3801: 5601, 72020: 108020, 76020: 112020}, 'qualifications': {15: 0}},
    strings={'source3601': 'source5401', 'motion0375': 'motion100', '_u3601': '_u5401'},
    exact_strings={'saved_env_diff': 'saved_roundtrip_env_diff',
        'original200_and_undeclared_transfer_rejected': 'original_source200_rejected'})
_original_prepare, _original_run = _api.prepare, _api.run


def prepare(output, previous):
    plan, expected = _original_prepare(output, previous)
    decision, profile = runtime.repair_ready(), runtime.profile()
    plan.update(reward_key='tracking_ang_vel', reward_value=expected['rewards']['tracking_ang_vel'],
        authoritative_current_delta=runtime.DELTA,
        exploration_guard=dict(minimum_leg_std=profile['minimum_leg_std'],
            minimum_wheel_std=profile['minimum_wheel_std'], global_minimum_std=profile['global_minimum_std'],
            saved_initial_std_reset=False),
        source_full27_passed=27, source_full_height_passed=252, source_full_height_trials=255,
        source_moving_height_summary=decision['source_moving_height_summary'],
        source_moving_height_raw_integrity=decision['source_moving_height_raw_integrity'],
        source_moving_numeric_readiness=True,
        current_wheel_error_budget=decision['source_all_errors_including_full_speed'],
        current_all_error_types_review=decision['source_all_errors_including_full_speed'],
        source_wider_frozen_review=decision['source_wider_frozen_review'],
        source_combined_turns_summary=decision['source_combined_turns_summary'],
        source_forward_steering_steps_summary=decision['source_forward_steering_steps_summary'],
        source_wider_stability_and_height_failures_preserved=True,
        current_std_floor_pressure_review=str(Path(previous) / 'std_floor_pressure_prefix_0200_readonly_20261019.json'),
        required_full_height_review_scales=[.25, .375, .5, .625, .75, 1.],
        required_combined_forward_and_reverse_turn_review=True,
        actual_training_speed_range_m_s=[-.8, .8], actual_training_yaw_range_rad_s=[-.3, .3],
        unchanged_full_height_ranges_and_physical_command_bounds=True,
        only_training_sampler_amplitude_changes=True,
        complete_original_block_entry_metadata=True)
    paths = [SCRIPT, runtime.SCRIPT, qualifier.SCRIPT, adapter.SCRIPT, Path(template.__file__)]
    for key in ('current_wheel_error_budget', 'current_std_floor_pressure_review',
            'source_combined_turns_summary', 'source_forward_steering_steps_summary'):
        paths.append(Path(plan[key]))
    if any(not p.is_file() for p in paths):
        raise ValueError('Preserve all own completed source diagnostics')
    plan['source_files'].update({str(p): sha(p) for p in paths})
    plan['metadata_adapter_evidence'] = ADAPTER_EVIDENCE
    write_json(Path(output) / 'experiment_plan.json', plan)
    return plan, expected


_api.prepare = prepare
command, review, qualifications = _api.command, _api.review, _api.qualifications
helper, previous_controller = _api.helper, _api.original_controller


def run(output, previous, unit):
    entry = json.loads((QUALIFICATION / 'full_original_block_entry_preflight.json').read_text())
    if (not entry['qualified'] or entry['controller_sha256'] != sha(SCRIPT)
            or entry['source_actual_updates'] != 5401 or entry['target_actual_updates'] != 5601
            or entry['only_command_change'] != runtime.DELTA['commands']
            or entry['PPO_updates_performed'] or entry['simulator_started']):
        raise ValueError('Require the actual full original sampler-block entry qualification')
    return _original_run(output, previous, unit)


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
