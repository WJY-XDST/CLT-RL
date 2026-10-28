"""Retain the original200-update controller for the single yaw quadratic penalty trial."""
import argparse
import json
from pathlib import Path

from auto_train_mine import write_json
from joint_pd_continuation import ROOT,sha
import plain27_sampler_metadata_adapter as adapter
import train_source6001_orientation060_200 as original
import joint_pd_source6201_yawcost20_runtime as runtime
import qualify_source6201_yawcost20_transfer as qualifier

SCRIPT=Path(__file__).resolve()
RESULTS=ROOT/'mine_model/results'
QUALIFICATION=qualifier.OUTPUT
previous=original.previous
helper=original.helper
_api=adapter.private_module(original.SCRIPT,'_source6201_yawcost20_original200_controller')
_api.SCRIPT,_api.runtime,_api.qualifier,_api.QUALIFICATION=SCRIPT,runtime,qualifier,QUALIFICATION
ADAPTER=adapter.retarget(_api,('prepare','run'),
    integers={'*':{6001:6201,6201:6401,120020:124020,124020:128020}},
    strings={'orientation060':'yawcost20','_u6001':'_u6201','reviewed6001':'reviewed6201'},
    keyword_names={'selected_single_orientation_weight_trial':'selected_single_yaw_penalty_trial'})
_original_prepare=_api.prepare


def prepare(output,previous_output):
    plan,expected=_original_prepare(output,previous_output)
    decision=runtime.repair_ready()
    plan.update(source_all657_passed=591,source_original_and_literal_nonzero10_both_passed=464,
        source_full_height_passed=288,source_sustained120_passed=48,
        source_supplemental_interior54_passed=54,source_supplemental_interior54_body_failed=0,
        source_posture_reference_already_repaired=True,
        source_left_interior24_passed=18,source_left_interior24_body_failed=0,
        source_left_interior_paired_review=decision['source_left_interior_paired_review'],
        required_retained_left_interior24_cases=decision['required_retained_left_interior24_cases'],
        required_retained_left_interior24_seeds=decision['required_retained_left_interior_seeds'],
        required_left_interior24_review_after_actual_updates=200,
        required_deterministic_moving_and_steering_phase_literal_raw_review=True,
        source_stationary_actual_bodyP95_deg=.8454687765510931,
        stationary_mixed_pose_ratio_is_not_an_angle=True,
        original_full200_CPU_state_and_PPO_operations_preserved=True,
        source_target_clock_metadata_adapter=ADAPTER,
        source_summary_metadata_rebound_to_current_completed_reports=True)
    for p in (SCRIPT,original.SCRIPT,adapter.SCRIPT,runtime.SCRIPT,qualifier.SCRIPT):
        plan['source_files'][str(p)]=sha(p)
    if any(sha(p)!=digest for p,digest in plan['source_files'].items()):
        raise ValueError('Current retained-source evidence changed')
    write_json(Path(output)/'experiment_plan.json',plan)
    return plan,expected


_api.prepare=prepare
command,review,run=_api.command,_api.review,_api.run


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--previous',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--unit')
    cli.add_argument('--prepare-only',action='store_true')
    args=cli.parse_args()
    if args.prepare_only:
        prepare(args.output,args.previous)
    else:
        run(args.output,args.previous,args.unit)
