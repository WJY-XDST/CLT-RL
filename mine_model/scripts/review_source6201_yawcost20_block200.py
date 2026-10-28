"""Preserve all original reviews and both retained interior sets for6401.

This worker only qualifies or collects frozen evidence after the owned200
updates. It cannot accept a policy, extend training or change a threshold.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from joint_pd_continuation import ROOT,sha
import plain27_sampler_metadata_adapter as adapter
import review_wheel125_training_block as core
import review_source6001_orientation060_block200 as previous
import verify_completed_plain27_joint_pd_trial as final_check
import joint_pd_source6201_yawcost20_runtime as runtime
import qualify_source6201_yawcost20_transfer as qualifier
import audit_original_and_literal_relative_errors as literal
import audit_independent_fullspeed_sustained_raw as long_raw
import audit_completed_motion_phase_errors as phase_raw

SCRIPT=Path(__file__).resolve()
RESULTS=ROOT/'mine_model/results'
QUALIFICATION=qualifier.OUTPUT/'post200_original_review_qualification.json'
OUTPUT=RESULTS/'joint_pd_yawcost20_review6401_20261021'
STATIC=RESULTS/'joint_pd_static_retention_yawcost20_20261021_source6401'
_base=adapter.private_module(core.SCRIPT,'_yawcost20_original_all_scope_review')
_base.SCRIPT,_base.QUALIFICATION,_base.OUTPUT,_base.STATIC=SCRIPT,QUALIFICATION,OUTPUT,STATIC
_base.runtime,_base.final_check=runtime,final_check
BASE_ADAPTER=adapter.retarget(_base,('run',),
    integers={'*':{6001:6401,120020:128020}},
    strings={'source6001_wheel125':'source6401_yawcost20','source6001':'source6401'})
_api=adapter.private_module(previous.SCRIPT,'_yawcost20_retained_known_interior_review')
_api.SCRIPT,_api.QUALIFICATION,_api.OUTPUT,_api.STATIC=SCRIPT,QUALIFICATION,OUTPUT,STATIC
_api.runtime,_api.qualifier,_api._api=runtime,qualifier,_base
QUALIFIER_ADAPTER=adapter.retarget(_api,('qualify',),
    integers={'*':{6001:6201,6201:6401,120020:124020,124020:128020}},
    exact_strings={'joint_pd_wheel125_review6001_20261020':
        'joint_pd_orientation060_review6201_20261020'})
_api.ADAPTER_EVIDENCE=BASE_ADAPTER+QUALIFIER_ADAPTER


def read(path):
    return json.loads(Path(path).read_text())


def qualify():
    proof=_api.qualify()
    decision=runtime.repair_ready()
    path=Path(decision['required_retained_left_interior24_cases'])
    manifest=read(path)
    pair=read(decision['source_left_interior_paired_review'])
    raw=read(pair['results']['candidate6201']['raw_verified'])
    if (len(manifest['cases'])!=8 or manifest['seeds']!=[311483,311484,311485]
            or manifest['checkpoint_sha256']!=runtime.ANCHOR_SHA
            or not pair['paired_left_interior_verified']
            or not raw['verified'] or raw['total_trials']!=24
            or raw['all7_metrics_and_limits_raw_reproduction_max_error']!=0.):
        raise ValueError('Retain all source6201 left-interior cases, strict gates and paired seeds')
    proof.update(retained_left_cases=manifest['cases'],retained_left_seeds=manifest['seeds'],
        retained_left_source_directory=str(path.parent),
        retained_left_source_raw=pair['results']['candidate6201']['raw_verified'],
        known54_and_left24_both_required=True,
        deterministic_moving_and_steering_literal_phase_review_required=True,
        independent120_raw_and_literal657_required=True,
        original_Gaussian_stress_survival_rule_unchanged=True,
        complete_review_expected=str(OUTPUT/'complete_with_retained_interior_reviews.json'))
    for p in (SCRIPT,previous.SCRIPT,core.SCRIPT,adapter.SCRIPT,literal.SCRIPT,
            long_raw.SCRIPT,phase_raw.SCRIPT,path,Path(decision['source_left_interior_paired_review']),
            Path(pair['results']['candidate6201']['raw_verified'])):
        proof['source_pins'][str(p)]=sha(p)
    core.intact(proof['source_pins'])
    QUALIFICATION.write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
    return proof


def run():
    result=_api.run()
    proof=read(QUALIFICATION)
    training=Path(proof['training'])
    source=Path(result['checkpoint'])
    final=read(training/'final_checkpoint_verification_readonly_20261019.json')
    if (final['actual_ppo_updates']!=6401 or final['all17_Adam_step']!=128020
            or final['saved_configuration_differences']!=['/log_dir','/rewards/yaw_rate_error_sq']):
        raise ValueError('Require the exact retained-state200 single-yaw-penalty descendant')
    # Both interior command sets retain original physical metrics and None semantics.
    known_raw=OUTPUT/'known_interior/all54_raw_verified.json'
    known_literal=OUTPUT/'known_interior/all54_literal10.json'
    literal.audit(known_raw,known_literal)
    directory=_base.native(source,OUTPUT/'retained_left_interior/candidate',20.,
        proof['retained_left_seeds'],proof['retained_left_cases'])
    reader,_,_=core.combined.reader(proof['retained_left_cases'],proof['retained_left_seeds'])
    report=reader(directory,source)
    report_path=directory/'original_metric_left_interior_diagnosis.json'
    core.save(report_path,report)
    with np.load(directory/'policy_inputs.npz',allow_pickle=False) as a,np.load(
            Path(proof['retained_left_source_directory'])/'policy_inputs.npz',allow_pickle=False) as b:
        if not np.array_equal(a['observations'][0],b['observations'][0]):
            raise ValueError('Retained left-interior initial policy inputs changed')
    spec=OUTPUT/'retained_left_interior/input_specification.json'
    core.save(spec,[dict(scope='retained_left_interior8',kind='fixed_command_probe',
        directory=str(directory),report=str(report_path))])
    left_raw=OUTPUT/'retained_left_interior/all24_raw_verified.json'
    left=core.errors.audit(training/'final_checkpoint_verification_readonly_20261019.json',spec,left_raw)
    left_literal=OUTPUT/'retained_left_interior/all24_literal10.json'
    left_rel=literal.audit(left_raw,left_literal)
    original_raw=training/'all_errors_all657_source6401_readonly_20261020.json'
    original_literal=training/'literal_nonzero_error10_all657_source6401_readonly_20261020.json'
    rel=literal.audit(original_raw,original_literal)
    phase_path=training/'motion_and_steering_all810_phases_raw_verified_readonly_20261020.json'
    phases=phase_raw.audit([OUTPUT/'moving_height/deterministic',
        OUTPUT/'moving_height/saved_Gaussian',OUTPUT/'forward_steering/deterministic'],phase_path)
    phases.update(actual_ppo_updates=6401,actual_update_clock_verified_from_final_checkpoint_report=True)
    phase_path.write_text(json.dumps(phases,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    phase_literal=training/'literal_nonzero_error10_motion_and_steering_all810_readonly_20261020.json'
    literal.audit(phase_path,phase_literal)
    long_path=training/'independent120_raw_verified_paired_source6201_readonly_20261020.json'
    decision=runtime.repair_ready()
    long=long_raw.audit(training,OUTPUT,long_path,Path(decision['source_original_sustained120']))
    # Pair the complete original scope by exact identity, preserving every failure.
    old=read(decision['source_all_errors'])
    new=read(original_raw)
    key=lambda t:(t['scope'],t['case']['name'],t['seed'])
    before={key(t):t for t in old['trials']}
    after={key(t):t for t in new['trials']}
    if before.keys()!=after.keys() or len(after)!=657:
        raise ValueError('Original657 case/seed identities changed')
    for k in after:
        if before[k]['case']!=after[k]['case']:
            raise ValueError('Original speed/yaw/height commands changed')
    paired=training/'paired_original_all657_source6201_candidate6401_readonly_20261020.json'
    core.save(paired,dict(source_actual_updates=6201,candidate_actual_updates=6401,
        total_trials=657,source_original_passed=old['total_passed'],candidate_original_passed=new['total_passed'],
        rescues=[list(k) for k in after if not before[k]['passed'] and after[k]['passed']],
        regressions=[list(k) for k in after if before[k]['passed'] and not after[k]['passed']],
        source_raw=decision['source_all_errors'],candidate_raw=str(original_raw),
        source_pins={str(p):sha(p) for p in
            (SCRIPT,Path(decision['source_all_errors']),original_raw)},
        original_cases_and_seeds_unchanged=True,training_modified=False,policy_accepted=False))
    result.update(retained_left24_raw_verified=str(left_raw),retained_left24_literal10=str(left_literal),
        retained_left24_survivors=left['total_survivors'],retained_left24_original_passed=left['total_passed'],
        retained_left24_original_and_literal_passed=left_rel['original_gates_and_nonzero_relative10_both_passed'],
        known54_literal10=str(known_literal),original657_literal10=str(original_literal),
        original657_and_literal_both_passed=rel['original_gates_and_nonzero_relative10_both_passed'],
        original657_paired_comparison=str(paired),motion_and_steering810_raw=str(phase_path),
        motion_and_steering810_literal10=str(phase_literal),independent120_raw=str(long_path),
        independent120_survivors=long['survivors'],independent120_original_passed=long['passed'],
        all_original_and_retained_reviews_complete=True,policy_accepted=False,
        can_select_or_extend_training=False)
    core.intact(proof['source_pins'])
    core.save(OUTPUT/'complete_with_retained_interior_reviews.json',result)
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualify-only',action='store_true')
    args=cli.parse_args()
    d=qualify() if args.qualify_only else run()
    print(json.dumps({k:d[k] for k in ('qualified','actual_updates',
        'all_original_and_retained_reviews_complete','policy_accepted') if k in d}))
