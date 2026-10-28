"""Retain the full original review after the owned4096 ordinary200 trial.

Reuse original numeric readers, cases, seeds, windows and failed-None
semantics. Adapt only source/target labels, the declared count differences,
and the truthful4096 final-verification filename/alignment key.
"""
import argparse
import ast
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import joint_pd_AF7001_yawsigma0025_runtime as native_runtime
import source7001_yawcost200_unchanged_context as old_context
import review_source7001_yawcost200_unchanged200 as old_review
import review_wheel125_training_block as core
import review_source6001_orientation060_block200 as known_template
import review_source6201_lin300_block200 as retained_template
import verify_AF7001_yawsigma0025_completed200 as final_check
import audit_completed_tracking_reports as original_errors
import audit_independent_fullspeed_sustained_raw as original_long
import review_preserved6801_leg005_fullspeed_height_transitions_v2 as signed_template
import review_source7001_yawcost200_unchanged_fullspeed_height as old_signed
import probe_source7001_yawcost200_unchanged_small_yaw120_v2 as small_template

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
PREPARATION = RESULTS / 'joint_pd_AF7001_yawsigma0025_qualification_v2_20261023'
LAUNCH = RESULTS / 'joint_pd_AF7001_yawsigma0025_owned_block200_launch.json'
QUALIFICATION = PREPARATION / 'post200_original4096_review_qualification_after_finalizer_repair_v2.json'
SIGNED_QUALIFICATION = PREPARATION / 'post200_signed4096_review_qualification_after_finalizer_repair_v2.json'
SMALL_QUALIFICATION = PREPARATION / 'post200_small_yaw120_4096_review_qualification_after_finalizer_repair_v2.json'
OUTPUT = RESULTS / 'joint_pd_AF7001_yawsigma0025_review7201_20261023'
STATIC = RESULTS / 'joint_pd_static_retention_AF7001_yawsigma0025_source7201_20261023'
EXTRA = RESULTS / 'joint_pd_AF7001_yawsigma0025_fullspeed_height_source7001_candidate7201_20261023'
SMALL = RESULTS / 'joint_pd_AF7001_yawsigma0025_small_yaw120_20261023'
UNIT = 'clt-joint-pd-af7001-yawsigma0025-full-reviews-20261023.service'
FINAL_NAME = 'final_checkpoint_verification_readonly_20261019.json'
METADATA = {
    'final_checkpoint_verification_readonly_20261019.json': FINAL_NAME,
    'all200_all4096_alignment_max_errors': 'all200_all8192_alignment_max_errors',
}


def read(path):
    return json.loads(Path(path).read_text())


def ready():
    # The selected source remains the exact completely reviewed7001 model.
    old = old_context.ready()
    native_runtime.review_ready(Path(old['source_checkpoint']))
    return dict(old, next_configuration_changes=native_runtime.DELTA)


runtime = SimpleNamespace(**dict(old_context.api_runtime().__dict__,
    SCRIPT=native_runtime.SCRIPT, FLAG=native_runtime.FLAG,
    profile=native_runtime.profile, require_source=native_runtime.require_source,
    replay_command=native_runtime.replay_command, repair_ready=ready, DELTA=native_runtime.DELTA))

errors = adapter.private_module(original_errors.SCRIPT, '_AF7001_yawsigma0025_tracking_raw_reader')
long_raw = adapter.private_module(original_long.SCRIPT, '_AF7001_yawsigma0025_sustained_raw_reader')
ADAPTERS = adapter.retarget(errors, ('audit',), exact_strings=METADATA)
ADAPTERS += adapter.retarget(long_raw, ('audit',), exact_strings=METADATA)
static_raw = adapter.private_module(core.static_raw.SCRIPT, '_AF7001_yawsigma0025_static_raw_reader')
ADAPTERS += adapter.retarget(static_raw, ('audit',), exact_strings=METADATA)
core_api = SimpleNamespace(**dict(core.__dict__, errors=errors, static_raw=static_raw))


def labels(function, clocks):
    result = {**old_review.strings(function, clocks), **METADATA}
    for key, value in list(result.items()):
        result[key] = value.replace('yawcost200_unchanged', 'AF7001_yawsigma0025')
    return result


base = adapter.private_module(core.SCRIPT, '_AF7001_yawsigma0025_original_full_scope')
base.SCRIPT, base.QUALIFICATION, base.OUTPUT, base.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
base.runtime, base.final_check, base.errors, base.static_raw = runtime, final_check, errors, static_raw
ADAPTERS += adapter.retarget(base, ('run',), integers={'*':{6001:7201,120020:144020}},
    exact_strings=labels(core.run, {6001:7201}))
known = adapter.private_module(known_template.SCRIPT, '_AF7001_yawsigma0025_original_known54')
known.SCRIPT, known.QUALIFICATION, known.OUTPUT, known.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
known.runtime, known._api, known.original = runtime, base, core_api
ADAPTERS += adapter.retarget(known, ('run',), exact_strings=METADATA)
retained = adapter.private_module(retained_template.SCRIPT, '_AF7001_yawsigma0025_original_retained24')
retained.SCRIPT, retained.QUALIFICATION, retained.OUTPUT, retained.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
retained.runtime, retained.final_check, retained._base, retained._api = runtime, final_check, base, known
retained.core, retained.long_raw = core_api, long_raw


def install_retained():
    tree = ast.parse(inspect.getsource(retained_template.run))
    before, changes = ast.dump(tree), []
    guards = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
        and ast.unparse(node.left) == "final['saved_configuration_differences']"]
    if len(guards) != 1 or ast.literal_eval(guards[0].comparators[0]) != ['/log_dir','/rewards/tracking_lin_vel']:
        raise ValueError('Requalify the exact original configuration guard')
    changes.append((guards[0], 'comparators', copy.deepcopy(guards[0].comparators)))
    guards[0].comparators = [ast.parse("['/scene/num_envs','/log_dir','/rewards/tracking_sigma_ang','/terrain/num_envs']", mode='eval').body]
    replacements = labels(retained_template.run, {6201:7001,6401:7201})
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (6201,6401,128020):
            changes.append((node,'value',node.value));node.value={6201:7001,6401:7201,128020:144020}[node.value]
        elif isinstance(node, ast.Constant) and isinstance(node.value,str) and node.value in replacements:
            changes.append((node,'value',node.value));node.value=replacements[node.value]
    modified=copy.deepcopy(tree)
    for node, field, value in reversed(changes):setattr(node,field,value)
    if ast.dump(tree)!=before:raise ValueError('Original numeric review operations changed')
    exec(compile(ast.fix_missing_locations(modified),str(retained_template.SCRIPT),'exec'),retained.__dict__)
    return dict(original_run_AST_restored_identical=True,
        declared_metadata_changes=len(changes), exact_configuration_differences=['/scene/num_envs','/log_dir','/rewards/tracking_sigma_ang','/terrain/num_envs'])


ADAPTERS.append(install_retained())
signed = adapter.private_module(signed_template.SCRIPT, '_AF7001_yawsigma0025_original_signed_height')
signed.SCRIPT, signed.QUALIFICATION, signed.OUTPUT, signed.AFTER_UNIT = SCRIPT, SIGNED_QUALIFICATION, EXTRA, read(LAUNCH)['unit']
signed.runtime = runtime
signed.SOURCE = old_signed.previous.OUTPUT / 'candidate7001'
signed_labels = {**old_signed.labels, **METADATA}
ADAPTERS += adapter.retarget(signed, ('run',),
    integers={'*':{6801:7001,7001:7201,140020:144020}},exact_strings=signed_labels,
    keyword_names={'source6801':'source7001','candidate7001':'candidate7201'})
small = adapter.private_module(small_template.SCRIPT, '_AF7001_yawsigma0025_original_small_yaw120')
small.SCRIPT, small.QUALIFICATION, small.OUTPUT = SCRIPT, SMALL_QUALIFICATION, SMALL
small.runtime = runtime
ADAPTERS += adapter.retarget(small, ('run',),exact_strings=METADATA)


def qualify():
    decision=ready();launch=read(LAUNCH);training=Path(launch['output']);plan_path=training/'experiment_plan.json';plan=read(plan_path)
    if (any(p.exists() for p in (QUALIFICATION,SIGNED_QUALIFICATION,SMALL_QUALIFICATION,OUTPUT,STATIC,EXTRA,SMALL))
            or plan['native_num_envs']!=8192 or plan['fixed_standing_environments']!=2048
            or plan['profile']!=native_runtime.profile() or plan['next_configuration_changes']!=native_runtime.DELTA
            or plan['source_sha256']!=runtime.ANCHOR_SHA or plan['source_actual_ppo_updates']!=7001
            or plan['target_actual_ppo_updates']!=7201 or plan['expected_terminal_Adam_step']!=144020):
        raise ValueError('Bind the exact newly owned4096 count-only trial and fresh outputs')
    source=Path(plan['source_checkpoint']);fixture=old_context.SOURCE_REVIEW/'full_height_1.0/candidate'
    old=read(old_review.QUALIFICATION)
    reader,captured,evidence=core.combined.reader(old['full_height_cases']['1.0'],old['full_seeds'])
    rebuilt=reader(fixture,source);prior=read(fixture/'motion025_full_height_diagnosis.json')
    if len(captured)!=1 or rebuilt['trials']!=prior['trials'] or rebuilt['total_trials']!=51:
        raise ValueError('Original source metrics and gates must reproduce exactly')
    pinned=[SCRIPT,adapter.SCRIPT,core.SCRIPT,known_template.SCRIPT,retained_template.SCRIPT,
        original_errors.SCRIPT,original_long.SCRIPT,core.static_raw.SCRIPT,final_check.SCRIPT,native_runtime.SCRIPT,
        signed_template.SCRIPT,small_template.SCRIPT,old_signed.SCRIPT,
        old_review.QUALIFICATION,old_signed.QUALIFICATION,small_template.QUALIFICATION,
        LAUNCH,plan_path,source,native_runtime.DECISION,native_runtime.ADMISSION,fixture/'manifest.json',fixture/'trace.csv']
    pins={**old['source_pins'],**plan['source_files'],**read(native_runtime.DECISION)['source_pins'],
        **{str(p):sha(p) for p in pinned}}
    core.intact(pins)
    proof=copy.deepcopy(old)
    proof.update(qualified=True,observed_at=datetime.now().astimezone().isoformat(),
        training_launch=str(LAUNCH),training=str(training),after_unit=launch['unit'],output=str(OUTPUT),static_output=str(STATIC),
        target_actual_updates=7201,target_all17_Adam_step=144020,prepared_plan_path=str(plan_path),
        native_num_envs=8192,fixed_standing_environments=2048,complete_review_expected=str(OUTPUT/'complete_with_retained_interior_reviews.json'),
        count_metadata_adapters=ADAPTERS,original_real51_metrics_flags_reproduced=True,
        original_metrics_thresholds_absolute_floors_windows_and_failed_None_preserved=True,
        unchanged_numeric_reader_metadata_adapter=evidence,source_pins=pins,policy_accepted=False)
    core.save(QUALIFICATION,proof)
    for old_path,destination,updates in [
        (old_signed.QUALIFICATION,SIGNED_QUALIFICATION,dict(output=str(EXTRA),original_complete_required=str(OUTPUT/'complete_with_retained_interior_reviews.json'))),
        (small_template.QUALIFICATION,SMALL_QUALIFICATION,dict(output=str(SMALL),complete_original_review_required=str(OUTPUT/'complete_with_retained_interior_reviews.json'),signed_height_review_required=str(EXTRA/'complete_paired_fullspeed_height_transitions.json'))),
    ]:
        item=copy.deepcopy(read(old_path));item.update(updates)
        item.update(training=str(training),after_unit=launch['unit'],observed_at=proof['observed_at'],source_pins=pins,
            native_num_envs=8192,original_source_cache_and_cases_preserved=True,only_final_population_metadata_adapted=True)
        core.save(destination,item)
    return proof


def run():
    proof=read(QUALIFICATION);core.intact(proof['source_pins'])
    # The inherited readers wait for the owned training unit and verify the
    # complete200/4096 alignment, saved state and immutable source evidence.
    result=retained.run()
    extra=signed.run()
    supplement=small.run()
    core.intact(proof['source_pins'])
    combined=dict(completed=True,observed_at=datetime.now().astimezone().isoformat(),
        checkpoint=result['checkpoint'],checkpoint_sha256=result['checkpoint_sha256'],actual_updates=7201,
        original735_static810_original120_complete=result['all_original_and_retained_reviews_complete'],
        signed_fullspeed_height_complete=extra['completed'],small_yaw120_complete=supplement['completed'],
        original_complete=str(OUTPUT/'complete_with_retained_interior_reviews.json'),
        signed_complete=str(EXTRA/'complete_paired_fullspeed_height_transitions.json'),
        small_yaw_complete=str(SMALL/'complete_paired_small_yaw_sustained120.json'),
        numeric_gates_and_all_failures_preserved=True,source_pins=proof['source_pins'],
        can_select_or_extend_training=False,training_modified=False,policy_accepted=False)
    core.save(OUTPUT/'complete_all_scopes_AF7001_yawsigma0025.json',combined)
    return combined


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--qualify-only',action='store_true');args=parser.parse_args()
    result=qualify() if args.qualify_only else run()
    print(json.dumps({k:result[k] for k in ('qualified','completed','policy_accepted') if k in result}))
