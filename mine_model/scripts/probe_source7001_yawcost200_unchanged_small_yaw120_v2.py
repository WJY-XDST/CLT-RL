"""Pair unchanged7201 with cached7001 small-yaw120 source.

Reuse the source6801 native run and the original numerical scorer. The same
held-out seeds give paired initial states; one new candidate run is needed.
The complete saved source recipe is retained. No original acceptance,
window, physical reset, action or observation is changed.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from joint_pd_continuation import sha
from review_plain27_single_reward_block200 import rebound
import source7001_yawcost200_unchanged_context as context
from types import SimpleNamespace
import qualify_retained_plain27_source as cpu_qualifier
runtime = context.api_runtime()
qualifier = SimpleNamespace(OUTPUT=context.QUALIFICATION, SCRIPT=cpu_qualifier.SCRIPT)
import review_source7001_yawcost200_unchanged200 as post
import review_source7001_yawcost200_unchanged_fullspeed_height as signed
import probe_preserved6801_leg005_small_yaw_sustained120 as template

SCRIPT = Path(__file__).resolve()
QUALIFICATION = qualifier.OUTPUT / 'small_yaw_sustained120_qualification.json'
OUTPUT = runtime.RESULTS / 'joint_pd_yawcost200_unchanged_small_yaw120_20261022'
AFTER_UNIT = 'clt-joint-pd-source7201-yawcost200-unchanged-signed-20261022.service'
CACHE = runtime.RESULTS / 'joint_pd_preserved6801_yawcost200_small_yaw_sustained120_20261022'
SOURCE_DIRECTORY = CACHE / 'candidate7001/candidate'
SEEDS = template.SEEDS
read, core = template.read, template.core


def qualify():
    if QUALIFICATION.exists() or OUTPUT.exists():
        raise ValueError('Keep each previous qualification and native output')
    decision = runtime.repair_ready()
    source, _, cfg = runtime.anchor()
    launch_path = qualifier.OUTPUT / 'training_launch.json'
    launch = read(launch_path)
    plan_path = Path(read(context.QUALIFICATION / 'full_original_block_entry_preflight_v2.json')['prepared_plan_path'])
    plan = read(plan_path)
    cache_complete = read(CACHE / 'complete_paired_small_yaw_sustained120.json')
    cached = read(CACHE / 'candidate7001/all54_raw_and_windows_verified.json')
    manifest = read(SOURCE_DIRECTORY / 'manifest.json')
    finished = read(SOURCE_DIRECTORY / 'completed.json')
    cases = template.cases_for(cfg)
    if (launch['source_checkpoint_sha256'] != runtime.ANCHOR_SHA
            or launch['source_actual_updates'] != 7001 or launch['target_actual_updates'] != 7201
            or launch['planned_actual_updates'] != 200 or launch['target_all17_Adam_step'] != 144020
            or plan['next_configuration_changes'] != runtime.DELTA
            or not cache_complete['completed'] or cached['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or manifest['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or manifest['cases'] != cases or manifest['seeds'] != SEEDS
            or manifest['seconds_per_case'] != 120. or not finished['completed']
            or finished['samples'] != 324000 or not cached['original98_overlapping_windows_preserved']
            or cached['native_wheel_reference_reproduction_max_error_rad_s'] > 1.e-7
            or cached['optimizer_executed'] or cached['training_modified']):
        raise ValueError('Require the exact completed cached source and selected reward-only200 trial')
    pins = {**plan['source_files'], **read(post.QUALIFICATION)['source_pins']}
    for p in (SCRIPT, template.SCRIPT, runtime.SCRIPT, runtime.previous.SCRIPT, qualifier.SCRIPT,
            post.SCRIPT, post.QUALIFICATION, signed.SCRIPT, signed.QUALIFICATION,
            CACHE / 'complete_paired_small_yaw_sustained120.json',
            CACHE / 'candidate7001/all54_raw_and_windows_verified.json',
            launch_path, plan_path, source, Path(decision['source_original_sustained120'])):
        pins[str(p)] = sha(p)
    for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz'):
        pins[str(SOURCE_DIRECTORY / name)] = sha(SOURCE_DIRECTORY / name)
    proof = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        after_unit=AFTER_UNIT, training=launch['output'], output=str(OUTPUT),
        source_checkpoint=str(source), source_checkpoint_sha256=runtime.ANCHOR_SHA,
        source_actual_updates=7001, target_actual_updates=7201, target_all17_Adam_step=144020,
        cases=cases, seeds=SEEDS, seconds_per_case=120., stable_seconds=100.,
        original_windows_per_trial=98, episodes_per_policy=54, total_paired_episodes=108,
        cached_source_native_directory=str(SOURCE_DIRECTORY), cached_source_runs_reused=1,
        new_candidate_native_runs=1, original_heldout_source_seeds_reused_for_paired_comparison=True,
        no_new_fresh_seed_claim_for_cached_source=True,
        candidate_expected_reward_difference=runtime.DELTA['rewards'],
        source_reference_resolved_env=str(SOURCE_DIRECTORY / 'resolved_env.json'),
        complete_original_review_required=str(post.OUTPUT / 'complete_with_retained_interior_reviews.json'),
        signed_height_review_required=str(signed.OUTPUT / 'complete_paired_fullspeed_height_transitions.json'),
        original735_static810_120s_and_signed_reviews_not_replaced=True,
        original_numeric_score_function_unmodified=True, diagnostic_prefix_ends_s=[20., 60., 120.],
        source_pins=pins, training_modified=False, optimizer_executed=False, simulator_started=False,
        acceptance_changed=False, policy_accepted=False, can_select_or_extend_training=False)
    core.intact(pins)
    core.save(QUALIFICATION, proof)
    return proof


def run():
    proof = read(QUALIFICATION)
    core.intact(proof['source_pins'])
    complete = read(proof['complete_original_review_required'])
    extra = read(proof['signed_height_review_required'])
    final = read(Path(proof['training']) / 'final_checkpoint_verification_readonly_20261019.json')
    if (OUTPUT.exists() or not complete['all_original_and_retained_reviews_complete']
            or not extra['completed'] or not final['verified'] or final['actual_ppo_updates'] != 7201
            or final['all17_Adam_step'] != 144020 or complete['checkpoint_sha256'] != final['checkpoint_sha256']):
        raise ValueError('Complete all original candidate reviews before the small-yaw supplement')
    candidate = Path(final['checkpoint'])
    runtime.require_source(candidate)
    OUTPUT.mkdir()
    core.save(OUTPUT / 'plan.json', proof)
    reference = read(proof['source_reference_resolved_env'])
    source_report, initial = template.assess(SOURCE_DIRECTORY, Path(proof['source_checkpoint']), reference, proof)
    if source_report != read(CACHE / 'candidate7001/all54_raw_and_windows_verified.json'):
        raise ValueError('The original reader no longer reproduces the complete cached source report')
    (OUTPUT / 'source7001').mkdir()
    core.save(OUTPUT / 'source7001/all54_raw_and_windows_verified.json', source_report)
    native = rebound(core.native, runtime=runtime, QUALIFICATION=QUALIFICATION)
    directory = native(candidate, OUTPUT / 'candidate7201/candidate', 120., SEEDS, proof['cases'])
    expected = copy.deepcopy(reference)
    expected['rewards'][runtime.REWARD_KEY] = runtime.REWARD_VALUE
    report, current_initial = template.assess(directory, candidate, expected, proof)
    if not np.array_equal(initial, current_initial):
        raise ValueError('Source/candidate initial observations differ')
    core.save(OUTPUT / 'candidate7201/all54_raw_and_windows_verified.json', report)
    core.intact(proof['source_pins'])
    fields = ('checkpoint', 'checkpoint_sha256', 'total_trials', 'survivors',
        'passed', 'original_AND_literal10_passed', 'raw_OR_resets')
    result = dict(completed=True, observed_at=datetime.now().astimezone().isoformat(),
        results={label: {k: value[k] for k in fields} for label, value in
            (('source7001', source_report), ('candidate7201', report))},
        source_candidate_initial_observations_bitwise_paired=True,
        original735_static810_original120_and_signed_reviews_not_replaced=True,
        original98_windows_all7_metrics_limits_and_failed_None_preserved=True,
        cached_source_numeric_report_reproduced_exactly=True, cached_source_native_runs_reused_unchanged=1,
        new_candidate_native_runs=1, complete_saved_recipe_unchanged=True,
        source_pins=proof['source_pins'], training_modified=False, optimizer_executed=False,
        acceptance_changed=False, can_select_or_extend_training=False, policy_accepted=False)
    core.save(OUTPUT / 'complete_paired_small_yaw_sustained120.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true')
    result = qualify() if parser.parse_args().qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'completed', 'policy_accepted') if k in result}))
