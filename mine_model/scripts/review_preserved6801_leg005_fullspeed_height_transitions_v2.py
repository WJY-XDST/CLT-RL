"""Pair the exact cached6801 signed-height traces with the new leg0057001 candidate.

Reuse the completed source evidence unchanged; collect two new candidate
runs. The original nine trajectories, three seeds, ten phases, seven metrics,
and Gaussian survival stress criterion are retained.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from joint_pd_continuation import sha
from review_plain27_single_reward_block200 import rebound
import joint_pd_preserved6801_leg005_runtime as runtime
import qualify_preserved6801_leg005_transfer as qualifier
import review_preserved6801_leg005_block200_v2 as post
import review_targeted_turn030_fullspeed_height_transitions_20261021 as previous
import review_lin300_fullspeed_height_transitions as original

SCRIPT = Path(__file__).resolve()
QUALIFICATION = qualifier.OUTPUT / 'fullspeed_height_v2/qualification.json'
OUTPUT = runtime.RESULTS / 'joint_pd_preserved6801_leg005_v2_fullspeed_height_transitions_source6801_candidate7001_20261021'
AFTER_UNIT = 'clt-joint-pd-preserved6801-leg005-v2-original-and-retained-reviews-20261021.service'
SOURCE = previous.OUTPUT / 'candidate6801'
read, core = original.read, original.core


def qualify():
    if QUALIFICATION.exists() or OUTPUT.exists():
        raise ValueError('Preserve every published review and failure')
    runtime.repair_ready()
    source, _, cfg = runtime.anchor()
    launch_path = qualifier.OUTPUT / 'training_launch_v2.json'
    launch = read(launch_path)
    old = read(previous.QUALIFICATION)
    complete = read(previous.OUTPUT / 'complete_paired_fullspeed_height_transitions.json')
    cached = complete['results']['candidate6801']
    if (not old['qualified'] or old['command_plan'] != original.plan() or old['seeds'] != original.SEEDS
            or not complete['completed'] or cached['checkpoint_sha256'] != runtime.ANCHOR_SHA
            or launch['source_actual_updates'] != 6801 or launch['target_actual_updates'] != 7001
            or cfg['min_root_height'] != .22 or cfg['low_height_terminal_time_s'] != .15
            or any(m['survivors'] != 27 or m['resets'] for m in cached['modes'])):
        raise ValueError('Require the exact completed6801 source and original signed-height coverage')
    proof = copy.deepcopy(old)
    pins = dict(old['source_pins'])
    for mode in old['modes']:
        manifest = read(SOURCE / mode / 'manifest.json')
        if manifest['checkpoint_sha256'] != runtime.ANCHOR_SHA or manifest['seeds'] != original.SEEDS:
            raise ValueError('Cached source identity or seeds changed')
        for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv',
                     'policy_inputs.npz', 'motion_height_steps_diagnosis.json'):
            path = SOURCE / mode / name
            pins[str(path)] = sha(path)
    for p in (SCRIPT, previous.SCRIPT, original.SCRIPT, runtime.SCRIPT, qualifier.SCRIPT,
              post.SCRIPT, post.QUALIFICATION, previous.QUALIFICATION, launch_path,
              previous.OUTPUT / 'complete_paired_fullspeed_height_transitions.json',
              Path(cached['raw_report']), Path(cached['literal_report'])):
        pins[str(p)] = sha(p)
    proof.update(observed_at=datetime.now().astimezone().isoformat(), after_unit=AFTER_UNIT,
        training=launch['output'], output=str(OUTPUT),
        source_checkpoint=str(source), source_checkpoint_sha256=runtime.ANCHOR_SHA,
        source_actual_updates=6801, candidate_actual_updates=7001, candidate_all17_Adam_step=140020,
        cached_source_directory=str(SOURCE), cached_source_results=cached,
        new_native_runs=2, cached_source_native_runs_reused=2,
        original_complete_required=str(post.OUTPUT / 'complete_with_retained_interior_reviews.json'),
        source_pins=pins, policy_accepted=False, can_select_or_extend_training=False)
    core.intact(pins)
    QUALIFICATION.parent.mkdir()
    core.save(QUALIFICATION, proof)
    return proof


def run():
    proof = read(QUALIFICATION)
    core.intact(proof['source_pins'])
    core.wait_unit(AFTER_UNIT, QUALIFICATION.parent / 'wait_status.json')
    complete = read(proof['original_complete_required'])
    training = Path(proof['training'])
    final = read(training / 'final_checkpoint_verification_readonly_20261019.json')
    if (OUTPUT.exists() or not complete['all_original_and_retained_reviews_complete']
            or not final['verified'] or final['actual_ppo_updates'] != 7001
            or final['all17_Adam_step'] != 140020
            or complete['checkpoint_sha256'] != final['checkpoint_sha256']):
        raise ValueError('Finish every owned original review before this signed supplement')
    candidate = Path(final['checkpoint'])
    runtime.require_source(candidate)
    OUTPUT.mkdir()
    core.save(OUTPUT / 'plan.json', proof)
    entries = []
    native = rebound(core.native, runtime=runtime, QUALIFICATION=QUALIFICATION)
    for mode, sampled in [('deterministic', False), ('saved_Gaussian', True)]:
        directory = native(candidate, OUTPUT / 'candidate7001' / mode, 50.,
            proof['seeds'], proof['command_plan'], sampled)
        assessed = original.reader(proof['command_plan'], proof['seeds'])(directory, candidate, sampled)
        with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as a, np.load(
                SOURCE / mode / 'policy_inputs.npz', allow_pickle=False) as b:
            if not np.array_equal(a['observations'][0], b['observations'][0]):
                raise ValueError('Candidate/source initial policy observations are not bitwise paired')
        entries.append(dict(mode=mode, episodes=27, survivors=assessed['survivors'],
            resets=assessed['resets'], all_numeric_phase_gates_passed=assessed['all_numeric_phase_gates_passed']))
        core.intact(proof['source_pins'])
    raw_path = OUTPUT / 'candidate7001/all540_phases_raw_verified.json'
    raw = original.phase_raw.audit([OUTPUT / 'candidate7001' / mode for mode in proof['modes']], raw_path)
    raw.update(actual_ppo_updates=7001, actual_update_clock_verified_from_completed_checkpoint=True)
    raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    literal_path = OUTPUT / 'candidate7001/all540_phases_literal10.json'
    relative = original.literal.audit(raw_path, literal_path)
    core.intact(proof['source_pins'])
    result = dict(completed=True, observed_at=datetime.now().astimezone().isoformat(),
        results=dict(source6801=proof['cached_source_results'], candidate7001=dict(
            checkpoint=str(candidate), checkpoint_sha256=sha(candidate), modes=entries,
            raw_report=str(raw_path), literal_report=str(literal_path), original_phase_passed=raw['total_passed'],
            original_and_literal_phase_passed=relative['original_gates_and_nonzero_relative10_both_passed'])),
        candidate_initial_observations_bitwise_paired=True, original_scope_not_replaced=True,
        cached_source_native_runs_reused_unchanged=2, new_candidate_native_runs=2,
        sampled_numeric_results_descriptive_survival_stress_criterion_unchanged=True,
        training_modified=False, optimizer_executed=False, acceptance_changed=False,
        policy_accepted=False, can_select_or_extend_training=False)
    core.save(OUTPUT / 'complete_paired_fullspeed_height_transitions.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true')
    result = qualify() if parser.parse_args().qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'completed', 'policy_accepted') if k in result}))
