"""Admission for one reviewed unchanged7001 to7201 ordinary PPO block."""
import json
from pathlib import Path
from types import SimpleNamespace

from joint_pd_continuation import ROOT, sha
import joint_pd_preserved6801_yawcost200_runtime as family

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
TRAINING = RESULTS / 'joint_pd_optimization_20261022_135230_239804_preserved6801_yawcost200_200'
SOURCE_REVIEW = RESULTS / 'joint_pd_preserved6801_yawcost200_review7001_20261022'
SOURCE_EXTRA = RESULTS / 'joint_pd_preserved6801_yawcost200_fullspeed_height_transitions_source6801_candidate7001_20261022'
DECISION = RESULTS / 'joint_pd_source7001_yawcost200_unchanged200_decision_20261022.json'
QUALIFICATION = RESULTS / 'joint_pd_source7001_yawcost200_unchanged_qualification_20261022'
OUTPUT = RESULTS / 'joint_pd_yawcost200_unchanged_review7201_20261022'
STATIC = RESULTS / 'joint_pd_static_retention_yawcost200_unchanged_20261022_source7201'
EXTRA = RESULTS / 'joint_pd_yawcost200_unchanged_fullspeed_height_source7001_candidate7201_20261022'
ANCHOR_SHA = 'af47077587044230d5814f5e0e76a633366260698471da6162f3411494e64f49'
DELTA = dict(rewards={}, commands={}, exploration={}, controller={}, physics={})


def read(path):
    return json.loads(Path(path).read_text())


def ready():
    decision = read(DECISION)
    final = read(decision['source_final_verification'])
    full = read(decision['source_full_review'])
    fixed = read(decision['source_all735_summary'])['current']
    static = read(decision['source_static_retention_report'])
    long = read(decision['source_original_sustained120'])
    extra = read(decision['source_fullspeed_height_transition_review'])
    modes = extra['results']['candidate7001']['modes']
    small = read(decision['source_small_yaw_sustained120_complete'])
    if (not small['completed'] or small['results']['candidate7001']['checkpoint_sha256'] != ANCHOR_SHA
            or small['results']['candidate7001']['survivors'] != 54
            or not small['source_candidate_initial_observations_bitwise_paired']):
        raise ValueError('Complete the exact selected source small-yaw120 review first')
    if (not decision['reviewed_unchanged_consolidation_selected']
            or decision['source_checkpoint_sha256'] != ANCHOR_SHA
            or decision['source_actual_updates'] != 7001 or decision['source_all17_Adam_step'] != 140020
            or decision['planned_actual_updates'] != 200 or decision['target_actual_updates'] != 7201
            or decision['next_configuration_changes'] != DELTA
            or not final['verified'] or final['actual_ppo_updates'] != 7001
            or final['all17_Adam_step'] != 140020 or final['checkpoint_sha256'] != ANCHOR_SHA
            or not full['all_original_and_retained_reviews_complete'] or full['actual_updates'] != 7001
            or fixed['checkpoint_sha256'] != ANCHOR_SHA or fixed['fixed_trials'] != 735
            or fixed['survivors'] != 735 or fixed['original_passed'] != 735
            or fixed['original_AND_literal10_passed'] != 645
            or not static['raw_integrity_verified'] or not static['static_retention_verified']
            or static['checkpoint_sha256'] != ANCHOR_SHA
            or not long['raw_integrity_verified'] or long['actual_updates'] != 7001
            or long['original_gates_and_nonzero_relative10_both_passed'] != 54
            or long['survivors'] != 54 or long['raw_OR_resets']
            or not extra['completed'] or len(modes) != 2
            or any(m['survivors'] != 27 or m['resets'] for m in modes)
            or not decision['all_failed_and_regressed_trials_preserved']
            or decision['relative_error_target'] != .1 or decision['acceptance_changed']
            or decision['optimizer_reset'] or decision['std_reset'] or decision['original_model200_reload']
            or decision['faster_curriculum_authorized']
            or any(sha(p) != digest for p, digest in decision['source_pins'].items())):
        raise ValueError('Require complete own7001 evidence and an unchanged exact200 selection')
    if decision.get('actual_LR_reset', False):
        raise ValueError('Keep the actual saved adaptive learning rate')
    return decision


def anchor():
    decision = ready()
    path = Path(decision['source_checkpoint'])
    saved, cfg = family.require_source(path)
    if sha(path) != ANCHOR_SHA:
        raise ValueError('Keep the completely reviewed7001 checkpoint')
    return path, saved, cfg


def api_runtime():
    # The native family/serialized profile remains unchanged. Only these
    # outer review/admission labels refer to the next7001 source block.
    return SimpleNamespace(**dict(family.__dict__, ANCHOR_SHA=ANCHOR_SHA,
        DECISION=DECISION, TRAINING=TRAINING, DELTA=DELTA, repair_ready=ready,
        anchor=anchor, previous=family))
