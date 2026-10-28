"""Select an immediate, existing FK reward only after measured height stagnation.

This module changes training-stage experiment selection, never control,
physical parameters, PPO, checkpoint state, or the original acceptance gate.
"""
import json
from pathlib import Path

import joint_pd_stationary_stage as stage
from joint_pd_priority import STATIC_CASES, trials

KEY = 'joint_target_leg_length_error_sq'
VALUE = -500.


def completed_output(state):
    """Completed saves use last_output; running saves use current_output."""
    if state.get('state') != 'block_review_complete' or not state.get('history'):
        raise ValueError('Require a completed reviewed block')
    expected = Path(state['history'][-1]['output'])
    declared = state.get('last_output', state.get('current_output'))
    if declared is None or Path(declared) != expected:
        raise ValueError('Completed output differs from the latest reviewed history')
    if state.get('current_output') is not None and Path(state['current_output']) != expected:
        raise ValueError('Do not skip another current or unreviewed output')
    return expected


def review(after, before):
    """A better worst height must not hide a still-failed height regression."""
    decision = stage.review(after, before)
    height_regressions = [label for label in decision['full_scope_comparison']['regressions']['height']
                          if label.split('/')[0] in STATIC_CASES]
    decision.update(stationary_height_regressions=height_regressions,
        promote=bool(decision['promote'] and not height_regressions),
        original_per_condition_height_limits_unchanged=True,
        failed_height_case_cannot_be_averaged_away=True)
    return decision


def select_balanced_source(history, load_report):
    """Reassess preserved history without rewriting any historical decisions."""
    if not history:
        raise ValueError('Require a complete stationary review history')
    incumbent = Path(history[0]['source'])
    before = load_report(incumbent)
    if not stage.summary(before)['stationary_pose_passed']:
        raise ValueError('The preserved history baseline lost stationary posture')
    decisions = []
    for item in history:
        output = Path(item['output'])
        after = load_report(output)
        decision = review(after, before)
        decisions.append(dict(output=str(output), compared_with=str(incumbent), **decision))
        if decision['promote']:
            incumbent, before = output, after
    return incumbent, decisions


def completed_height_trials(history):
    """Use real independently reviewed, all-static 100-update experiments."""
    rows = []
    for item in history:
        output = Path(item['output'])
        plan = json.loads((output/'experiment_plan.json').read_text())
        if plan.get('training_motion_scale') != 0. or plan.get('bounded_training_updates') != 100:
            continue
        changes = plan['only_reward_change']
        if len(changes) == 1 and 'base_height_error_sq' in changes:
            decision = item.get('balanced_selection_review', item)
            rows.append(dict(output=str(output), coefficient=changes['base_height_error_sq'][1],
                source=plan['source_sha256'], promote=decision['promote'],
                original_controller_promote=item['promote'],
                before=item['source_stationary']['worst_stationary_height_relative_mae'],
                after=item['candidate_stationary']['worst_stationary_height_relative_mae']))
    return rows


def next_reward(cfg, evidence, history, used):
    s = stage.summary(evidence)
    if not s['stationary_pose_passed']:
        return stage.next_reward(cfg, evidence, history, used)
    static = [(case, trial) for (name, _), (case, trial) in trials(evidence).items()
              if name in STATIC_CASES]
    height_failed = any(trial['metrics']['height_mae_m'] >= trial['limits']['height_mae_m']-1.e-12
                        for _, trial in static)
    if not height_failed:
        return stage.next_reward(cfg, evidence, history, used)
    last = history[-1].get('balanced_selection_review', history[-1]) if history else None
    if last and last['promote']:
        before = last['source_stationary']['worst_stationary_height_relative_mae']
        if s['worst_stationary_height_relative_mae'] < .99*before:
            return {}
    actual_trials = completed_height_trials(history)
    stagnant = [item for item in actual_trials if not item['promote'] or item['after'] >= .99*item['before']]
    # Two independently replayed strengths are evidence against continuing an
    # unchanged scalar-cost ladder. They are not a total-update/time endpoint.
    diagnosed = len({item['coefficient'] for item in stagnant}) >= 2
    delta = {KEY: VALUE}
    if diagnosed and cfg['rewards'].get(KEY, 0.) == 0. and tuple(delta.items()) not in used:
        return delta
    if diagnosed:
        raise RuntimeError('Height remains stagnant after diagnosed actual-height and immediate FK-credit experiments; recheck code and native data before choosing another objective')
    return stage.next_reward(cfg, evidence, history, used)
