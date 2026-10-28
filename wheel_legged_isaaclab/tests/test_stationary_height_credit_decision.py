import json
from pathlib import Path

import pytest

import joint_pd_stationary_stage as stage
import stationary_height_credit_decision as credit
from test_stationary_motion_curriculum import completed_report
from test_joint_pd_priority import trial


def cfg():
    return dict(rewards=dict(orientation=-20., standing_velocity=-100., yaw_rate_error_sq=-1.,
                             base_height_error_sq=-100., joint_target_leg_length_error_sq=0.))


def history(tmp_path, coefficients=(-250., -500.), motion=0.):
    result = []
    evidence = completed_report()
    for index, value in enumerate(coefficients):
        output = tmp_path/str(index)
        output.mkdir()
        (output/'experiment_plan.json').write_text(json.dumps(dict(training_motion_scale=motion,
            bounded_training_updates=100, source_sha256='source',
            only_reward_change={'base_height_error_sq': [-100., value]})))
        decision = stage.review(evidence, evidence)
        result.append(dict(output=str(output), **decision))
    return result


def test_two_failed_static_strengths_choose_existing_immediate_credit_without_another_ladder(tmp_path):
    rows = history(tmp_path)
    assert credit.next_reward(cfg(), completed_report(), rows, set()) == {
        'joint_target_leg_length_error_sq': -500.}


@pytest.mark.parametrize('coefficients,motion', [((-250.,), 0.), ((-250., -250.), 0.), ((-250., -500.), 1.)])
def test_one_strength_repetitions_or_moving_history_do_not_justify_new_experiment(tmp_path, coefficients, motion):
    rows = history(tmp_path, coefficients, motion)
    assert credit.next_reward(cfg(), completed_report(), rows, set()) == {'base_height_error_sq': -250.}


def test_height_credit_never_overrides_lost_standing_pose(tmp_path):
    rows = history(tmp_path)
    evidence = completed_report()
    trial(evidence, 'height_low')['metrics']['body_p95_deg'] = 4.
    assert credit.next_reward(cfg(), evidence, rows, set()) == {'orientation': -40.}


def test_meaningful_verified_height_progress_continues_same_objective(tmp_path):
    rows = history(tmp_path)
    before, after = completed_report(), completed_report(.04)
    rows[-1].update(stage.review(after, before))
    assert credit.next_reward(cfg(), after, rows, set()) == {}


def test_failed_immediate_credit_cannot_repeat_same_source_reward_seed(tmp_path):
    rows = history(tmp_path)
    used = {(('joint_target_leg_length_error_sq', -500.),)}
    with pytest.raises(RuntimeError, match='recheck code and native data'):
        credit.next_reward(cfg(), completed_report(), rows, used)


def test_stagnant_learned_immediate_credit_requires_diagnosis_instead_of_another_height_scale(tmp_path):
    rows = history(tmp_path)
    current = cfg()
    current['rewards']['joint_target_leg_length_error_sq'] = -500.
    with pytest.raises(RuntimeError, match='recheck code and native data'):
        credit.next_reward(current, completed_report(), rows, set())


def test_better_worst_height_cannot_hide_another_still_failed_height_regression():
    before, after = completed_report(), completed_report()
    for case in before['conditions']:
        if case['name'] == 'height_low':
            for row in case['trials']:
                row['metrics']['height_mae_m'] = .08
    for case in after['conditions']:
        if case['name'] == 'height_low':
            for row in case['trials']:
                row['metrics']['height_mae_m'] = .07
        elif case['name'] == 'height_high':
            for row in case['trials']:
                row['metrics']['height_mae_m'] = .065
    assert stage.review(after, before)['promote']
    decision = credit.review(after, before)
    assert not decision['promote']
    assert all(label.startswith('height_high/') for label in decision['stationary_height_regressions'])
    assert not decision['policy_accepted']


def test_improved_height_at_both_endpoints_keeps_original_stability_priority():
    before, after = completed_report(), completed_report(.04)
    assert credit.review(after, before)['promote']
    trial(after, 'stand')['metrics']['body_p95_deg'] = 3.
    assert not credit.review(after, before)['promote']


def test_reassessing_saved_sources_does_not_rewrite_old_promotions():
    before, bad, good = completed_report(), completed_report(), completed_report(.04)
    for case in bad['conditions']:
        if case['name'] == 'height_low':
            for row in case['trials']:
                row['metrics']['height_mae_m'] = .05
        elif case['name'] == 'height_high':
            for row in case['trials']:
                row['metrics']['height_mae_m'] = .065
    source, candidate, better = map(Path, ('source', 'candidate', 'better'))
    rows = [dict(source=str(source), output=str(candidate), promote=True),
            dict(source=str(candidate), output=str(better), promote=True)]
    values = {source: before, candidate: bad, better: good}
    chosen, decisions = credit.select_balanced_source(rows, values.__getitem__)
    assert chosen == better and not decisions[0]['promote'] and decisions[1]['promote']
    assert rows[0]['promote'] is True and rows[1]['source'] == str(candidate)
    assert decisions[1]['compared_with'] == str(source)


def test_old_minimax_promotion_does_not_start_unchanged_training_from_balanced_source(tmp_path):
    rows = history(tmp_path)
    rows[-1]['promote'] = True
    rows[-1]['source_stationary']['worst_stationary_height_relative_mae'] = .5
    rows[-1]['balanced_selection_review'] = dict(promote=False)
    assert credit.next_reward(cfg(), completed_report(), rows, set()) == {
        'joint_target_leg_length_error_sq': -500.}


def test_completed_coordinator_uses_real_last_output_schema_not_running_current_output():
    state = dict(state='block_review_complete', last_output='/runs/completed',
                 history=[dict(output='/runs/completed')])
    assert credit.completed_output(state) == Path('/runs/completed')


def test_older_completed_current_output_alias_is_consistent_with_history():
    state = dict(state='block_review_complete', current_output='/runs/completed',
                 history=[dict(output='/runs/completed')])
    assert credit.completed_output(state) == Path('/runs/completed')


@pytest.mark.parametrize('change', [dict(state='training_and_review'), dict(last_output='/runs/other'),
    dict(current_output='/runs/unfinished'), dict(history=[])])
def test_incomplete_or_conflicting_output_cannot_be_hidden_by_last_output(change):
    state = dict(state='block_review_complete', last_output='/runs/completed',
                 history=[dict(output='/runs/completed')])
    state.update(change)
    with pytest.raises(ValueError):
        credit.completed_output(state)
