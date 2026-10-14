import json

import pytest

from joint_pd_optimization import prepared_reward_trial


def test_only_the_duplicate_worst_tracking_term_may_be_disabled(tmp_path):
    path=tmp_path/'trial.json'
    path.write_text(json.dumps({'rewards.tracking_goal_violation':0.}))
    assert prepared_reward_trial(path)=={'rewards.tracking_goal_violation':0.}


@pytest.mark.parametrize('value', [.1, -2.01, float('nan'), float('inf')])
def test_nonpositive_finite_bounded_admission_is_preserved(tmp_path,value):
    path=tmp_path/'trial.json'
    path.write_text(json.dumps({'rewards.tracking_goal_violation':value}))
    with pytest.raises(ValueError):
        prepared_reward_trial(path)


def test_reward_simplification_cannot_modify_gate_or_multiple_coefficients(tmp_path):
    path=tmp_path/'trial.json'
    for change in ({'relative_error':.2},
                   {'rewards.tracking_goal_violation':0.,'rewards.tracking_goal_component_violation':0.}):
        path.write_text(json.dumps(change))
        with pytest.raises(ValueError):
            prepared_reward_trial(path)
