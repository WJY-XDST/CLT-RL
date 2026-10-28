"""Feature trials must change representation only and never repeat stagnation."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'mine_model/scripts'))
import continue_joint_pd_height_features as launch


def command():
    return ['run_python.sh', str(launch.observed.ENTRY), '--observation-output', 'output/native_learning_signal',
            '--', '--checkpoint_path', 'retained.pt', '--target_iteration', '526',
            'agent.algorithm.entropy_coef=0.001', 'agent.policy.class_name=ActorCritic',
            'env.commands.training_motion_scale=0.0']


def test_representation_command_preserves_checkpoint_clock_and_all_other_arguments():
    before = command()
    after = launch.feature_command(before)
    assert before == command()
    changes = [(a, b) for a, b in zip(before, after) if a != b]
    assert changes == [(str(launch.observed.ENTRY), str(launch.ENTRY)),
                       ('agent.policy.class_name=ActorCritic',
                        'agent.policy.class_name=JointPdHeightFeatureActorCritic')]


@pytest.mark.parametrize('flag', ['--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint'])
def test_feature_trial_refuses_resets(flag):
    with pytest.raises(ValueError, match='Never reset'):
        launch.feature_command([*command(), flag])


@pytest.mark.parametrize('extra', ['agent.policy.class_name=ActorCritic', str(launch.observed.ENTRY)])
def test_feature_command_refuses_ambiguous_duplicate_selection(extra):
    with pytest.raises(ValueError):
        launch.feature_command([*command(), extra])


def history(tmp_path, *, feature, promote=False, before=.32, after=.32, strict=False):
    output = tmp_path / ('feature' if feature else 'legacy')
    output.mkdir()
    (output / 'experiment_plan.json').write_text(json.dumps(
        {'height_feature_profile': launch.PROFILE} if feature else {}))
    return [dict(output=str(output), promote=promote,
        source_stationary={'worst_stationary_height_relative_mae': before},
        candidate_stationary={'stationary_all_original_metrics_passed': strict,
            'stationary_pose_passed': True, 'worst_stationary_height_relative_mae': after})]


def test_legacy_cost_attempt_is_not_an_identical_representation_trial(tmp_path):
    used = {(('joint_target_leg_length_error_sq', -500.),)}
    delta = launch.next_reward({'rewards': {'joint_target_leg_length_error_sq': 0.}}, {},
                              history(tmp_path, feature=False), used)
    assert delta == {'joint_target_leg_length_error_sq': -500.}
    assert not used


def test_rejected_feature_trial_requires_diagnosis_before_another_trial(tmp_path):
    with pytest.raises(RuntimeError, match='inspect full native and replay'):
        launch.next_reward({}, {}, history(tmp_path, feature=True, promote=False), set())


def test_tiny_worst_height_gain_cannot_trigger_blind_continuation(tmp_path):
    with pytest.raises(RuntimeError, match='stagnant'):
        launch.next_reward({}, {}, history(tmp_path, feature=True, promote=True, after=.3199), set())


def test_substantial_reviewed_gain_continues_without_more_reward_changes(tmp_path):
    assert launch.next_reward({}, {}, history(tmp_path, feature=True, promote=True, after=.30), set()) == {}


def test_strict_static_pass_allows_independent_reverification(tmp_path):
    assert launch.next_reward({}, {}, history(tmp_path, feature=True, promote=True, after=.05, strict=True), set()) == {}
