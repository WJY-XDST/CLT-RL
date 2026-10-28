import pytest

from stationary_entropy_zero import entropy_zero_command, canonical_command


def test_entropy_zero_keeps_checkpoint_std_optimizer_and_all_other_arguments():
    original = ['run_python.sh', 'observed.py', '--checkpoint_path', '/saved/model.pt',
        '--min_exploration_std', '.01', '--num_envs', '8192', 'agent.algorithm.lam=0.95',
        'agent.algorithm.entropy_coef=0.001', 'env.rewards.joint_target_leg_length_error_sq=-500.0']
    result = entropy_zero_command(original)
    assert result != original and original[-2] == 'agent.algorithm.entropy_coef=0.001'
    assert [i for i, (a, b) in enumerate(zip(original, result)) if a != b] == [9]
    assert result[9] == 'agent.algorithm.entropy_coef=0.0'


@pytest.mark.parametrize('fields', [[], ['agent.algorithm.entropy_coef=0.003'],
    ['agent.algorithm.entropy_coef=0.001', 'agent.algorithm.entropy_coef=0.001']])
def test_missing_duplicate_or_unmatched_entropy_cannot_be_silently_changed(fields):
    with pytest.raises(ValueError):
        entropy_zero_command(['python', *fields])


@pytest.mark.parametrize('flag', ['--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint'])
def test_matching_recipe_cannot_reset_any_preserved_state(flag):
    with pytest.raises(ValueError):
        entropy_zero_command(['python', 'agent.algorithm.entropy_coef=0.001', flag])


def test_only_run_identity_paths_are_ignored_in_matched_command_comparison():
    a = ['python', '--run_name', 'control', '--observation-output', '/a', '--stop_file', '/a/STOP',
         '--checkpoint_path', '/saved/model.pt', 'agent.algorithm.lam=0.95']
    b = ['python', '--run_name', 'trial', '--observation-output', '/b', '--stop_file', '/b/STOP',
         '--checkpoint_path', '/saved/model.pt', 'agent.algorithm.lam=0.95']
    assert canonical_command(a) == canonical_command(b)
    b[-1] = 'agent.algorithm.lam=1.0'
    assert canonical_command(a) != canonical_command(b)
