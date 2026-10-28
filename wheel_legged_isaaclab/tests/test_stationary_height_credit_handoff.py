import copy

import pytest

from handoff_stationary_height_credit import EXHAUSTION, validate_completed_block


def completed():
    state = dict(state='block_review_complete')
    status = dict(training_active=False, state='agent_review_required', actual_ppo_updates=526,
                  checkpoint_sha256='checkpoint')
    plan = dict(source_actual_ppo_updates=426, target_actual_ppo_updates=526, training_motion_scale=0.)
    proof = dict(actual_ppo_updates=526, optimizer_states=17, optimizer_step=10520,
        all_tensors_finite=True, actor_rows_updated=[True]*6, shared_actor_updated=True,
        critic_updated=True, checkpoint_sha256='checkpoint')
    native = dict(checkpoint_sha256='checkpoint', actual_observed_updates=100, optimizer_minibatches=2000,
        actual_ppo_updates=526, PPO_updates_repeated=0, all_probe_RNG_unchanged=True,
        original_control_and_PPO_loss_unchanged=True,
        all_environment_alignment_max_errors={str(k): 0. for k in range(9)})
    return [state, status, plan, proof, native, EXHAUSTION]


def test_complete_diagnosis_retains_native_model_and_optimizer_without_repeating_updates():
    validate_completed_block(*completed())


@pytest.mark.parametrize('index,key,value', [
    (0, 'state', 'training_and_review'), (1, 'training_active', True),
    (1, 'actual_ppo_updates', 525), (1, 'checkpoint_sha256', 'changed'),
    (2, 'source_actual_ppo_updates', 425), (2, 'training_motion_scale', .125),
    (3, 'optimizer_states', 16), (3, 'optimizer_step', 10500),
    (3, 'all_tensors_finite', False), (3, 'actor_rows_updated', [True]*5),
    (3, 'actor_rows_updated', [True]*5+[False]), (3, 'critic_updated', False),
    (4, 'actual_observed_updates', 99), (4, 'optimizer_minibatches', 1980),
    (4, 'actual_ppo_updates', 525), (4, 'PPO_updates_repeated', 100),
    (4, 'all_probe_RNG_unchanged', False), (4, 'original_control_and_PPO_loss_unchanged', False),
    (4, 'all_environment_alignment_max_errors', {}),
    (4, 'all_environment_alignment_max_errors', {str(k): float(k == 4)*1.e-6 for k in range(9)}),
])
def test_partial_corrupt_moving_or_modified_learning_cannot_trigger_successor(index, key, value):
    values = copy.deepcopy(completed())
    values[index][key] = value
    with pytest.raises(ValueError):
        validate_completed_block(*values)


@pytest.mark.parametrize('journal', ['', 'Out of memory', 'KeyError: commands', 'explicit_stop_respected'])
def test_unrelated_failure_or_old_stop_text_is_never_a_height_diagnosis(journal):
    values = completed()
    values[-1] = journal
    with pytest.raises(ValueError, match='No verified completed-block'):
        validate_completed_block(*values)
