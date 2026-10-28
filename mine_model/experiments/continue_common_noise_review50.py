"""Continue a calibrated trial without reapplying its anchor's metadata deltas.

The existing runtime still creates every native command and performs training
and replay. This entry point only distinguishes changes made earlier from the
effective changes in the next segment. It does not change PPO or task methods.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path

import joint_pd_calibrated_common_noise_runtime as runtime

common_train = runtime.profile_for_count.__globals__['train']
assert Path(common_train.__code__.co_filename).name == 'joint_pd_calibrated_common_noise_runtime.py'


@contextmanager
def unique_inherited_physics_flag():
    """Let the calibration layer author one already-authorized CLI flag.

    Descendant checkpoints already carry this flag in continuation_command().
    The calibration layer appends it as well and rejects the duplicate. Strip
    only the inherited occurrence before that layer appends the same flag.
    """
    namespace = runtime.command.__globals__
    original = namespace['old_command']

    def command(*args, **kwargs):
        argv, final, saved = original(*args, **kwargs)
        assert argv.count('--allow_physics_contract_change') <= 1
        return [arg for arg in argv if arg != '--allow_physics_contract_change'], final, saved

    namespace['old_command'] = command
    try:
        yield
    finally:
        namespace['old_command'] = original


def run(request_path):
    import train_saved_joint_pd_review50 as workflow

    request = json.loads(request_path.read_text())
    runtime.configure(request['recipe'])
    source = Path(request['source_checkpoint'])
    saved, current = runtime.require_source(source)
    expected = runtime.expected_configuration(current, 8192)
    assert current == expected, 'This entry point only continues an unchanged calibrated trial'
    assert saved['infos'][runtime.KEY] == runtime.profile_for_count(8192)
    assert runtime.saved_updates(source) == request['source_actual_updates']
    assert request['target_actual_updates'] == request['source_actual_updates'] + 50
    floors = runtime.recipe()['exploration']
    assert all(saved['infos'][runtime.KEY][name] == value for name, value in floors.items())
    empty_delta = {key: {} for key in ('physics', 'controller', 'rewards', 'commands', 'exploration')}
    original = workflow.write_json

    def write(path, value):
        if Path(path).name == 'experiment_plan.json':
            value = copy.deepcopy(value)
            value['cumulative_trial_changes_from_retained_anchor'] = copy.deepcopy(
                value['next_configuration_changes'])
            value['next_configuration_changes'] = copy.deepcopy(empty_delta)
            for name in ('physics', 'controller', 'reward', 'command', 'exploration'):
                value['only_' + name + '_change'] = {}
            value['source_exploration'] = copy.deepcopy(floors)
            value['only_runtime_sampling_change'] = {}
            value['saved_leg_PD_action_maps_exploration_and_reward_structure_retained'] = True
            value['original_observation_and_action_maps_and_saved_controller_retained'] = True
            value['same_configuration_as_immediate_source'] = True
            value['continuation_segment_from_previous_trial'] = request['continuation_from_output']
            value['cumulative_trial_updates_at_segment_end'] = request['cumulative_trial_updates_at_segment_end']
            value['source_initial_distribution_17Adam_and_actualLR_preserved'] = True
            value['metadata_delta_scope'] = 'immediate source; anchor changes retained separately'
        elif Path(path).name == 'resume_verification.json':
            value = copy.deepcopy(value)
            value['calibrated_asset_and_full_height_unchanged'] = True
            value['physical_control_contract_unchanged'] = True
            value['declared_mass_gravity_and_common_wheel_gain_transition_verified'] = False
            value['declared_physics_changes'] = {}
            value['declared_control_changes'] = {}
            value['same_configuration_as_immediate_source'] = True
            value['previous_calibration_retained'] = True
        return original(path, value)

    workflow.write_json = write
    try:
        with unique_inherited_physics_flag():
            return common_train(request_path)
    finally:
        workflow.write_json = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    run(args.request.resolve())
