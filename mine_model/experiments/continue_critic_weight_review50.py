"""Continue a calibrated trial without reapplying its anchor's metadata deltas.

The existing runtime still creates every native command and performs training
and replay. This entry point only distinguishes changes made earlier from the
effective changes in the next segment. It does not change PPO or task methods.
"""
import argparse
import copy
import json
import yaml
from pathlib import Path

import joint_pd_calibrated_critic_weight_runtime as runtime

common_train = runtime.profile_for_count.__globals__['train']
assert Path(common_train.__code__.co_filename).name == 'joint_pd_calibrated_critic_weight_runtime.py'



def run(request_path, *, qualify_only=False):
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
    empty_delta = {key: {} for key in ('physics', 'controller', 'rewards', 'commands', 'exploration', 'algorithm')}
    agent = yaml.safe_load((source.parent / 'params/agent.yaml').read_text())
    assert agent['algorithm']['class_name'] == 'PPO'
    assert agent['algorithm']['value_loss_coef'] == 5.
    assert agent['algorithm']['desired_kl'] == .01
    assert saved['optimizer_state_dict']['param_groups'][0]['lr'] > 0.
    if qualify_only:
        argv, final, _ = runtime.command(source, current, 'candidate_unchanged_CPU_boundary',
            50, current['commands'], Path(request['output']) / 'NO_STOP_REQUEST')
        assert final == request['target_actual_updates'] - 1
        assert argv.count('--allow_physics_contract_change') == 1
        return dict(qualified=True, source_checkpoint=str(source),
            source_checkpoint_sha256=runtime.sha(source),
            source_actual_updates=runtime.saved_updates(source),
            all17_Adam_step=runtime.saved_updates(source)*20,
            actual_saved_adaptive_LR=saved['optimizer_state_dict']['param_groups'][0]['lr'],
            saved_sixstd=saved['model_state_dict']['std'].tolist(),
            target_actual_updates=request['target_actual_updates'],
            effective_changes_relative_to_immediate_source=empty_delta,
            actual_value_loss_coef_retained=5., native_num_envs=8192,
            command=argv, command_executed=False, optimizer_executed=False,
            simulator_started=False, policy_accepted=False)
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
            value['source_PPO_algorithm_configuration'] = copy.deepcopy(agent['algorithm'])
            value['only_PPO_algorithm_change'] = {}
            value['source_standard_PPO_parameters_unchanged'] = True
            value['leg_exploration_floor_unchanged'] = True
            value['saved_leg_PD_action_maps_and_exploration_retained'] = True
            value['same_standard_value_loss_coef_as_immediate_source'] = 5.
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
        return common_train(request_path)
    finally:
        workflow.write_json = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--qualify-only', action='store_true')
    args = parser.parse_args()
    result = run(args.request.resolve(), qualify_only=args.qualify_only)
    if args.qualify_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
