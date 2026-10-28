"""Original fixed-input action/FK audit with the exact D075/D100 contract."""
import argparse
import copy
import json
from pathlib import Path

import compare_AF7001_wheelD075_wheel005_recorded_joint_targets as original
import joint_pd_wheel005_source7201_D100_runtime as runtime
import plain27_sampler_metadata_adapter as adapter


SCRIPT = Path(__file__).resolve()
api = adapter.private_module(original.SCRIPT, '_source7201_D100_original_fixed_state_reader')
api.SCRIPT, api.runtime = SCRIPT, runtime


def expected_controller_and_population_difference(configs, plan):
    before, after = configs
    assert plan['source_sha256'] == runtime.ANCHOR_SHA
    assert plan['source_actual_ppo_updates'] == 7201
    assert plan['next_configuration_changes'] == runtime.DELTA
    assert not plan['only_reward_change'] and not plan['only_command_change']
    assert before['scene']['num_envs'] == before['terrain']['num_envs'] == 8192
    assert after['scene']['num_envs'] == after['terrain']['num_envs'] == 8192
    assert before['wheel_damping'] == before['robot']['actuators']['wheels']['damping'] == .75
    assert after['wheel_damping'] == after['robot']['actuators']['wheels']['damping'] == 1.
    expected = runtime.expected_configuration(before, 8192)
    expected['log_dir'] = after['log_dir']
    assert after == expected, 'No unrelated physical, reward, observation or motor field may change'
    return {'/log_dir', '/wheel_damping', '/robot/actuators/wheels/damping'}


api.expected_controller_and_population_difference = expected_controller_and_population_difference
METADATA = adapter.retarget(api, ('compare',), integers={'*': {4096: 8192}},
    strings={'wheelD075_fixed_source_comparison_prefix_': 'wheelD100_fixed_source_comparison_prefix_'})


def qualify(plan):
    before = runtime.baseline()[2]
    after = runtime.expected_configuration(before, 8192)
    after['log_dir'] = 'CPU_CONTRACT_FIXTURE'
    assert expected_controller_and_population_difference((before, after), plan) == {
        '/log_dir', '/wheel_damping', '/robot/actuators/wheels/damping'}
    rejected = []
    for name, change in [('unrelated_leg_effort', {'leg_effort_limit': 35.}),
                         ('wrong_drive', {'wheel_damping': 1.1}),
                         ('extra_configuration_key', {'unexpected': True})]:
        invalid = copy.deepcopy(after)
        invalid.update(change)
        try:
            expected_controller_and_population_difference((before, invalid), plan)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Unrelated change escaped: ' + name)
    return dict(qualified=True, rejected_negative_cases=rejected, metadata=METADATA,
        original_all17_Adam_RNG_input_identity_mapping_and_FK_reader_reused=True,
        actual_source_population=8192, actual_candidate_population=8192,
        PPO_updates=0, simulator_started=False, policy_accepted=False)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    path, report = api.compare(args.directory, args.through)
    print(json.dumps(dict(report=str(path),
        source_actor_parity=report['source_actor_recorded_mean_parity_max_error'],
        source_FK_parity=report['source_recorded_FK_length_parity_max_error_m'],
        training_modified=False, closed_loop_acceptance=False)))
