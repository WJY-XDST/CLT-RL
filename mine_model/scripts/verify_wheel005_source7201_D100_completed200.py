"""Original full200/state checks with the exact declared .75/1.0 contract.

The exception is installed before training; any unrelated control, saved
configuration, clock, model, Adam, precision or native-data change rejects.
"""
import argparse
from pathlib import Path

import yaml

import joint_pd_wheel005_source7201_D100_runtime as runtime
import plain27_sampler_metadata_adapter as adapter
import verify_AF7001_wheelD075_completed200 as original
import train_joint_pd_standing_first as standing
import watch_joint_pd_scratch as physical
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()
api = adapter.private_module(original.SCRIPT, '_source7201_D100_original_full200_verifier')
api.SCRIPT, api.runtime = SCRIPT, runtime
boundary = adapter.private_module(standing.__file__, '_source7201_D100_original_boundary')


def check_contract_pair(before, after):
    assert set(before) == set(after), 'Unexpected control contract key'
    delta = {key: [before[key], after[key]] for key in before
             if key != 'fresh_weights' and before[key] != after[key]}
    assert delta == {'wheel_damping': [.75, 1.]}, delta
    return delta


def declared_contract(run, reward_profile=None, *, fresh_weights=True, source_checkpoint=None):
    assert source_checkpoint is not None and fresh_weights is False
    before = yaml.safe_load((Path(source_checkpoint).parent / 'params/control_contract.yaml').read_text())
    after = yaml.safe_load((Path(run) / 'params/control_contract.yaml').read_text())
    check_contract_pair(before, after)
    return physical.validate_contract(run, reward_profile,
        fresh_weights=fresh_weights, source_checkpoint=None)


boundary.validate_contract = declared_contract


def declared_configuration_differences(plan, before, after):
    count = plan['native_num_envs']
    assert count in (8192, 4096) and plan['fixed_standing_environments'] == count // 4
    assert before['wheel_damping'] == before['robot']['actuators']['wheels']['damping'] == .75
    assert after['wheel_damping'] == after['robot']['actuators']['wheels']['damping'] == 1.
    assert plan['next_configuration_changes'] == runtime.DELTA
    assert before['rewards'] == after['rewards'] and before['commands'] == after['commands']
    assert plan['only_exploration_change'] == {}
    result = {'/wheel_damping', '/robot/actuators/wheels/damping'}
    for name in ('scene', 'terrain'):
        assert before[name]['num_envs'] == 8192 and after[name]['num_envs'] == count
        if count != 8192:
            result.add('/' + name + '/num_envs')
    return result


api.declared_configuration_differences = declared_configuration_differences


def verify_boundary(source, checkpoint, target, expected_cfg, baseline):
    proof = boundary.verify_update(source, checkpoint, target, expected_cfg, baseline)
    proof.update(physical_control_contract_unchanged=False,
        physical_control_contract_unchanged_except_explicitly_declared_wheel_damping=True,
        declared_control_contract_difference={'wheel_damping': [.75, 1.]},
        complete_saved_configuration_matches_declared_recipe=True,
        only_exploration_change={}, original_numeric_and_saved_state_guards_retained=True)
    return proof


def verify(directory, destination=None):
    plan = original.original.json.loads((Path(directory) / 'experiment_plan.json').read_text())
    assert plan['source_sha256'] == runtime.ANCHOR_SHA
    assert plan['source_actual_ppo_updates'] == 7201 and plan['target_actual_ppo_updates'] == 7401
    assert plan['only_exploration_change'] == {}
    guard = original.original.json.loads((Path(directory) /
        'native_learning_signal/leg_exploration_floor_runtime.json').read_text())
    assert guard['profile'] == runtime.profile()
    return api.verify(directory, destination)


def qualify_contract():
    source = runtime.baseline()[0]
    path = source.parent / 'params/control_contract.yaml'
    before = yaml.safe_load(path.read_text())
    after = dict(before, wheel_damping=1.)
    assert check_contract_pair(before, after) == {'wheel_damping': [.75, 1.]}
    rejected = []
    for name, fields in [('different_wheel_damping', dict(wheel_damping=1.1)),
                         ('changed_leg_effort', dict(leg_effort_limit=35.)),
                         ('extra_control_key', dict(unexpected=True))]:
        try:
            check_contract_pair(before, dict(after, **fields))
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Unrelated physical change escaped: ' + name)
    return dict(qualified=True, accepted_exact_difference={'wheel_damping': [.75, 1.]},
        rejected_negative_cases=rejected, original_boundary_function_reused=True,
        original_full200_numeric_model_17Adam_and9_alignment_guards_reused=True,
        source_pins={str(p): sha(p) for p in
            (SCRIPT, original.SCRIPT, Path(standing.__file__), Path(physical.__file__), path)},
        PPO_updates=0, simulator_started=False, policy_accepted=False)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', nargs='?', type=Path)
    cli.add_argument('--qualify-contract-only', action='store_true')
    args = cli.parse_args()
    if args.qualify_contract_only:
        print(original.original.json.dumps(qualify_contract()))
    else:
        assert args.directory is not None
        print(verify(args.directory))
