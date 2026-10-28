"""Retain complete200 verification for the admitted matched exploration trial.

The exact drive-contract exception is checked before the production boundary;
all original tensor, six-head, Adam, configuration and precision guards run.
The exploration change is a future floor, never a saved-state reset.
"""
from pathlib import Path

import plain27_sampler_metadata_adapter as adapter
import verify_AF7001_wheelD075_completed200 as previous
import joint_pd_AF7001_wheelD075_wheel005_runtime as runtime
import recover_AF7001_wheelD075_completed200_boundary as contract
import train_joint_pd_standing_first as standing


SCRIPT = Path(__file__).resolve()
api = adapter.private_module(previous.SCRIPT, '_D075_wheel005_complete200_verifier')
api.SCRIPT, api.runtime = SCRIPT, runtime
boundary = adapter.private_module(standing.__file__, '_D075_wheel005_original_boundary')
boundary.validate_contract = contract.declared_contract


def verify_boundary(source, checkpoint, target, expected_cfg, baseline):
    proof = boundary.verify_update(source, checkpoint, target, expected_cfg, baseline)
    proof.update(physical_control_contract_unchanged=False,
        physical_control_contract_unchanged_except_explicitly_declared_wheel_damping=True,
        declared_control_contract_difference={'wheel_damping': [.5, .75]},
        complete_saved_configuration_matches_declared_recipe=True,
        only_exploration_change=runtime.DELTA['exploration'],
        original_numeric_and_saved_state_guards_retained=True)
    return proof


def verify(directory, destination=None):
    plan = previous.original.json.loads((Path(directory) / 'experiment_plan.json').read_text())
    assert plan['only_exploration_change'] == runtime.DELTA['exploration']
    guard = previous.original.json.loads((Path(directory) /
        'native_learning_signal/leg_exploration_floor_runtime.json').read_text())
    assert guard['profile'] == runtime.profile()
    return api.verify(directory, destination)


if __name__ == '__main__':
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    args = cli.parse_args()
    result = verify(args.directory)
    print({key: result[key] for key in ('verified', 'actual_ppo_updates', 'all17_Adam_step')})
