"""Preserve a completed ordinary-PPO prefix with unchanged physical control.

Reuse the complete prefix/NN/Adam/LR verification and real CPU runner load.
Only the contract validator and truthful command-change metadata differ from
the retained wheel-damping trial archiver. No PPO or simulation is started.
"""
import argparse
from pathlib import Path
import json

import joint_pd_uniform_command_runtime as runtime
import preserve_partial_uniform_control_trial as retained
from joint_pd_continuation import sha
from watch_joint_pd_scratch import validate_contract

SCRIPT = Path(__file__).resolve()


def preserve(output, recipe, destination):
    previous_validator, previous_writer = retained.declared_contract_validator, retained.write_json

    def validator(plan, selected_runtime):
        assert selected_runtime is runtime
        assert not plan['only_controller_change'] and not runtime.recipe()['control_changes']
        return validate_contract, dict(all_physical_control_contract_fields_unchanged=True,
                                       only_declared_global_command_change=runtime.recipe()['command_changes'])

    def write(path, value):
        if isinstance(value, dict) and value.get('partial_trial_not_completed200'):
            value.update(physical_control_contract_unchanged=True,
                         only_declared_global_wheel_damping_changed=False,
                         only_declared_global_command_change=runtime.recipe()['command_changes'])
            value['source_pins'][str(SCRIPT)] = sha(SCRIPT)
        return previous_writer(path, value)

    retained.declared_contract_validator, retained.write_json = validator, write
    try:
        return retained.preserve(output, recipe, destination)
    finally:
        retained.declared_contract_validator, retained.write_json = previous_validator, previous_writer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    result = preserve(args.output.resolve(), args.recipe.resolve(), args.archive.resolve())
    print(json.dumps({k: result[k] for k in ('verified', 'actual_ppo_updates',
          'completed_native_updates', 'all17_Adam_step', 'checkpoint_sha256')}))
