"""Keep the v3 ordinary PPO block and admit a declared drive change at finalization.

Only checkpoint-contract metadata validation changes. A versioned validator
accepts the immutable recipe's one coupled wheel damping change and retains
every original mode, effort, precision, tensor, configuration and update check.
The native PPO, all11 learning readers and all4 replay readers are unchanged.
"""
import argparse
from pathlib import Path
import types

import train_uniform_control_joint_pd_block200_v3 as original
import train_joint_pd_standing_first as boundary
import finalize_uniform_control_completed200_readonly as admission
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()


def run(pointer):
    pointer = Path(pointer).resolve()
    request = original.read(pointer)
    output = Path(request["planned_production_output"])
    previous_verify, previous_writer = boundary.verify_update, original.write_json

    def verify(source, checkpoint, target, expected_cfg, baseline):
        plan = original.read(output / "experiment_plan.json")
        assert Path(source).resolve() == Path(plan["source_checkpoint"]).resolve()
        assert sha(source) == plan["source_sha256"]
        assert target == plan["target_actual_ppo_updates"] == plan["source_actual_ppo_updates"] + 200
        validator, proof = admission.declared_contract_validator(plan, original.runtime)
        examples = admission.qualify_contract_examples(validator, source, checkpoint, output)
        qualified = types.FunctionType(previous_verify.__code__,
            dict(previous_verify.__globals__, validate_contract=validator),
            previous_verify.__name__, previous_verify.__defaults__, previous_verify.__closure__)
        result = qualified(source, checkpoint, target, expected_cfg, baseline)
        result.update(physical_control_contract_unchanged=False,
            only_explicitly_declared_global_drive_change=True,
            all_other_physical_contract_fields_unchanged=True,
            declared_control_contract_admission=proof)
        write_json(output / "prospective_declared_control_finalization_verified.json", dict(
            verified=True, admission=proof, negative_contract_qualification=examples,
            original_checkpoint_and_all17_Adam_update_validator_unchanged=True,
            original200_PPO_updates_not_repeated=True, actual_PPO_updates_added_here=0,
            tracking_metrics_or_thresholds_changed=False, policy_accepted=False))
        return result

    def write(path, value):
        if Path(path) == output / "experiment_plan.json":
            value = dict(value)
            # Retired descriptive banners do not describe a later recipe.
            for key in ("height3_orientation40_velocity025_unchanged",
                        "height3_orientation40_quality50_yaw025_unchanged",
                        "unchanged_recipe_continuation",
                        "existing_reward_command_controller_physics_and_reset_configurations_unchanged",
                        "existing_exploration_commands_controller_physics_and_reset_configurations_unchanged"):
                value.pop(key, None)
            value["prospective_declared_control_finalization_runtime"] = str(SCRIPT)
            value["original_native_PPO_and_all11_and_all4_numeric_readers_unchanged"] = True
            value["only_control_contract_metadata_admission_is_extended"] = True
        return previous_writer(path, value)

    boundary.verify_update, original.write_json = verify, write
    try:
        return original.run(pointer)
    finally:
        boundary.verify_update, original.write_json = previous_verify, previous_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.pointer)
    except BaseException as error:
        pointer = original.read(args.pointer)
        destination = Path(pointer["planned_production_output"])
        destination.mkdir(exist_ok=True)
        write_json(destination / "controller_inspection_required.json", dict(
            error=repr(error), checkpoint_files_preserved=True,
            training_not_automatically_restarted=True, goal_remains_active=True,
            policy_accepted=False))
        raise
