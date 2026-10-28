"""CPU qualify exact state recovery and the prospective declared-drive finalizer."""
import argparse
from pathlib import Path

import yaml

import qualify_uniform_control_joint_pd_runtime_v3 as original
import train_uniform_control_joint_pd_block200_v4 as controller
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()


def qualify(recipe, output):
    result = original.qualify(recipe, output)
    runtime = original.runtime
    source, _, _ = runtime.baseline()
    changes = runtime.recipe()["control_changes"]
    assert set(changes) == {"wheel_damping", "robot.actuators.wheels.damping"}
    plan = dict(only_controller_change=changes, next_configuration_changes=dict(controller=changes),
                source_checkpoint=str(source), source_sha256=sha(source))
    validator, proof = controller.admission.declared_contract_validator(plan, runtime)
    fixture = output / "declared_controller_sidecar_fixture"
    params = fixture / "params"
    params.mkdir(parents=True)
    contract = yaml.safe_load((source.parent / "params/control_contract.yaml").read_text())
    assert contract["wheel_damping"] == changes["wheel_damping"][0]
    contract["wheel_damping"] = changes["wheel_damping"][1]
    (params / "control_contract.yaml").write_text(yaml.safe_dump(contract))
    (params / "precision_mode.yaml").write_text((source.parent / "params/precision_mode.yaml").read_text())
    examples = controller.admission.qualify_contract_examples(validator, source, fixture / "NO_TRAINED_WEIGHTS.pt", output)
    result.update(prospective_control_contract_admission=proof,
                  negative_control_contract_qualification=examples,
                  metadata_fixtures_contain_no_trained_weights=True,
                  native_PPO_code_changed=False, original11_and_all4_numeric_readers_changed=False,
                  new_PPO_updates=0)
    for path in (SCRIPT, controller.SCRIPT, controller.admission.SCRIPT,
                 SCRIPT.with_name("prepare_uniform_control_trial_v4.py")):
        result["source_pins"][str(path)] = sha(path)
    write_json(output / "qualification.json", result)
    return result


if __name__ == "__main__":
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.recipe.resolve(), args.output.resolve())
    print(json.dumps(dict(qualified=result["qualified"], results=result["results"],
                         prospective_declared_control_verified=True,
                         negative_controls=len(result["negative_control_contract_qualification"]),
                         new_PPO_updates=0)))
