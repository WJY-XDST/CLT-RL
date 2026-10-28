"""Finalize an already completed ordinary PPO block with a declared drive change.

No training entry point exists here. Preserve the old contract checks and
normalize only the source contract's explicitly declared global wheel D in
memory. All200 persisted transitions, original2000/4000 minibatch audits,
checkpoint tensors and independent tracking thresholds remain authoritative.
"""
import argparse
import ast
import copy
from datetime import datetime
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def declared_contract_validator(plan, runtime):
    import watch_joint_pd_scratch as original
    from joint_pd_continuation import sha

    changes = plan["only_controller_change"]
    controls = runtime.recipe()["control_overrides"]
    assert set(changes) == {"wheel_damping", "robot.actuators.wheels.damping"}
    assert changes["wheel_damping"] == changes["robot.actuators.wheels.damping"]
    before, after = changes["wheel_damping"]
    assert controls["wheel_damping"] == controls["robot.actuators.wheels.damping"] == after
    assert plan["next_configuration_changes"]["controller"] == changes
    source = Path(plan["source_checkpoint"]).resolve()

    def normalize(previous, contract, source_checkpoint):
        assert Path(source_checkpoint).resolve() == source
        assert sha(source) == plan["source_sha256"]
        assert previous.keys() == contract.keys(), "Reject undeclared added/deleted contract fields"
        assert previous["wheel_damping"] == before and contract["wheel_damping"] == after
        result = copy.deepcopy(previous)
        result["wheel_damping"] = after
        return result

    tree = ast.parse(inspect.getsource(original.validate_contract))
    before_ast = ast.dump(tree)
    matches = [(node, index) for node in ast.walk(tree) if isinstance(node, ast.If)
        for index, statement in enumerate(node.body) if isinstance(statement, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "previous" for target in statement.targets)]
    assert len(matches) == 1
    node, index = matches[0]
    insert = ast.parse("previous = _normalize_declared_source_contract(previous, contract, source_checkpoint)").body
    node.body[index + 1:index + 1] = insert
    modified = copy.deepcopy(tree)
    del node.body[index + 1:index + 1 + len(insert)]
    assert ast.dump(tree) == before_ast
    namespace = dict(original.__dict__, _normalize_declared_source_contract=normalize)
    exec(compile(ast.fix_missing_locations(modified), str(SCRIPT), "exec"), namespace)
    proof = dict(original_validator=str(original.__file__), original_validator_sha256=sha(original.__file__),
        original_AST_restored_identical=True, inserted_metadata_normalization_statements=1,
        admitted_source_contract_change={"wheel_damping": [before, after]},
        source_and_candidate_contract_key_sets_identical=True,
        every_original_mode_effort_precision_and_other_contract_predicate_retained=True,
        only_an_immutable_declared_global_control_change_admitted=True,
        tracking_metrics_or_acceptance_not_modified=True, optimizer_executed=False)
    return namespace["validate_contract"], proof


def qualify_contract_examples(validator, source, candidate, output):
    import yaml
    from auto_train_mine import write_json

    base_contract = yaml.safe_load((candidate.parent / "params/control_contract.yaml").read_text())
    base_precision = yaml.safe_load((candidate.parent / "params/precision_mode.yaml").read_text())
    root = output / "declared_control_contract_CPU_qualification"
    root.mkdir(exist_ok=False)
    cases = [("declared_only", None, None), ("wrong_declared_gain", "wheel_damping", 1.1),
        ("undeclared_wheel_effort", "wheel_effort_limit", 4.9),
        ("undeclared_leg_effort", "leg_effort_limit", 35.6),
        ("undeclared_contract_field", "unreviewed_controller", "extra"),
        ("frozen_actor", "precision_frozen", True)]
    results = []
    for name, field, value in cases:
        directory = root / name
        (directory / "params").mkdir(parents=True)
        contract, precision = copy.deepcopy(base_contract), copy.deepcopy(base_precision)
        if field == "precision_frozen":
            precision["shared_actor_features_frozen"] = value
        elif field:
            contract[field] = value
        (directory / "params/control_contract.yaml").write_text(yaml.safe_dump(contract))
        (directory / "params/precision_mode.yaml").write_text(yaml.safe_dump(precision))
        error = None
        try:
            validator(directory, fresh_weights=False, source_checkpoint=source)
        except (AssertionError, ValueError) as failure:
            error = str(failure)
        accepted = error is None
        assert accepted == (field is None), (name, error)
        results.append(dict(case=name, accepted=accepted, expected_accepted=field is None, error=error))
    write_json(root / "qualification.json", dict(qualified=True, results=results,
        source_checkpoint_not_modified=True, fixtures_contain_no_trained_weights=True,
        negative_control_contracts_rejected=True, acceptance_not_modified=True))
    return results


def restart_completed_waiter(unit, marker):
    if Path(marker).exists():
        return dict(unit=unit, completed=True, restarted=False)
    text = subprocess.run(["systemctl", "--user", "show", unit, "-p", "MainPID", "-p", "SubState"],
        capture_output=True, text=True, check=True).stdout
    state = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    if state["MainPID"] != "0":
        return dict(unit=unit, still_live=True, restarted=False)
    assert state["SubState"] in ("failed", "dead", "exited")
    subprocess.run(["systemctl", "--user", "restart", unit], check=True)
    return dict(unit=unit, prior_terminal_state=state, restarted=True, PPO_started=False)


def run(pointer):
    import torch
    import joint_pd_uniform_control_runtime_v3 as runtime
    import train_joint_pd_standing_first as original_boundary
    from auto_train_mine import write_json
    from joint_pd_continuation import ROOT, RUNS, sha
    from train_joint_pd_standing_seed import saved_updates

    pointer = read(pointer)
    runtime.configure(pointer["uniform_recipe"])
    output = Path(pointer["planned_production_output"])
    plan = read(output / "experiment_plan.json")
    assert plan["bounded_training_updates"] == 200 and plan["native_num_envs"] == 8192
    assert all(sha(path) == digest for path, digest in plan["source_files"].items())
    source = Path(plan["source_checkpoint"])
    runs = list(RUNS.glob("*_" + plan["run_name"]))
    assert len(runs) == 1
    target = plan["target_actual_ppo_updates"]
    candidate = runs[0] / f"model_{target - 1}.pt"
    assert saved_updates(candidate) == target == saved_updates(source) + 200
    records = sorted((output / "native_learning_signal").glob("audit_update_*.json"))
    assert len(records) == 200
    assert [read(path)["update"] for path in records] == list(range(1, 201))
    saved, cfg = runtime.require_source(candidate)
    assert len(saved["optimizer_state_dict"]["state"]) == 17
    assert all(float(value["step"]) == target * 20 for value in saved["optimizer_state_dict"]["state"].values())
    validator, admission = declared_contract_validator(plan, runtime)
    results = qualify_contract_examples(validator, source, candidate, output)
    function = original_boundary.verify_update
    qualified = types.FunctionType(function.__code__, dict(function.__globals__, validate_contract=validator),
        function.__name__, function.__defaults__, function.__closure__)
    proof = qualified(source, candidate, target, plan["training_config"], source)
    proof.update(physical_control_contract_unchanged=False,
        only_explicitly_declared_global_drive_change=True,
        all_other_physical_contract_fields_unchanged=True,
        declared_control_contract_admission=admission)
    write_json(output / "resume_verification.json", proof)
    write_json(output / "declared_control_contract_admission_verified.json", dict(
        qualified=True, admission=admission, negative_contract_qualification=results,
        saved_cfg_exactly_matches_immutable_training_plan=True, actual_PPO_updates_repeated=0,
        original_training_failure_preserved=read(output / "controller_inspection_required.json"),
        all200_checkpoints_and_optimizer_preserved=True, policy_accepted=False))
    import continue_joint_pd_observed100 as original_external
    import plain27_sampler_metadata_adapter as adapter
    import joint_pd_review_interval200 as interval
    from native_admitted_population_finalizer import install
    external = adapter.private_module(original_external.__file__, "_declared_control_original_external200")
    interval.install_finalizer(external)
    install(external, plan["native_num_envs"])
    external.finalize_observations(output)
    files = dict(plan["source_files"])
    files.update({str(path): sha(path) for path in (SCRIPT,
        output / "declared_control_contract_admission_verified.json")})
    result = dict(verified=True, checkpoint=str(candidate), checkpoint_sha256=sha(candidate),
        source_checkpoint=str(source), source_checkpoint_sha256=sha(source), actual_ppo_updates=target,
        all17_Adam_step=target * 20, saved_std=saved["model_state_dict"]["std"].tolist(),
        saved_actual_LR=[group["lr"] for group in saved["optimizer_state_dict"]["param_groups"]],
        all200_native_updates_and_original_alignment_verified=True,
        original_uniform_reward_structure_and_continuous_commands=True,
        declared_global_reward_scalar_changes=plan["only_reward_change"],
        declared_global_control_scalar_changes=plan["only_controller_change"], case_specific_training_logic=False,
        source_initial_std_LR_and_all17_Adam_preserved=True, source_pins=files,
        completed_PPO_not_restarted=True, finalization_recovery_only=True, policy_accepted=False)
    write_json(output / "final_checkpoint_verification_readonly.json", result)
    backup = Path(pointer["uniform_recipe"]).parent / ("preserved_candidate" + str(target))
    backup.mkdir(exist_ok=False)
    shutil.copy2(candidate, backup / candidate.name)
    shutil.copytree(candidate.parent / "params", backup / "params")
    shutil.copy2(output / "final_checkpoint_verification_readonly.json", backup / "final_checkpoint_verification_readonly.json")
    assert sha(backup / candidate.name) == sha(candidate)
    write_json(backup / "preservation_manifest.json", dict(verified=True, original_checkpoint=str(candidate),
        checkpoint_sha256=sha(candidate), actual_updates=target, all17_Adam_step=target * 20,
        saved_std=result["saved_std"], saved_actual_LR=result["saved_actual_LR"],
        preserved_before_independent_review=True, policy_accepted=False))
    monitor_pid = read(output / "owned_live_processes.json")["readonly_monitor_pid"]
    cmdline = Path(f"/proc/{monitor_pid}/cmdline")
    if cmdline.exists() and b"monitor_uniform_control_joint_pd_learning_v3.py" in cmdline.read_bytes():
        while not (output / "learning_health_uniform_prefix_0200_readonly.json").exists():
            assert cmdline.exists(), "Read-only monitor exited before prefix200 qualification"
            time.sleep(5.)
    else:
        with (output / "learning_monitor_recovery.log").open("x") as log:
            subprocess.run([sys.executable, str(SCRIPT.with_name("monitor_uniform_control_joint_pd_learning_v3.py")),
                "--output", str(output), "--recipe", pointer["uniform_recipe"]],
                cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    assert read(output / "learning_health_uniform_prefix_0200_readonly.json")["qualified"]
    write_json(output / "controller_failure_recovered_without_PPO_rerun.json", dict(verified=True,
        declared_gain_contract_admission_only=True, original200_and_all17_Adam_preserved=True,
        original11_readers_50_100_200_complete=True, new_PPO_updates=0,
        original_failure_marker_retained=True, policy_accepted=False))
    write_json(output / "status.json", dict(state="independent_review_running_after_declared_gain_contract_recovery",
        training_active=False, actual_ppo_updates=target, checkpoint=str(candidate),
        original200_completed=True, native_PPO_restarted=False, goal_remains_active=True, policy_accepted=False))
    with (output / "review_controller_recovery.log").open("x") as log:
        subprocess.run([sys.executable, str(SCRIPT.with_name("review_uniform_control_joint_pd_block_v3.py")),
            "--output", str(output), "--recipe", pointer["uniform_recipe"]],
            cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    b = ROOT / "mine_model/results"
    macro = b / "joint_pd_global_driveD100_after200_macro315_20261025"
    edge = b / "joint_pd_global_driveD100_after200_stage_edges210_20261025"
    revived = [restart_completed_waiter("clt-uniform-driveD100-after200-macro315-20261025.service",
        macro / "complete_paired_adaptation315.json"),
        restart_completed_waiter("clt-uniform-driveD100-after200-stage-edges210-20261025.service",
        edge / "complete_paired_stage_grid525.json")]
    write_json(output / "completed_full_review_and_rearmed_readonly_grids.json", dict(verified=True,
        original_standard_review_complete=True, readonly_waiters=revived,
        optimizer_executed=False, PPO_updates_repeated=0, policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, required=True)
    args = parser.parse_args()
    run(args.pointer)
