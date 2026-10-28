"""One source-preserving200-update global scalar experiment after real capacity.

The block is a review interval, not a total-training endpoint. A final model
still requires all operating conditions, retention, and sustained validation.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joint_pd_uniform_scalar_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, sha
from train_joint_pd_standing_seed import saved_updates
from check_uniform_scalar_environment_capacity import require_capacity_selection


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def owner(unit):
    result = subprocess.run(["systemctl", "--user", "show", unit,
        "-p", "MainPID", "-p", "Result", "-p", "SubState"], capture_output=True, text=True, check=True)
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def run(pointer_path):
    pointer = read(pointer_path)
    runtime.configure(pointer["uniform_recipe"])
    capacity_path = Path(pointer["capacity"]) / "complete.json"
    while not capacity_path.exists():
        state = owner(pointer["capacity_unit"])
        assert state["MainPID"] != "0", state
        time.sleep(10.)
    capacity = read(capacity_path)
    runtime.NUM_ENVS = count = capacity["selected_num_envs"]
    assert count in (8192, 4096)
    request = read(pointer["request"])
    source = Path(request["source_checkpoint"])
    saved, cfg = runtime.require_source(source)
    source_verification = read(request["source_final_verification"])
    assert source_verification["verified"] and source_verification["checkpoint_sha256"] == sha(source)
    assert source_verification["actual_ppo_updates"] == saved_updates(source)
    require_capacity_selection(capacity_path, block_id=pointer["block_id"], source=source, num_envs=count)
    output = Path(pointer["planned_production_output"])
    assert not output.exists()
    source_actual = saved_updates(source)
    target = source_actual + runtime.recipe()["planned_updates"]
    expected = runtime.expected_configuration(cfg, count)
    delta = dict(rewards=runtime.recipe()["reward_changes"], commands={}, exploration=runtime.recipe()["effective_change_relative_to_source"], controller={}, physics={})
    original_plan = read(Path(request["source_final_verification"]).parent / "experiment_plan.json")
    plan = copy.deepcopy(original_plan)
    files = dict(read(pointer["request"])["source_pins"])
    for path in (SCRIPT, capacity_path, Path(pointer_path),
        SCRIPT.with_name("monitor_uniform_scalar_joint_pd_learning.py"),
        SCRIPT.with_name("compare_uniform_scalar_recorded_joint_targets.py"),
        SCRIPT.with_name("review_uniform_scalar_joint_pd_block.py")):
        files[str(path)] = sha(path)
    assert all(sha(path) == digest for path, digest in files.items())
    plan.update(created_at=datetime.now().astimezone().isoformat(), source_checkpoint=str(source),
        source_sha256=sha(source), source_actual_ppo_updates=source_actual,
        source_Adam_step=source_actual * 20, source_Adam_states=17,
        source_optimizer_LR=[group["lr"] for group in saved["optimizer_state_dict"]["param_groups"]],
        source_std=saved["model_state_dict"]["std"].tolist(),
        source_replay=request["source_mapping_replay"], source_review_same51=request["source_same51_replay"],
        target_actual_ppo_updates=target, expected_terminal_Adam_step=target * 20,
        bounded_training_updates=200, native_num_envs=count, fixed_standing_environments=count // 4,
        motion_training_environments=count * 3 // 4, native_transition_batch=count * 48,
        run_name=f"uniform_reward_scalars_n{count}_{output.name}_u{source_actual}",
        training_config=expected, profile=runtime.profile(), next_configuration_changes=delta,
        only_reward_change=delta["rewards"], only_command_change={}, only_exploration_change=delta["exploration"],
        only_controller_change={}, only_runtime_sampling_change={}, only_runtime_reward_change={},
        typed_saved_reward_scalars_unchanged=False, typed_saved_commands_unchanged=True,
        exploration_guard=dict(runtime.recipe()["exploration"], saved_initial_std_reset=False), checkpoint_save_interval=1,
        source_files=files, capacity_admission=str(capacity_path), ordinary_full_policy_PPO=True,
        auxiliary_losses_enabled=False, action_map_or_controller_changed=False,
        optimizer_reset=False, learned_std_reset=False, actual_saved_LR_reset=False,
        all_six_Gaussian_std_entries_remain_trainable=True, no_total_training_update_limit=True,
        source_model_and_optimizer_reloaded_from_original200=False,
        case_specific_training_logic=False, relative_error_target=.1, acceptance_changed=False,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    assert Path(plan["source_replay"]).is_dir()
    output.mkdir()
    write_json(output / "experiment_plan.json", plan)
    # Reuse the measured source/count resource check; it does not change PPO.
    from train_AF7001_wheelD075_wheel005_block200 import wait_resources
    wait_resources(output, plan)
    argv, final, _ = runtime.command(source, cfg, plan["run_name"], 200, cfg["commands"], output / "STOP")
    assert final == target - 1
    write_json(output / "launch.json", dict(command=argv, source_sha256=sha(source)))
    monitor_command = [sys.executable, str(SCRIPT.with_name("monitor_uniform_scalar_joint_pd_learning.py")),
        "--output", str(output), "--recipe", pointer["uniform_recipe"]]
    with (output / "training.log").open("x") as log, (output / "learning_monitor.log").open("x") as monitor_log:
        child = subprocess.Popen(argv, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        monitor = subprocess.Popen(monitor_command, cwd=ROOT, env=os.environ.copy(), stdout=monitor_log, stderr=subprocess.STDOUT)
        write_json(output / "owned_live_processes.json", dict(training_child_pid=child.pid,
            readonly_monitor_pid=monitor.pid, unit=os.environ.get("CLT_UNIFORM_TRAINING_UNIT"),
            source_actual_updates=source_actual, target_actual_updates=target, policy_accepted=False))
        while True:
            committed = len(list((output / "native_learning_signal").glob("audit_update_*.json")))
            code = child.poll()
            write_json(output / "status.json", dict(observed_at=datetime.now().astimezone().isoformat(),
                state="training" if code is None else "verifying_saved200", training_active=code is None,
                child_pid=child.pid, committed_actual_updates=committed, actual_ppo_updates=source_actual + committed,
                target_actual_ppo_updates=target, no_total_training_update_limit=True, goal_remains_active=True,
                case_specific_training_logic=False, policy_accepted=False))
            if code is not None:
                assert code == 0, code
                break
            time.sleep(10.)
        assert all(sha(path) == digest for path, digest in files.items())
        runs = list(RUNS.glob("*_" + plan["run_name"]))
        assert len(runs) == 1
        checkpoint = runs[0] / f"model_{target - 1}.pt"
        assert saved_updates(checkpoint) == target
        import train_joint_pd_standing_first as boundary
        proof = boundary.verify_update(source, checkpoint, target, expected, source)
        write_json(output / "resume_verification.json", proof)
        import continue_joint_pd_observed100 as original_external
        import plain27_sampler_metadata_adapter as adapter
        import joint_pd_review_interval200 as interval
        from native_admitted_population_finalizer import install
        external = adapter.private_module(original_external.__file__, "_uniform_original_external200")
        interval.install_finalizer(external)
        install(external, count)
        external.finalize_observations(output)
        after, _ = runtime.require_source(checkpoint)
        states = after["optimizer_state_dict"]["state"]
        assert len(states) == 17 and all(float(value["step"]) == target * 20 for value in states.values())
        result = dict(verified=True, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            source_checkpoint=str(source), source_checkpoint_sha256=sha(source), actual_ppo_updates=target,
            all17_Adam_step=target * 20, saved_std=after["model_state_dict"]["std"].tolist(),
            saved_actual_LR=[group["lr"] for group in after["optimizer_state_dict"]["param_groups"]],
            all200_native_updates_and_original_alignment_verified=True,
            original_uniform_reward_structure_and_continuous_commands=True,
            declared_global_reward_scalar_changes=delta["rewards"], case_specific_training_logic=False,
            source_initial_std_LR_and_all17_Adam_preserved=True, source_pins=files, policy_accepted=False)
        write_json(output / "final_checkpoint_verification_readonly.json", result)
        monitor_code = monitor.wait(timeout=1800)
        assert monitor_code == 0, monitor_code
    write_json(output / "status.json", dict(state="independent_review_running", training_active=False,
        actual_ppo_updates=target, checkpoint=str(checkpoint), goal_remains_active=True, policy_accepted=False))
    review_command = [sys.executable, str(SCRIPT.with_name("review_uniform_scalar_joint_pd_block.py")),
        "--output", str(output), "--recipe", pointer["uniform_recipe"]]
    with (output / "review_controller.log").open("x") as log:
        subprocess.run(review_command, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.pointer.resolve())
    except BaseException as error:
        pointer = read(args.pointer)
        destination = Path(pointer["planned_production_output"])
        destination.mkdir(exist_ok=True)
        write_json(destination / "controller_inspection_required.json", dict(
            error=repr(error), checkpoint_files_preserved=True, training_not_automatically_restarted=True,
            goal_remains_active=True, policy_accepted=False))
        raise
