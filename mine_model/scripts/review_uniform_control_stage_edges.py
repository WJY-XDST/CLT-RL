"""Complete current-stage replay coverage with both signed speed boundaries.

The owned PPO block and its original reviews are untouched. Replay every
height/yaw combination at both speed bounds, retaining all old passing cases.
No command from this test grid enters training, a reward, or a controller.
"""
import argparse
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def qualify(args):
    import joint_pd_uniform_control_runtime_v3 as runtime
    from auto_train_mine import write_json
    from joint_pd_continuation import ROOT, sha
    from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases

    assert not args.output.exists()
    runtime.configure(args.recipe)
    source, _, cfg = runtime.baseline()
    macro_path = args.macro / "qualification.json"
    macro = read(macro_path)
    assert macro["qualified"] and macro["source_checkpoint_sha256"] == sha(source)
    assert macro["recipe"] == str(args.recipe.resolve())
    assert macro["trials"] == 315
    scale = cfg["commands"]["training_motion_scale"]
    low, high = cfg["commands"]["ranges_height"]
    forward = max(abs(x) for x in cfg["commands"]["ranges_lin_vel_x"]) * scale
    reverse = max(abs(x) for x in cfg["commands"]["ranges_reverse_lin_vel_x"]) * scale
    assert forward == reverse > 0.
    existing = [case for batch in macro["batches"] for case in batch["cases"]]
    heights = sorted({case["height"] for case in existing})
    yaws = sorted({case["yaw"] for case in existing})
    speeds = sorted({case["speed"] for case in existing})
    assert len(heights) == 7 and len(yaws) == 5 and len(speeds) == 3
    assert heights[0] == low and heights[-1] == high
    assert speeds == [-forward / 4., 0., forward / 4.]
    assert max(map(abs, yaws)) == max(abs(x) for x in cfg["commands"]["ranges_ang_vel_yaw"]) * scale
    batches = [dict(name=f"speed_bound_{iv}", cases=[dict(name=f"edge_h{ih}_v{iv}_w{iw}",
        height=h, speed=v, yaw=w) for ih, h in enumerate(heights) for iw, w in enumerate(yaws)])
        for iv, v in enumerate((-reverse, forward))]
    for batch in batches:
        assert command_probe_cases(batch["cases"], SimpleNamespace(**cfg["commands"])) == [
            tuple(case[key] for key in ("name", "speed", "yaw", "height")) for case in batch["cases"]]
    target_damping = runtime.recipe()["control_overrides"]["wheel_damping"]
    transfer_needed = cfg["wheel_damping"] != target_damping
    freeze_script = SCRIPT.with_name("diagnose_uniform_wheel_drive_transfer.py")
    pins = {str(path.resolve()): sha(path) for path in (SCRIPT, runtime.SCRIPT, args.recipe,
        macro_path, source, source.parent / "params/env.yaml", source.parent / "params/agent.yaml",
        freeze_script, SCRIPT.with_name("compare_uniform_joint_pd_replay_retention.py"),
        SCRIPT.with_name("review_historical_source3001_conditional_yaw200.py"))}
    freeze = None
    if transfer_needed:
        assert args.frozen_transfer_spec
        template = read(args.frozen_transfer_spec)
        assert template["qualified"] and template["parent_checkpoint_sha256"] == sha(source)
        assert template["parent_actual_updates"] == runtime.saved_updates(source)
        assert template["baseline_wheel_damping"] == cfg["wheel_damping"]
        assert template["probe_wheel_damping"] == target_damping
        freeze = copy.deepcopy(template)
        freeze.update(batches=batches, seeds=macro["seeds"], seconds=macro["seconds"], trials=210,
            purpose="Frozen parent replay at declared global target D for complete current-stage edge comparison; no D0.5 rerun or PPO")
        pins[str(args.frozen_transfer_spec.resolve())] = sha(args.frozen_transfer_spec)
    spec = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        recipe=str(args.recipe.resolve()), training=macro["training"], macro=str(args.macro.resolve()),
        after_owned_unit=args.after_owned_unit, parent_checkpoint=str(source), parent_sha256=sha(source),
        parent_actual_updates=runtime.saved_updates(source), global_wheel_damping=target_damping,
        batches=batches, seeds=macro["seeds"], seconds=macro["seconds"],
        old_macro_trials=315, new_edge_trials=210, combined_stage_trials=525,
        heights=heights, combined_signed_speeds=[-reverse, *speeds, forward], yaws=yaws,
        frozen_parent_transfer_needed=transfer_needed,
        command_grid_enters_training=False, acceptance_changed=False, policy_accepted=False)
    args.output.mkdir()
    for batch in batches:
        path = args.output / (batch["name"] + "_commands.json")
        write_json(path, batch["cases"])
        pins[str(path)] = sha(path)
    if freeze is not None:
        freeze["source_pins"].update(pins)
        path = args.output / "frozen_parent_transfer_request.json"
        write_json(path, freeze)
        pins[str(path)] = sha(path)
        spec["frozen_parent_transfer_request"] = str(path)
    spec["source_pins"] = pins
    write_json(args.output / "qualification.json", spec)
    print(json.dumps({key: spec[key] for key in ("qualified", "new_edge_trials", "combined_stage_trials", "combined_signed_speeds")}))


def run(args):
    spec = read(args.output / "qualification.json")
    macro_complete = Path(spec["macro"]) / "complete_paired_adaptation315.json"
    while not macro_complete.exists():
        state = subprocess.run(["systemctl", "--user", "show", spec["after_owned_unit"],
            "-p", "MainPID", "-p", "SubState", "-p", "Result"], capture_output=True, text=True, check=True).stdout
        owner = dict(line.split("=", 1) for line in state.splitlines() if "=" in line)
        (args.output / "waiting_owner.json").write_text(json.dumps(owner, indent=2) + "\n")
        if owner.get("MainPID") == "0" and owner.get("SubState") in ("exited", "dead", "failed"):
            raise RuntimeError("Owned macro review ended without completion; inspect without restarting PPO")
        time.sleep(10.)
    import joint_pd_uniform_control_runtime_v3 as runtime
    import review_historical_source3001_conditional_yaw200 as reader
    from compare_uniform_joint_pd_replay_retention import compare, summarize_pairs
    from auto_train_mine import write_json
    from joint_pd_continuation import ROOT, sha

    assert all(sha(path) == digest for path, digest in spec["source_pins"].items())
    runtime.configure(spec["recipe"])
    previous = read(macro_complete)
    assert previous["complete"] and previous["source_actual_updates"] == spec["parent_actual_updates"]
    proof_path = Path(spec["training"]) / "final_checkpoint_verification_readonly.json"
    proof = read(proof_path)
    assert proof["verified"] and proof["checkpoint_sha256"] == previous["candidate_checkpoint_sha256"]
    assert proof["actual_ppo_updates"] == spec["parent_actual_updates"] + 200
    candidate = Path(proof["checkpoint"])
    runtime.require_source(candidate)
    source = Path(spec["parent_checkpoint"])
    runtime.require_source(source)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    reports, comparisons, pins = {}, [], dict(spec["source_pins"])
    if spec["frozen_parent_transfer_needed"]:
        overrides = args.output / "declared_frozen_parent_global_drive.json"
        write_json(overrides, {"wheel_damping": spec["global_wheel_damping"],
            "robot.actuators.wheels.damping": spec["global_wheel_damping"]})
    for batch in spec["batches"]:
        commands = args.output / (batch["name"] + "_commands.json")
        for role, checkpoint in (("parent", source), ("candidate", candidate)):
            label = role + "_" + batch["name"]
            destination = args.output / label
            argv = runtime.replay_command(checkpoint, destination)
            if role == "parent" and spec["frozen_parent_transfer_needed"]:
                assert argv[2:4] == ["replay", "--recipe"]
                argv[1:5] = [str(SCRIPT.with_name("diagnose_uniform_wheel_drive_transfer.py")),
                    "replay", "--request", spec["frozen_parent_transfer_request"]]
                argv += ["--overrides_json", str(overrides)]
            argv[argv.index("--seconds") + 1] = str(spec["seconds"])
            at = argv.index("--seeds") + 1
            argv[at:at + 3] = list(map(str, spec["seeds"]))
            argv += ["--command-probe-json", str(commands)]
            write_json(args.output / (label + "_launch.json"), dict(command=argv,
                checkpoint_sha256=sha(checkpoint), frozen_diagnostic_only=True, new_PPO_updates=0))
            write_json(args.output / "status.json", dict(state=label, diagnostic_active=True,
                training_modified=False, policy_accepted=False, goal_remains_active=True))
            with (args.output / (label + ".log")).open("x") as stream:
                subprocess.run(argv, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            reports[label] = reader.assess(destination)["summary"]
            key = ("frozen_uniform_wheel_drive_native_proof.json" if role == "parent" and
                spec["frozen_parent_transfer_needed"] else "uniform_global_control_native_contract.json")
            path = destination / key
            physical = read(path)
            if key.startswith("frozen"):
                assert physical["actual_wheel_damping_min"] == physical["actual_wheel_damping_max"] == spec["global_wheel_damping"]
            else:
                assert physical["actual_PhysX_wheel_damping_min"] == physical["actual_PhysX_wheel_damping_max"] == spec["global_wheel_damping"]
            pins[str(path)] = sha(path)
            write_json(args.output / "progress.json", dict(completed_scopes=list(reports), summaries=reports,
                training_modified=False, policy_accepted=False))
        result, inputs = compare(args.output / ("parent_" + batch["name"]),
            args.output / ("candidate_" + batch["name"]), batch["name"])
        comparisons.append(result)
        pins.update(inputs)
    original_pairs = [row for comparison in previous["comparisons"] for row in comparison["all_pairs"]]
    new_pairs = [row for comparison in comparisons for row in comparison["all_pairs"]]
    pairs = original_pairs + new_pairs
    assert len(original_pairs) == 315 and len(new_pairs) == 210 and len(pairs) == 525
    commands = {(row["case"]["height"], row["case"]["speed"], row["case"]["yaw"]) for row in pairs}
    expected = {(h, v, w) for h in spec["heights"] for v in spec["combined_signed_speeds"] for w in spec["yaws"]}
    assert commands == expected and len(commands) == 175
    for command in commands:
        assert {row["seed"] for row in pairs if tuple(row["case"][k] for k in ("height", "speed", "yaw")) == command} == set(spec["seeds"])
    grouped = {axis: [dict(command_value=value,
        **summarize_pairs([row for row in pairs if row["case"][axis] == value]))
        for value in sorted({row["case"][axis] for row in pairs})] for axis in ("height", "speed", "yaw")}
    pins.update({str(path): sha(path) for path in (macro_complete, proof_path, candidate)})
    assert all(sha(path) == digest for path, digest in pins.items())
    write_json(args.output / "complete_paired_stage_grid525.json", dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), combined525_summary=summarize_pairs(pairs),
        new_edge210_summary=summarize_pairs(new_pairs), original_macro315_report=str(macro_complete),
        new_edge_comparisons=comparisons, all525_pairs=pairs, grouped_diagnosis=grouped,
        all_height_speed_yaw_combinations_in_current_stage_grid_retained=True,
        same_global_controller_in_parent_and_candidate=True, global_wheel_damping=spec["global_wheel_damping"],
        source_actual_updates=spec["parent_actual_updates"], candidate_actual_updates=proof["actual_ppo_updates"],
        candidate_checkpoint_sha256=sha(candidate), training_sampler_not_modified=True,
        optimizer_executed=False, acceptance_changed=False,
        only_current_stage_coverage_not_final_full_plus_minus08=True,
        canonical17_sustained_does_not_cover_every_grid_combination=True,
        final_full_range_and_independent_sustained_verification_still_required=True,
        continuation_requires_agent_data_review=True, source_pins=pins,
        policy_accepted=False, hardware_enabled=False, shutdown=False, goal_remains_active=True))
    write_json(args.output / "status.json", dict(state="complete_agent_decision_required", diagnostic_active=False,
        training_modified=False, policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--macro", type=Path)
    parser.add_argument("--after-owned-unit")
    parser.add_argument("--frozen-transfer-spec", type=Path)
    args = parser.parse_args()
    if args.qualify_only:
        assert args.recipe and args.macro and args.after_owned_unit
        qualify(args)
    else:
        run(args)
