"""Compare an adapted policy with its frozen-policy global-drive baseline.

All combinations, formerly passing trials, thresholds and seeds are retained.
The grid is replay data only. This process has no PPO or source-selection path.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def qualify(args):
    import joint_pd_uniform_control_runtime_v3 as runtime
    from compare_uniform_joint_pd_replay_retention import load
    from joint_pd_continuation import sha
    from auto_train_mine import write_json

    assert not args.output.exists()
    runtime.configure(args.recipe)
    frozen = read(args.baseline / "qualification.json")
    finished = args.baseline / "complete_frozen_uniform_wheel_drive315.json"
    assert read(finished)["complete"]
    source, _, _ = runtime.baseline()
    assert sha(source) == frozen["parent_checkpoint_sha256"]
    assert runtime.recipe()["control_overrides"]["wheel_damping"] == frozen["probe_wheel_damping"]
    pins = {str(path.resolve()): sha(path) for path in (SCRIPT, runtime.SCRIPT,
        args.recipe, args.baseline / "qualification.json", finished,
        SCRIPT.with_name("compare_uniform_joint_pd_replay_retention.py"),
        SCRIPT.with_name("review_historical_source3001_conditional_yaw200.py"))}
    for batch in frozen["batches"]:
        directory = args.baseline / ("probe_" + batch["name"])
        report, manifest, cfg, _, _ = load(directory)
        assert manifest["cases"] == batch["cases"] and manifest["seeds"] == frozen["seeds"]
        assert manifest["checkpoint_sha256"] == sha(source)
        assert cfg["wheel_damping"] == cfg["robot"]["actuators"]["wheels"]["damping"] == frozen["probe_wheel_damping"]
        assert report["summary"]["trials"] == len(batch["cases"]) * len(frozen["seeds"])
    spec = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        recipe=str(args.recipe.resolve()), baseline=str(args.baseline.resolve()),
        training=str(args.training.resolve()), after_owned_unit=args.after_owned_unit,
        source_checkpoint_sha256=sha(source), source_actual_updates=runtime.saved_updates(source),
        batches=frozen["batches"], seconds=frozen["seconds"], seeds=frozen["seeds"],
        trials=frozen["trials"], global_wheel_damping=frozen["probe_wheel_damping"], source_pins=pins,
        formerly_passing_trials_included=True, grid_enters_training=False,
        optimizer_executed=False, policy_accepted=False)
    args.output.mkdir()
    write_json(args.output / "qualification.json", spec)
    print(json.dumps({key: spec[key] for key in ("qualified", "trials", "source_actual_updates", "global_wheel_damping")}))


def run(args):
    spec = read(args.output / "qualification.json")
    training = Path(spec["training"])
    complete = training / "independent_review/complete_comparison_agent_decision_required.json"
    while not complete.exists():
        state = subprocess.run(["systemctl", "--user", "show", spec["after_owned_unit"],
            "-p", "MainPID", "-p", "SubState", "-p", "Result"], capture_output=True, text=True, check=True).stdout
        owner = dict(line.split("=", 1) for line in state.splitlines() if "=" in line)
        (args.output / "waiting_owner.json").write_text(json.dumps(owner, indent=2) + "\n")
        if owner.get("MainPID") == "0" and owner.get("SubState") in ("exited", "dead", "failed"):
            raise RuntimeError("Owned200/review ended without its completion; inspect, do not restart PPO")
        time.sleep(10.)
    import joint_pd_uniform_control_runtime_v3 as runtime
    import review_historical_source3001_conditional_yaw200 as reader
    from compare_uniform_joint_pd_replay_retention import compare, summarize_pairs
    from joint_pd_continuation import ROOT, sha
    from auto_train_mine import write_json

    runtime.configure(spec["recipe"])
    assert all(sha(path) == digest for path, digest in spec["source_pins"].items())
    proof_path = training / "final_checkpoint_verification_readonly.json"
    proof, reviewed = read(proof_path), read(complete)
    assert proof["verified"] and proof["all200_native_updates_and_original_alignment_verified"] and reviewed["complete"]
    assert proof["source_checkpoint_sha256"] == spec["source_checkpoint_sha256"]
    assert proof["actual_ppo_updates"] == reviewed["candidate_actual_updates"] == spec["source_actual_updates"] + 200
    source = Path(proof["checkpoint"])
    _, cfg = runtime.require_source(source)
    assert cfg["wheel_damping"] == cfg["robot"]["actuators"]["wheels"]["damping"] == spec["global_wheel_damping"]
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    comparisons, pins, summaries = [], dict(spec["source_pins"]), {}
    for batch in spec["batches"]:
        label = batch["name"]
        commands = args.output / (label + "_commands.json")
        write_json(commands, batch["cases"])
        destination = args.output / ("candidate_" + label)
        argv = runtime.replay_command(source, destination)
        argv[argv.index("--seconds") + 1] = str(spec["seconds"])
        at = argv.index("--seeds") + 1
        argv[at:at + 3] = list(map(str, spec["seeds"]))
        argv += ["--command-probe-json", str(commands)]
        write_json(args.output / (label + "_launch.json"), dict(command=argv,
            source_checkpoint_sha256=sha(source), frozen_policy=True, controller_same_as_frozen_baseline=True))
        write_json(args.output / "status.json", dict(state=label, diagnostic_active=True,
            training_modified=False, policy_accepted=False, goal_remains_active=True))
        with (args.output / (label + ".log")).open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        summaries[label] = reader.assess(destination)["summary"]
        physical = read(destination / "uniform_global_control_native_contract.json")
        assert physical["actual_PhysX_wheel_damping_min"] == physical["actual_PhysX_wheel_damping_max"] == spec["global_wheel_damping"]
        baseline = Path(spec["baseline"]) / ("probe_" + label)
        comparison, inputs = compare(baseline, destination, label)
        comparisons.append(comparison)
        pins.update(inputs)
        write_json(args.output / "progress.json", dict(completed_scopes=list(summaries), summaries=summaries,
            paired_summary=summarize_pairs([row for item in comparisons for row in item["all_pairs"]]),
            training_modified=False, policy_accepted=False))
    pairs = [row for item in comparisons for row in item["all_pairs"]]
    assert len(pairs) == spec["trials"] == len({(row["case"]["name"], row["seed"]) for row in pairs})
    grouped = {axis: [dict(command_value=value,
        **summarize_pairs([row for row in pairs if row["case"][axis] == value]))
        for value in sorted({row["case"][axis] for row in pairs})] for axis in ("height", "speed", "yaw")}
    pins.update({str(path): sha(path) for path in (complete, proof_path, source)})
    assert all(sha(path) == digest for path, digest in pins.items())
    write_json(args.output / "complete_paired_adaptation315.json", dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), comparisons=comparisons,
        paired315_summary=summarize_pairs(pairs), grouped_diagnosis=grouped,
        same_controller_in_both_sides=True, global_wheel_damping=spec["global_wheel_damping"],
        source_actual_updates=spec["source_actual_updates"], candidate_actual_updates=proof["actual_ppo_updates"],
        candidate_checkpoint_sha256=sha(source), all_original_scopes_and_sustained_review_already_complete=True,
        all_baseline_passes_retained_in_report=True, original_thresholds_and_strict_relative10_unchanged=True,
        grid_enters_training=False, optimizer_executed=False, training_modified=False,
        continuation_requires_agent_data_review=True, final_full_plus_minus08_acceptance_still_required=True,
        source_pins=pins, policy_accepted=False, hardware_enabled=False, shutdown=False, goal_remains_active=True))
    write_json(args.output / "status.json", dict(state="complete_agent_decision_required", diagnostic_active=False,
        policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--after-owned-unit")
    args = parser.parse_args()
    if args.qualify_only:
        assert args.recipe and args.training and args.baseline and args.after_owned_unit
        qualify(args)
    else:
        run(args)
