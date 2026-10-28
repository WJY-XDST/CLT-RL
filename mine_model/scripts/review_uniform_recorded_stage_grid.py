"""Replay a descendant against every previously recorded current-stage cell.

Reuse verified source telemetry instead of rerunning identical frozen trials.
The full Cartesian grid is evaluation data only; the original metric reader
and all formerly passing cells are retained. No training or source selection.
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
    from auto_train_mine import write_json
    from compare_uniform_joint_pd_replay_retention import load
    from joint_pd_continuation import sha

    assert not args.output.exists()
    runtime.configure(args.recipe)
    source, _, cfg = runtime.baseline()
    pointer = read(args.pointer)
    assert pointer["uniform_recipe"] == str(args.recipe.resolve())
    assert pointer["source_actual_updates"] == runtime.saved_updates(source)
    assert pointer["target_actual_updates"] == runtime.saved_updates(source) + 200
    pins = {str(p.resolve()): sha(p) for p in [SCRIPT, runtime.SCRIPT, args.recipe, args.pointer,
        source, source.parent / "params/env.yaml", source.parent / "params/agent.yaml",
        SCRIPT.with_name("compare_uniform_joint_pd_replay_retention.py"),
        SCRIPT.with_name("review_historical_source3001_conditional_yaw200.py")]}
    batches, cells, trials = [], set(), 0
    for i, folder in enumerate(args.source):
        report, manifest, actual_cfg, _, inputs = load(folder)
        assert manifest["checkpoint_sha256"] == sha(source)
        assert not manifest["action_bias_probes"] and not manifest["wheel_probe"]
        assert manifest["seconds_per_case"] == 20. and manifest["policy_dt"] == .02
        assert actual_cfg["wheel_damping"] == actual_cfg["robot"]["actuators"]["wheels"]["damping"] == cfg["wheel_damping"]
        assert manifest["reset_velocity_uniform_amplitude"] == .05
        if batches:
            assert manifest["seeds"] == batches[0]["seeds"]
        assert len(manifest["cases"]) <= 50
        for case in manifest["cases"]:
            cell = tuple(case[k] for k in ("height", "speed", "yaw"))
            assert cell not in cells
            cells.add(cell)
        batches.append(dict(name=f"recorded_batch_{i}", source=str(folder.resolve()),
            cases=manifest["cases"], seeds=manifest["seeds"], seconds=manifest["seconds_per_case"],
            source_summary=report["summary"]))
        trials += report["summary"]["trials"]
        pins.update(inputs)
    axes = [sorted({c[i] for c in cells}) for i in range(3)]
    assert cells == {(h, v, w) for h in axes[0] for v in axes[1] for w in axes[2]}
    assert len(axes[0]) == 7 and len(axes[1]) == len(axes[2]) == 5
    assert trials == len(cells) * len(batches[0]["seeds"]) == 525
    scale = cfg["commands"]["training_motion_scale"]
    assert axes[0][0] == cfg["commands"]["ranges_height"][0]
    assert axes[0][-1] == cfg["commands"]["ranges_height"][1]
    assert axes[1][0] == cfg["commands"]["ranges_reverse_lin_vel_x"][0] * scale
    assert axes[1][-1] == cfg["commands"]["ranges_lin_vel_x"][1] * scale
    assert axes[2][0] == cfg["commands"]["ranges_ang_vel_yaw"][0] * scale
    assert axes[2][-1] == cfg["commands"]["ranges_ang_vel_yaw"][1] * scale
    assert all(sha(p) == digest for p, digest in pins.items())
    args.output.mkdir()
    for batch in batches:
        path = args.output / (batch["name"] + "_commands.json")
        write_json(path, batch["cases"])
        pins[str(path)] = sha(path)
    spec = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        recipe=str(args.recipe.resolve()), source_checkpoint=str(source), source_checkpoint_sha256=sha(source),
        source_actual_updates=runtime.saved_updates(source), training=pointer["planned_production_output"],
        after_owned_unit=args.after_owned_unit, batches=batches, trials=trials,
        all_current_stage_boundary_and_interior_cartesian_cells_included=True,
        source_native_trials_reused_without_rerun=True, source_pins=pins,
        grid_enters_training=False, optimizer_executed=False, policy_accepted=False)
    write_json(args.output / "qualification.json", spec)
    print(json.dumps(dict(qualified=True, trials=trials, source_actual_updates=spec["source_actual_updates"],
        new_native_replays=len(batches), source_native_replays=0)))


def run(output):
    spec = read(output / "qualification.json")
    training = Path(spec["training"])
    marker = training / "independent_review/complete_comparison_agent_decision_required.json"
    while not marker.exists():
        raw = subprocess.run(["systemctl", "--user", "show", spec["after_owned_unit"],
            "-p", "MainPID", "-p", "SubState"], capture_output=True, text=True, check=True).stdout
        owner = dict(row.split("=", 1) for row in raw.splitlines() if "=" in row)
        if owner.get("MainPID") == "0" and owner.get("SubState") in ("failed", "dead", "exited"):
            raise RuntimeError("Owned200/review ended without completion; inspect without restarting PPO")
        time.sleep(10.)
    import joint_pd_uniform_control_runtime_v3 as runtime
    import review_historical_source3001_conditional_yaw200 as reader
    from compare_uniform_joint_pd_replay_retention import compare, summarize_pairs
    from auto_train_mine import write_json
    from joint_pd_continuation import ROOT, sha

    runtime.configure(spec["recipe"])
    assert all(sha(p) == digest for p, digest in spec["source_pins"].items())
    proof_path = training / "final_checkpoint_verification_readonly.json"
    proof = read(proof_path)
    assert proof["verified"] and proof["all200_native_updates_and_original_alignment_verified"]
    assert proof["source_checkpoint_sha256"] == spec["source_checkpoint_sha256"]
    assert proof["actual_ppo_updates"] == spec["source_actual_updates"] + 200
    candidate = Path(proof["checkpoint"])
    runtime.require_source(candidate)
    comparisons, summaries, pins = [], {}, dict(spec["source_pins"])
    for batch in spec["batches"]:
        label = batch["name"]
        destination = output / label
        argv = runtime.replay_command(candidate, destination)
        argv[argv.index("--seconds") + 1] = str(batch["seconds"])
        at = argv.index("--seeds") + 1
        argv[at:at+3] = list(map(str, batch["seeds"]))
        argv += ["--command-probe-json", str(output / (label + "_commands.json"))]
        write_json(output / (label + "_launch.json"), dict(command=argv, frozen_policy=True,
            source_checkpoint_sha256=spec["source_checkpoint_sha256"], candidate_checkpoint_sha256=sha(candidate)))
        write_json(output / "status.json", dict(state=label, diagnostic_active=True,
            training_modified=False, policy_accepted=False, goal_remains_active=True))
        with (output / (label + ".log")).open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        summaries[label] = reader.assess(destination)["summary"]
        physical = read(destination / "uniform_global_control_native_contract.json")
        assert physical["actual_PhysX_wheel_damping_min"] == physical["actual_PhysX_wheel_damping_max"] == runtime.recipe()["control_overrides"]["wheel_damping"]
        paired, inputs = compare(Path(batch["source"]), destination, label)
        comparisons.append(paired)
        pins.update(inputs)
        write_json(output / "progress.json", dict(completed_scopes=list(summaries), summaries=summaries,
            paired_summary=summarize_pairs([r for c in comparisons for r in c["all_pairs"]]),
            training_modified=False, policy_accepted=False))
    pairs = [r for c in comparisons for r in c["all_pairs"]]
    assert len(pairs) == spec["trials"] == 525
    assert len({(*[r["case"][k] for k in ("height", "speed", "yaw")], r["seed"]) for r in pairs}) == 525
    pins.update({str(p): sha(p) for p in [marker, proof_path, candidate]})
    assert all(sha(p) == digest for p, digest in pins.items())
    write_json(output / "complete_paired_current_stage525.json", dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), comparisons=comparisons,
        paired525_summary=summarize_pairs(pairs), all525_pairs=pairs,
        grouped_diagnosis={axis:[dict(command_value=value,
            **summarize_pairs([r for r in pairs if r["case"][axis] == value]))
            for value in sorted({r["case"][axis] for r in pairs})] for axis in ("height", "speed", "yaw")},
        source_actual_updates=spec["source_actual_updates"], candidate_actual_updates=proof["actual_ppo_updates"],
        source_checkpoint_sha256=spec["source_checkpoint_sha256"], candidate_checkpoint_sha256=sha(candidate),
        original4_reviews_completed_before_grid=True, source_native_trials_reused_without_rerun=True,
        all_formerly_passing_and_failed_cells_retained=True, same_physical_controller=True,
        grid_enters_training=False, numerical_comparison_reader_unchanged=True,
        source_pins=pins, continuation_requires_agent_data_review=True,
        current_stage_only_not_full_plus_minus08_acceptance=True,
        all_known_failure_sustained_and_final_full_range_still_required=True,
        optimizer_executed=False, training_modified=False, acceptance_changed=False,
        policy_accepted=False, hardware_enabled=False, shutdown=False, goal_remains_active=True))
    write_json(output / "status.json", dict(state="complete_agent_decision_required", diagnostic_active=False,
        policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--pointer", type=Path)
    parser.add_argument("--source", type=Path, nargs="+")
    parser.add_argument("--after-owned-unit")
    args = parser.parse_args()
    if args.qualify_only:
        assert args.recipe and args.pointer and args.source and args.after_owned_unit
        qualify(args)
    else:
        run(args.output)
