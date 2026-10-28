"""Prescreen a wider global command envelope with an immutable saved actor.

Nine ordinary motion roles are crossed with every existing diagnostic height.
These commands are evaluation data only: no sampler, reward or controller is
changed. A prescreen never accepts a policy or starts PPO automatically.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args):
    import joint_pd_uniform_control_runtime_v3 as runtime
    from types import SimpleNamespace
    from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases

    runtime.configure(args.recipe)
    _, cfg = runtime.require_source(args.checkpoint)
    all4 = read(args.training / "independent_review/complete_comparison_agent_decision_required.json")
    grid_path = args.grid / "complete_paired_current_stage525.json"
    grid = read(grid_path)
    gap = read(args.saved_gaussian)
    assert all4["complete"] and grid["complete"] and gap["complete"]
    digest = sha(args.checkpoint)
    assert grid["candidate_checkpoint_sha256"] == gap["checkpoint_sha256"] == digest
    previous = cfg["commands"]["training_motion_scale"]
    assert 0 < previous < args.motion_scale <= 1.
    v = max(abs(x) for x in cfg["commands"]["ranges_lin_vel_x"]) * args.motion_scale
    w = max(abs(x) for x in cfg["commands"]["ranges_ang_vel_yaw"]) * args.motion_scale
    heights = sorted({row["case"]["height"] for row in grid["all525_pairs"]})
    assert len(heights) == 7 and [heights[0], heights[-1]] == list(cfg["commands"]["ranges_height"])
    roles = (("stand", 0., 0.), ("forward", v, 0.), ("reverse", -v, 0.),
             ("spin_left", 0., w), ("spin_right", 0., -w),
             ("forward_left", v, w), ("forward_right", v, -w),
             ("reverse_left", -v, w), ("reverse_right", -v, -w))
    cases = [dict(name=f"envelope_h{i}_{name}", height=h, speed=speed, yaw=yaw)
             for i, h in enumerate(heights) for name, speed, yaw in roles]
    # Preserve the existing evaluator's case-count guard. Batching retains
    # every Cartesian height/role combination instead of dropping conditions.
    batches = [cases[index:index+35] for index in range(0, len(cases), 35)]
    for batch in batches:
        expected = [(r["name"], r["speed"], r["yaw"], r["height"]) for r in batch]
        assert command_probe_cases(batch, SimpleNamespace(**cfg["commands"])) == expected
    args.output.mkdir(parents=True, exist_ok=False)
    command_file = args.output / "commands.json"
    write(command_file, cases)
    jobs = []
    for index, batch in enumerate(batches):
        name = f"batch_{index}"
        commands = args.output / (name + "_commands.json")
        write(commands, batch)
        destination = args.output / name
        argv = runtime.replay_command(args.checkpoint, destination)
        argv[argv.index("--seconds") + 1] = "20.0"
        start = argv.index("--seeds") + 1
        argv[start:start + 3] = ["883", "884", "885"]
        argv += ["--command-probe-json", str(commands.resolve())]
        jobs.append(dict(name=name, replay=str(destination.resolve()), command=argv,
                         command_file=str(commands.resolve()), case_count=len(batch)))
    pins = [SCRIPT, args.recipe, args.checkpoint, args.checkpoint.parent / "params/env.yaml",
            args.checkpoint.parent / "params/agent.yaml", grid_path, args.saved_gaussian,
            args.training / "independent_review/complete_comparison_agent_decision_required.json",
            command_file, runtime.SCRIPT, *[Path(job["command_file"]) for job in jobs]]
    plan = dict(qualified=True, created_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest,
        source_actual_updates=gap["actual_updates"], recipe=str(args.recipe.resolve()),
        batches=jobs, previous_training_motion_scale=previous,
        diagnostic_next_motion_scale=args.motion_scale, speed_limit_m_s=v,
        yaw_limit_rad_s=w, height_range_m=cfg["commands"]["ranges_height"],
        heights=heights, case_count=len(cases), trial_count=len(cases)*3,
        command_file=str(command_file.resolve()), seconds_per_case=20.,
        prior_physical_minimum_nonzero_yaw_rate=cfg["commands"]["yaw_min_abs_rate"]*previous,
        proposed_future_global_raw_minimum_yaw_rate=(cfg["commands"]["yaw_min_abs_rate"]*previous/args.motion_scale),
        proposed_future_minimum_is_support_preservation_not_a_case_selector=True,
        native_training_commands_unchanged=True, frozen_probe_only=True,
        cases_enter_training=False, source_pins={str(p.resolve()): sha(p) for p in pins},
        policy_accepted=False, next_PPO_selected=False)
    write(args.output / "qualification.json", plan)
    print(json.dumps({k:plan[k] for k in ("qualified", "trial_count", "speed_limit_m_s", "yaw_limit_rad_s")}))


def run(directory):
    directory = directory.resolve()
    plan = read(directory / "qualification.json")
    assert plan["qualified"] and plan["frozen_probe_only"]
    assert all(sha(p) == digest for p, digest in plan["source_pins"].items())
    # The original deterministic numerical reader and all seven constraints
    # execute unmodified, in the SDK environment, after native replay exits.
    code = ("from pathlib import Path;import sys;"
            "from review_historical_source3001_conditional_yaw200 import assess;"
            "assess(Path(sys.argv[1]))")
    results, all_trials, additional_pins = [], [], {}
    for job in plan["batches"]:
        replay = Path(job["replay"])
        assert not replay.exists(), "Observe an existing replay; never repeat or overwrite it"
        with (directory / (job["name"] + "_native.log")).open("x") as stream:
            subprocess.run(job["command"], cwd=ROOT, env=os.environ.copy(),
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        with (directory / (job["name"] + "_original_reader.log")).open("x") as stream:
            subprocess.run([str(ROOT / "run_python.sh"), "-c", code, job["replay"]],
                           cwd=ROOT, env=os.environ.copy(), stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        manifest = read(replay / "manifest.json")
        assert manifest["checkpoint_sha256"] == plan["checkpoint_sha256"]
        assert manifest["cases"] == read(job["command_file"])
        assert manifest["seeds"] == [883, 884, 885]
        assert not any(manifest.get(key) for key in ("sampled_policy_actions", "weights_updated",
            "policy_input_remapping", "action_bias_probes", "wheel_probe", "wheel_reference_limit_probe"))
        result = read(replay / "original_and_literal10_conditional_assessment.json")
        assert result["verified"] and result["summary"]["trials"] == job["case_count"]*3
        results.append(dict(name=job["name"], summary=result["summary"]))
        all_trials.extend(result["trials"])
        additional_pins.update({str(replay / name): sha(replay / name) for name in
            ("manifest.json", "completed.json", "trace.csv", "original_and_literal10_conditional_assessment.json")})
        write(directory / "progress.json", dict(completed_batches=results,
              policy_accepted=False, next_PPO_selected=False))
    assert len(all_trials) == plan["trial_count"]
    assert len({(row["case"]["name"], row["seed"]) for row in all_trials}) == plan["trial_count"]
    summary = dict(trials=len(all_trials), survivors=sum(row["survived"] for row in all_trials),
        original_passed=sum(row["original_passed"] for row in all_trials),
        original_AND_literal10_passed=sum(row["original_AND_literal10_passed"] for row in all_trials),
        missing_failed_trial_metrics=sum(row["metrics"] is None for row in all_trials),
        worst_nonzero_relative_percent={axis:max((row["nonzero_relative_percent"][axis]
            for row in all_trials if row["nonzero_relative_percent"][axis] is not None), default=None)
            for axis in ("speed", "yaw", "height")},
        worst_body_P95_deg=max((row["metrics"]["body_p95_deg"] for row in all_trials
            if row["metrics"] is not None), default=None))
    assert all(sha(p) == digest for p, digest in plan["source_pins"].items())
    report = dict(complete=True, observed_at=datetime.now(timezone.utc).isoformat(),
        checkpoint_sha256=plan["checkpoint_sha256"], summary=summary,
        batches=results, all_trials=all_trials,
        frozen_probe_only=True, all_failures_retained=True,
        original_numeric_reader_unchanged=True, original_training_unchanged=True,
        runtime_or_optimizer_modified=False, acceptance_changed=False,
        policy_accepted=False, next_PPO_selected=False,
        source_pins={**plan["source_pins"], **additional_pins,
            str(directory / "qualification.json"):sha(directory / "qualification.json")},
        limitations=["A wider-envelope prescreen is not full acceptance or a PPO improvement.",
            "Existing low-speed/interior-yaw failures remain in the completed525 evidence.",
            "No evaluation coordinate or failure list selects training or control logic."])
    write(directory / "complete_frozen_macro_envelope.json", report)
    print(json.dumps(dict(complete=True, summary=summary, policy_accepted=False)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--grid", type=Path)
    parser.add_argument("--saved-gaussian", type=Path)
    parser.add_argument("--motion-scale", type=float)
    args = parser.parse_args()
    if args.mode == "prepare":
        assert all(getattr(args, key) is not None for key in
            ("recipe", "checkpoint", "training", "grid", "saved_gaussian", "motion_scale"))
        prepare(args)
    else:
        run(args.output)
