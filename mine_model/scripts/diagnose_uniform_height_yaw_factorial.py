"""Cross height, signed speed and signed yaw only in frozen diagnostics.

The original PPO command sampler is untouched. Compare all combinations on
the reviewed parent and its candidate after the owned full review finishes.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve()


def qualify(args):
    import joint_pd_uniform_scalar_runtime as runtime
    from auto_train_mine import write_json
    from joint_pd_continuation import sha
    from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases

    assert not args.output.exists()
    runtime.configure(args.recipe)
    checkpoint, _, cfg = runtime.baseline()
    low, high = cfg["commands"]["ranges_height"]
    center = cfg["rewards"]["base_height_target"]
    heights = sorted({low, high, center, *(low + (high - low) * (.05 + .9 * i / 3.) for i in range(4))})
    assert low <= center <= high and len(heights) == 7
    scale = cfg["commands"]["training_motion_scale"]
    speed = max(abs(value) for value in cfg["commands"]["ranges_lin_vel_x"]) * scale / 4.
    yaw = max(abs(value) for value in cfg["commands"]["ranges_ang_vel_yaw"]) * scale
    cases = [dict(name=f"factorial_h{ih}_v{iv}_w{iw}", height=h, speed=v, yaw=w)
             for ih, h in enumerate(heights) for iv, v in enumerate((-speed, 0., speed))
             for iw, w in enumerate((-yaw, -yaw * 2. / 3., 0., yaw * 2. / 3., yaw))]
    # Keep the existing <=50-case diagnostic guard. Split along the complete
    # signed speed axis, retaining every height/yaw combination in each batch.
    batches = [dict(name=f"speed_axis_{index}", cases=[case for case in cases if case["speed"] == v])
               for index, v in enumerate((-speed, 0., speed))]
    assert sum(len(batch["cases"]) for batch in batches) == len(cases)
    for batch in batches:
        assert command_probe_cases(batch["cases"], SimpleNamespace(**cfg["commands"])) == [
            tuple(case[key] for key in ("name", "speed", "yaw", "height")) for case in batch["cases"]]
    pins = dict(runtime.recipe()["source_pins"])
    inputs = [SCRIPT, runtime.SCRIPT, args.recipe, args.parent_recipe, checkpoint,
        checkpoint.parent / "params/env.yaml", checkpoint.parent / "params/agent.yaml",
        SCRIPT.with_name("compare_uniform_joint_pd_replay_retention.py")]
    pins.update({str(path): sha(path) for path in inputs})
    spec = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        recipe=str(args.recipe), parent_recipe=str(args.parent_recipe), training_output=str(args.training),
        parent_checkpoint=str(checkpoint), parent_checkpoint_sha256=sha(checkpoint),
        parent_actual_updates=runtime.saved_updates(checkpoint), after_owned_unit=args.after_owned_unit,
        heights=heights, cases=cases, batches=batches, seeds=[883, 884, 885], seconds=20.,
        commands=len(cases), trials=len(cases) * 3, source_pins=pins,
        all_combinations_included=True, command_grid_enters_training=False,
        policy_weights_or_optimizer_updated=False, acceptance_changed=False, policy_accepted=False)
    args.output.mkdir()
    write_json(args.output / "qualification.json", spec)
    return spec


def run(args):
    spec = json.loads((args.output / "qualification.json").read_text())
    proof_path = Path(spec["training_output"]) / "final_checkpoint_verification_readonly.json"
    completion = Path(spec["training_output"]) / "independent_review/complete_comparison_agent_decision_required.json"
    while not completion.exists():
        status = subprocess.run(["systemctl", "--user", "show", spec["after_owned_unit"],
            "-p", "MainPID", "-p", "SubState", "-p", "Result"], capture_output=True, text=True, check=True).stdout
        owner = dict(line.split("=", 1) for line in status.splitlines() if "=" in line)
        (args.output / "waiting_owner.json").write_text(json.dumps(owner, indent=2) + "\n")
        if owner.get("MainPID") == "0" and owner.get("SubState") in ("exited", "dead", "failed"):
            raise RuntimeError("Owned review ended without completion; diagnose without restarting PPO")
        time.sleep(5.)
    import joint_pd_uniform_scalar_runtime as runtime
    import review_historical_source3001_conditional_yaw200 as readers
    from compare_uniform_joint_pd_replay_retention import compare, summarize_pairs
    from auto_train_mine import write_json
    from joint_pd_continuation import ROOT, sha

    assert all(sha(path) == value for path, value in spec["source_pins"].items())
    runtime.configure(spec["recipe"])
    proof, done = [json.loads(path.read_text()) for path in (proof_path, completion)]
    assert proof["verified"] and done["complete"]
    assert proof["actual_ppo_updates"] == done["candidate_actual_updates"] == spec["parent_actual_updates"] + 200
    checkpoint = Path(proof["checkpoint"])
    runtime.require_source(checkpoint)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    reports, comparisons, paired_pins = {}, [], {}
    for batch in spec["batches"]:
        commands = args.output / (batch["name"] + "_commands.json")
        write_json(commands, batch["cases"])
        for role, source, recipe in (("parent", Path(spec["parent_checkpoint"]), spec["parent_recipe"]),
                                       ("candidate", checkpoint, spec["recipe"])):
            label = role + "_" + batch["name"]
            directory = args.output / label
            argv = runtime.replay_command(source, directory)
            argv[argv.index("--recipe") + 1] = recipe
            argv[argv.index("--seconds") + 1] = str(spec["seconds"])
            at = argv.index("--seeds") + 1
            argv[at:at + 3] = list(map(str, spec["seeds"]))
            argv += ["--command-probe-json", str(commands)]
            write_json(args.output / (label + "_launch.json"), dict(command=argv,
                checkpoint_sha256=sha(source), diagnostic_only=True))
            write_json(args.output / "status.json", dict(state=label, diagnostic_active=True,
                training_modified=False, optimizer_executed=False, policy_accepted=False, goal_remains_active=True))
            with (args.output / (label + ".log")).open("x") as stream:
                subprocess.run(argv, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
            reports[label] = readers.assess(directory)["summary"]
            manifest = json.loads((directory / "manifest.json").read_text())
            assert manifest["checkpoint_sha256"] == sha(source) and manifest["cases"] == batch["cases"]
            assert manifest["seeds"] == spec["seeds"] and not manifest["sampled_policy_actions"]
            write_json(args.output / "progress.json", dict(completed_scopes=list(reports), summaries=reports,
                training_modified=False, policy_accepted=False))
        comparison, inputs = compare(args.output / ("parent_" + batch["name"]),
            args.output / ("candidate_" + batch["name"]), batch["name"])
        comparisons.append(comparison)
        paired_pins.update(inputs)
    all_pairs = [row for comparison in comparisons for row in comparison["all_pairs"]]
    assert len(all_pairs) == spec["trials"] == 315
    assert len({(row["case"]["name"], row["seed"]) for row in all_pairs}) == 315
    assert all(sha(path) == value for path, value in spec["source_pins"].items())
    assert sha(spec["parent_checkpoint"]) == spec["parent_checkpoint_sha256"]
    assert sha(checkpoint) == proof["checkpoint_sha256"]
    write_json(args.output / "complete_paired_height_yaw_factorial315.json", dict(complete=True,
        observed_at=datetime.now().astimezone().isoformat(), comparisons=comparisons,
        paired315_summary=summarize_pairs(all_pairs),
        original_sustained_review_completed_before_this_diagnosis=True,
        source_pins={**spec["source_pins"], **paired_pins, str(completion): sha(completion), str(proof_path): sha(proof_path)},
        all_combinations_included=True, command_grid_enters_training=False, optimizer_executed=False,
        training_modified=False, acceptance_changed=False, policy_accepted=False,
        final_full_plus_minus08_acceptance_still_required=True, goal_remains_active=True))
    write_json(args.output / "status.json", dict(state="complete_agent_analysis_required", diagnostic_active=False,
        training_modified=False, policy_accepted=False, goal_remains_active=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify-only", action="store_true")
    parser.add_argument("--training", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--parent-recipe", type=Path)
    parser.add_argument("--after-owned-unit")
    args = parser.parse_args()
    if args.qualify_only:
        assert all((args.training, args.recipe, args.parent_recipe, args.after_owned_unit))
        spec = qualify(args)
        print(json.dumps({key: spec[key] for key in ("qualified", "commands", "trials", "heights", "parent_actual_updates")}))
    else:
        run(args)
