"""Compare the original failed cells, all51, neighbours and sustained spins."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
Q = ROOT / "mine_model/results/joint_pd_source3001_conditional_yaw_restart_20261024"
SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def owner(unit):
    result = subprocess.run(["systemctl", "--user", "show", unit,
        "-p", "MainPID", "-p", "SubState", "-p", "Result"], capture_output=True, text=True, check=True)
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def assess(directory):
    import numpy as np
    import joint_pd_acceptance_10pct_20261017 as acceptance
    import assess_height_leg_command_tracking as leg_assessor
    from auto_train_mine import write_json
    from joint_pd_continuation import sha

    manifest, complete = read(directory / "manifest.json"), read(directory / "completed.json")
    assert complete["completed"] and not manifest["sampled_policy_actions"] and not manifest["weights_updated"]
    assert manifest["leg_control_mode"] == "implicit_joint_position" and manifest["policy_dt"] == .02
    trace = np.genfromtxt(directory / "trace.csv", delimiter=",", names=True)
    expected_samples = len(manifest["cases"]) * len(manifest["seeds"]) * round(manifest["seconds_per_case"] / .02)
    assert len(trace) == complete["samples"] == expected_samples
    assert all(np.isfinite(trace[key]).all() for key in trace.dtype.names)
    criteria = acceptance.profile()["criteria"]
    assert criteria["relative_error"] == .1
    trials = []
    for index, case in enumerate(manifest["cases"]):
        for seed in manifest["seeds"]:
            part = trace[(trace["case_id"] == index) & (trace["seed"] == seed)]
            survived = not any(np.any(part[key]) for key in ("episode", "terminated", "timeout"))
            tail = part[part["time_s"] >= manifest["seconds_per_case"] - 3.]
            metric = acceptance.assessor.metrics_for_window(tail, case, criteria) if survived else None
            reference = tail if survived else part[part["episode"] == 0]
            leg_reference = max(.001, min(np.mean(abs(reference[f"{side}_length_ref_m"])) for side in ("left", "right")))
            wheel_reference = min(np.mean(abs(reference[f"{side}_wheel_ref_rad_s"])) for side in ("left", "right"))
            limits = dict(speed_mae_m_s=max(criteria["speed_absolute_m_s"], .1 * abs(case["speed"])),
                yaw_mae_rad_s=max(criteria["yaw_absolute_rad_s"], .1 * abs(case["yaw"])),
                height_mae_m=.1 * case["height"], leg_length_mae_m=.1 * leg_reference,
                leg_angle_mae_rad=.1 * criteria["leg_angle_design_range_rad"],
                wheel_speed_mae_rad_s=max(criteria["wheel_absolute_rad_s"], .1 * wheel_reference),
                body_p95_deg=criteria["body_p95_deg"])
            flags = acceptance.assessor.tracking_components(survived, metric,
                {key: value - 1.e-12 for key, value in limits.items()})
            static = (leg_assessor.trial_metrics(part, case, manifest["seconds_per_case"], criteria)
                      if case["speed"] == 0. and case["yaw"] == 0. else None)
            static_ok = static is None or all(static[key] for key in
                ("height_posture_passed", "leg_command_passed", "stationary_drift_passed"))
            relative = {axis: 100. * metric[field] / abs(case[axis]) if metric is not None and case[axis] else None
                        for axis, field in (("speed", "speed_mae_m_s"), ("yaw", "yaw_mae_rad_s"), ("height", "height_mae_m"))}
            original = bool(flags["task_goal_passed"] and flags["reference_tracking_passed"] and static_ok)
            literal = original and all(value < 10. for value in relative.values() if value is not None)
            trials.append(dict(case=case, seed=seed, survived=survived, metrics=metric, limits=limits,
                flags=flags, actual_static_leg_commands=static,
                original_passed=original, original_AND_literal10_passed=literal,
                nonzero_relative_percent=relative,
                yaw_signed_bias_rad_s=float(np.mean(tail["yaw_rate_rad_s"] - case["yaw"])) if survived else None,
                yaw_actual_mean_rad_s=float(np.mean(tail["yaw_rate_rad_s"])) if survived else None,
                yaw_standard_deviation_rad_s=float(np.std(tail["yaw_rate_rad_s"])) if survived else None))
    summary = dict(trials=len(trials), survivors=sum(t["survived"] for t in trials),
        original_passed=sum(t["original_passed"] for t in trials),
        original_AND_literal10_passed=sum(t["original_AND_literal10_passed"] for t in trials),
        worst_nonzero_relative_percent={axis: max((t["nonzero_relative_percent"][axis] for t in trials
            if t["nonzero_relative_percent"][axis] is not None), default=None) for axis in ("speed", "yaw", "height")},
        worst_body_P95_deg=max((t["metrics"]["body_p95_deg"] for t in trials if t["metrics"] is not None), default=None))
    report = dict(verified=True, summary=summary, trials=trials,
        original_absolute_floors_and_literal10_separate=True, failed_metrics_not_zeroed=True,
        source_pins={str(path): sha(path) for path in
                     (SCRIPT, directory / "trace.csv", directory / "manifest.json", directory / "completed.json")},
        acceptance_changed=False, policy_accepted=False)
    write_json(directory / "original_and_literal10_conditional_assessment.json", report)
    return report


def run(unit):
    pointer = read(Q / "prepared_capacity_pointer.json")
    production = Path(pointer["planned_production_output"])
    final = production / "final_checkpoint_verification_readonly_20261024.json"
    while not final.exists():
        state = owner(unit)
        assert state["MainPID"] != "0", state
        time.sleep(15.)
    import joint_pd_source3001_conditional_yaw_runtime as runtime
    import diagnose_motion025_full_height as original
    from auto_train_mine import write_json
    from joint_pd_continuation import sha
    proof = read(final)
    assert proof["verified"] and proof["actual_ppo_updates"] == 3201 and proof["all17_Adam_step"] == 64020
    checkpoint = Path(proof["checkpoint"])
    runtime.require_source(checkpoint)
    output = Q / "review3201"
    output.mkdir(exist_ok=False)
    baseline = assess(Path(read(Q / "native_frozen_qualification.json")["same51_replay"]))
    neighbours = [dict(name=f"heldout_h{height}_v{speed}_w{yaw}", speed=speed, yaw=yaw, height=height)
                  for height in (.29, .335, .37) for speed in (-.05, 0., .05) for yaw in (-.075, -.05, .05, .075)]
    focuses = [dict(name="nominal_left0075", speed=0., yaw=.075, height=.3),
               dict(name="high_left0075", speed=0., yaw=.075, height=.37534645531),
               dict(name="high_right0075", speed=0., yaw=-.075, height=.37534645531)]
    reports = {}
    for name, cases, seeds, seconds in (("same51", original.cases(), [483, 484, 485], 20.),
        ("heldout_neighbours108", neighbours, [883, 884, 885], 20.),
        ("focused_sustained120", focuses, [983, 984, 985], 120.)):
        destination, commands = output / name, output / (name + "_commands.json")
        write_json(commands, cases)
        argv = runtime.replay_command(checkpoint, destination)
        argv[argv.index("--seconds") + 1] = str(seconds)
        index = argv.index("--seeds") + 1
        argv[index:index + 3] = list(map(str, seeds))
        argv += ["--command-probe-json", str(commands)]
        write_json(output / (name + "_launch.json"), dict(command=argv, checkpoint_sha256=sha(checkpoint)))
        with (output / (name + ".log")).open("x") as stream:
            subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT, check=True)
        reports[name] = assess(destination)
    before = {(t["case"]["name"], t["seed"]): t for t in baseline["trials"]}
    after = {(t["case"]["name"], t["seed"]): t for t in reports["same51"]["trials"]}
    assert before.keys() == after.keys()
    paired = []
    for key in before:
        a, b = before[key], after[key]
        paired.append(dict(case=a["case"], seed=a["seed"], before=a, after=b,
            historical3001_original_failed=not a["original_passed"],
            yaw_MAE_change_rad_s=b["metrics"]["yaw_mae_rad_s"] - a["metrics"]["yaw_mae_rad_s"] if b["metrics"] is not None else None))
    decision = dict(complete=True, observed_at=datetime.now().astimezone().isoformat(),
        source_actual_updates=3001, candidate_actual_updates=3201,
        candidate_checkpoint=str(checkpoint), candidate_checkpoint_sha256=sha(checkpoint),
        source51=baseline["summary"], candidate_summaries={name: report["summary"] for name, report in reports.items()},
        original51_pairs=paired, current_models_preserved=True,
        select_continuation_or_change_reward_only_after_per_condition_agent_review=True,
        stage025_does_not_establish_full_plus_minus08_acceptance=True,
        standing_height_speed_priority_retained=True, policy_accepted=False,
        hardware_enabled=False, shutdown=False, goal_remains_active=True)
    write_json(output / "complete_conditional_comparison_and_decision_required.json", decision)
    print(json.dumps({key: decision[key] for key in ("complete", "source51", "candidate_summaries")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit", required=True)
    args = parser.parse_args()
    run(args.unit)
