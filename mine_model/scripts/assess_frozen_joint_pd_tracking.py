"""Generic frozen replay scoring, with no trial-specific launcher.

The numeric assess() body is preserved from the existing reader. Legacy report
filenames remain for compatibility; no condition selects commands or controls.
"""
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


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
