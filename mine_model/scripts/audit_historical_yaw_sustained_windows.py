"""Use the original overlapping-window scorer on declared historical replays.

Keep the seven original metrics, floors and strict comparisons. The legacy
command-probe flag means this is a scoped diagnostic, never full acceptance.
Report literal nonzero10% separately from the original absolute floors.
"""
import argparse
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path

import numpy as np

import joint_pd_acceptance_10pct_20261017 as acceptance
import summarize_mine_trial as original
import plain27_sampler_metadata_adapter as adapter
from auto_train_mine import write_json
from joint_pd_continuation import sha


SCRIPT = Path(__file__).resolve()


def audit(directory):
    directory = Path(directory).resolve()
    paths = {name: directory / name for name in
             ("manifest.json", "completed.json", "trace.csv", "original_and_literal10_conditional_assessment.json")}
    manifest, completed, baseline = [json.loads(paths[name].read_text()) for name in
        ("manifest.json", "completed.json", "original_and_literal10_conditional_assessment.json")]
    assert baseline["verified"] and completed["completed"]
    assert not manifest["sampled_policy_actions"] and not manifest["weights_updated"]
    assert manifest["leg_control_mode"] == "implicit_joint_position" and manifest["policy_dt"] == .02
    assert not any(manifest.get(key) for key in ("action_bias_probes", "wheel_probe", "wheel_reference_limit_probe"))
    assert not manifest["policy_input_remapping"] and manifest["command_scale"] == 1.
    profile = acceptance.profile()
    criteria = profile["criteria"]
    duration, dt = manifest["seconds_per_case"], manifest["policy_dt"]
    assert duration >= 20. and criteria["relative_error"] == .1
    assert len(baseline["trials"]) == len(manifest["cases"]) * len(manifest["seeds"])
    converted = []
    for case in manifest["cases"]:
        rows = [row for row in baseline["trials"] if row["case"] == case]
        assert len(rows) == len(manifest["seeds"]) and {r["seed"] for r in rows} == set(manifest["seeds"])
        trials = [dict(seed=row["seed"], survived=row["survived"], metrics=copy.deepcopy(row["metrics"]),
            limits=copy.deepcopy(row["limits"]), passed=row["original_passed"], **copy.deepcopy(row["flags"])) for row in rows]
        converted.append(dict(case, trials=trials, survivors=sum(t["survived"] for t in trials)))
    report = dict(conditions=converted, acceptance_criteria=criteria,
                  trials=len(baseline["trials"]), survivors=baseline["summary"]["survivors"],
                  policy_accepted=False, diagnostic_command_manifest_not_full_acceptance=True)
    api = adapter.private_module(original.__file__, "_historical_original_overlapping_window_numeric_reader")
    proof = adapter.retarget(api, ("assess_stability",), strings={
        "assessment_stability.json": "historical_original7_overlap_windows_readonly.json"})
    # Only the input-report schema is adapted. The original metric calculation,
    # overlapping-window selection and limit comparisons remain the same AST.
    api.assess = lambda directory, report_name: copy.deepcopy(report)
    numeric_path = directory / "historical_original7_overlap_windows_readonly.json"
    assert not numeric_path.exists()
    windowed = api.assess_stability(directory, stable_seconds=duration - 17., window_seconds=3.)
    raw = np.genfromtxt(paths["trace.csv"], delimiter=",", names=True)
    assert len(raw) == completed["samples"] == len(manifest["cases"]) * len(manifest["seeds"]) * round(duration / dt)
    assert all(np.isfinite(raw[name]).all() for name in raw.dtype.names)
    starts = np.arange(17., duration - 3. + dt / 2., 1.)
    rows = []
    for cid, case in enumerate(windowed["conditions"]):
        for trial in case["trials"]:
            part = raw[(raw["case_id"] == cid) & (raw["seed"] == trial["seed"])]
            survived = not any(np.any(part[key]) for key in ("episode", "terminated", "timeout"))
            assert survived == trial["survived"]
            original_trial = next(row for row in baseline["trials"] if row["case"] == manifest["cases"][cid] and row["seed"] == trial["seed"])
            static = original_trial["actual_static_leg_commands"]
            static_ok = static is None or all(static[key] for key in
                ("height_posture_passed", "leg_command_passed", "stationary_drift_passed"))
            relative = {axis: 100. * trial["metrics"][field] / abs(case[axis])
                        if survived and case[axis] else None
                        for axis, field in (("speed", "speed_mae_m_s"), ("yaw", "yaw_mae_rad_s"), ("height", "height_mae_m"))}
            worst = None
            if survived:
                values = []
                for start in starts:
                    window = part[(part["time_s"] >= start - dt * .1) & (part["time_s"] < start + 3. - dt * .1)]
                    assert len(window) == round(3. / dt)
                    values.append(original.metrics_for_window(window, case, criteria)["yaw_mae_rad_s"])
                index = int(np.argmax(values))
                assert abs(values[index] - trial["metrics"]["yaw_mae_rad_s"]) < 1.e-12
                worst = dict(start_s=float(starts[index]), end_s=float(starts[index] + 3.), yaw_MAE_rad_s=values[index])
            floor_pass = bool(trial["passed"] and static_ok)
            literal = floor_pass and all(value < 10. for value in relative.values() if value is not None)
            rows.append(dict(case=manifest["cases"][cid], seed=trial["seed"], survived=survived,
                original_floor_aware_all_windows_passed=floor_pass,
                original_AND_literal10_all_windows_passed=literal,
                nonzero_relative_percent=relative, maximum_original_window_metrics=trial["metrics"],
                original_limits=trial["limits"], original_static_checks_retained=static_ok,
                failed_metrics=trial["failed_metrics"], failed_prerequisites=trial["failed_prerequisites"],
                worst_yaw_window=worst))
    pins = {str(path): sha(path) for path in paths.values()}
    pins.update({str(path): sha(path) for path in (SCRIPT, Path(original.__file__), numeric_path,
        Path(adapter.__file__), Path(profile["acceptance_file"]))})
    value = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        duration_s=duration, steady_interval_s=[17., duration], window_s=3., stride_s=1.,
        windows_per_trial=len(starts), all_original_metric_window_and_comparison_AST_retained=proof,
        complete_trace_survival_checked=True, trial_count=len(rows), survivors=sum(r["survived"] for r in rows),
        original_floor_aware_passed=sum(r["original_floor_aware_all_windows_passed"] for r in rows),
        original_AND_literal10_passed=sum(r["original_AND_literal10_all_windows_passed"] for r in rows),
        original_floor_aware_all_windows_passed=all(r["original_floor_aware_all_windows_passed"] for r in rows),
        original_AND_literal10_all_windows_passed=all(r["original_AND_literal10_all_windows_passed"] for r in rows),
        trials=rows, source_pins=pins, original_absolute_floors_and_literal10_reported_separately=True,
        scoped_replay_only_not_full_speed_acceptance=True, simulator_started=False,
        optimizer_executed=False, training_modified=False, acceptance_changed=False, policy_accepted=False)
    destination = directory / "historical_literal10_overlap_windows_readonly.json"
    assert not destination.exists()
    write_json(destination, value)
    return destination, value


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    args = cli.parse_args()
    path, value = audit(args.directory)
    print(json.dumps(dict(verified=True, path=str(path), trials=value["trial_count"],
        windows_per_trial=value["windows_per_trial"], original_floor_aware_passed=value["original_floor_aware_passed"],
        original_AND_literal10_passed=value["original_AND_literal10_passed"], policy_accepted=False)))
