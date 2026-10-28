"""Pair all completed frozen D1/D1.5 scopes; do not select or train a policy.

The owned diagnostic has already applied the unchanged qualified original raw
reader to each new replay. This collector preserves those original flags and
adds literal nonzero10% accounting, full configuration and initial-input pairs.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "mine_model/results/joint_pd_frozen_C400_7401_D150_diagnostic_20261024"
DESTINATION = OUTPUT / "all318_D1_D150_paired_readonly.json"
GATE = ROOT / "mine_model/config/tracking_acceptance.json"
GATE_SHA = "08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d"
CHECKPOINT_SHA = "7d82cee7bc43b9b13ea22016d844b72f63e3df4ccdfe021e526907011746b688"
CHANNELS = (("speed_mae_m_s", "speed"), ("yaw_mae_rad_s", "yaw"), ("height_mae_m", "height"))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def differences(before, after, path=""):
    if isinstance(before, dict) and isinstance(after, dict):
        return [item for key in before.keys() | after.keys()
                for item in differences(before.get(key), after.get(key), f"{path}.{key}" if path else key)]
    return [] if before == after else [(path, before, after)]


def classify(row):
    original = bool(row["stage_constraints_passed"])
    if row["metrics"] is None:
        if original or row["survived"]:
            raise ValueError("Preserve failed first-episode undefined metrics")
        return dict(original=False, strict=False, relative_percent=None, failed_nonzero_relative_metrics=None)
    relative = {metric: 100. * row["metrics"][metric] / abs(row["case"][target])
                if row["case"][target] else None for metric, target in CHANNELS}
    failed = [metric for metric, value in relative.items() if value is not None and not value < 10.]
    return dict(original=original, strict=original and not failed,
                relative_percent=relative, failed_nonzero_relative_metrics=failed)


def summary(rows):
    classified = [classify(row) for row in rows]
    yaw_rows = [(row, flag) for row, flag in zip(rows, classified) if row["case"]["yaw"]]
    failed = Counter(metric for flag in classified for metric in flag["failed_nonzero_relative_metrics"] or [])
    return dict(
        trials=len(rows), survivors=sum(row["survived"] for row in rows),
        undefined_failed_metrics=sum(row["metrics"] is None for row in rows),
        original_passed=sum(flag["original"] for flag in classified),
        original_AND_literal10_passed=sum(flag["strict"] for flag in classified),
        nonzero_relative_failed_metric_counts=dict(failed),
        nonzero_yaw_trials=len(yaw_rows),
        nonzero_yaw_strict10_passed=sum(flag["relative_percent"] is not None
                                     and flag["relative_percent"]["yaw_mae_rad_s"] < 10.
                                     for _, flag in yaw_rows),
        nonzero_yaw_mean_MAE_rad_s=float(np.mean([row["metrics"]["yaw_mae_rad_s"]
            for row, _ in yaw_rows if row["metrics"] is not None])) if yaw_rows else None,
        worst_nonzero_relative_percent={metric: max((flag["relative_percent"][metric]
            for flag in classified if flag["relative_percent"] is not None
            and flag["relative_percent"][metric] is not None), default=None) for metric, _ in CHANNELS},
        worst_body_P95_deg=max((row["metrics"]["body_p95_deg"] for row in rows
                               if row["metrics"] is not None), default=None),
    )


def audit():
    if DESTINATION.exists() or sha(GATE) != GATE_SHA:
        raise ValueError("Preserve immutable reports and the original10% gate")
    proof, complete, baseline = [read(OUTPUT / name) for name in
        ("CPU_preflight.json", "completed.json", "baseline100_native_qualification.json")]
    if (not proof["qualified_CPU_only"] or not complete["completed"]
            or proof["checkpoint_sha256"] != complete["checkpoint_sha256"]):
        raise ValueError("Require the completed owned frozen diagnostic")
    if (proof["checkpoint_sha256"] != CHECKPOINT_SHA or complete["checkpoint_sha256"] != CHECKPOINT_SHA
            or proof["source_actual_updates"] != 7401 or proof["all17_Adam_step"] != 148020
            or complete["PPO_updates"] != 0 or complete["probe_trials"] != 318
            or not baseline["qualified"] or not baseline["original51_initial27_inputs_and_means_bitwise_equal"]
            or not baseline["all_original_flags_identical"]
            or baseline["max_original_metric_reproduction_error"] != 0.
            or baseline["performance_gates_relaxed"]):
        raise ValueError("Require unchanged source state and exact fresh baseline reproduction")
    pins = {str(path): sha(path) for path in (
        Path(__file__).resolve(), GATE, OUTPUT / "CPU_preflight.json", OUTPUT / "completed.json",
        OUTPUT / "baseline100_native_qualification.json",
    )}
    pins.update(proof["source_pins"])
    if sha(proof["checkpoint"]) != CHECKPOINT_SHA:
        raise ValueError("Preserve the complete frozen NN/optimizer/std checkpoint")
    grouped = {"D1": [], "D150": []}
    scopes, paired, rescues, regressions = [], [], [], []
    for scope in proof["scopes"]:
        source, target = Path(scope["fixture"]), OUTPUT / ("damping150_" + scope["name"])
        original, changed = read(scope["original_report"]), read(target / "metrics_diagnostic_only.json")
        old_manifest, new_manifest = read(source / "manifest.json"), read(target / "manifest.json")
        observer, finished = read(target / "frozen_wheel_damping_native_observation.json"), read(target / "completed.json")
        if (not scope["original_numeric_metrics_flags_reproduced_exactly"]
                or original["checkpoint_sha256"] != CHECKPOINT_SHA
                or changed["checkpoint_sha256"] != CHECKPOINT_SHA
                or not finished["completed"] or finished["samples"] != scope["count"] * 1000
                or not changed["frozen_controller_diagnostic_only"] or changed["declared_wheel_damping"] != 1.5
                or changed["baseline_source_fixture"] != str(source)
                or changed["criteria"] != original["criteria"] or changed["criteria"]["relative_error"] != .1):
            raise ValueError("Require original scoring and complete unmodified-source frozen scopes")
        for field in ("cases", "seeds", "policy_dt", "seconds_per_case", "sampled_policy_actions",
                      "leg_control_mode", "checkpoint_sha256"):
            if field in old_manifest or field in new_manifest:
                if old_manifest.get(field) != new_manifest.get(field):
                    raise ValueError("Frozen command/seed/clock/action contract changed: " + field)
        delta = differences(read(source / "resolved_env.json"), read(target / "resolved_env.json"))
        if set(delta) != {("wheel_damping", 1., 1.5), ("robot.actuators.wheels.damping", 1., 1.5)}:
            raise ValueError("Admit exactly the declared synchronized wheel-gain change")
        if (observer["checkpoint_sha256"] != CHECKPOINT_SHA or observer["native_environments"] != scope["count"]
                or observer["actual_wheel_damping_min"] != 1.5 or observer["actual_wheel_damping_max"] != 1.5
                or observer["actual_USD_drive_damping"] != [1.5, 1.5]
                or not observer["all_other_declared_control_fields_retained"] or not observer["RNG_states_unchanged"]
                or observer["observer_changes_actions_or_rewards"] or observer["optimizer_loaded_into_runner"]
                or observer["PPO_updates_performed"] != 0):
            raise ValueError("Require the actual gain and unchanged native physical/force contract")
        with np.load(source / "policy_inputs.npz", allow_pickle=False) as before, np.load(
                target / "policy_inputs.npz", allow_pickle=False) as after:
            if (not np.array_equal(before["observations"][0], after["observations"][0])
                    or not np.array_equal(before["means"][0], after["means"][0])):
                raise ValueError("Require paired initial27 observations and original6-head means")
        if len(original["trials"]) != scope["count"] or len(changed["trials"]) != scope["count"]:
            raise ValueError("Do not omit failed cases")
        for before, after in zip(original["trials"], changed["trials"]):
            if before["case"] != after["case"] or before["seed"] != after["seed"]:
                raise ValueError("Require matched original cases and seeds")
            for field in ("speed_mae_m_s", "yaw_mae_rad_s", "height_mae_m", "leg_angle_mae_rad", "body_p95_deg"):
                if before["limits"][field] != after["limits"][field]:
                    raise ValueError("Preserve the fixed original floor-aware limits")
            old, new = classify(before), classify(after)
            record = dict(scope=scope["name"], case=before["case"], seed=before["seed"],
                          before=before, after=after, before_classification=old, after_classification=new)
            paired.append(record)
            if not old["strict"] and new["strict"]:
                rescues.append(record)
            if old["strict"] and not new["strict"]:
                regressions.append(record)
        grouped["D1"].extend(original["trials"])
        grouped["D150"].extend(changed["trials"])
        scopes.append(dict(scope=scope["name"], count=scope["count"],
                           original_27_inputs_and_6_means_bitwise_paired=True,
                           entire_resolved_configuration_delta=delta,
                           D1=summary(original["trials"]), D150=summary(changed["trials"])))
        files = [Path(scope["original_report"])]
        files += [directory / name for directory in (source, target) for name in
                  ("manifest.json", "completed.json", "resolved_env.json", "policy_inputs.npz", "trace.csv")]
        files += [target / name for name in
                  ("metrics_diagnostic_only.json", "frozen_wheel_damping_native_observation.json")]
        pins.update({str(path): sha(path) for path in files})
    if len(paired) != 318 or any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError("Require every paired trial and unchanged immutable evidence")
    result = dict(
        observed_at=datetime.now().astimezone().isoformat(), verified=True,
        checkpoint_sha256=CHECKPOINT_SHA, actual_ppo_updates=7401, all17_Adam_step=148020,
        matched_trials=318, summaries={name: summary(rows) for name, rows in grouped.items()},
        scopes=scopes, literal_rescues=rescues, literal_regressions=regressions,
        paired_trials=paired, source_pins=pins,
        all_original_windows_and_failed_None_metrics_preserved=True,
        reference_dependent_leg_and_wheel_limits_not_substituted=True,
        original_raw_reader_CPU_qualification_and_native_reports_retained=True,
        initial_observation_pairs=318, complete_configuration_delta_only_wheel_D1_to_D150=True,
        simulator_started=False, optimizer_executed=False, training_modified=False, acceptance_changed=False,
        can_select_gain_for_training=False, can_select_source=False, policy_accepted=False,
        limitation="This318-case fixed-command diagnostic is not full735, static, motion810, signed-height or sustained120 acceptance. It cannot promote a checkpoint or start PPO.",
    )
    with DESTINATION.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(DESTINATION), matched_trials=318,
                          summaries=result["summaries"], rescues=len(rescues), regressions=len(regressions))))


if __name__ == "__main__":
    audit()
