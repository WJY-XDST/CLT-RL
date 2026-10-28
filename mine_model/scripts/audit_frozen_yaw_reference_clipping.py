"""Read all frozen cases: distinguish policy reference bias from drive gap.

Keep the original seven metrics and strict ten-percent calculation unchanged.
No replay, policy update, checkpoint edit, parameter selection or acceptance.
"""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np


def audit(directory):
    import auto_train_mine
    import review_historical_source3001_conditional_yaw200 as original
    from joint_pd_continuation import sha

    directory = directory.resolve()
    paths = [directory / name for name in ("manifest.json", "resolved_env.json", "completed.json",
                                          "trace.csv", "policy_inputs.npz")]
    pins = {str(p): sha(p) for p in paths}
    manifest, cfg = [json.loads(p.read_text()) for p in paths[:2]]
    assert not manifest["weights_updated"] and not manifest["sampled_policy_actions"]
    assert manifest["action_bias_probes"] is None
    assert not manifest["wheel_probe"] and not manifest["policy_input_remapping"]
    captured = []
    saved_writer = auto_train_mine.write_json
    try:
        # The original function imports this writer locally. Redirect only its
        # output, in this CPU process, so historical result directories stay
        # unchanged; every original numerical statement still executes.
        auto_train_mine.write_json = lambda path, value: captured.append((path, value))
        assessment = original.assess(directory)
    finally:
        auto_train_mine.write_json = saved_writer
    assert len(captured) == 1 and captured[0][1] is assessment
    with paths[3].open() as stream:
        header = next(csv.reader(stream))
    names = ["time_s", "case_id", "seed", "left_wheel_ref_rad_s", "right_wheel_ref_rad_s",
             "left_wheel_rad_s", "right_wheel_rad_s", "yaw_rate_rad_s",
             "left_length_m", "right_length_m", "left_angle_rad", "right_angle_rad"]
    columns = [header.index(k) for k in names]
    count = len(manifest["cases"]) * len(manifest["seeds"])
    frames = round(manifest["seconds_per_case"] / manifest["policy_dt"])
    trace = np.loadtxt(paths[3], delimiter=",", skiprows=1, usecols=columns).reshape(frames, count, len(names))
    selected = trace[:, 0, 0] >= manifest["seconds_per_case"] - 3.
    tail = trace[selected]
    assert tail.shape[0] == round(3 / manifest["policy_dt"]) + 1
    with np.load(paths[4]) as recorded:
        means = recorded["means"][selected]
    assert means.shape == (*tail.shape[:2], 6)
    ratio = cfg["wheel_radius"] / cfg["wheel_track_width"]
    ref = (tail[:, :, 4] - tail[:, :, 3]) * ratio
    wheel = (tail[:, :, 6] - tail[:, :, 5]) * ratio
    reconstructed = (np.clip(means[:, :, 5].astype(np.float64), -1, 1)
                     - np.clip(means[:, :, 2].astype(np.float64), -1, 1)) * cfg["action_scale_vel"] * ratio
    assert np.max(np.abs(ref - reconstructed)) < 2.e-6
    indexed = {(r["case"]["name"], r["seed"]): r for r in assessment["trials"]}
    rows = []
    for ic, case in enumerate(manifest["cases"]):
        for iseed, seed in enumerate(manifest["seeds"]):
            i = ic * len(manifest["seeds"]) + iseed
            trial = indexed[(case["name"], seed)]
            bias = trial["yaw_signed_bias_rad_s"]
            components = [case["yaw"] - ref[:, i], ref[:, i] - wheel[:, i], wheel[:, i] - tail[:, i, 7]]
            closure = np.max(np.abs(sum(components) - (case["yaw"] - tail[:, i, 7])))
            assert closure < 1.e-12
            clips = (np.abs(means[:, i]) > 1).mean(0)
            rows.append(dict(case=case, seed=seed, survived=trial["survived"],
                original_passed=trial["original_passed"],
                original_AND_literal10_passed=trial["original_AND_literal10_passed"],
                nonzero_relative_percent=trial["nonzero_relative_percent"],
                original_metrics=trial["metrics"],
                strict_yaw_failed=case["yaw"] != 0 and trial["nonzero_relative_percent"]["yaw"] >= 10.,
                mean_raw_actor_actions=means[:, i].mean(0, dtype=np.float64).tolist(),
                per_head_unclamped_mean_beyond_action_envelope_fraction=clips.tolist(),
                wheel_heads_ever_beyond_envelope=bool(np.any(clips[[2, 5]] > 0)),
                leg_heads_persistently_beyond_envelope=bool(np.any(clips[[0, 1, 3, 4]] > .95)),
                body_yaw_signed_bias_rad_s=bias,
                signed_command_minus_reference_rad_s=float(components[0].mean()),
                signed_reference_minus_measured_wheels_rad_s=float(components[1].mean()),
                signed_measured_wheels_minus_body_rad_s=float(components[2].mean()),
                derived_left_minus_right_foreaft_mm=float(1000 * (
                    -tail[:, i, 8] * np.sin(tail[:, i, 10]) + tail[:, i, 9] * np.sin(tail[:, i, 11])).mean()),
                signed_component_closure_max_error_rad_s=float(closure)))
    assert all(sha(p) == digest for p, digest in pins.items())
    return dict(directory=str(directory), checkpoint_sha256=manifest["checkpoint_sha256"],
        wheel_damping=cfg["wheel_damping"], summary=assessment["summary"], all_trials=rows,
        original_assessor_output_redirected_only=True, original_numerical_assessor_unchanged=True,
        actor_action_to_recorded_reference_max_error_rad_s=float(np.max(np.abs(ref - reconstructed))),
        source_pins=pins)


def main():
    from joint_pd_continuation import sha
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", type=Path, action="append", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert not args.output.exists()
    results = [audit(p) for p in args.replay]
    rows = [r for result in results for r in result["all_trials"]]
    failed = [r for r in rows if r["strict_yaw_failed"]]
    summary = dict(trials=len(rows), survivors=sum(r["survived"] for r in rows),
        original_passed=sum(r["original_passed"] for r in rows),
        original_AND_literal10_passed=sum(r["original_AND_literal10_passed"] for r in rows),
        rotating_trials=sum(r["case"]["yaw"] != 0 for r in rows), strict_yaw_failed=len(failed),
        yaw_failed_wheel_heads_ever_beyond_envelope=sum(r["wheel_heads_ever_beyond_envelope"] for r in failed),
        yaw_failed_leg_heads_persistently_beyond_envelope=sum(r["leg_heads_persistently_beyond_envelope"] for r in failed))
    report = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(), summary=summary,
        results=results, strict_yaw_failed_trials=failed,
        source_pins={str(Path(__file__).resolve()): sha(__file__)},
        limitations=["Rolling yaw proxies are not measurements of contact force or lateral slip.",
            "A clipped joint target envelope is distinct from a physical mechanical stop.",
            "Association with clipping or a leg offset does not establish a causal training fix.",
            "No command case selects different control, reward or sampling logic."],
        historical_result_files_modified=False, optimizer_executed=False, simulation_started=False,
        training_modified=False, acceptance_changed=False, policy_accepted=False)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(verified=True, output=str(args.output), summary=summary), ensure_ascii=False))


if __name__ == "__main__":
    main()
