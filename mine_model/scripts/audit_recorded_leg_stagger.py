"""Read every saved replay tail to examine leg-angle asymmetry.

This is an observational audit. It neither evaluates the policy nor changes a
controller, reward, command distribution or acceptance decision. The fore-aft
quantity is reconstructed from measured joint FK, not a logged body position.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    return statistics.mean(values)


def summarize(rows):
    rotating = [r for r in rows if r["case"]["yaw"] != 0]
    return dict(trials=len(rows),
        mean_abs_leg_angle_difference_deg=mean(abs(r["leg_angle_difference_deg"]) for r in rows),
        mean_abs_derived_left_minus_right_foreaft_mm=mean(abs(r["derived_left_minus_right_foreaft_mm"]) for r in rows),
        signed_foreaft_mm_range=[min(r["derived_left_minus_right_foreaft_mm"] for r in rows),
                                max(r["derived_left_minus_right_foreaft_mm"] for r in rows)],
        mean_signed_yaw_bias_rad_s=mean(r["yaw_signed_bias_rad_s"] for r in rows),
        mean_yaw_mae_rad_s=mean(r["yaw_mae_rad_s"] for r in rows),
        rotating_yaw_beyond10_count=sum(r["yaw_relative_percent"] > 10 for r in rotating),
        original_and_literal10_passed=sum(r["original_AND_literal10_passed"] for r in rows),
        survived=sum(r["survived"] for r in rows))


def audit(directory):
    directory = Path(directory).resolve()
    mf = directory / "manifest.json"
    af = directory / "original_and_literal10_conditional_assessment.json"
    tf = directory / "trace.csv"
    manifest, assessment = read(mf), read(af)
    pins = {str(f): sha(f) for f in (mf, af, tf)}
    assert not manifest["weights_updated"] and not manifest["policy_input_remapping"]
    assert not manifest["sampled_policy_actions"]
    assert manifest["leg_control_mode"] == "implicit_joint_position"
    trials = {(r["case"]["name"], r["seed"]): r for r in assessment["trials"]}
    columns = ["left_angle_rad", "right_angle_rad", "left_length_m", "right_length_m",
               "yaw_rate_rad_s", "speed_m_s", "pitch_rad", "roll_rad"]
    tails = {}
    with tf.open() as stream:
        reader = csv.reader(stream)
        header = next(reader)
        at = {name: header.index(name) for name in
              ["time_s", "case_id", "seed", "episode", "terminated", *columns]}
        for raw in reader:
            if float(raw[at["time_s"]]) < manifest["seconds_per_case"] - 3 - 1e-8:
                continue
            case = manifest["cases"][int(raw[at["case_id"]])]
            key = case["name"], int(raw[at["seed"]])
            values = [float(raw[at[name]]) for name in columns]
            if not all(math.isfinite(v) for v in values):
                raise ValueError("Non-finite FK telemetry")
            tails.setdefault(key, []).append(values)
    rows = []
    for key, trial in trials.items():
        samples = tails[key]
        averages = [mean(v[i] for v in samples) for i in range(len(columns))]
        expected = round(3 / manifest["policy_dt"]) + 1
        assert len(samples) == expected, (key, len(samples), expected)
        angles = [v[0] - v[1] for v in samples]
        # five_bar_state documents theta=atan2(-X,-Z). Equal hip X is
        # independently checked in the composed-USD audit supplied to main().
        # Hence the wheel-end X difference is -L_left*sin(theta_left)
        # +L_right*sin(theta_right), before any body-to-world rotation.
        foreaft = [-v[2] * math.sin(v[0]) + v[3] * math.sin(v[1]) for v in samples]
        rows.append(dict(case=trial["case"], seed=trial["seed"], samples=len(samples),
            survived=trial["survived"], original_passed=trial["original_passed"],
            original_AND_literal10_passed=trial["original_AND_literal10_passed"],
            theta_left_mean_rad=averages[0], theta_right_mean_rad=averages[1],
            leg_angle_difference_deg=math.degrees(mean(angles)),
            derived_left_minus_right_foreaft_mm=1000 * mean(foreaft),
            yaw_signed_bias_rad_s=trial["yaw_signed_bias_rad_s"],
            yaw_mae_rad_s=trial["metrics"]["yaw_mae_rad_s"],
            yaw_relative_percent=trial["nonzero_relative_percent"]["yaw"],
            speed_mae_m_s=trial["metrics"]["speed_mae_m_s"],
            body_p95_deg=trial["metrics"]["body_p95_deg"]))
    assert len(rows) == len(trials) == len(manifest["cases"]) * len(manifest["seeds"])
    assert all(sha(path) == digest for path, digest in pins.items())
    return dict(directory=str(directory), checkpoint_sha256=manifest["checkpoint_sha256"],
        summary=summarize(rows),
        grouped={axis: [dict(value=value, **summarize([r for r in rows if r["case"][axis] == value]))
            for value in sorted({r["case"][axis] for r in rows})] for axis in ("height", "speed", "yaw")},
        all_trials=rows, source_pins=pins)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--replay", type=Path, action="append", required=True)
    cli.add_argument("--geometry-audit", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    assert not args.output.exists()
    geometry = read(args.geometry_audit)
    hips = [j for j in geometry["joints"] if j["name"] in ("LB_joint", "LL_joint", "RB_joint", "RL_joint")]
    assert len(hips) == 4 and all(abs(j["anchor_base_m"][0]) < 1e-7 for j in hips)
    assert abs(geometry["left_minus_right_wheel_origin_base_m"][0]) < 1e-7
    results = [audit(p) for p in args.replay]
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        results=results, geometry_audit=str(args.geometry_audit.resolve()),
        source_pins={str(args.geometry_audit.resolve()): sha(args.geometry_audit),
                     str(Path(__file__).resolve()): sha(__file__)},
        derived_foreaft_is_in_body_frame=True, actual_wheel_body_transforms_logged=False,
        no_geometry_required_foreaft_stagger_in_authored_pose=True,
        limitations=["An observational association is not a causal intervention.",
            "The derived FK displacement is not independently logged wheel-body translation.",
            "Authored left/right wheel body origins differ laterally; they are not tire contact centers.",
            "Different heights, velocities and policy outputs can confound cross-case associations.",
            "No reward or parameter is selected automatically from this audit."],
        optimizer_executed=False, simulation_started=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(verified=True, output=str(args.output),
        summaries=[dict(directory=r["directory"], **r["summary"]) for r in results]), ensure_ascii=False))


if __name__ == "__main__":
    main()
