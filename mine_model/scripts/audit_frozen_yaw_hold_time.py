"""Stream whole frozen replay grids to compare early and late yaw error.

All command values are diagnostic data. No policy, simulator or training
configuration is loaded or modified. Failed traces remain explicitly marked.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics


SCRIPT = Path(__file__).resolve()
WINDOWS = ((2., 5.), (7., 10.), (12., 15.), (17., 20.))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit(directory, label):
    files = [directory / name for name in ("manifest.json", "completed.json", "trace.csv")]
    manifest, complete = [json.loads(path.read_text()) for path in files[:2]]
    assert complete["completed"] and manifest["seconds_per_case"] >= 20.
    assert not manifest["weights_updated"] and not manifest["sampled_policy_actions"]
    assert not manifest.get("command_sequence")
    assert manifest["leg_control_mode"] == "implicit_joint_position"
    pins = {str(path): sha(path) for path in files}
    rows = {(index, seed): dict(case=case, seed=seed, survived=True, samples=[[] for _ in WINDOWS])
            for index, case in enumerate(manifest["cases"]) for seed in manifest["seeds"]}
    count = 0
    with files[-1].open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        indices = {name: header.index(name) for name in
                   ("time_s", "case_id", "seed", "episode", "terminated", "timeout", "yaw_rate_rad_s", "yaw_cmd_rad_s")}
        for values in reader:
            count += 1
            record = rows[(int(values[indices["case_id"]]), int(values[indices["seed"]]))]
            if any(float(values[indices[name]]) != 0. for name in ("episode", "terminated", "timeout")):
                record["survived"] = False
            time_s = float(values[indices["time_s"]])
            for window, (start, stop) in enumerate(WINDOWS):
                if start - 1.e-9 <= time_s <= stop + 1.e-9:
                    command = float(values[indices["yaw_cmd_rad_s"]])
                    assert abs(command - record["case"]["yaw"]) < 1.e-6
                    record["samples"][window].append(float(values[indices["yaw_rate_rad_s"]]) - command)
    assert count == complete["samples"]
    result = []
    for record in rows.values():
        metrics = []
        for (start, stop), samples in zip(WINDOWS, record.pop("samples")):
            assert len(samples) == round((stop - start) / manifest["policy_dt"]) + 1
            bias = statistics.mean(samples)
            mse = statistics.mean(value * value for value in samples)
            metrics.append(dict(interval_s=[start, stop], frames=len(samples),
                yaw_MAE_rad_s=statistics.mean(abs(value) for value in samples),
                yaw_signed_bias_rad_s=bias, yaw_MSE=mse, yaw_temporal_variance=max(0., mse - bias * bias)))
        result.append(dict(**record, windows=metrics,
            last_minus_first_yaw_MAE_rad_s=metrics[-1]["yaw_MAE_rad_s"] - metrics[0]["yaw_MAE_rad_s"],
            last_minus_first_yaw_signed_bias_rad_s=metrics[-1]["yaw_signed_bias_rad_s"] - metrics[0]["yaw_signed_bias_rad_s"]))
    summary = {}
    for name, selected in (("all_trials", result), ("nonzero_yaw", [row for row in result if row["case"]["yaw"]])):
        survivors = [row for row in selected if row["survived"]]
        assert survivors
        summary[name] = dict(trials=len(selected), complete_trace_survivors=len(survivors),
            survivor_mean_yaw_MAE_by_window_rad_s=[statistics.mean(row["windows"][i]["yaw_MAE_rad_s"] for row in survivors) for i in range(len(WINDOWS))],
            survivor_max_abs_late_minus_early_bias_rad_s=max(abs(row["last_minus_first_yaw_signed_bias_rad_s"]) for row in survivors),
            survivor_mean_abs_late_minus_early_bias_rad_s=statistics.mean(abs(row["last_minus_first_yaw_signed_bias_rad_s"]) for row in survivors))
    assert all(sha(path) == value for path, value in pins.items())
    return dict(label=label, directory=str(directory), checkpoint_sha256=manifest["checkpoint_sha256"],
        intervals_s=WINDOWS, summary=summary, trials=result), pins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert len(args.replays) == len(args.labels) == len(set(args.labels))
    assert not args.output.exists()
    reports, pins = [], {str(SCRIPT): sha(SCRIPT)}
    for directory, label in zip(args.replays, args.labels):
        report, source = audit(directory.resolve(), label)
        reports.append(report)
        pins.update(source)
    assert all(sha(path) == value for path, value in pins.items())
    args.output.write_text(json.dumps(dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        reports=reports, source_pins=pins, memory_usage_not_proportional_to_full_trace_size=True,
        simulator_started=False, optimizer_executed=False, training_modified=False, acceptance_changed=False,
        policy_accepted=False, limitations=["Time association is diagnostic; it is not a causal command-duration intervention.",
            "Frozen deterministic traces do not reproduce Gaussian training trajectories or their changing command distribution.",
            "These early/late comparisons do not replace the full-duration survival and original sustained acceptance checks."]), indent=2) + "\n")
    print(json.dumps(dict(verified=True, output=str(args.output), summaries={row["label"]: row["summary"] for row in reports})))
