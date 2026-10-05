"""Analyze MuJoCo sim2sim traces with phase-level tracking metrics."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED = {
    "time", "phase", "cmd_x_target", "cmd_height", "vel_x_heading", "base_height",
    "height_error_mm", "pitch_deg", "roll_deg", "left_theta0_deg", "right_theta0_deg",
}


def numbers(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def analyze(trace, tail_seconds=2.0):
    path = Path(trace).expanduser().resolve()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    times = numbers(rows, "time")
    if len(times) < 2:
        dt = tail_seconds
    else:
        dt = float(np.median(np.diff(times)))
    tail_count = max(1, round(tail_seconds / dt))
    report = {"trace": str(path), "samples": len(rows), "duration_s": float(times[-1]),
              "sample_dt_s": dt, "tail_window_s": tail_seconds, "phases": []}
    phase_ids = sorted({int(float(row["phase"])) for row in rows})
    for phase in phase_ids:
        phase_rows = [row for row in rows if int(float(row["phase"])) == phase]
        tail = phase_rows[-tail_count:]
        speed_target = numbers(tail, "cmd_x_target")
        speed = numbers(tail, "vel_x_heading")
        height_error = numbers(tail, "height_error_mm")
        left = numbers(tail, "left_theta0_deg")
        right = numbers(tail, "right_theta0_deg")
        pitch = numbers(tail, "pitch_deg")
        roll = numbers(tail, "roll_deg")
        angle_diff = left - right
        report["phases"].append({
            "phase": phase,
            "target_speed_m_s": float(speed_target[-1]),
            "target_height_m": float(numbers(tail, "cmd_height")[-1]),
            "samples": len(tail),
            "phase_complete": len(phase_rows) * dt >= tail_count * dt,
            "speed_mean_m_s": float(np.mean(speed)),
            "speed_mae_m_s": float(np.mean(np.abs(speed - speed_target))),
            "speed_max_abs_error_m_s": float(np.max(np.abs(speed - speed_target))),
            "speed_overshoot_m_s": float(np.max(np.maximum(np.abs(speed) - np.abs(speed_target), 0))),
            "height_mean_m": float(np.mean(numbers(tail, "base_height"))),
            "height_mae_mm": float(np.mean(np.abs(height_error))),
            "height_max_abs_error_mm": float(np.max(np.abs(height_error))),
            "left_leg_angle_mean_deg": float(np.mean(left)),
            "right_leg_angle_mean_deg": float(np.mean(right)),
            "left_leg_angle_peak_abs_deg": float(np.max(np.abs(left))),
            "right_leg_angle_peak_abs_deg": float(np.max(np.abs(right))),
            "leg_angle_difference_mean_deg": float(np.mean(angle_diff)),
            "leg_angle_difference_mae_deg": float(np.mean(np.abs(angle_diff))),
            "leg_angle_difference_peak_abs_deg": float(np.max(np.abs(angle_diff))),
            "pitch_mean_deg": float(np.mean(pitch)),
            "pitch_peak_abs_deg": float(np.max(np.abs(pitch))),
            "roll_mean_deg": float(np.mean(roll)),
            "roll_peak_abs_deg": float(np.max(np.abs(roll))),
        })
    report["overall"] = {
        "speed_mae_m_s_mean": float(np.mean([item["speed_mae_m_s"] for item in report["phases"]])),
        "height_mae_mm_max": float(max(item["height_mae_mm"] for item in report["phases"])),
        "leg_angle_difference_peak_abs_deg_max": float(max(item["leg_angle_difference_peak_abs_deg"] for item in report["phases"])),
        "pitch_peak_abs_deg_max": float(max(item["pitch_peak_abs_deg"] for item in report["phases"])),
        "roll_peak_abs_deg_max": float(max(item["roll_peak_abs_deg"] for item in report["phases"])),
    }
    return report, rows


def write_outputs(report, rows, output):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    fields = list(report["phases"][0])
    with (output / "phase_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["phases"])
    time = numbers(rows, "time")
    phase = np.asarray([int(float(row["phase"])) for row in rows])
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    plots = [
        (("cmd_x_target", "vel_x_heading"), "Speed (m/s)"),
        (("left_theta0_deg", "right_theta0_deg"), "Virtual leg angle (deg)"),
        (("pitch_deg", "roll_deg"), "Body attitude (deg)"),
        (("cmd_height", "base_height"), "Height (m)"),
    ]
    for axis, (names, ylabel) in zip(axes, plots):
        for name in names:
            axis.plot(time, numbers(rows, name), label=name, linewidth=1)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=.3)
        axis.legend(loc="best")
    for boundary in np.where(np.diff(phase) != 0)[0]:
        for axis in axes:
            axis.axvline(time[boundary], color="0.5", linestyle="--", linewidth=.6)
    axes[-1].set_xlabel("Simulation time (s)")
    fig.suptitle("Isaac Lab policy in MuJoCo: tracking and posture")
    fig.tight_layout()
    fig.savefig(output / "analysis.png", dpi=160)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tail_seconds <= 0:
        parser.error("--tail-seconds must be positive")
    report, rows = analyze(args.trace, args.tail_seconds)
    output = args.output or args.trace.parent / "analysis"
    write_outputs(report, rows, output)
    print(json.dumps(report, indent=2))
    print(f"Artifacts: {Path(output).resolve()}")


if __name__ == "__main__":
    main()
