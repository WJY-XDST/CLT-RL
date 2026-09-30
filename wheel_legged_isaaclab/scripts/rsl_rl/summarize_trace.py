"""Summarize continuous command phases from play.py CSV traces (no simulator needed)."""

import argparse
import csv
import json
import math
import statistics
from itertools import groupby
from pathlib import Path


def summarize(path: Path, settle_seconds: float) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    if not rows:
        raise ValueError(f"Empty trace: {path}")
    if any(not math.isfinite(value) for row in rows for value in row.values()):
        raise ValueError(f"Non-finite trace data: {path}")
    required = {"terminated", "time_out", "cmd_x_target", "height_cmd"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Trace lacks command/reset diagnostics: {path}")
    if len({row["env_id"] for row in rows}) != 1:
        raise ValueError("Each input trace must describe one environment.")

    result = {
        "trace": str(path),
        "samples": len(rows),
        "settle_seconds": settle_seconds,
        "terminated": sum(int(row["terminated"]) for row in rows),
        "time_out": sum(int(row["time_out"]) for row in rows),
        "phases": [],
    }
    commands = lambda row: (round(row["cmd_x_target"], 6), round(row["height_cmd"], 6))
    for (speed, height), group in groupby(rows, key=commands):
        phase = list(group)
        settled = [row for row in phase if row["sim_time_s"] - phase[0]["sim_time_s"] >= settle_seconds]
        report = {
            "target_speed": speed,
            "target_height": height,
            "start_time": phase[0]["sim_time_s"],
            "samples": len(phase),
            "settled_samples": len(settled),
            "terminated": sum(int(row["terminated"]) for row in phase),
            "time_out": sum(int(row["time_out"]) for row in phase),
            "minimum_height_all": min(row["base_height"] for row in phase),
        }
        if settled:
            mean = lambda key: statistics.mean(row[key] for row in settled)
            report.update(
                speed_mean=mean("vel_x_heading"),
                speed_mae=statistics.mean(abs(row["vel_x_heading"] - speed) for row in settled),
                horizontal_speed_mean=statistics.mean(
                    math.hypot(row["vel_x_heading"], row["vel_y_heading"]) for row in settled
                ),
                height_mean=mean("base_height"),
                height_mae=statistics.mean(abs(row["base_height"] - height) for row in settled),
                pitch_abs_max_deg=max(abs(row["pitch_est_rad"]) for row in settled) * 180.0 / math.pi,
                roll_abs_max_deg=max(abs(row["roll_est_rad"]) for row in settled) * 180.0 / math.pi,
                leg_angle_difference_mean_deg=statistics.mean(
                    abs(row["theta_left"] - row["theta_right"]) for row in settled
                ) * 180.0 / math.pi,
                left_target_mean=mean("target_length_left"),
                right_target_mean=mean("target_length_right"),
                target_difference_mean=statistics.mean(
                    abs(row["target_length_left"] - row["target_length_right"]) for row in settled
                ),
            )
        result["phases"].append(report)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--settle-seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.settle_seconds < 0.0:
        parser.error("--settle-seconds must be non-negative")
    text = json.dumps([summarize(path, args.settle_seconds) for path in args.traces], indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
