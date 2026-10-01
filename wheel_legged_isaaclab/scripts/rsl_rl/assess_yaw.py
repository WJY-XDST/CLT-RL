"""Validate yaw-rate tracking, centered spins and retained travel/height/posture."""

import csv
import math
import statistics
from pathlib import Path

from assess_tracking import percentile


def yaw_sequence(extended=False):
    if extended:
        return [(0., 0., .18)] + [
            (speed, yaw, height) for height in (.16, .18, .20)
            for speed in (0., .4, -.4, .8, -.8) for yaw in (.15, -.15, .3, -.3, .5, -.5)
        ]
    return [(speed, yaw, .18) for speed, yaw in (
        (0, 0), (0, .3), (0, 0), (0, -.3), (0, 0),
        (.4, 0), (.4, .3), (.4, 0), (.4, -.3), (.4, 0),
        (-.4, 0), (-.4, .3), (-.4, 0), (-.4, -.3), (0, 0),
    )]


def assess_yaw(path, extended=False):
    expected = yaw_sequence(extended)
    with Path(path).open(newline="") as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    if len(rows) != len(expected) * 250:
        raise ValueError("Incomplete yaw evaluation matrix")
    if len({r["env_id"] for r in rows}) != 1:
        raise ValueError("Yaw evaluation requires one environment")
    for i, row in enumerate(rows):
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError("Nonfinite yaw trace")
        if row["step"] != i or abs(row["sim_time_s"] - i * .02) > 1e-5:
            raise ValueError("Missing or mistimed yaw sample")
        if any(row[k] not in (0, 1) for k in ("terminated", "time_out", "heading_hold")):
            raise ValueError("Invalid reset or heading flag")
    phases = []
    for index, (speed, yaw, height) in enumerate(expected):
        phase = rows[index * 250:(index + 1) * 250]
        for row in phase:
            if abs(row["cmd_x_target"] - speed) > 1e-5 or abs(row["height_cmd"] - height) > 1e-5:
                raise ValueError("Unexpected speed/height command")
            if yaw and (abs(row["cmd_yaw"] - yaw) > 1e-5 or row["heading_hold"]):
                raise ValueError("Turn command was overridden or missing")
            if not yaw and not row["heading_hold"]:
                raise ValueError("Zero-yaw phase must hold the reached heading")
        steady = phase[-100:]
        yaw_errors = [abs(r["yaw_rate_body"] - yaw) for r in steady]
        speed_errors = [abs(r["vel_x_heading"] - speed) if speed else
                        math.hypot(r["vel_x_heading"], r["vel_y_heading"]) for r in steady]
        body = [math.degrees(max(abs(r["pitch_est_rad"]), abs(r["roll_est_rad"]))) for r in steady]
        symmetry = [math.degrees(abs(r["theta_left"] - r["theta_right"])) for r in steady]
        wheel_center = [abs(.0675 * (r["wheel_vel_left"] + r["wheel_vel_right"]) / 2) for r in steady]
        ratios = {
            "yaw": percentile(yaw_errors) / (.05 * abs(yaw) if yaw else .03),
            "speed": percentile(speed_errors) / (.05 * abs(speed) if speed else .02),
            "height": percentile([abs(r["base_height"] - height) for r in steady]) / (.05 * height),
            "body": max(body) / 3., "symmetry": max(symmetry) / 3.,
            "spin_center": percentile(wheel_center) / .02 if not speed and yaw else 0.,
            "transient_body": max(math.degrees(max(abs(r["pitch_est_rad"]), abs(r["roll_est_rad"])))
                                  for r in (phase if index else steady)) / 10.,
        }
        resets = sum(int(r["terminated"]) + int(r["time_out"]) for r in phase)
        phases.append({"phase": index, "speed": speed, "yaw": yaw, "height": height,
                       "yaw_rate_mean": statistics.mean(r["yaw_rate_body"] for r in steady),
                       "yaw_mae": statistics.mean(yaw_errors), "speed_mae": statistics.mean(speed_errors),
                       "left_wheel_mean": statistics.mean(r["wheel_vel_left"] for r in steady),
                       "right_wheel_mean": statistics.mean(r["wheel_vel_right"] for r in steady),
                       "ratios": ratios, "resets": resets,
                       "passed": resets == 0 and all(v < 1 for v in ratios.values())})
    worst = {k: max(p["ratios"][k] for p in phases) for k in phases[0]["ratios"]}
    return {"trace": str(path), "phases": phases, "worst_ratios": worst,
            "score": max(worst.values()), "resets": sum(p["resets"] for p in phases),
            "passed": all(p["passed"] for p in phases), "extended": extended}
