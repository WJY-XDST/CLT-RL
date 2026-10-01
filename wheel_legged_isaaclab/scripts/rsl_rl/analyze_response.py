"""Measure command-step settling, ramp duration, overshoot and steady accuracy."""

import argparse
import csv
import json
import math
from pathlib import Path

from assess_tracking import percentile


def settling_time(errors, band, dt=0.02, hold_seconds=0.5):
    """First entry after the last violation; require a final uninterrupted hold."""
    if not errors or any(not math.isfinite(value) for value in errors):
        raise ValueError("Settling requires finite samples")
    outside = [i for i, error in enumerate(errors) if error > band]
    first = outside[-1] + 1 if outside else 0
    if (len(errors) - first) * dt < hold_seconds - 1e-9:
        return None
    return round(first * dt, 6)


def analyze(path, phase_seconds=5.0):
    dt = 0.02
    count = round(phase_seconds / dt)
    with Path(path).open(newline="") as stream:
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(stream)]
    if not rows or len(rows) % count:
        raise ValueError("Trace must contain complete command phases")
    for i, r in enumerate(rows):
        if not all(math.isfinite(v) for v in r.values()):
            raise ValueError("Non-finite data")
        if r["step"] != i or abs(r["sim_time_s"] - i * dt) > 1e-5:
            raise ValueError("Missing or mistimed sample")
    if len({r["env_id"] for r in rows}) != 1:
        raise ValueError("Exactly one environment is required")
    phases = []
    for index in range(len(rows) // count):
        phase = rows[index * count:(index + 1) * count]
        v, h = round(phase[0]["cmd_x_target"], 6), round(phase[0]["height_cmd"], 6)
        if any(abs(r["cmd_x_target"] - v) > 1e-5 or abs(r["height_cmd"] - h) > 1e-5 for r in phase):
            raise ValueError("Commands changed inside a phase")
        speed_band = .05 * abs(v) if v else .02
        errors = {
            "speed": [abs(r["vel_x_heading"] - v) if v else math.hypot(r["vel_x_heading"], r["vel_y_heading"]) for r in phase],
            "height": [abs(r["base_height"] - h) for r in phase],
            "body": [max(abs(r["pitch_est_rad"]), abs(r["roll_est_rad"])) * 180 / math.pi for r in phase],
            "symmetry": [abs(r["theta_left"] - r["theta_right"]) * 180 / math.pi for r in phase],
        }
        bands = {"speed": speed_band, "height": h * .05, "body": 3.0, "symmetry": 3.0}
        resets = sum(int(r["terminated"]) + int(r["time_out"]) for r in phase)
        times = {key: settling_time(values, bands[key], dt) if not resets else None for key, values in errors.items()}
        ramp = settling_time([abs(r["cmd_x"] - v) for r in phase], 1e-5, dt)
        # Retain the complete transient, but judge steady accuracy on the final two seconds.
        ratios = {key: (percentile(values[-100:]) if key in ("speed", "height") else max(values[-100:])) / bands[key]
                  for key, values in errors.items()}
        previous_v = phases[-1]["target_speed"] if phases else v
        previous_h = phases[-1]["target_height"] if phases else h
        direction_v = 1 if v > previous_v else -1
        direction_h = 1 if h > previous_h else -1
        phases.append({"phase": index, "target_speed": v, "target_height": h,
                       "speed_changed": index > 0 and v != previous_v,
                       "height_changed": index > 0 and h != previous_h,
                       "resets": resets, "settling_seconds": times, "command_ramp_seconds": ramp,
                       "speed_lag_after_ramp_seconds": round(max(0, times["speed"] - ramp), 6)
                           if times["speed"] is not None and ramp is not None else None,
                       "speed_overshoot_m_s": max(0.0, max(direction_v * (r["vel_x_heading"] - v) for r in phase))
                           if index and v != previous_v else None,
                       "height_overshoot_m": max(0.0, max(direction_h * (r["base_height"] - h) for r in phase))
                           if index and h != previous_h else None,
                       "body_peak_deg": max(errors["body"]), "symmetry_peak_deg": max(errors["symmetry"]),
                       "height_peak_relative": max(errors["height"]) / h,
                       "steady_ratios": ratios,
                       "steady_passed": not resets and all(value < 1 for value in ratios.values())})
    transitions = phases[1:]
    def worst(values):
        return None if not values or any(v is None for v in values) else max(values)
    return {"trace": str(path), "samples": len(rows), "phase_seconds": phase_seconds,
            "settling_definition": "last exit from tolerance band followed by >=0.5 s in band; first startup phase excluded from worst transition times",
            "resets": sum(p["resets"] for p in phases), "steady_passed": all(p["steady_passed"] for p in phases),
            "worst_speed_step_seconds": worst([p["settling_seconds"]["speed"] for p in transitions if p["speed_changed"]]),
            "worst_speed_lag_after_ramp_seconds": worst([p["speed_lag_after_ramp_seconds"] for p in transitions if p["speed_changed"]]),
            "worst_height_step_seconds": worst([p["settling_seconds"]["height"] for p in transitions if p["height_changed"]]),
            "worst_body_recovery_seconds": worst([p["settling_seconds"]["body"] for p in transitions]),
            "worst_symmetry_recovery_seconds": worst([p["settling_seconds"]["symmetry"] for p in transitions]),
            "transition_body_peak_deg": max((p["body_peak_deg"] for p in transitions), default=0),
            "transition_symmetry_peak_deg": max((p["symmetry_peak_deg"] for p in transitions), default=0),
            "transition_height_peak_relative": max((p["height_peak_relative"] for p in transitions), default=0),
            "phases": phases}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [analyze(path) for path in args.traces]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{k: v for k, v in r.items() if k != "phases"} for r in reports], indent=2))
