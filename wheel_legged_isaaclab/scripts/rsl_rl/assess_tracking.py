"""Strict, per-phase replay acceptance; runs without Isaac Sim or torch."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

SPEEDS = (0.0, 0.2, 0.4, 0.6, 0.8, 0.0, -0.2, -0.4, -0.6, -0.8, 0.0)
HEIGHTS = (0.16, 0.18, 0.20)
PHASE_SECONDS = 5.0
DT = 0.02
SETTLE_SECONDS = 3.0
LIMITS = {"speed_relative": 0.05, "height_relative": 0.05,
          "standing_m_s": 0.02, "angle_deg": 3.0,
          "transient_body_deg": 10.0, "height_overshoot_relative": 0.05}


def sequence(reverse=False):
    speeds = tuple(reversed(SPEEDS)) if reverse else SPEEDS
    heights = tuple(reversed(HEIGHTS)) if reverse else HEIGHTS
    return [(speed, height) for speed in speeds for height in heights]


def percentile(values, fraction=0.95):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def assess_trace(path, reverse=False):
    with Path(path).open(newline="") as stream:
        rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    expected = sequence(reverse)
    per_phase = round(PHASE_SECONDS / DT)
    if len(rows) != len(expected) * per_phase:
        raise ValueError(f"Incomplete trace: {path}: {len(rows)} rows, expected {len(expected) * per_phase}")
    required = {"step", "sim_time_s", "env_id", "cmd_x_target", "height_cmd", "terminated", "time_out",
                "vel_x_heading", "vel_y_heading", "base_height", "pitch_est_rad", "roll_est_rad",
                "theta_left", "theta_right", "length_left", "length_right",
                "target_length_left", "target_length_right"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing columns: {sorted(required - rows[0].keys())}")
    if len({row["env_id"] for row in rows}) != 1:
        raise ValueError("Trace must contain exactly one environment")
    for i, row in enumerate(rows):
        if not all(math.isfinite(v) for v in row.values()):
            raise ValueError(f"Non-finite sample {i}")
        if row["step"] != i or abs(row["sim_time_s"] - i * DT) > 1e-5:
            raise ValueError(f"Missing, duplicated or mistimed sample {i}")
        if row["terminated"] not in (0, 1) or row["time_out"] not in (0, 1):
            raise ValueError(f"Invalid reset marker at {i}")
    phases = []
    for index, (speed, height) in enumerate(expected):
        phase = rows[index * per_phase:(index + 1) * per_phase]
        if any(abs(row["cmd_x_target"] - speed) > 1e-5
               or abs(row["height_cmd"] - height) > 1e-5 for row in phase):
            raise ValueError(f"Unexpected command or reset command corruption in phase {index}")
        settled = phase[round(SETTLE_SECONDS / DT):]
        speed_errors = [abs(r["vel_x_heading"] - speed) for r in settled]
        drift = [math.hypot(r["vel_x_heading"], r["vel_y_heading"]) for r in settled]
        height_errors = [abs(r["base_height"] - height) for r in settled]
        body = [max(abs(r["pitch_est_rad"]), abs(r["roll_est_rad"])) * 180 / math.pi for r in settled]
        legs = [abs(r["theta_left"] - r["theta_right"]) * 180 / math.pi for r in settled]
        # Initial drop/reset perturbation is reported separately, not a command transition.
        transient = phase if index else settled
        body_peak = max(max(abs(r["pitch_est_rad"]), abs(r["roll_est_rad"])) * 180 / math.pi
                        for r in transient)
        overshoot = 0.0
        if index:
            previous_height = expected[index - 1][1]
            direction = 1 if height > previous_height else -1
            overshoot = max(0.0, max(direction * (r["base_height"] - height) for r in phase)) / height
        speed_scale = abs(speed) * LIMITS["speed_relative"] if speed else LIMITS["standing_m_s"]
        speed_samples = speed_errors if speed else drift
        ratios = {
            "speed": max(statistics.mean(speed_samples), percentile(speed_samples)) / speed_scale,
            "height": max(statistics.mean(height_errors), percentile(height_errors)) / (height * LIMITS["height_relative"]),
            "body": max(body) / LIMITS["angle_deg"],
            "symmetry": max(legs) / LIMITS["angle_deg"],
            "transient_body": body_peak / LIMITS["transient_body_deg"],
            "height_overshoot": overshoot / LIMITS["height_overshoot_relative"],
        }
        phases.append({
            "phase": index, "speed": speed, "height": height,
            "speed_mae": statistics.mean(speed_errors), "speed_p95": percentile(speed_errors),
            "speed_relative_mae": statistics.mean(speed_errors) / abs(speed) if speed else None,
            "speed_relative_p95": percentile(speed_errors) / abs(speed) if speed else None,
            "standing_horizontal_p95": percentile(drift) if not speed else None,
            "height_relative_mae": statistics.mean(height_errors) / height,
            "height_relative_p95": percentile(height_errors) / height,
            "body_abs_max_deg": max(body), "leg_difference_max_deg": max(legs),
            "transient_body_abs_max_deg": body_peak, "height_overshoot_relative": overshoot,
            "leg_length_mae_m": {side: statistics.mean(abs(r[f"length_{side}"] - r[f"target_length_{side}"])
                                                       for r in settled) for side in ("left", "right")},
            "ratios": ratios, "passed": all(v < 1 for v in ratios.values()),
        })
    resets = sum(int(r["terminated"] + r["time_out"]) for r in rows)
    return {"trace": str(path), "samples": len(rows), "resets": resets,
            "terminated": sum(int(r["terminated"]) for r in rows),
            "timeouts": sum(int(r["time_out"]) for r in rows), "phases": phases,
            "passed": resets == 0 and all(p["passed"] for p in phases)}


def assess_suite(traces, reverse=False):
    if len(traces) < 3 or len({str(Path(p).resolve()) for p in traces}) != len(traces):
        raise ValueError("At least three distinct seed traces are required")
    results = [assess_trace(path, reverse) for path in traces]
    worst = {key: max(p["ratios"][key] for r in results for p in r["phases"])
             for key in results[0]["phases"][0]["ratios"]}
    # Ratios below one already satisfy the gate. Excess focuses selection on remaining errors.
    score = statistics.mean(max(1.0, value) for value in worst.values())
    return {"limits": LIMITS, "angle_limit_is_provisional_absolute_not_percent": True,
            "steady_window_seconds": [SETTLE_SECONDS, PHASE_SECONDS],
            "reverse_order": reverse, "resets": sum(r["resets"] for r in results),
            "worst_ratios": worst, "score": score,
            "passed": all(r["passed"] for r in results), "traces": results}


def clearly_better(candidate, incumbent):
    """Require a complete, reset-free suite, improvement and no material phase regression."""
    if candidate["resets"]:
        return False
    old = incumbent["traces"]
    new = candidate["traces"]
    if (len(old) != len(new) or candidate["reverse_order"] != incumbent["reverse_order"]
            or candidate.get("seeds") != incumbent.get("seeds")):
        raise ValueError("Cannot compare different evaluation suites")
    for before, after in zip(old, new):
        if len(before["phases"]) != len(after["phases"]):
            raise ValueError("Cannot compare different phase counts")
        for a, b in zip(before["phases"], after["phases"]):
            if (a["speed"], a["height"]) != (b["speed"], b["height"]):
                raise ValueError("Cannot compare different commands")
            if not incumbent["resets"] and any(
                b["ratios"][key] > max(1.0, value * 1.10) for key, value in a["ratios"].items()
            ):
                return False
    if incumbent["resets"]:
        return True
    return candidate["passed"] and not incumbent["passed"] or candidate["score"] < incumbent["score"] * 0.90


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess_suite(args.traces, args.reverse)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "traces"}, indent=2))


if __name__ == "__main__":
    main()
