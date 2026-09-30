"""Plot measured replay signals, commands, and reset markers from play.py CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.trace.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("The trace is empty.")
    data = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    t = data["sim_time_s"]
    reset = (data["terminated"] + data["time_out"]) > 0
    fig, axes = plt.subplots(6, 1, figsize=(16, 17), sharex=True, constrained_layout=True)

    def line(ax, key, label, scale=1.0, style="-", color=None):
        # The simulator has already reset state on a done row; break the line there.
        values = data[key].copy() * scale
        if key not in ("cmd_x_target", "cmd_x", "height_cmd"):
            values[reset] = np.nan
        ax.plot(t, values, style, label=label, linewidth=1.05, color=color)

    line(axes[0], "cmd_x_target", "Speed target", style="--", color="black")
    line(axes[0], "cmd_x", "Ramped command", color="tab:orange")
    line(axes[0], "vel_x_heading", "Measured forward speed", color="tab:blue")
    line(axes[0], "vel_y_heading", "Measured lateral speed", color="tab:green")
    axes[0].set_ylabel("Speed (m/s)")
    line(axes[1], "height_cmd", "Body height target", style="--", color="black")
    line(axes[1], "base_height", "Measured body height", color="tab:blue")
    axes[1].set_ylabel("Height (m)")
    line(axes[2], "pitch_est_rad", "Pitch estimate", scale=180 / np.pi)
    line(axes[2], "roll_est_rad", "Roll estimate", scale=180 / np.pi)
    axes[2].axhspan(-3, 3, color="green", alpha=0.07, label="Reference: +/-3 deg")
    axes[2].set_ylabel("Body angle (deg)")
    line(axes[3], "theta_left", "Left leg angle", scale=180 / np.pi)
    line(axes[3], "theta_right", "Right leg angle", scale=180 / np.pi)
    difference = np.abs(data["theta_left"] - data["theta_right"]) * 180 / np.pi
    difference[reset] = np.nan
    axes[3].plot(t, difference, label="Absolute leg angle difference", linewidth=1.0)
    axes[3].set_ylabel("Leg angle (deg)")
    for side, color in (("left", "tab:blue"), ("right", "tab:orange")):
        line(axes[4], f"length_{side}", f"{side.title()} actual", scale=1000, color=color)
        line(axes[4], f"target_length_{side}", f"{side.title()} target", scale=1000,
             style="--", color=color)
        error = (data[f"length_{side}"] - data[f"target_length_{side}"]) * 1000
        error[reset] = np.nan
        axes[5].plot(t, error, label=f"{side.title()} actual - target", color=color, linewidth=1)
    axes[4].set_ylabel("Leg length (mm)")
    axes[5].set_ylabel("Leg error (mm)")
    axes[5].set_xlabel("Simulation time (s)")
    changes = np.flatnonzero((np.diff(data["cmd_x_target"]) != 0)
                            | (np.diff(data["height_cmd"]) != 0)) + 1
    for ax in axes:
        for when in t[changes]:
            ax.axvline(when, color="gray", linewidth=0.5, alpha=0.3)
        for when in t[reset]:
            ax.axvline(when, color="red", linewidth=1.5)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper left", ncol=4, fontsize=8)
        ax.set_xlim(t[0], t[-1])
    fig.suptitle(f"Replay: {args.trace.stem} | {len(rows)} samples | "
                 f"terminated={int(data['terminated'].sum())}, "
                 f"timeouts={int(data['time_out'].sum())}\n"
                 "Gray lines: command changes; red lines: resets. All transients included.")
    output = args.output or args.trace.with_suffix(".png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output.resolve())


if __name__ == "__main__":
    main()
