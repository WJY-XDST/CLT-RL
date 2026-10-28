"""Account for yaw references, measured motion and actual reward in every case.

Alternative scalar reward costs are evaluated on the same recorded states.
They are neither a policy-gradient estimate nor a prediction of retraining.
No simulator, controller, policy, optimizer or command sampler is changed.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from joint_pd_continuation import sha
from auto_train_mine import write_json


SCRIPT = Path(__file__).resolve()


def audit(directories, destination, coefficients):
    results, pins = [], {str(SCRIPT): sha(SCRIPT)}
    for directory in directories:
        directory = Path(directory).resolve()
        files = [directory / name for name in ("manifest.json", "completed.json", "resolved_env.json", "trace.csv")]
        manifest, complete, cfg = [json.loads(p.read_text()) for p in files[:3]]
        assert complete["completed"] and not manifest["weights_updated"] and not manifest["sampled_policy_actions"]
        assert manifest["leg_control_mode"] == "implicit_joint_position" and manifest["policy_dt"] == .02
        raw = np.genfromtxt(files[3], delimiter=",", names=True)
        assert len(raw) == complete["samples"] and all(np.isfinite(raw[k]).all() for k in raw.dtype.names)
        radius, track = cfg["wheel_radius"], cfg["wheel_track_width"]
        original_coefficient = cfg["rewards"]["yaw_rate_error_sq"]
        bound = cfg["rewards"]["yaw_rate_penalty_clip"] * .02
        error = raw["yaw_rate_rad_s"] - raw["yaw_cmd_rad_s"]
        predicted_cost = np.maximum(original_coefficient * error ** 2 * .02, -bound)
        parity = float(np.max(abs(predicted_cost - raw["reward_yaw_rate_error_sq"])))
        assert parity < 1.e-7
        rows = []
        for cid, case in enumerate(manifest["cases"]):
            for seed in manifest["seeds"]:
                full = raw[(raw["case_id"] == cid) & (raw["seed"] == seed)]
                assert len(full) == round(manifest["seconds_per_case"] / .02)
                part = full[full["time_s"] >= manifest["seconds_per_case"] - 3.]
                command = part["yaw_cmd_rad_s"]
                reference = (part["right_wheel_ref_rad_s"] - part["left_wheel_ref_rad_s"]) * radius / track
                wheel = (part["right_wheel_rad_s"] - part["left_wheel_rad_s"]) * radius / track
                body = part["yaw_rate_rad_s"]
                components = (command - reference, reference - wheel, wheel - body)
                closure = float(np.max(abs(sum(components) - (command - body))))
                assert closure < 1.e-12
                cost_proposals = []
                for coefficient in coefficients:
                    proposed = np.maximum(coefficient * (body - command) ** 2 * .02, -bound)
                    cost_proposals.append(dict(coefficient=coefficient,
                        mean_cost_per_step=float(proposed.mean()),
                        mean_additional_cost_per_step=float((proposed - part["reward_yaw_rate_error_sq"]).mean()),
                        steady_clipped_fraction=float(np.mean(proposed <= -bound)),
                        all_time_clipped_fraction=float(np.mean(coefficient *
                            (full["yaw_rate_rad_s"] - full["yaw_cmd_rad_s"]) ** 2 * .02 <= -bound))))
                squared_error = (body - command) ** 2
                kernel = np.exp(-squared_error / cfg["rewards"]["tracking_sigma_ang"])
                assert kernel.min() > 1.e-6
                bonus_at_perfect_error = part["reward_track_ang_vel"] / kernel
                rows.append(dict(case=case, seed=seed,
                    full_trace_survived=not any(np.any(full[k]) for k in ("episode", "terminated", "timeout")),
                    body_yaw_mean_rad_s=float(body.mean()), body_yaw_MAE_rad_s=float(abs(body - command).mean()),
                    body_yaw_relative_percent=100. * float(abs(body - command).mean()) / abs(case["yaw"]) if case["yaw"] else None,
                    reference_rolling_yaw_proxy_mean_rad_s=float(reference.mean()),
                    measured_wheel_rolling_yaw_proxy_mean_rad_s=float(wheel.mean()),
                    command_minus_reference_mean_rad_s=float(components[0].mean()),
                    reference_minus_measured_wheel_proxy_mean_rad_s=float(components[1].mean()),
                    measured_wheel_proxy_minus_body_mean_rad_s=float(components[2].mean()),
                    signed_component_closure_max_error_rad_s=closure,
                    wheel_reference_absolute_max_rad_s=float(max(abs(part[k]).max() for k in
                        ("left_wheel_ref_rad_s", "right_wheel_ref_rad_s"))),
                    actual_mean_reward_terms_per_step={key: float(part["reward_" + key].mean()) for key in
                        ("track_ang_vel", "yaw_rate_error_sq", "standing_velocity", "standing_wheel_tracking",
                         "action_rate", "action_smooth", "base_height", "orientation")},
                    additional_existing_yaw_bonus_at_zero_error_with_other_signals_fixed=float(
                        (bonus_at_perfect_error - part["reward_track_ang_vel"]).mean()),
                    hypothetical_uniform_scalar_costs_on_same_recorded_states=cost_proposals))
        results.append(dict(directory=str(directory), trials=len(rows),
            checkpoint_sha256=manifest["checkpoint_sha256"], original_coefficient=original_coefficient,
            actual_recorded_yaw_cost_formula_max_error=parity, rows=rows))
        pins.update({str(p): sha(p) for p in files})
    assert all(sha(p) == digest for p, digest in pins.items())
    report = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(), results=results,
        alternative_coefficients_apply_identically_to_every_recorded_command=True,
        original_standing_velocity_penalty_contains_horizontal_velocity_only=True,
        no_yaw_signal_added_to_standing_velocity=True, source_pins=pins,
        limitations=["Rolling yaw is a kinematic proxy; its residuals do not identify physical slip, solver force or motor torque.",
            "Per-term reward magnitudes and state derivatives are not policy gradients or causal learning attributions.",
            "A hypothetical coefficient evaluated on a frozen trace cannot predict changed actions, stability or tracking.",
            "A coefficient trial needs its own exact-state load, fresh real capacity tests,200 updates and independent retention/replay."],
        optimizer_executed=False, simulator_started=False, training_modified=False,
        policy_accepted=False, acceptance_changed=False)
    assert not Path(destination).exists()
    write_json(destination, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coefficients", type=float, nargs="+", default=[-10., -50.])
    args = parser.parse_args()
    result = audit(args.directories, args.output, args.coefficients)
    print(json.dumps(dict(verified=True, report=str(args.output), trials=sum(r["trials"] for r in result["results"]),
        optimizer_executed=False, simulator_started=False)))
