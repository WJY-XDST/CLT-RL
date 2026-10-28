"""Audit the saved command/actor/wheel/body chain without running a simulator.

Use all 324 original combined-turn trials per checkpoint. Rolling wheel yaw
is a kinematic accounting proxy, not a measurement of slip or solver torque.
This diagnostic cannot accept a policy or select a new PPO configuration.
"""
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "mine_model/results"
Q = BASE / "joint_pd_source7401_D100_C400_leg0075_qualification_20261024"
OUTPUT = Q / "all648_combined_yaw_command_actor_wheel_body_chain_readonly_20261024.json"
SOURCES = {
    "source7401": (
        BASE / "joint_pd_wheel005_source7201_D100_yaw400_review7401_20261024",
        BASE / "joint_pd_optimization_20261024_065638_819339_c26a87bd_wheel005_source7201_D100_yaw400_200/all_errors_all657_source7401_readonly_20261024.json",
    ),
    "candidate7601": (
        BASE / "joint_pd_source7401_D100_C400_leg0075_review7601_20261024",
        BASE / "joint_pd_optimization_20261024_100639_266840_56c5900d_source7401_D100_C400_leg0075_200/all_errors_all657_source7601_readonly_20261024.json",
    ),
}
FIELDS = (
    "time_s", "case_id", "seed", "episode", "terminated", "timeout",
    "speed_cmd_m_s", "yaw_cmd_rad_s", "height_cmd_m", "speed_m_s",
    "yaw_rate_rad_s", "height_m", "pitch_rad", "roll_rad",
    "left_wheel_ref_rad_s", "right_wheel_ref_rad_s",
    "left_wheel_rad_s", "right_wheel_rad_s", "action_clipping_fraction",
    "torque_saturation_fraction", "peak_torque_nm",
)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit():
    assert not OUTPUT.exists(), "Preserve prior immutable diagnostics"
    results, mirrors, pins = {}, {}, {str(Path(__file__).resolve()): sha(__file__)}
    errors = dict(native_mean_to_wheel_reference=0., actor_yaw_command=0.,
                  actor_measured_yaw_previous_step=0., three_original_MAEs=0.,
                  signed_proxy_chain_closure=0.)
    for tag, (directory, raw_path) in SOURCES.items():
        original = read(raw_path)
        assert original["verified"] and original["total_trials"] == 657
        assert original["all7_metrics_and_limits_raw_reproduction_max_error"] == 0.
        pins[str(raw_path)] = sha(raw_path)
        rows = []
        for height in ("low", "nominal", "high"):
            folder = directory / "combined/candidate" / height
            manifest, cfg, done = (read(folder / name) for name in
                                   ("manifest.json", "resolved_env.json", "completed.json"))
            assert manifest["checkpoint_sha256"] == original["checkpoint_sha256"]
            assert not manifest["sampled_policy_actions"] and done["completed"]
            assert not manifest["policy_input_remapping"] and not manifest["action_bias_probes"]
            assert cfg["wheel_control_mode"] == "implicit_velocity" and cfg["wheel_damping"] == 1.
            assert cfg["obs_scales"]["ang_vel"] == .25
            cases, seeds = manifest["cases"], manifest["seeds"]
            assert len(cases) == 36 and seeds == [683, 684, 685]
            data = np.genfromtxt(folder / "trace.csv", delimiter=",", names=True, usecols=FIELDS)
            assert len(data) == 108000
            assert all(np.isfinite(data[k]).all() for k in FIELDS)
            width = len(cases) * len(seeds)
            matrix = {k: data[k].reshape(1000, width) for k in FIELDS}
            assert np.array_equal(matrix["case_id"], np.broadcast_to(np.repeat(np.arange(36), 3), (1000, width)))
            assert np.array_equal(matrix["seed"], np.broadcast_to(np.tile(seeds, 36), (1000, width)))
            assert np.allclose(matrix["time_s"], (np.arange(1, 1001) * .02)[:, None], atol=1.e-8, rtol=0.)
            assert not any(matrix[k].any() for k in ("episode", "terminated", "timeout"))
            with np.load(folder / "policy_inputs.npz", allow_pickle=False) as archive:
                obs, means = archive["observations"], archive["means"]
            assert obs.shape == (1000, width, 27) and means.shape == (1000, width, 6)
            reference = np.clip(means[:, :, (2, 5)], -cfg["wheel_action_limit"], cfg["wheel_action_limit"]) * cfg["action_scale_vel"]
            actual_reference = np.stack((matrix["left_wheel_ref_rad_s"], matrix["right_wheel_ref_rad_s"]), axis=-1)
            errors["native_mean_to_wheel_reference"] = max(errors["native_mean_to_wheel_reference"], float(abs(reference - actual_reference).max()))
            errors["actor_yaw_command"] = max(errors["actor_yaw_command"], float(abs(obs[:, :, 7] / .25 - matrix["yaw_cmd_rad_s"]).max()))
            errors["actor_measured_yaw_previous_step"] = max(errors["actor_measured_yaw_previous_step"], float(abs(obs[1:, :, 2] / .25 - matrix["yaw_rate_rad_s"][:-1]).max()))
            for cid, case in enumerate(cases):
                for sid, seed in enumerate(seeds):
                    index = cid * 3 + sid
                    tail = data[(data["case_id"] == cid) & (data["seed"] == seed) & (data["time_s"] >= 17.)]
                    assert len(tail) == 151
                    old = [r for r in original["trials"] if r["scope"] == "combined_" + height
                           and r["case"] == case and r["seed"] == seed]
                    assert len(old) == 1 and old[0]["survived"]
                    for key, field, command in (("speed_mae_m_s", "speed_m_s", "speed_cmd_m_s"),
                                               ("yaw_mae_rad_s", "yaw_rate_rad_s", "yaw_cmd_rad_s"),
                                               ("height_mae_m", "height_m", "height_cmd_m")):
                        error = abs(float(abs(tail[field] - tail[command]).mean()) - old[0]["metrics"][key])
                        errors["three_original_MAEs"] = max(errors["three_original_MAEs"], error)
                    command = float(tail["yaw_cmd_rad_s"].mean())
                    body = float(tail["yaw_rate_rad_s"].mean())
                    # Same measured radius and nominal contact-center track as the
                    # earlier frozen diagnostics. No no-slip assumption is made.
                    ref_proxy = float(((tail["right_wheel_ref_rad_s"] - tail["left_wheel_ref_rad_s"]) * .06 / .4199).mean())
                    measured_proxy = float(((tail["right_wheel_rad_s"] - tail["left_wheel_rad_s"]) * .06 / .4199).mean())
                    chain = (command - ref_proxy, ref_proxy - measured_proxy, measured_proxy - body)
                    errors["signed_proxy_chain_closure"] = max(errors["signed_proxy_chain_closure"], abs(sum(chain) - (command - body)))
                    rows.append(dict(case=case, seed=seed, actor_environment_index=index,
                        actual_yaw_mean=body, commanded_yaw_mean=command,
                        yaw_MAE=old[0]["metrics"]["yaw_mae_rad_s"],
                        yaw_relative_percent=old[0]["nonzero_command_relative_error_percent"]["yaw_mae_rad_s"],
                        reference_rolling_yaw_proxy=ref_proxy, measured_rolling_yaw_proxy=measured_proxy,
                        signed_command_to_reference_proxy_gap=chain[0],
                        signed_reference_to_measured_proxy_gap=chain[1], signed_measured_proxy_to_body_gap=chain[2],
                        minimum_wheel_target_headroom_rad_s=float(cfg["wheel_action_limit"] * cfg["action_scale_vel"] - max(abs(tail[k]).max() for k in ("left_wheel_ref_rad_s", "right_wheel_ref_rad_s"))),
                        tail_reported_action_clipping_fraction_max=float(tail["action_clipping_fraction"].max()),
                        tail_reported_approximate_torque_saturation_fraction_max=float(tail["torque_saturation_fraction"].max()),
                        approximate_peak_torque_telemetry_nm_max=float(tail["peak_torque_nm"].max()),
                        pitch_abs_mean_deg=float(np.rad2deg(abs(tail["pitch_rad"])).mean()),
                        roll_abs_mean_deg=float(np.rad2deg(abs(tail["roll_rad"])).mean())))
            for name in ("manifest.json", "resolved_env.json", "completed.json", "trace.csv", "policy_inputs.npz"):
                pins[str(folder / name)] = sha(folder / name)
        assert len(rows) == 324
        results[tag] = rows
        mirrors[tag] = []
        for height in sorted({r["case"]["height"] for r in rows}):
            for speed in sorted({r["case"]["speed"] for r in rows}):
                for yaw in (.075, .15, .3):
                    pair = [[r for r in rows if r["case"]["height"] == height and r["case"]["speed"] == speed
                             and r["case"]["yaw"] == sign * yaw] for sign in (1, -1)]
                    assert all(len(part) == 3 for part in pair)
                    mean = lambda part, key: float(np.mean([r[key] for r in part]))
                    left, right = (mean(part, "actual_yaw_mean") for part in pair)
                    lr, rr = (mean(part, "reference_rolling_yaw_proxy") for part in pair)
                    mirrors[tag].append(dict(height_command=height, speed_command=speed, absolute_yaw_command=yaw,
                        positive_yaw_body_mean=left, negative_yaw_body_mean=right,
                        body_even_offset=(left + right) / 2., body_odd_gain=(left - right) / (2. * yaw),
                        reference_even_offset=(lr + rr) / 2., reference_odd_gain=(lr - rr) / (2. * yaw)))
    assert errors["native_mean_to_wheel_reference"] <= 1.e-6
    assert errors["actor_yaw_command"] <= 1.e-7
    assert errors["actor_measured_yaw_previous_step"] <= 2.e-7
    assert errors["three_original_MAEs"] <= 1.e-14
    assert errors["signed_proxy_chain_closure"] <= 1.e-14
    assert all(sha(path) == digest for path, digest in pins.items())
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        trials_per_checkpoint=324, original_tail_frames=151, original_tail_seconds=[17, 20],
        input_and_output_alignment_errors=errors, results=results, mirrored_pair_decomposition=mirrors,
        source_pins=pins, scope="All original three-height combined turns, both speed/yaw signs; frozen read-only accounting.",
        limitations=["Rolling wheel yaw is a kinematic accounting proxy, not a measurement of ground slip or actual solver torque.",
                     "Approximate torque telemetry cannot prove the combined implicit-drive and feedforward solver force is unsaturated.",
                     "This identifies where tracking is lost; it does not causally identify the historical PPO reward or sampling term that learned the deficient mean."],
        simulator_started=False, optimizer_executed=False, training_modified=False,
        parameter_change_selected=False, acceptance_changed=False, policy_accepted=False)
    with OUTPUT.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    critical = {tag: [r for r in pairs if abs(r["speed_command"]) == .8 and r["absolute_yaw_command"] == .075]
                for tag, pairs in mirrors.items()}
    print(json.dumps(dict(report=str(OUTPUT), verified=True, errors=errors, critical_mirrored_pairs=critical), ensure_ascii=False))
    return result


if __name__ == "__main__":
    audit()
