"""Read evolving-policy yaw statistics in stage-relative diagnostic bins.

The original reader, its physical-age mask, its numbers and all training code
remain unchanged. These bins are analysis only, never command sampling,
reward weighting, action compensation or an acceptance test.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, through, output):
    directory, output = directory.resolve(), output.resolve()
    assert through in (50, 100, 200) and not output.exists()
    original_path = directory / f"sampled_yaw_reference_components_prefix_{through:04d}_readonly.json"
    original = json.loads(original_path.read_text())
    assert original["verified"] and original["original_settled_masks_and_population_counts_exact"]
    pins = dict(original["source_pins"])
    assert all(sha(path) == digest for path, digest in pins.items())
    plan_path = directory / "experiment_plan.json"
    plan = json.loads(plan_path.read_text())
    cfg, ids = plan["training_config"], original["sampled_environment_ids"]
    commands_cfg = cfg["commands"]
    stage = commands_cfg["training_motion_scale"]
    speed_limit = stage * max(abs(v) for key in ("ranges_lin_vel_x", "ranges_reverse_lin_vel_x")
                              for v in commands_cfg[key])
    yaw_limit = stage * max(abs(v) for v in commands_cfg["ranges_ang_vel_yaw"])
    low, high = commands_cfg["ranges_height"]
    assert speed_limit > 0 and yaw_limit > 0 and high > low
    fields = ("native_commands", "native_yaw", "mu", "actions", "native_raw_actions", "dones")
    parts = {key: [] for key in fields}
    for update in range(1, through + 1):
        path = directory / f"native_learning_signal/rollout_{update:04d}.npz"
        assert sha(path) == pins[str(path)]
        with np.load(path, allow_pickle=False) as arrays:
            for key in fields:
                value = arrays[key]
                assert value.shape[:2] == (48, len(ids)) and np.isfinite(value).all()
                parts[key].append(value.copy())
    data = {key: np.concatenate(values) for key, values in parts.items()}
    assert np.array_equal(data["actions"], data["native_raw_actions"])
    command_native = data["native_commands"]
    command = command_native.astype(np.float64)
    done = data["dones"][..., 0].astype(bool)
    dt = cfg["sim"]["dt"] * cfg["decimation"]
    age, stable_age = np.empty(done.shape), np.empty(done.shape)
    age[0], stable_age[0] = dt, 0.
    for t in range(1, len(age)):
        age[t] = np.where(done[t - 1], dt, age[t - 1] + dt)
        changed = np.any(abs(command_native[t] - command_native[t - 1]) > 1.e-7, axis=-1)
        stable_age[t] = np.where(changed | done[t - 1], 0., stable_age[t - 1] + dt)
    settled = (age > 2.) & (stable_age >= 1.) & ~done
    static = np.broadcast_to(np.asarray(ids)[None, :] < plan["fixed_standing_environments"], done.shape)
    ratio, scale, limit = cfg["wheel_radius"] / cfg["wheel_track_width"], cfg["action_scale_vel"], cfg["wheel_action_limit"]

    def reference(key):
        wheels = np.clip(data[key][..., [2, 5]].astype(np.float64), -limit, limit) * scale
        return ratio * (wheels[..., 1] - wheels[..., 0])

    mean, sampled = reference("mu"), reference("actions")
    body, target = data["native_yaw"].astype(np.float64), command[..., 1]
    components = (mean - target, sampled - mean, body - sampled)
    closure = float(abs(sum(components) - (body - target)).max())
    assert closure < 1.e-12
    v, w = command[..., 0] / speed_limit, target / yaw_limit
    h = (command[..., 2] - low) / (high - low)
    assert abs(v).max() <= 1. + 1.e-5 and abs(w).max() <= 1. + 1.e-5
    assert h.min() >= -1.e-5 and h.max() <= 1. + 1.e-5
    height_bin = np.clip(np.searchsorted([1 / 3, 2 / 3], h, side="right"), 0, 2)
    velocity_bin = np.where(v <= -.5, 0, np.where(v < 0., 1, np.where(v < .5, 3, 4)))
    velocity_bin[abs(command[..., 0]) < 1.e-7] = 2
    magnitude_bin = np.searchsorted([1 / 3, 2 / 3], abs(w), side="right")
    yaw_bin = np.where(w < 0., 2 - magnitude_bin, 4 + magnitude_bin)
    yaw_bin[abs(target) < 1.e-7] = 3
    windows = {"entire_prefix": np.ones_like(done)}
    first, last = np.zeros_like(done), np.zeros_like(done)
    first[:20 * 48], last[-20 * 48:] = True, True
    windows.update(first20_updates=first, last20_updates=last)
    groups, reproductions = {}, {}
    for name, window in windows.items():
        population = settled & window
        reproductions[name] = {}
        for label, selected in (("dedicated_static", static), ("motion_population", ~static)):
            mask = population & selected
            recorded = original["groups"][name][label]
            assert int(mask.sum()) == recorded["recorded_settled_transitions"]
            error = 0. if not mask.any() else abs(float(abs(body[mask] - target[mask]).mean())
                                                   - recorded["metrics"]["body_yaw_MAE_rad_s"])
            assert error < 1.e-7
            reproductions[name][label] = dict(samples=int(mask.sum()), original_MAE_error=error)
        rows = []
        for ih in range(3):
            for iv in range(5):
                for iy in range(7):
                    mask = population & (height_bin == ih) & (velocity_bin == iv) & (yaw_bin == iy)
                    count = int(mask.sum())
                    metrics = None if not count else dict(
                        command_mean=command[mask].mean(0).tolist(),
                        body_yaw_MAE_rad_s=float(abs(body[mask] - target[mask]).mean()),
                        body_yaw_signed_bias_rad_s=float((body[mask] - target[mask]).mean()),
                        mean_reference_minus_command_MAE_rad_s=float(abs(components[0][mask]).mean()),
                        sampled_minus_mean_reference_MAE_rad_s=float(abs(components[1][mask]).mean()),
                        body_minus_sampled_reference_MAE_rad_s=float(abs(components[2][mask]).mean()))
                    rows.append(dict(height_bin=ih, velocity_bin=iv, yaw_bin=iy, samples=count, metrics=metrics))
        assert sum(row["samples"] for row in rows) == int(population.sum())
        groups[name] = rows
    assert all(sha(path) == digest for path, digest in pins.items())
    pins.update({str(path): sha(path) for path in (Path(__file__).resolve(), original_path, plan_path)})
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        through=through, actual_num_envs=plan["native_num_envs"], recorded_environment_count=len(ids),
        motion_scale=stage, applied_speed_limit_m_s=speed_limit, applied_yaw_limit_rad_s=yaw_limit,
        height_range_m=[low, high],
        height_bins="normalized thirds: low, middle, high",
        velocity_bins="negative_fast[-1,-.5],negative_slow(-.5,0),zero,positive_slow(0,.5),positive_fast[.5,1]",
        yaw_bins="negative_high,negative_middle,negative_low,zero,positive_low,positive_middle,positive_high; absolute normalized thirds",
        original_population_and_body_MAE_reproduced=reproductions,
        signed_component_closure_max_error_rad_s=closure, groups=groups,
        all_settled_samples_partitioned_exactly_once=True,
        legacy_empty_groups=[key for key, value in original["groups"]["entire_prefix"].items()
                             if not value["recorded_settled_transitions"]],
        training_modified=False, controller_changed=False, masks_enter_training=False,
        acceptance_changed=False, optimizer_executed=False, simulator_started=False, policy_accepted=False,
        limitations=["Evolving Gaussian-policy training on recorded environments, not frozen independent replay.",
                     "These bins diagnose the current stage; zero counts do not establish missing full-population support.",
                     "Body-versus-sampled-reference includes dynamics and filtering; it is not a measured wheel servo error.",
                     "Temporal MAE comparisons are not matched-state causal experiments."], source_pins=pins)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("--through", type=int, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.directory, args.through, args.output)
    print(json.dumps(dict(verified=result["verified"], output=str(args.output),
                         applied_speed_limit_m_s=result["applied_speed_limit_m_s"],
                         all_samples_partitioned=result["all_settled_samples_partitioned_exactly_once"])))
