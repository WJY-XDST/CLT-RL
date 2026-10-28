"""Offline geometry: distinguish a joint action box from physical leg stops.

Inverse triangle geometry is used only to ask whether an upright hypothetical
pose fits the saved joint-target map. No controller, replay, IK execution mode,
PPO update, limit change or parameter selection is installed by this audit.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import torch

from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state
from wheel_legged_gym_isaaclab.leg_control import joint_position_targets
from wheel_legged_gym_isaaclab.robot_model import MODEL


def inverse_geometry(length, angle, geometry):
    d = length / geometry["scale"]
    angles = []
    phi = torch.atan2(-length * angle.sin(), -length * angle.cos())
    for i, sign in ((0, 1.), (1, -1.)):
        upper, lower = geometry["upper"][i], geometry["lower"][i]
        cosine = (d.square() + upper ** 2 - lower ** 2) / (2 * d * upper)
        assert bool((cosine.abs() < 1).all())
        value = phi + sign * cosine.acos() - geometry["offsets"][i]
        angles.append(torch.atan2(value.sin(), value.cos()))
    return torch.stack((angles[0], angles[1], -angles[0], -angles[1]), -1)


def run(args):
    assert not args.output.exists()
    read = lambda p: json.loads(Path(p).read_text())
    config = read(args.config)
    cfg = config.get("training_config", config)
    prior = read(args.joint_limit_report)
    assert prior["native_num_envs"] in (4096, 8192)
    assert prior["original_native_motor_target_parity_max_error_rad"] < 1.e-7
    grid = read(args.source_grid)
    heights = sorted({r["case"]["height"] for item in grid["results"] for r in item["all_trials"]})
    assert heights[0] == cfg["commands"]["ranges_height"][0]
    assert heights[-1] == cfg["commands"]["ranges_height"][1]
    assert cfg["leg_control_mode"] == "implicit_joint_position"
    geometry = cfg["five_bar_geometry"]
    offset = torch.tensor(cfg["leg_joint_position_offset"], dtype=torch.float64)
    scale = torch.tensor(cfg["leg_joint_position_scale"], dtype=torch.float64)
    limits = torch.tensor(prior["USD_composed_joint_limits_rad"], dtype=torch.float64)
    lower = torch.maximum(limits[:, 0] + cfg["leg_joint_target_limit_margin"], offset - scale)
    upper = torch.minimum(limits[:, 1] - cfg["leg_joint_target_limit_margin"], offset + scale)
    rows, maximum_inverse_error = [], 0.
    theta = torch.linspace(-.3, .3, 6001, dtype=torch.float64)
    rng = torch.get_rng_state().clone()
    for height in heights:
        length = (height + MODEL["hip_z"] - cfg["wheel_radius"]) / theta.cos()
        q = inverse_geometry(length, theta, geometry)
        actual_l, actual_t, _, valid = five_bar_state(q[:, [0, 2]], q[:, [1, 3]], **geometry)
        error = max(float((actual_l - length[:, None]).abs().max()),
                    float((actual_t - theta[:, None]).abs().max()))
        maximum_inverse_error = max(maximum_inverse_error, error)
        assert valid.all() and error < 1.e-10
        feasible = ((q >= lower - 1.e-12) & (q <= upper + 1.e-12)).all(-1)
        indices = feasible.nonzero().flatten()
        assert len(indices) and bool(feasible[indices[0]:indices[-1]+1].all())
        representatives = []
        for angle in (-.1, -.05, 0., .05, .1):
            at = int(round((angle + .3) / .0001))
            target, _ = joint_position_targets(((q[at] - offset) / scale)[None],
                cfg["leg_joint_position_offset"], cfg["leg_joint_position_scale"], limits[None],
                cfg["leg_joint_target_limit_margin"])
            clipped_l, clipped_t, _, good = five_bar_state(target[:, [0, 2]], target[:, [1, 3]], **geometry)
            assert good.all()
            projected_height = clipped_l * clipped_t.cos() + cfg["wheel_radius"] - MODEL["hip_z"]
            representatives.append(dict(hypothetical_common_leg_angle_rad=angle,
                desired_length_m=float(length[at]), desired_raw_joint_angles_rad=q[at].tolist(),
                required_normalized_action=((q[at] - offset) / scale).tolist(),
                inside_current_joint_action_box=bool(feasible[at]),
                inside_USD_joint_margin=bool(((q[at] >= limits[:, 0] + cfg["leg_joint_target_limit_margin"])
                    & (q[at] <= limits[:, 1] - cfg["leg_joint_target_limit_margin"])).all()),
                clipped_FK_upright_height_proxy_m=projected_height[0].tolist()))
        rows.append(dict(root_height_m=height,
            hypothetical_upright_common_leg_angle_feasible_interval_rad=[float(theta[indices[0]]), float(theta[indices[-1]])],
            angular_grid_resolution_rad=.0001, representatives=representatives))
    box = []
    axis = torch.linspace(-1., 1., 161, dtype=torch.float64)
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    actions = torch.stack((x.flatten(), y.flatten(), -x.flatten(), -y.flatten()), -1)
    for value in (.3, .35):
        q = offset + value * actions
        length, _, _, valid = five_bar_state(q[:, [0, 2]], q[:, [1, 3]], **geometry)
        box.append(dict(hypothetical_common_scale_rad=value, sampled_motor_pairs=len(q),
            valid_FK_fraction=float(valid.double().mean()),
            geometric_length_min_m=float(length.min()), geometric_length_max_m=float(length.max()),
            FK_reference_below_MCU_length_min_fraction=float((length < cfg["l0_ref_min"]).double().mean()),
            FK_reference_above_MCU_length_max_fraction=float((length > cfg["l0_ref_max"]).double().mean()),
            physical_motion_or_contact_not_simulated=True))
    assert torch.equal(rng, torch.get_rng_state())
    files = [Path(__file__).resolve(), args.config.resolve(), args.joint_limit_report.resolve(),
             args.source_grid.resolve(), Path(five_bar_state.__code__.co_filename),
             Path(joint_position_targets.__code__.co_filename)]
    pins = {str(f): sha(f) for f in files}
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        original_forward_geometry_used_for_inverse_cross_check=True,
        inverse_to_original_FK_max_error=maximum_inverse_error,
        saved_joint_action_scales_rad=scale.tolist(), saved_joint_offsets_rad=offset.tolist(),
        saved_USD_joint_limits_rad=limits.tolist(), all_height_results=rows,
        hypothetical_independent_motor_boxes=box, CPU_Torch_RNG_unchanged=True,
        limitations=["The assumed body is upright and wheel centers lie on a level plane; no actual balance or trajectory is predicted.",
            "Inverse geometry is an offline reachability query only and is never used to generate policy actions.",
            "The joint action box is distinct from CAD stop contacts and MCU virtual-length operating limits.",
            "A wider independent motor box can command FK references outside the length range even while USD joint limits are respected.",
            "Collision, stop force, swept clearance, torque and dynamic stability are not established by FK.",
            "This audit does not prove the tracking goal unreachable or select a new action scale."],
        source_pins=pins, simulator_started=False, optimizer_executed=False,
        controller_modified=False, action_envelope_modified=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(verified=True, output=str(args.output), maximum_inverse_error=maximum_inverse_error,
        feasible_angles=[dict(height=r["root_height_m"], angles=r["hypothetical_upright_common_leg_angle_feasible_interval_rad"]) for r in rows],
        hypothetical_boxes=box), ensure_ascii=False))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--config", type=Path, required=True)
    cli.add_argument("--joint-limit-report", type=Path, required=True)
    cli.add_argument("--source-grid", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    run(cli.parse_args())
