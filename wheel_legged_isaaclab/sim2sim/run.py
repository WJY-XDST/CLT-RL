"""Replay a frozen Isaac Lab actor in MuJoCo and retain numerical evidence."""

import argparse
import csv
import hashlib
import json
import shutil
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from core import Simulation, build_model, load_actor, load_config
from keyboard import KeyboardConfig, MujocoKeyboardController, STATE_SOURCE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint trained for the imported closed-five-bar vehicle.")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--jit", action="store_true", help="Checkpoint is an exported TorchScript actor.")
    parser.add_argument("--mjcf", type=Path, help="Previously converted model; new models require its USD audit manifest.")
    parser.add_argument("--physics-substeps", type=int, help="Internal MuJoCo integration steps per 200 Hz controller update (mine default: 50).")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-keyboard", "--no_keyboard", action="store_true", help="Disable the automatic focused keyboard panel for GUI replay.")
    parser.add_argument("--keyboard-speed", type=float, default=.8, help="W/S target speed in m/s (0 < value <= 0.8).")
    parser.add_argument("--keyboard-yaw-rate", type=float, default=.3, help="A/D target yaw rate in rad/s (default: 0.3; explicit values up to 0.6 may exceed training range).")
    parser.add_argument("--keyboard-height-step", type=float, default=.005, help="Q/E single-press height goal step in m.")
    parser.add_argument("--keyboard-height-rate", type=float, default=.02, help="Q/E held height goal rate in m/s.")
    parser.add_argument("--keyboard-height-slew-rate", type=float, default=.02, help="Height command ramp rate in m/s, continuing after key release.")
    parser.add_argument("--demo", action="store_true", help="45-second forward/reverse/height demonstration.")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--fixed-command", nargs=3, type=float, metavar=("VX", "YAW_RATE", "HEIGHT"))
    parser.add_argument("--velocity-cycle", nargs="+", type=float)
    parser.add_argument("--height-cycle", nargs="+", type=float)
    parser.add_argument("--yaw-cycle", nargs="+", type=float)
    parser.add_argument("--phase-duration", type=float)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--reset-velocity", type=float, default=0., help="Uniform root velocity perturbation magnitude (default: zero).")
    parser.add_argument("--reset-rng", choices=['numpy', 'isaac-torch'], default='numpy', help="isaac-torch matches evaluate_mine_policy.py initial velocity draws.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--snapshot", action="store_true", help="Save an offscreen PNG; set MUJOCO_GL=egl if headless.")
    parser.add_argument("--no-heading-hold", action="store_true", help="Zero yaw command stays a yaw-rate command.")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    config_folder = args.checkpoint.parent / 'params' if (args.checkpoint.parent / 'params').is_dir() else args.checkpoint.parent
    env_path = (args.env_config or config_folder / "env.yaml").expanduser().resolve()
    agent_path = (args.agent_config or config_folder / "agent.yaml").expanduser().resolve()
    cfg = load_config(env_path)
    mine = cfg['leg_model'] == 'mine_five_bar'
    height = cfg['rewards']['base_height_target']
    args.fixed_command = args.fixed_command or [0., 0., height]
    args.phase_duration = args.phase_duration if args.phase_duration is not None else 8.
    if args.demo:
        args.velocity_cycle = args.velocity_cycle or [0, .3, .6, -.3, -.6, 0, 0, 0, 0]
        args.height_cycle = args.height_cycle or [height] * 7 + list(cfg['commands']['ranges_height'])
        args.yaw_cycle = args.yaw_cycle or [0, 0, 0, 0, 0, .3, -.3, 0, 0]
    args.no_heading_hold = True  # Preserve constant-yaw-rate training commands.
    keyboard_enabled = not args.headless and not args.no_keyboard
    keyboard_config = None
    if keyboard_enabled:
        try:
            keyboard_config = KeyboardConfig(
                speed=args.keyboard_speed, yaw_rate=args.keyboard_yaw_rate,
                height_min=cfg['commands']['ranges_height'][0], height_max=cfg['commands']['ranges_height'][1],
                initial_height=args.fixed_command[2], height_step=args.keyboard_height_step,
                height_rate=args.keyboard_height_rate, height_slew_rate=args.keyboard_height_slew_rate)
            if keyboard_config.speed > .8 or keyboard_config.yaw_rate > .6:
                raise ValueError('Keyboard speed/yaw rate must not exceed 0.8 m/s / 0.6 rad/s')
        except ValueError as error:
            parser.error(str(error))
        for value, field in ((keyboard_config.speed, 'ranges_lin_vel_x'), (keyboard_config.yaw_rate, 'ranges_ang_vel_yaw')):
            bounds = cfg['commands'].get(field)
            # Grouped walking samples reverse commands separately from the
            # positive forward range. Do not label those samples out-of-range.
            grouped_speed = field == 'ranges_lin_vel_x' and cfg['commands'].get('grouped_training', False)
            outside = bounds and (value > max(abs(v) for v in bounds) if grouped_speed else
                                  (-value < bounds[0] or value > bounds[1]))
            if outside:
                print(f"WARNING: keyboard ±{value:g} exceeds training {field}={bounds}; "
                      "this is exploratory replay, not guaranteed tracking.", flush=True)
    if args.max_steps is None:
        args.max_steps = round(9 * args.phase_duration / (cfg['sim']['dt'] * cfg['decimation'])) if args.demo else (30000 if keyboard_enabled else 1000)
    if args.max_steps < 1 or args.phase_duration <= 0 or args.reset_velocity < 0:
        parser.error("max-steps/phase-duration must be positive; reset-velocity must be non-negative")
    dt = cfg["sim"]["dt"] * cfg["decimation"]
    for values, bounds, name in ((args.height_cycle or [args.fixed_command[2]], cfg["commands"]["ranges_height"], "height"),
                                 (args.velocity_cycle or [args.fixed_command[0]], [-.8, .8], "speed"),
                                 (args.yaw_cycle or [args.fixed_command[1]], [-.5, .5], "yaw rate")):
        if any(not np.isfinite(v) or not bounds[0] <= v <= bounds[1] for v in values):
            parser.error(f"{name} must be finite and in {bounds}")
    output = args.output or Path(__file__).parent / "outputs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(env_path, output / "env_source.yaml")
    if agent_path.exists():
        shutil.copyfile(agent_path, output / "agent_source.yaml")
    torch.set_num_threads(1)
    actor = torch.jit.load(str(args.checkpoint), map_location="cpu").eval() if args.jit else load_actor(args.checkpoint, agent_path)
    if args.mjcf:
        source = args.mjcf.expanduser().resolve()
        if cfg['leg_model'] == 'mine_five_bar':
            audit = json.loads(source.with_suffix('.json').read_text())
            if Path(audit['source_usd']) != Path(cfg['robot']['spawn']['usd_path']).resolve():
                raise ValueError('MJCF source differs from the checkpoint training asset')
            if hashlib.sha256(source.read_bytes()).hexdigest() != audit['mjcf_sha256']:
                raise ValueError('Converted MJCF changed since audit')
            for mesh, expected in audit['mesh_sha256'].items():
                if hashlib.sha256(Path(mesh).read_bytes()).hexdigest() != expected:
                    raise ValueError('Converted mesh changed: ' + mesh)
            for layer, expected in audit['usd_layers'].items():
                if hashlib.sha256(Path(layer).read_bytes()).hexdigest() != expected:
                    raise ValueError('Converted USD dependency changed: ' + layer)
            shutil.copyfile(source.with_suffix('.json'), output / 'model_audit.json')
        model = mujoco.MjModel.from_xml_path(str(source))
        shutil.copyfile(source, output / 'scene.xml')
    else:
        model = build_model(cfg, output / "scene.xml")
    substeps = args.physics_substeps if args.physics_substeps is not None else 50
    if substeps < 1:
        parser.error('physics-substeps must be positive')
    cfg['_mujoco_physics_substeps'] = substeps
    model.opt.timestep = cfg['sim']['dt'] / substeps
    sim = Simulation(cfg, model)
    rng = np.random.default_rng(args.seed)
    rotation = sim.data.xmat[sim.base_id].reshape(3, 3)
    linear_com = rng.uniform(-args.reset_velocity, args.reset_velocity, 3)
    angular_world = rng.uniform(-args.reset_velocity, args.reset_velocity, 3)
    if args.reset_rng == 'isaac-torch':
        initial_velocity = ((2 * torch.rand(6, generator=torch.Generator().manual_seed(args.seed)) - 1) * args.reset_velocity).numpy()
        linear_com, angular_world = initial_velocity[:3], initial_velocity[3:]
    # Isaac Lab reset writes COM velocities; MuJoCo qvel uses the link origin.
    sim.data.qvel[:3] = linear_com - np.cross(angular_world, rotation @ model.body_ipos[sim.base_id])
    sim.data.qvel[3:6] = rotation.T @ angular_world
    mujoco.mj_forward(model, sim.data)
    requested = np.array(args.fixed_command, dtype=float)
    sim.command = np.array([0., requested[1], requested[2]])
    heading_target = sim.body_state()[3]
    previous_yaw_request = requested[1]
    metadata = {"checkpoint": str(args.checkpoint), "sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "env_config": str(env_path), "agent_config": str(agent_path),
                "env_config_sha256": hashlib.sha256(env_path.read_bytes()).hexdigest(),
                "agent_config_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest() if agent_path.exists() else None,
                "mujoco": mujoco.__version__, "torch": torch.__version__, "seed": args.seed,
                "physics_dt": cfg["sim"]["dt"], "control_dt": dt, "arguments": vars(args),
                "mujoco_integration_dt": model.opt.timestep, "physics_substeps": substeps,
                "keyboard_enabled": keyboard_enabled,
                "keyboard_config": vars(keyboard_config) if keyboard_config else None,
                "initialization": "Imported USD/checkpoint pose; explicit uniform velocity perturbation; no automatic resets",
                "initial_com_linear_velocity": linear_com.tolist(), "initial_world_angular_velocity": angular_world.tolist(),
                "contact": "composed training USD; ordinary hulls/base box; protected CAD CoACD pieces; stop-shank self contact",
                "limitation": "PhysX/MuJoCo contact solvers are different; this is transfer validation, not identical dynamics."}
    metadata["source_sha256"] = {str(path.relative_to(Path(__file__).resolve().parents[2])):
                                 hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in (Path(__file__), Path(__file__).with_name("core.py"),
                                              Path(__file__).with_name("keyboard.py"), STATE_SOURCE,
                                              Path(__file__).resolve().parents[1] / "wheel_legged_gym_isaaclab/robot_model.py")}
    if cfg['leg_model'] == 'mine_five_bar':
        for path in (Path(__file__).with_name('mine_model.py'), Path(__file__).resolve().parents[1] / 'wheel_legged_gym_isaaclab/five_bar_vmc.py'):
            metadata['source_sha256'][str(path.relative_to(Path(__file__).resolve().parents[2]))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    rows, status, failure_time = [], "completed", None
    with ExitStack() as stack:
        viewer, keyboard = None, None
        if not args.headless:
            from mujoco import viewer as mj_viewer
            viewer = stack.enter_context(mj_viewer.launch_passive(model, sim.data))
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 1.4, -22, 135
            viewer.opt.geomgroup[3] = 0
            if keyboard_enabled:
                keyboard = MujocoKeyboardController(keyboard_config)
                stack.callback(keyboard.close)
        handle = stack.enter_context((output / "trace.csv").open("w", newline=""))
        writer = None
        with torch.inference_mode():
            for step in range(args.max_steps):
                start = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    status = "viewer_closed"
                    break
                phase = int(round(step * dt, 8) / args.phase_duration)
                for index, cycle in enumerate((args.velocity_cycle, args.yaw_cycle, args.height_cycle)):
                    requested[index] = cycle[phase % len(cycle)] if cycle else args.fixed_command[index]
                if keyboard is not None:
                    manual = keyboard.advance(dt, sim.command[2])
                    if manual is not None:
                        requested[:] = manual
                manual_active = keyboard is not None and keyboard.state.active
                current_yaw = sim.body_state()[3]
                if requested[1] == 0 and previous_yaw_request != 0:
                    heading_target = current_yaw
                previous_yaw_request = requested[1]
                limit = cfg["commands"]["linear_acceleration_limit"] * dt
                sim.command[0] += np.clip(requested[0] - sim.command[0], -limit, limit)
                sim.command[1:] = requested[1:]
                if requested[1] == 0 and not args.no_heading_hold and not manual_active:
                    error = (heading_target - current_yaw + np.pi) % (2 * np.pi) - np.pi
                    sim.command[1] = np.clip(cfg["commands"]["heading_kp"] * error,
                                            -cfg["commands"]["heading_rate_limit"], cfg["commands"]["heading_rate_limit"])
                sim.set_action(actor(sim.observation()))
                sim.step()
                if not np.isfinite(sim.data.qpos).all() or not np.isfinite(sim.data.qvel).all() or sim.data.warning.number.any():
                    status, failure_time = "numerical_failure", sim.data.time
                    break
                angular, _, velocity, yaw, pitch, roll = sim.body_state()
                _, _, length, angle, *_ = sim.coordinates()
                row = {"step": step + 1, "time": sim.data.time, "phase": phase,
                       "cmd_x_target": requested[0], "cmd_x": sim.command[0], "cmd_yaw": sim.command[1],
                       "yaw_rate_requested": requested[1],
                       "command_source": "keyboard" if manual_active else ("cycle" if any((args.velocity_cycle, args.yaw_cycle, args.height_cycle)) else "fixed"),
                       "keyboard_active": int(manual_active),
                       "keyboard_height_target": keyboard.state.target_height if manual_active else sim.command[2],
                       "keyboard_jump_requests": keyboard.state.jump_requests if keyboard is not None else 0,
                       "cmd_height": sim.command[2], "vel_x_heading": velocity[0], "vel_y_heading": velocity[1],
                       "base_height": sim.data.qpos[2], "height_error_mm": 1000 * (sim.data.qpos[2] - requested[2]),
                       "yaw_deg": np.degrees(yaw), "pitch_deg": np.degrees(pitch), "roll_deg": np.degrees(roll),
                       "yaw_rate": angular[2], "left_l0": length[0, 0].item(), "right_l0": length[0, 1].item(),
                       "left_theta0_deg": np.degrees(angle[0, 0].item()), "right_theta0_deg": np.degrees(angle[0, 1].item()),
                       "left_l0_ref": sim.length_ref[0], "right_l0_ref": sim.length_ref[1]}
                row.update({f"torque_{i}": value for i, value in enumerate(sim.torques)})
                row.update({f"action_{i}": value for i, value in enumerate(sim.raw.numpy()[0])})
                if cfg['leg_model'] == 'mine_five_bar':
                    gaps = [np.linalg.norm(sim.data.site_xpos[model.eq_obj1id[i]] - sim.data.site_xpos[model.eq_obj2id[i]])
                            for i in range(model.neq)]
                    row['closure_gap_max_mm'] = 1000 * max(gaps, default=0.)
                    row['contact_penetration_max_mm'] = 1000 * max(0., max((-contact.dist for contact in sim.data.contact), default=0.))
                    row['left_wheel_rad_s'] = sim.data.qvel[sim.vids[2]]
                    row['right_wheel_rad_s'] = -sim.data.qvel[sim.vids[5]]
                    row['five_bar_valid'] = bool(sim.kinematics_valid.all())
                    for side, side_id, wheel_action in [('left', 0, 2), ('right', 1, 5)]:
                        row[side + '_angle_ref_rad'] = float(sim.angle_ref[side_id])
                        row[side + '_wheel_ref_rad_s'] = float(sim.actions[0, wheel_action].item() * cfg['action_scale_vel'])
                    stop_contacts = [contact for contact in sim.data.contact
                                     if {model.geom_contype[contact.geom1], model.geom_contype[contact.geom2]} == {4, 8}]
                    row['stop_contact_count'] = len(stop_contacts)
                    row['stop_penetration_max_mm'] = 1000 * max(0., max((-contact.dist for contact in stop_contacts), default=0.))
                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=row)
                    writer.writeheader()
                writer.writerow(row)
                rows.append(row)
                if step % 50 == 0:
                    print(f"t={sim.data.time:6.2f}s vx={velocity[0]:+.3f} height={sim.data.qpos[2]:.3f}m "
                          f"error={row['height_error_mm']:+.1f}mm pitch={row['pitch_deg']:+.1f} roll={row['roll_deg']:+.1f}", flush=True)
                if sim.data.qpos[2] < cfg.get('min_root_height', .07) or abs(pitch) > cfg["max_body_pitch"] or abs(roll) > cfg["max_body_roll"]:
                    status, failure_time = "posture_failure", sim.data.time
                    break
                if viewer is not None:
                    if keyboard is not None and step % 5 == 0:
                        keyboard.show_state(sim.command, velocity[0], sim.data.qpos[2], row['pitch_deg'], row['roll_deg'])
                    viewer.cam.lookat[:] = sim.data.xpos[sim.base_id]
                    viewer.sync()
                    time.sleep(max(0, dt - (time.monotonic() - start)))
        if args.snapshot:
            with mujoco.Renderer(model, height=480, width=640) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = sim.data.xpos[sim.base_id]
                camera.distance, camera.azimuth, camera.elevation = 1.4, 135, -22
                visual_options = mujoco.MjvOption()
                visual_options.geomgroup[3] = 0
                renderer.update_scene(sim.data, camera, scene_option=visual_options)
                from PIL import Image
                Image.fromarray(renderer.render()).save(output / "snapshot.png")
    summary = {"status": status, "failure_time_s": failure_time, "steps": len(rows), "duration_s": sim.data.time,
               "conditions": []}
    for phase in sorted({row["phase"] for row in rows}):
        phase_rows = [row for row in rows if row["phase"] == phase]
        tail = phase_rows[-max(1, round(2 / dt)):]
        summary["conditions"].append({"phase": phase, "target_vx": tail[-1]["cmd_x_target"],
                                      "constant_targets": len({(r['cmd_x_target'], r['yaw_rate_requested'], r['cmd_height']) for r in tail}) == 1,
                                      "command_sources": sorted({r['command_source'] for r in tail}),
                                      "target_height": tail[-1]["cmd_height"],
                                      "target_yaw_rate": tail[-1]["yaw_rate_requested"],
                                      "phase_complete": len(phase_rows) * dt >= args.phase_duration - dt / 2,
                                      "samples": len(tail), "window_duration_s": len(tail) * dt,
                                      "mean_speed_m_s": float(np.mean([r["vel_x_heading"] for r in tail])),
                                      "mean_height_m": float(np.mean([r["base_height"] for r in tail])),
                                      "speed_mae_m_s": float(np.mean([abs(r["vel_x_heading"] - r["cmd_x_target"]) for r in tail])),
                                      "height_mae_mm": float(np.mean([abs(r["height_error_mm"]) for r in tail])),
                                      "yaw_rate_mae_rad_s": float(np.mean([abs(r["yaw_rate"] - r["yaw_rate_requested"]) for r in tail])),
                                      "max_abs_pitch_deg": max(abs(r["pitch_deg"]) for r in tail),
                                      "max_abs_roll_deg": max(abs(r["roll_deg"]) for r in tail)})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        times = [r["time"] for r in rows]
        for axis, fields, label in zip(axes, (("cmd_x_target", "vel_x_heading"), ("cmd_height", "base_height"),
                                               ("pitch_deg", "roll_deg")), ("Speed (m/s)", "Height (m)", "Attitude (deg)")):
            for field in fields:
                axis.plot(times, [r[field] for r in rows], label=field)
            axis.set_ylabel(label)
            axis.grid(alpha=.3)
            axis.legend()
        axes[-1].set_xlabel("Simulation time (s)")
        fig.tight_layout()
        fig.savefig(output / "tracking.png", dpi=150)
        plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(f"Artifacts: {output}")
    return 0 if status in ("completed", "viewer_closed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
