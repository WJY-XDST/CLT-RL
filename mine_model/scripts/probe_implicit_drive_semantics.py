"""Measure two free rotors through real IsaacLab actuator initialization.

Run only through run_python.sh --headless, serialized by the native SDK lock.
No PPO, robot checkpoint, robot asset, hardware or acceptance gate changes.
"""
import argparse
import hashlib
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
manifest = json.loads(args.manifest.read_text())
for row in manifest['assets'].values():
    if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() != row['sha256']:
        raise ValueError('Probe asset changed after preparation')
args.output.mkdir(exist_ok=False)
app = AppLauncher(args).app

try:
    import math
    import torch
    from pxr import UsdPhysics
    from isaacsim.core.utils.stage import get_current_stage
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg
    from auto_train_mine import write_json

    dt = manifest['dt_s']
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
        dt=dt, gravity=(0., 0., 0.), device=args.device,
        physx=sim_utils.PhysxCfg(solver_type=1)))
    robots = {}
    for kind, asset in manifest['assets'].items():
        cfg = ArticulationCfg(prim_path='/World/Probe_' + kind,
            spawn=sim_utils.UsdFileCfg(usd_path=asset['path'],
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False)),
            init_state=ArticulationCfg.InitialStateCfg(pos=(0., 0., 0.),
                joint_pos={'wheel_joint': 0.}, joint_vel={'wheel_joint': 0.}),
            actuators={'wheel': ImplicitActuatorCfg(joint_names_expr=['wheel_joint'],
                stiffness=0., damping=manifest['runtime_damping'],
                effort_limit_sim=manifest['runtime_effort_limit_Nm'], armature=0.,
                friction=0., dynamic_friction=0., viscous_friction=0.)})
        robots[kind] = Articulation(cfg)
    sim.reset()
    runtime = {}
    for kind, robot in robots.items():
        robot.reset()
        if robot.num_joints != 1 or robot.joint_names != ['wheel_joint']:
            raise ValueError('Probe rig is not the intended one-DOF articulation')
        view = robot.root_physx_view
        drive = UsdPhysics.DriveAPI(get_current_stage().GetPrimAtPath('/World/Probe_' + kind + '/wheel_joint'), 'angular')
        runtime[kind] = dict(joints=robot.joint_names,
            USD_drive_type_after_initialization=str(drive.GetTypeAttr().Get()),
            backend_stiffness=view.get_dof_stiffnesses().cpu().tolist(),
            backend_damping=view.get_dof_dampings().cpu().tolist(),
            backend_effort_limit=view.get_dof_max_forces().cpu().tolist(),
            initial_velocity=robot.data.joint_vel.cpu().tolist())
        if float(view.get_dof_dampings().flatten()[0]) != manifest['runtime_damping']:
            raise ValueError('Actuator damping was not applied through the runtime backend')
    samples = []
    target = manifest['target_velocity_rad_s']
    for step in range(round(manifest['duration_s'] / dt)):
        for robot in robots.values():
            robot.set_joint_velocity_target(torch.full_like(robot.data.joint_vel, target))
            robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_vel))
            robot.write_data_to_sim()
        sim.step(render=False)
        row = dict(time_s=(step + 1) * dt)
        for kind, robot in robots.items():
            robot.update(dt)
            value = float(robot.data.joint_vel[0, 0])
            if not math.isfinite(value):
                raise ValueError('Nonfinite probe velocity')
            row[kind + '_velocity_rad_s'] = value
        samples.append(row)
    early = samples[round(.02 / dt) - 1]
    contrast = (early['force_velocity_rad_s'] > 1.5
                and early['acceleration_velocity_rad_s'] < .1)
    result = dict(completed=True, runtime=runtime, target_velocity_rad_s=target,
        free_rotor_axial_inertia_kg_m2=manifest['axial_inertia_kg_m2'],
        measured_at20ms=early, measured_at1s=samples[-1],
        distinct_force_and_acceleration_response=contrast, samples=samples,
        expected_continuous_20ms=dict(
            acceleration=target * (1. - math.exp(-manifest['runtime_damping'] * .02)),
            force=target * (1. - math.exp(-manifest['runtime_damping'] / manifest['axial_inertia_kg_m2'] * .02))),
        probe_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        assets=manifest['assets'], native_PPO_updates=0, robot_assets_changed=False,
        optimizer_or_checkpoint_changed=False, policy_accepted=False,
        limitation='Isolated zero-gravity no-contact unit-inertia probe. Confirms runtime drive semantics only; '
                    'not a full-robot controller comparison, proof of the entire tracking error cause, or acceptance.')
    write_json(args.output / 'result.json', result)
    print('DRIVE_SEMANTICS_PROBE_COMPLETE ' + json.dumps({k:v for k,v in result.items() if k != 'samples'}), flush=True)
except BaseException as error:
    (args.output / 'diagnosis_required.json').write_text(json.dumps(dict(
        reason=repr(error), native_PPO_updates=0, robot_assets_changed=False,
        optimizer_or_checkpoint_changed=False, policy_accepted=False), indent=2) + '\n')
    raise
finally:
    app.close()
