"""Offline USD-tree and recorded native initial-pose audit; no simulator or PPO.

Reconstructs the imported3D joint tree independently of the planar reset helper,
checks the omitted loop-joint anchor/axis constraints, tire-ground clearance and
whole-body COM projection. Native first observations establish what the actor
actually saw; geometric balance diagnostics do not identify a failure cause.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
USD_LIB = next((ROOT/'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent
if str(USD_LIB/'bin') not in os.environ.get('LD_LIBRARY_PATH','').split(':'):
    environment = dict(os.environ)
    environment['LD_LIBRARY_PATH'] = str(USD_LIB/'bin')+':'+environment.get('LD_LIBRARY_PATH','')
    os.execve(sys.executable,[sys.executable,__file__,*sys.argv[1:]],environment)
sys.path.insert(0,str(USD_LIB))

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdPhysics, UsdGeom


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transform(position, quaternion):
    result = np.eye(4)
    result[:3,3] = np.asarray(position)
    result[:3,:3] = Rotation.from_quat([*quaternion.GetImaginary(),quaternion.GetReal()]).as_matrix()
    return result


def audit(training_output, destination):
    training_output, destination = Path(training_output).resolve(),Path(destination).resolve()
    if destination.exists():
        raise ValueError('Preserve earlier diagnostic evidence')
    plan = json.loads((training_output/'experiment_plan.json').read_text())
    cfg = plan['training_config']
    if (cfg['leg_control_mode'] != 'implicit_joint_position'
            or cfg['joint_reset_profile'] != 'joint_target_nominal_v1'
            or cfg['robot']['init_state']['rot'] != [1.,0.,0.,0.]):
        raise ValueError('Explicitly handle any different initial control/pose convention')
    asset = Path(cfg['robot']['spawn']['usd_path'])
    stage = Usd.Stage.Open(str(asset))
    bodies = {str(p.GetPath()):p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    joints = [UsdPhysics.RevoluteJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    cache = UsdGeom.XformCache()
    base = next(path for path,p in bodies.items() if p.GetName() == 'base_link')
    authored_base = np.asarray(cache.GetLocalToWorldTransform(bodies[base])).T
    rest = {path:np.linalg.inv(authored_base)@np.asarray(cache.GetLocalToWorldTransform(p)).T
            for path,p in bodies.items()}
    tree = [j for j in joints if j.GetJointEnabledAttr().Get() and not j.GetExcludeFromArticulationAttr().Get()]
    q = cfg['robot']['init_state']['joint_pos']
    if len(bodies) != 15 or len(tree) != 14 or {j.GetPrim().GetName() for j in tree} != set(q):
        raise ValueError('The actual imported joint tree and all14 configured reset angles differ')

    def forward(angles):
        states, remaining = {base:np.eye(4)}, list(tree)
        while remaining:
            progressed = False
            for j in remaining[:]:
                parent,child = (str(j.GetBody0Rel().GetTargets()[0]),str(j.GetBody1Rel().GetTargets()[0]))
                if parent not in states:
                    continue
                t0 = transform(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get())
                t1 = transform(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get())
                motion = np.eye(4)
                axis = np.eye(3)['XYZ'.index(j.GetAxisAttr().Get())]
                motion[:3,:3] = Rotation.from_rotvec(axis*angles.get(j.GetPrim().GetName(),0.)).as_matrix()
                states[child] = states[parent]@t0@motion@np.linalg.inv(t1)
                remaining.remove(j)
                progressed = True
            if not progressed:
                raise ValueError('Could not traverse the complete imported tree')
        return states

    zero, nominal = forward({}), forward(q)
    zero_position_error = max(float(np.linalg.norm(zero[path][:3,3]-rest[path][:3,3])) for path in bodies)
    zero_rotation_error = max(float(Rotation.from_matrix(zero[path][:3,:3]@rest[path][:3,:3].T).magnitude()) for path in bodies)
    if zero_position_error > 1.e-6 or zero_rotation_error > 1.e-6:
        raise ValueError('Independent3D zero-angle FK does not reproduce authored USD frames')
    loops = []
    for j in joints:
        if not j.GetJointEnabledAttr().Get() or not j.GetExcludeFromArticulationAttr().Get():
            continue
        p0,p1 = (str(j.GetBody0Rel().GetTargets()[0]),str(j.GetBody1Rel().GetTargets()[0]))
        f0 = nominal[p0]@transform(j.GetLocalPos0Attr().Get(),j.GetLocalRot0Attr().Get())
        f1 = nominal[p1]@transform(j.GetLocalPos1Attr().Get(),j.GetLocalRot1Attr().Get())
        axis = 'XYZ'.index(j.GetAxisAttr().Get())
        angle = float(np.arccos(np.clip(np.dot(f0[:3,axis],f1[:3,axis]),-1.,1.)))
        loops.append(dict(joint=j.GetPrim().GetName(),anchor_gap_m=float(np.linalg.norm(f0[:3,3]-f1[:3,3])),
                          axis_error_rad=angle))
    if len(loops) != 4 or any(row['anchor_gap_m'] > 1.e-6 or row['axis_error_rad'] > 1.e-6 for row in loops):
        raise ValueError('Configured initial angles violate an actual USD closed loop')
    limits = []
    for j in tree:
        name = j.GetPrim().GetName()
        value = math.degrees(q[name])
        lower,upper = j.GetLowerLimitAttr().Get(),j.GetUpperLimitAttr().Get()
        if lower < upper and not lower <= value <= upper:
            raise ValueError('Configured reset joint lies outside authored limits: '+name)
        limits.append(dict(joint=name,angle_deg=value,within_authored_limits=True))

    mass,center = 0.,np.zeros(3)
    wheels = {}
    for path,body in bodies.items():
        api = UsdPhysics.MassAPI(body)
        m = api.GetMassAttr().Get()
        center += m*(nominal[path]@np.r_[list(api.GetCenterOfMassAttr().Get()),1.])[:3]
        mass += m
        if body.GetName() in ('LW_link','RW_link'):
            prim = stage.GetPrimAtPath(path+'/analytic_tire_collision')
            relative = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)@np.asarray(cache.GetLocalToWorldTransform(prim)).T
            frame = nominal[path]@relative
            shape = UsdGeom.Cylinder(prim)
            axis = frame[:3,'XYZ'.index(shape.GetAxisAttr().Get())]
            extent = float(shape.GetRadiusAttr().Get()*np.sqrt(max(0.,1.-axis[2]**2))
                +shape.GetHeightAttr().Get()*.5*abs(axis[2]))
            wheels[body.GetName()] = dict(center_in_base_m=frame[:3,3].tolist(),
                axis_in_base=axis.tolist(),lowest_world_z_m=cfg['robot']['init_state']['pos'][2]+frame[2,3]-extent)
    center /= mass
    wheel_center = np.mean([row['center_in_base_m'] for row in wheels.values()],axis=0)
    if len(wheels) != 2 or abs(mass-12.34) > 1.e-4:
        raise ValueError('Use the current measured vehicle and both analytic wheels')
    if max(abs(row['lowest_world_z_m']) for row in wheels.values()) > 1.e-5:
        raise ValueError('The configured initial root height leaves the analytic wheels off the ground')

    sample = training_output/'native_learning_signal/rollout_0001.npz'
    with np.load(sample,allow_pickle=False) as a:
        o = a['observations'][0]
        offset = np.asarray(cfg['leg_joint_position_offset'])
        actual_q = o[:,9:13]/cfg['obs_scales']['dof_pos']+offset
        q_error = float(abs(actual_q-offset).max())
        gravity_error = float(abs(o[:,3:6]-np.array([0.,0.,-1.])).max())
        velocity = o[:,13:17]/cfg['obs_scales']['dof_vel']
        if q_error > 1.e-7 or gravity_error > 1.e-7 or abs(velocity).max() > 1.e-7:
            raise ValueError('Native first observations differ from the configured level, nominal joint pose')
        angular = o[:,:3]/cfg['obs_scales']['ang_vel']
        first_delta = a['native_leg_targets'][0]-actual_q
        native = dict(sampled_environments=len(o),active_angles_deg=np.rad2deg(offset).tolist(),
            active_angles_vs_zero_action_max_error_rad=q_error,
            body_gravity_vs_level_max_error=gravity_error,initial_joint_velocity_abs_max=float(abs(velocity).max()),
            initial_body_angular_velocity_min_rad_s=angular.min(0).tolist(),
            initial_body_angular_velocity_max_rad_s=angular.max(0).tolist(),
            initial_action_history_abs_max=float(abs(o[:,21:27]).max()),
            first_policy_target_jump_deg_min=np.rad2deg(first_delta.min(0)).tolist(),
            first_policy_target_jump_deg_max=np.rad2deg(first_delta.max(0)).tolist())
    result = dict(observed_at=datetime.now().astimezone().isoformat(),
        current_root_height_m=cfg['robot']['init_state']['pos'][2],current_root_quaternion_wxyz=cfg['robot']['init_state']['rot'],
        all14_initial_joint_angles_deg=limits,all14_tree_joints_accounted_for=True,
        independent_USD_zero_pose_position_error_m=zero_position_error,
        independent_USD_zero_pose_rotation_error_rad=zero_rotation_error,nominal_loop_checks=loops,
        all_closed_loops_consistent=True,both_analytic_tires_on_ground=True,wheels=wheels,
        measured_vehicle_mass_kg=mass,whole_vehicle_COM_in_base_m=center.tolist(),
        COM_minus_mean_wheel_center_in_base_m=(center-wheel_center).tolist(),
        body_Y_rotation_to_align_COM_at_fixed_joint_pose_deg=math.degrees(math.atan2(
            -(center[0]-wheel_center[0]),center[2]-wheel_center[2])),
        COM_projection_angle_is_geometric_diagnostic_not_new_reset_setting=True,
        native_initial_observation=native,reset_velocity_fields=[cfg['reset_velocity_initial'],cfg['reset_velocity_final']],
        training_modified=False,simulator_started=False,PPO_updates_performed=0,policy_accepted=False,
        source_files={str(p):digest(p) for p in (asset,training_output/'experiment_plan.json',sample,Path(__file__))},
        limitation='USD3D kinematic closure and recorded32-environment initial state only. '
        'A COM/support-line offset or first-action target jump is a potential startup contributor, '
        'not proof of cause, live constraint force, rare-failure probability or new-pose validation.')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.training_output,args.output),indent=2))
