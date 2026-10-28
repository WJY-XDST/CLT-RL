"""Author two independent free-rotor test rigs; never edit the robot asset."""
import argparse
import json
import os
from pathlib import Path
import sys

from auto_train_mine import write_json
from diagnose_plain_joint_pd_height import cpu_usd_environment
from joint_pd_continuation import sha


def build(directory):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    directory = Path(directory).resolve()
    directory.mkdir(exist_ok=False)
    mass, radius, width = .665, .06, .0651
    axial = .5 * mass * radius**2
    transverse = mass * (3 * radius**2 + width**2) / 12
    assets = {}
    for drive_type in ('acceleration', 'force'):
        path = directory / f'free_rotor_{drive_type}.usda'
        stage = Usd.Stage.CreateNew(str(path))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.)
        root = UsdGeom.Xform.Define(stage, '/Rig')
        stage.SetDefaultPrim(root.GetPrim())
        base = UsdGeom.Xform.Define(stage, '/Rig/base').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(base)
        UsdPhysics.ArticulationRootAPI.Apply(base)
        base_mass = UsdPhysics.MassAPI.Apply(base)
        base_mass.CreateMassAttr(1.)
        base_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.01, .01, .01))
        base_mass.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        base_mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))
        wheel = UsdGeom.Xform.Define(stage, '/Rig/wheel').GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(wheel)
        wheel_mass = UsdPhysics.MassAPI.Apply(wheel)
        wheel_mass.CreateMassAttr(mass)
        wheel_mass.CreateDiagonalInertiaAttr(Gf.Vec3f(transverse, axial, transverse))
        wheel_mass.CreateCenterOfMassAttr(Gf.Vec3f(0.))
        wheel_mass.CreatePrincipalAxesAttr(Gf.Quatf(1.))
        visual = UsdGeom.Cylinder.Define(stage, '/Rig/wheel/visual')
        visual.CreateAxisAttr('Y')
        visual.CreateRadiusAttr(radius)
        visual.CreateHeightAttr(width)
        # No CollisionAPI: known inertia, no contact or gravity drag.
        fixed = UsdPhysics.FixedJoint.Define(stage, '/Rig/root_fix')
        fixed.CreateBody1Rel().SetTargets([base.GetPath()])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(0.))
        fixed.CreateLocalPos1Attr(Gf.Vec3f(0.))
        joint = UsdPhysics.RevoluteJoint.Define(stage, '/Rig/wheel_joint')
        joint.CreateBody0Rel().SetTargets([base.GetPath()])
        joint.CreateBody1Rel().SetTargets([wheel.GetPath()])
        joint.CreateAxisAttr('Y')
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.))
        joint.CreateLowerLimitAttr(float('-inf'))
        joint.CreateUpperLimitAttr(float('inf'))
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
        drive.CreateTypeAttr(drive_type)
        drive.CreateStiffnessAttr(0.)
        drive.CreateDampingAttr(0.)  # runtime actuator must set the requested gain
        drive.CreateMaxForceAttr(10.)
        stage.GetRootLayer().Save()
        reopened = Usd.Stage.Open(str(path))
        check = UsdPhysics.DriveAPI(reopened.GetPrimAtPath('/Rig/wheel_joint'), 'angular')
        assert check.GetTypeAttr().Get() == drive_type
        assert len([p for p in reopened.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]) == 1
        assets[drive_type] = dict(path=str(path), sha256=sha(path))
    result = dict(assets=assets, wheel_mass_kg=mass, wheel_radius_m=radius,
        wheel_width_m=width, axial_inertia_kg_m2=axial, dt_s=.005, duration_s=1.,
        runtime_stiffness=0., runtime_damping=.5, runtime_effort_limit_Nm=4.80944,
        target_velocity_rad_s=2., gravity=[0., 0., 0.], contact_shapes=0,
        builder_sha256=sha(Path(__file__)), simulator_started=False,
        robot_asset_changed=False, training_changed=False, policy_accepted=False,
        purpose='Isolated drive-semantic test using identical free rotors and actual Articulation/ImplicitActuator initialization. Not a robot tracking acceptance test.',
        theory_sources=['https://nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html'])
    write_json(directory / 'manifest.json', result)
    return result


if __name__ == '__main__':
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        if os.environ.get('CLT_CPU_USD_DIAG_READY'):
            raise RuntimeError('CPU USD library unavailable')
        os.execve(sys.executable, [sys.executable, *sys.argv], cpu_usd_environment())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.output)))
