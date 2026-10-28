"""Read composed wheel axes, limits and drive metadata without starting Kit.

Authored USD values are distinct from actuator settings applied at runtime.
This never changes an asset, gain, checkpoint, trainer or acceptance gate.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

from auto_train_mine import write_json
from diagnose_plain_joint_pd_height import cpu_usd_environment
from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


def finite_value(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else ('+inf' if value > 0 else '-inf')


def inspect(checkpoint):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    checkpoint = Path(checkpoint).resolve()
    cfg = validate_checkpoint_robot(checkpoint)
    asset = Path(cfg['robot']['spawn']['usd_path'])
    before = sha(asset)
    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise ValueError('Cannot read the composed paired asset')
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    boxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
    base = [prim for prim in stage.Traverse() if prim.GetName() == 'base_link']
    if len(base) != 1:
        raise ValueError('Select the unique physical base_link before interpreting axle directions')
    inverse_base = cache.GetLocalToWorldTransform(base[0]).GetInverse()
    names = cfg['wheel_joint_names']
    joints = {name: [p for p in stage.Traverse()
                     if p.GetName() == name and p.IsA(UsdPhysics.RevoluteJoint)] for name in names}
    if any(len(value) != 1 for value in joints.values()):
        raise ValueError('Require one composed revolute joint per configured wheel')
    rows = []
    for name in names:
        prim = joints[name][0]
        joint = UsdPhysics.RevoluteJoint(prim)
        body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            raise ValueError('Require both articulated wheel joint bodies')
        parent, wheel = stage.GetPrimAtPath(body0[0]), stage.GetPrimAtPath(body1[0])
        axis_name = str(joint.GetAxisAttr().Get())
        axis = {'X': Gf.Vec3d(1, 0, 0), 'Y': Gf.Vec3d(0, 1, 0), 'Z': Gf.Vec3d(0, 0, 1)}[axis_name]
        quat = joint.GetLocalRot0Attr().Get()
        rotation = Gf.Rotation(Gf.Quatd(float(quat.GetReal()), Gf.Vec3d(*quat.GetImaginary())))
        world = cache.GetLocalToWorldTransform(parent).TransformDir(rotation.TransformDir(axis))
        world.Normalize()
        relative = inverse_base.TransformDir(world)
        relative.Normalize()
        lower, upper = joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()
        free = (lower is not None and upper is not None and
                (lower > upper or (float(lower) < -1.e6 and float(upper) > 1.e6)))
        drive = UsdPhysics.DriveAPI(prim, 'angular')
        mass = UsdPhysics.MassAPI(wheel)
        bound = boxes.ComputeWorldBound(wheel).ComputeAlignedRange()
        attributes = {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                      if any(token in a.GetName().lower() for token in ('friction', 'limit', 'drive', 'armature'))}
        rows.append(dict(name=name, joint_path=str(prim.GetPath()),
            parent_body=str(body0[0]), wheel_body=str(body1[0]),
            authored_local_axis=axis_name, axis_in_world=list(world), axis_in_base_link=list(relative),
            parallel_to_base_lateral_axis=abs(relative[1]) > .999,
            authored_lower_limit_deg=finite_value(lower), authored_upper_limit_deg=finite_value(upper),
            effectively_unbounded_position=bool(free),
            authored_drive_stiffness=finite_value(drive.GetStiffnessAttr().Get()),
            authored_drive_damping=finite_value(drive.GetDampingAttr().Get()),
            authored_drive_max_force=finite_value(drive.GetMaxForceAttr().Get()),
            authored_wheel_mass_kg=finite_value(mass.GetMassAttr().Get()),
            world_geometry_bbox_size_m=list(bound.GetSize()), authored_attributes=attributes))
    if sha(asset) != before:
        raise ValueError('Asset changed during read-only inspection')
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        asset=str(asset), asset_sha256=before, base_frame=str(base[0].GetPath()), wheels=rows,
        paired_runtime_wheel_actuator=cfg['robot']['actuators']['wheels'],
        paired_wheel_control_mode=cfg['wheel_control_mode'],
        diagnostic_script_sha256=sha(Path(__file__)), simulator_started=False,
        assets_or_training_modified=False, policy_accepted=False,
        limitation='Composed authored asset and paired configuration only. Runtime actuator settings can override '
                   'authored drives; this is not a live PhysX drive-property readback or dynamic contact/friction test.')


if __name__ == '__main__':
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        if os.environ.get('CLT_CPU_USD_DIAG_READY'):
            raise RuntimeError('CPU USD libraries unavailable; no simulator started')
        os.execve(sys.executable, [sys.executable, *sys.argv], cpu_usd_environment())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve previous evidence; use a fresh output')
    result = inspect(args.checkpoint)
    write_json(args.output, result)
    print(json.dumps(result))
