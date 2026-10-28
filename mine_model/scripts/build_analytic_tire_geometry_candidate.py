"""Create an isolated round-tire diagnostic layer; never change training USD.

The source CAD, complete rigid-body mass/inertia, joints and geometry remain
referenced. Only the two active tire collision hulls are replaced with exact
cylinder primitives of the same radius, width, center and local axle.
"""
import argparse
import json
import os
from pathlib import Path
import sys

from auto_train_mine import write_json
from diagnose_plain_joint_pd_height import cpu_usd_environment
from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain


def dynamics(stage):
    from pxr import UsdGeom, UsdPhysics
    result = {}
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI) or prim.IsA(UsdPhysics.Joint):
            result[str(prim.GetPath())] = dict(type=prim.GetTypeName(),
                attributes={a.GetName(): str(a.Get()) for a in prim.GetAttributes()},
                relationships={r.GetName(): list(map(str, r.GetTargets())) for r in prim.GetRelationships()},
                world_transform=str(cache.GetLocalToWorldTransform(prim)))
    return result


def build(source, output):
    import numpy as np
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
    source, output = Path(source).resolve(), Path(output).resolve()
    if sha(source) != '7353cdd90aefb795a158176bad75bd210d27d5ba775905765c859fedbe73b71e':
        raise ValueError('Use the current exact measured-mass physical asset')
    chain, _ = verify_asset_chain(source)
    original = Usd.Stage.Open(str(source))
    before = dynamics(original)
    output.mkdir(exist_ok=False)
    path = output / 'measured_mass_analytic_tires_diagnostic.usda'
    stage = Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().subLayerPaths = [str(source)]
    stage.SetDefaultPrim(stage.GetPrimAtPath(original.GetDefaultPrim().GetPath()))
    cache = UsdGeom.XformCache()
    changes, redirects = [], {}
    for name in ('LW_joint', 'RW_joint'):
        joint = next(UsdPhysics.RevoluteJoint(p) for p in original.Traverse()
                     if p.GetName() == name and p.IsA(UsdPhysics.RevoluteJoint))
        body = original.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        mesh_path = str(body.GetPath()) + '/smooth_tire_collision'
        mesh = original.GetPrimAtPath(mesh_path)
        if not mesh.IsA(UsdGeom.Mesh) or UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get() is False:
            raise ValueError('Expected the currently active smooth tire collider')
        rotation = joint.GetLocalRot1Attr().Get()
        local = {'X': Gf.Vec3d(1, 0, 0), 'Y': Gf.Vec3d(0, 1, 0), 'Z': Gf.Vec3d(0, 0, 1)}[str(joint.GetAxisAttr().Get())]
        axis = np.asarray(Gf.Rotation(Gf.Quatd(float(rotation.GetReal()),
            Gf.Vec3d(*rotation.GetImaginary()))).TransformDir(local))
        if not np.allclose(axis, [0., 0., 1.], atol=1.e-8):
            raise ValueError('This paired candidate expects the audited local-Z wheel axle')
        matrix = np.asarray(cache.GetLocalToWorldTransform(mesh)
                            * cache.GetLocalToWorldTransform(body).GetInverse())
        points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
        points = points @ matrix[:3, :3] + matrix[3, :3]
        center = points.mean(axis=0)
        radii = np.linalg.norm((points - center)[:, :2], axis=-1)
        width = float(np.ptp(points[:, 2]))
        if not (np.allclose(radii, .06, atol=1.e-7) and abs(width - .02) < 1.e-7):
            raise ValueError('Preserve the actual tire radius and width')
        new_path = str(body.GetPath()) + '/analytic_tire_collision'
        stage.OverridePrim(mesh_path).GetAttribute('physics:collisionEnabled').Set(False)
        cylinder = UsdGeom.Cylinder.Define(stage, new_path)
        cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
        cylinder.CreateRadiusAttr(.06)
        cylinder.CreateHeightAttr(.02)
        cylinder.AddTranslateOp().Set(Gf.Vec3d(*center))
        cylinder.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim()).CreateCollisionEnabledAttr(True)
        # Preserve explicit physical-material bindings if present. Parent
        # bindings are inherited unchanged because the rigid body is shared.
        for rel in mesh.GetRelationships():
            if rel.GetName().startswith('material:binding'):
                cylinder.GetPrim().CreateRelationship(rel.GetName()).SetTargets(rel.GetTargets())
        redirects[mesh_path] = new_path
        changes.append(dict(wheel_joint=name, disabled_collision=mesh_path,
            new_collision=new_path, radius_m=.06, width_m=.02,
            center_in_wheel_body_m=center.tolist(), axle_in_wheel_body=axis.tolist()))
    membership = []
    for prim in original.Traverse():
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            extra = [Sdf.Path(redirects[str(t)]) for t in targets if str(t) in redirects]
            if extra:
                stage.OverridePrim(prim.GetPath()).CreateRelationship(rel.GetName()).SetTargets(targets + extra)
                membership.append(str(rel.GetPath()))
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(path))
    if dynamics(reopened) != before:
        raise ValueError('Candidate changed rigid-body dynamics, joint authoring or transforms')
    if any(sha(Path(p)) != digest for p, digest in chain.items()):
        raise ValueError('Source asset chain changed while preparing a private candidate')
    mass = sum(float(UsdPhysics.MassAPI(p).GetMassAttr().Get()) for p in reopened.Traverse()
               if p.HasAPI(UsdPhysics.MassAPI) and UsdPhysics.MassAPI(p).GetMassAttr().Get() is not None)
    if abs(mass - 12.34) > 1.e-5:
        raise ValueError('Keep measured total mass 12.34 kg')
    result = dict(source=str(source), source_sha256=sha(source), overlay=str(path), overlay_sha256=sha(path),
        changes=changes, explicit_collection_membership_redirects=membership,
        all_original_rigid_body_mass_COM_inertia_joints_transforms_bitwise_authoring_equal=True,
        total_mass_kg=mass, source_asset_chain_sha256=chain,
        required_process_setting={'/physics/collisionApproximateCylinders': False},
        purpose='Isolated frozen-policy physics diagnosis; not promoted training asset or acceptance',
        builder_sha256=sha(Path(__file__)), native_simulator_started=False,
        original_training_asset_changed=False, native_PPO_updates=0, policy_accepted=False)
    write_json(path.with_suffix('.json'), result)
    verified, _ = verify_asset_chain(path)
    result['complete_candidate_and_source_asset_chain_sha256'] = verified
    write_json(output / 'candidate_verification.json', result)
    return result


if __name__ == '__main__':
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        if os.environ.get('CLT_CPU_USD_DIAG_READY'):
            raise RuntimeError('CPU USD support unavailable; no native simulator started')
        os.execve(sys.executable, [sys.executable, *sys.argv], cpu_usd_environment())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.source, args.output)
    print(json.dumps({k: v for k, v in result.items() if 'chain_sha256' not in k}, indent=2))
