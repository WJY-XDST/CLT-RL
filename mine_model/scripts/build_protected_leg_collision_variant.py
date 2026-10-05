"""Simplify ordinary leg collisions while preserving confirmed stop CAD meshes."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(next((ROOT / 'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent))
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

PROTECTED = {'LB_link': 'part_413_4_28', 'RB_link': 'part_192_4_28',
             'LS_link': 'part_410___1_15', 'RS_link': 'part_156___1_15'}


def main():
    folder = ROOT / 'wheel_legged_isaaclab/assets/robots/mine/export_20261003_181515'
    source = folder / 'mine_closed_chain_base_box.usd'
    output = folder / 'mine_closed_chain_training_protected.usd'
    shared = folder / 'mine_protected_leg_instanceable_meshes.usd'
    for path in (output, shared, output.with_suffix('.json')):
        if path.exists():
            raise FileExistsError(path)
    baseline = Usd.Stage.Open(str(source))
    layer = Sdf.Layer.CreateNew(str(output))
    layer.subLayerPaths = [source.name]
    stage = Usd.Stage.Open(layer)
    stage.SetDefaultPrim(stage.GetPrimAtPath(baseline.GetDefaultPrim().GetPath()))
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(baseline))
    geometry = Usd.Stage.CreateNew(str(shared))
    geometry.SetDefaultPrim(UsdGeom.Xform.Define(geometry, '/Hulls').GetPrim())
    UsdGeom.SetStageMetersPerUnit(geometry, 1.)
    UsdGeom.SetStageUpAxis(geometry, UsdGeom.GetStageUpAxis(baseline))
    cache = UsdGeom.XformCache()
    records = []
    for body in baseline.Traverse():
        name = body.GetName()
        if not body.HasAPI(UsdPhysics.RigidBodyAPI) or name in ('base_link', 'LW_link', 'RW_link'):
            continue
        collision_path = str(body.GetPath()) + '/collisions'
        original = baseline.GetPrimAtPath(collision_path)
        protected_part = PROTECTED.get(name)
        selected = [p for p in Usd.PrimRange(original, Usd.TraverseInstanceProxies())
                    if p.IsA(UsdGeom.Mesh) and p.GetParent().GetParent().GetName() != protected_part]
        if not selected:
            raise ValueError('No ordinary geometry on ' + name)
        body_inverse = cache.GetLocalToWorldTransform(body).GetInverse()
        points = []
        for prim in selected:
            transform = cache.GetLocalToWorldTransform(prim) * body_inverse
            points.extend(transform.Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get())
        points = np.unique(np.round(np.asarray(points), 12), axis=0)
        hull = ConvexHull(points)
        vertices = points[hull.vertices]
        remap = {int(old): index for index, old in enumerate(hull.vertices)}
        triangles = []
        for tri, plane in zip(hull.simplices, hull.equations):
            ids = list(map(int, tri))
            if np.dot(np.cross(points[ids[1]] - points[ids[0]], points[ids[2]] - points[ids[0]]), plane[:3]) < 0:
                ids[1], ids[2] = ids[2], ids[1]
            triangles.extend(remap[i] for i in ids)
        root_path = '/Hulls/' + name
        wrapper = UsdGeom.Xform.Define(geometry, root_path).GetPrim()
        mesh = UsdGeom.Mesh.Define(geometry, root_path + '/mesh')
        mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in vertices])
        mesh.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
        mesh.CreateFaceVertexIndicesAttr(triangles)
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateVisibilityAttr('invisible')
        mesh.CreateExtentAttr([Gf.Vec3f(*vertices.min(axis=0)), Gf.Vec3f(*vertices.max(axis=0))])
        UsdPhysics.CollisionAPI.Apply(wrapper).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(wrapper).CreateApproximationAttr('convexHull')
        wrapper.AddAppliedSchema('PhysxConvexHullCollisionAPI')
        wrapper.CreateAttribute('physxConvexHullCollision:hullVertexLimit', Sdf.ValueTypeNames.Int, custom=False).Set(255)
        current = stage.GetPrimAtPath(collision_path)
        if protected_part:
            current.SetInstanceable(False)
            for part in current.GetChildren():
                if part.GetName() != protected_part:
                    part.SetActive(False)
            # Protected mesh topology, transform and attributes stay identical.
            prim_path = collision_path + '/' + protected_part + '/node_STL_BINARY_/mesh'
            before = baseline.GetPrimAtPath(prim_path)
            after = stage.GetPrimAtPath(prim_path)
            for attr in before.GetAttributes():
                if after.GetAttribute(attr.GetName()).Get() != attr.Get():
                    raise ValueError('Protected contact CAD changed: ' + str(attr.GetPath()))
        else:
            current.SetActive(False)
        replacement = UsdGeom.Xform.Define(stage, str(body.GetPath()) + '/simple_collision').GetPrim()
        replacement.GetReferences().AddReference(shared.name, root_path)
        replacement.SetInstanceable(True)
        records.append({'body': name, 'merged_mesh_count': len(selected), 'protected_part': protected_part,
                        'input_unique_vertices': len(points), 'output_vertices': len(vertices),
                        'hull_vertex_limit': 255, 'bounds_body_m': [points.min(axis=0).tolist(), points.max(axis=0).tolist()]})
    geometry.GetRootLayer().Save()
    unchanged = []
    for prim in baseline.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI) or prim.IsA(UsdPhysics.Joint):
            current = stage.GetPrimAtPath(prim.GetPath())
            for attr in prim.GetAttributes():
                if attr.GetName().startswith(('physics:', 'physxJoint:')) and current.GetAttribute(attr.GetName()).Get() != attr.Get():
                    raise ValueError('Mass or joint changed: ' + str(attr.GetPath()))
            for rel in prim.GetRelationships():
                if rel.GetName().startswith('physics:') and current.GetRelationship(rel.GetName()).GetTargets() != rel.GetTargets():
                    raise ValueError('Joint relation changed')
            unchanged.append(str(prim.GetPath()))
    layer.Save()
    reload_stage = Usd.Stage.Open(str(output))
    enabled = [p for p in reload_stage.Traverse(Usd.TraverseInstanceProxies())
               if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() != False]
    if len(enabled) != 21:
        raise ValueError(f'Expected 21 collision nodes, got {len(enabled)}')
    manifest = {'source': str(source), 'overlay': str(output), 'shared_mesh_asset': str(shared),
                'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, output, shared)},
                'records': records, 'protected_contact_parts': PROTECTED, 'collision_nodes': len(enabled),
                'unchanged_dynamics_and_joints': unchanged, 'status': 'diagnostic; capacity and policy tests required'}
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
