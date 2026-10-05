"""Build a base box while preserving mechanical leg stops and wheel collisions.

The overlay retains CAD visuals, explicit inertias, leg collisions and every
joint. Wheels keep the validated 96-sided collision. Optional leg merging is a
diagnostic only; it must never replace geometric stops without separate review.
Run with ./run_python.sh; original assets are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'wheel_legged_isaaclab/assets/robots/mine/export_20261003_181515/mine_closed_chain_tires_96.usd')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--merge-legs-diagnostic', action='store_true',
                        help='Diagnostic only: fills leg concavities; unsafe for mechanical stops')
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output_dir.resolve() if args.output_dir else source.parent
    output.mkdir(parents=True, exist_ok=True)
    variant = 'training_simple' if args.merge_legs_diagnostic else 'base_box'
    overlay = output / f'mine_closed_chain_{variant}.usd'
    shared = output / f'mine_{variant}_instanceable_meshes.usd'
    manifest_path = overlay.with_suffix('.json')
    for path in (overlay, shared, manifest_path):
        if path.exists():
            raise FileExistsError(path)

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    try:
        import numpy as np
        import omni.usd
        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
        from scipy.spatial import ConvexHull

        enable_extension('isaacsim.util.merge_mesh')
        app.update()
        from isaacsim.util.merge_mesh.mesh_merger import MeshMerger

        # Open the real asset in Isaac Sim; author changes only in a new layer.
        context = omni.usd.get_context()
        if not context.open_stage(str(source)):
            raise RuntimeError('Isaac Sim could not open the source asset')
        app.update()
        baseline = context.get_stage()
        work_layer = Sdf.Layer.CreateAnonymous('collision_merge_work.usda')
        baseline.GetSessionLayer().subLayerPaths.append(work_layer.identifier)
        baseline.SetEditTarget(work_layer)
        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths = [str(source)]
        stage.SetDefaultPrim(stage.GetPrimAtPath(baseline.GetDefaultPrim().GetPath()))
        UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(baseline))
        UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(baseline))
        geometry = Usd.Stage.CreateNew(str(shared))
        geometry.SetDefaultPrim(UsdGeom.Xform.Define(geometry, '/Collisions').GetPrim())
        UsdGeom.SetStageMetersPerUnit(geometry, 1.0)
        UsdGeom.SetStageUpAxis(geometry, UsdGeom.GetStageUpAxis(baseline))
        audit_instances = []
        for prim in baseline.Traverse():
            if prim.IsInstanceable():
                audit_instances.append({'path': str(prim.GetPath()), 'instanceable': True,
                                        'is_instance': prim.IsInstance(),
                                        'references': str(prim.GetMetadata('references'))})
        bodies = [p for p in baseline.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
        records = []
        cache = UsdGeom.XformCache()
        for body in bodies:
            name = body.GetName()
            if name in ('LW_link', 'RW_link'):
                continue
            if name != 'base_link' and not args.merge_legs_diagnostic:
                continue
            collisions = body.GetChild('collisions')
            original_meshes = [p for p in Usd.PrimRange(collisions, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
            if not original_meshes:
                raise ValueError('Missing collision geometry: ' + name)
            body_inverse = cache.GetLocalToWorldTransform(body).GetInverse()
            original_points = []
            for prim in original_meshes:
                transform = cache.GetLocalToWorldTransform(prim) * body_inverse
                original_points.extend(transform.Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get())
            original_points = np.asarray(original_points, dtype=np.float64)
            lower, upper = original_points.min(axis=0), original_points.max(axis=0)
            if not np.isfinite(original_points).all() or np.any(upper - lower <= 0):
                raise ValueError('Invalid collision bounds: ' + name)
            local_root = '/Collisions/' + name
            UsdGeom.Xform.Define(geometry, local_root)
            record = {'body': name, 'original_meshes': len(original_meshes),
                      'original_vertices': len(original_points),
                      'bounds_body_m': [lower.tolist(), upper.tolist()]}
            if name == 'base_link':
                shape = UsdGeom.Cube.Define(geometry, local_root + '/box')
                shape.CreateSizeAttr(1.0)
                shape.AddTranslateOp().Set(Gf.Vec3d(*((lower + upper) / 2)))
                shape.AddScaleOp().Set(Gf.Vec3f(*(upper - lower)))
                record.update(kind='bounding_box', dimensions_m=(upper - lower).tolist(), output_vertices=8)
            else:
                merger = MeshMerger(baseline)
                merger.deactivate_source = False
                merger.update_selection([str(collisions.GetPath())])
                if merger.total_meshes != len(original_meshes):
                    raise ValueError('Merge Tool did not select every collision mesh: ' + name)
                prior = set(p.GetPath() for p in baseline.GetPrimAtPath('/Merged').GetChildren()) if baseline.GetPrimAtPath('/Merged') else set()
                merger.merge_meshes()
                merged_prims = [p for p in baseline.GetPrimAtPath('/Merged').GetChildren() if p.GetPath() not in prior]
                if len(merged_prims) != 1:
                    raise ValueError('Unexpected Merge Tool output')
                merged = merged_prims[0]
                transform = UsdGeom.XformCache().GetLocalToWorldTransform(merged) * body_inverse
                points = np.asarray([transform.Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(merged).GetPointsAttr().Get()])
                if not np.allclose(points.min(axis=0), lower, atol=1e-7) or not np.allclose(points.max(axis=0), upper, atol=1e-7):
                    raise ValueError('Merge Tool changed body-frame bounds: ' + name)
                # Bake the hull rather than storing thousands of internal CAD
                # triangles. This retains the exterior support planes.
                points = np.unique(np.round(points, 12), axis=0)
                hull = ConvexHull(points)
                remap = {int(old): i for i, old in enumerate(hull.vertices)}
                triangles = []
                for triangle, plane in zip(hull.simplices, hull.equations):
                    tri = list(map(int, triangle))
                    if np.dot(np.cross(points[tri[1]] - points[tri[0]], points[tri[2]] - points[tri[0]]), plane[:3]) < 0:
                        tri[1], tri[2] = tri[2], tri[1]
                    triangles.extend(remap[i] for i in tri)
                vertices = points[hull.vertices]
                shape = UsdGeom.Mesh.Define(geometry, local_root + '/hull')
                shape.CreatePointsAttr([Gf.Vec3f(*v) for v in vertices])
                shape.CreateFaceVertexCountsAttr([3] * len(hull.simplices))
                shape.CreateFaceVertexIndicesAttr(triangles)
                shape.CreateSubdivisionSchemeAttr('none')
                shape.CreateExtentAttr([Gf.Vec3f(*lower), Gf.Vec3f(*upper)])
                UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr('convexHull')
                shape.GetPrim().AddAppliedSchema('PhysxConvexHullCollisionAPI')
                shape.GetPrim().CreateAttribute('physxConvexHullCollision:hullVertexLimit', Sdf.ValueTypeNames.Int, custom=False).Set(255)
                baseline.RemovePrim(merged.GetPath())
                record.update(kind='merged_convex_hull', merge_tool='isaacsim.util.merge_mesh.MeshMerger', output_vertices=len(vertices), output_triangles=len(hull.simplices))
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
            shape.CreateVisibilityAttr('invisible')
            # Retain the existing physics material when one is authored.
            material = None
            for prim in Usd.PrimRange(collisions, Usd.TraverseInstanceProxies()):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')[0]
                    if material:
                        break
            stage.GetPrimAtPath(collisions.GetPath()).SetActive(False)
            replacement = UsdGeom.Xform.Define(stage, str(body.GetPath()) + '/training_collision').GetPrim()
            replacement.GetReferences().AddReference(str(shared), local_root)
            if material:
                UsdShade.MaterialBindingAPI.Apply(replacement).Bind(material, materialPurpose='physics')
            replacement.SetInstanceable(True)
            records.append(record)
            print('COLLISION_BUILT ' + json.dumps(record), flush=True)
        geometry.GetRootLayer().Save()
        unchanged = []
        for prim in baseline.Traverse():
            if prim.HasAPI(UsdPhysics.MassAPI) or prim.IsA(UsdPhysics.Joint):
                current = stage.GetPrimAtPath(prim.GetPath())
                for attr in prim.GetAttributes():
                    if attr.GetName().startswith(('physics:', 'physxJoint:')) and current.GetAttribute(attr.GetName()).Get() != attr.Get():
                        raise ValueError('Dynamics changed: ' + str(attr.GetPath()))
                for rel in prim.GetRelationships():
                    if rel.GetName().startswith('physics:') and current.GetRelationship(rel.GetName()).GetTargets() != rel.GetTargets():
                        raise ValueError('Joint relation changed: ' + str(rel.GetPath()))
                unchanged.append(str(prim.GetPath()))
        # Use paths relative to the overlay, keeping the copied directory usable.
        import os
        stage.GetRootLayer().subLayerPaths = [os.path.relpath(source, output)]
        for record in records:
            prim = stage.GetPrimAtPath('/mine_robot/' + record['body'] + '/training_collision')
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddReference(shared.name, '/Collisions/' + record['body'])
        stage.GetRootLayer().Export(str(overlay))
        composed = Usd.Stage.Open(str(overlay))
        all_prims = list(composed.Traverse(Usd.TraverseInstanceProxies()))
        enabled = [p for p in all_prims if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() != False]
        expected = 17 if args.merge_legs_diagnostic else 29
        if len(enabled) != expected:
            raise ValueError(f'Expected {expected} collision nodes, got {len(enabled)}')
        protected = []
        if not args.merge_legs_diagnostic:
            for body in bodies:
                if body.GetName() == 'base_link':
                    continue
                before = list(Usd.PrimRange(body, Usd.TraverseInstanceProxies()))
                for original in before:
                    current = composed.GetPrimAtPath(original.GetPath())
                    if not current or current.GetTypeName() != original.GetTypeName():
                        raise ValueError('Protected leg/wheel prim changed')
                    for attr in original.GetAttributes():
                        if current.GetAttribute(attr.GetName()).Get() != attr.Get():
                            raise ValueError('Protected leg/wheel attribute changed: ' + str(attr.GetPath()))
                    for rel in original.GetRelationships():
                        if current.GetRelationship(rel.GetName()).GetTargets() != rel.GetTargets():
                            raise ValueError('Protected leg/wheel relation changed')
                protected.append(str(body.GetPath()))
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, overlay, shared)}
        manifest = {'source': str(source), 'overlay': str(overlay), 'shared_mesh_asset': str(shared),
                    'sha256': hashes, 'records': records, 'instance_audit_before': audit_instances,
                    'source_collision_nodes': 459, 'output_collision_nodes': len(enabled),
                    'unchanged_leg_and_wheel_subtrees': protected,
                    'unchanged_dynamics_and_joints': unchanged,
                    'status': 'diagnostic variant; validate dynamics before promotion',
                    'geometry_change': ('Base AABB replaces CAD parts; leg and wheel geometry is unchanged.'
                                        if not args.merge_legs_diagnostic else 'DIAGNOSTIC ONLY: Base AABB and merged leg hulls fill concavities and holes; do not use for mechanical stops.')}
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        print('COLLISION_VARIANT ' + str(overlay), flush=True)
    finally:
        app.close()


if __name__ == '__main__':
    main()
