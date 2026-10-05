"""A reversible high-resolution circular tire collision overlay; CAD dynamics stay fixed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(next((ROOT/'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent))
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sides',type=int,default=96)
    p.add_argument('--source',type=Path,default=ROOT/'wheel_legged_isaaclab/assets/robots/mine/export_20261003_181515/mine_closed_chain.usd')
    args=p.parse_args()
    if not 32<=args.sides<=128:raise ValueError('Expected 32 to 128 sides')
    source=args.source.resolve();out=source.with_name(f'mine_closed_chain_tires_{args.sides}.usd')
    if out.exists():raise FileExistsError(out)
    baseline=Usd.Stage.Open(str(source))
    stage=Usd.Stage.CreateInMemory();stage.GetRootLayer().subLayerPaths=[str(source)]
    UsdGeom.SetStageMetersPerUnit(stage,UsdGeom.GetStageMetersPerUnit(baseline))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.GetStageUpAxis(baseline))
    stage.SetDefaultPrim(stage.GetPrimAtPath(baseline.GetDefaultPrim().GetPath()))
    if UsdGeom.GetStageMetersPerUnit(stage)!=1.:raise ValueError('Expected metric robot stage')
    records=[]
    for name,part in (('LW_link','part_421_part'),('RW_link','part_422_part')):
        body=stage.GetPrimAtPath('/mine_robot/'+name)
        collisions=body.GetChild('collisions');collisions.SetInstanceable(False)
        original=stage.GetPrimAtPath(str(collisions.GetPath())+'/'+part+'/node_STL_BINARY_')
        if not original or not original.HasAPI(UsdPhysics.CollisionAPI):raise ValueError('Missing original tire collision')
        old_mesh=stage.GetPrimAtPath(str(original.GetPath())+'/mesh')
        cache=UsdGeom.XformCache();transform=cache.GetLocalToWorldTransform(old_mesh)*cache.GetLocalToWorldTransform(body).GetInverse()
        old_points=np.array([transform.Transform(Gf.Vec3d(*v)) for v in UsdGeom.Mesh(old_mesh).GetPointsAttr().Get()])
        radius=float(np.linalg.norm(old_points[:,:2],axis=1).max())
        if abs(radius-.06)>1e-6:raise ValueError('Tire radius differs from declared geometry')
        zmin,zmax=map(float,(old_points[:,2].min(),old_points[:,2].max()))
        angle=np.arange(args.sides)*2*np.pi/args.sides
        points=[Gf.Vec3f(.06*np.cos(a),.06*np.sin(a),z) for z in (zmin,zmax) for a in angle]
        counts=[];indices=[]
        for i in range(args.sides):
            j=(i+1)%args.sides;counts.append(4);indices.extend([i,j,j+args.sides,i+args.sides])
        counts.extend([args.sides,args.sides]);indices.extend(reversed(range(args.sides)));indices.extend(range(args.sides,2*args.sides))
        rounded=UsdGeom.Mesh.Define(stage,str(body.GetPath())+'/smooth_tire_collision')
        rounded.CreatePointsAttr(points);rounded.CreateFaceVertexCountsAttr(counts);rounded.CreateFaceVertexIndicesAttr(indices)
        rounded.CreateSubdivisionSchemeAttr('none');rounded.CreateVisibilityAttr('invisible')
        UsdPhysics.CollisionAPI.Apply(rounded.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(rounded.GetPrim()).CreateApproximationAttr('convexHull')
        rounded.GetPrim().AddAppliedSchema('PhysxConvexHullCollisionAPI')
        rounded.GetPrim().CreateAttribute('physxConvexHullCollision:hullVertexLimit',Sdf.ValueTypeNames.Int,custom=False).Set(255)
        material=UsdShade.MaterialBindingAPI(original).ComputeBoundMaterial(materialPurpose='physics')[0]
        if material:UsdShade.MaterialBindingAPI.Apply(rounded.GetPrim()).Bind(material,materialPurpose='physics')
        UsdPhysics.CollisionAPI(original).CreateCollisionEnabledAttr(False)
        records.append({'body':name,'disabled_collision':str(original.GetPath()),'new_collision':str(rounded.GetPath()),
                        'radius_m':.06,'axial_bounds_m':[zmin,zmax],'original_radial_sides':18,'new_radial_sides':args.sides,'hull_vertex_limit':255})
    # Explicit inertia and joints must be unchanged after the overlay composes.
    unchanged=[]
    for prim in baseline.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI) or prim.IsA(UsdPhysics.Joint):
            current=stage.GetPrimAtPath(prim.GetPath())
            for attr in prim.GetAttributes():
                if attr.GetName().startswith(('physics:','physxJoint:')):
                    if current.GetAttribute(attr.GetName()).Get()!=attr.Get():raise ValueError('Dynamics changed: '+str(attr.GetPath()))
            for rel in prim.GetRelationships():
                if rel.GetName().startswith('physics:') and current.GetRelationship(rel.GetName()).GetTargets()!=rel.GetTargets():raise ValueError('Joint relation changed')
            unchanged.append(str(prim.GetPath()))
    stage.GetRootLayer().subLayerPaths=[source.name]
    stage.GetRootLayer().Export(str(out))
    manifest={'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'overlay':str(out),'overlay_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
              'changes':records,'unchanged_dynamics_and_joints':unchanged,
              'scope':'Only tire collision tessellation; original visual CAD, masses, COM, inertia, axes and closure joints retained',
              'status':'diagnostic variant, not promoted'}
    out.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
