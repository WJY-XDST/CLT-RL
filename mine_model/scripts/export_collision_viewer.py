"""Export actual composed collision input geometry for a lightweight 3D view."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(next((ROOT/'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent))
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def main():
    state = json.loads((ROOT/'mine_model/results/auto_training_implicit_20pct_20261004/status.json').read_text())
    asset = Path(state['overrides']['robot.spawn.usd_path'])
    stage = Usd.Stage.Open(str(asset))
    cache = UsdGeom.XformCache()
    inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/mine_robot/base_link')).GetInverse()
    parts, joints = [], []
    cubes = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])
    cube_faces = [0,2,1,0,3,2,4,5,6,4,6,7,0,1,5,0,5,4,2,3,7,2,7,6,1,2,6,1,6,5,3,0,4,3,4,7]
    for collider in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not collider.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()==False:
            continue
        path = str(collider.GetPath())
        body = path.split('/')[2]
        category = ('base' if body=='base_link' else 'wheels' if body in ('LW_link','RW_link')
                    else 'stops' if any(n in path for n in ('part_413_4_28','part_192_4_28'))
                    else 'shanks' if any(n in path for n in ('part_410___1_15','part_156___1_15')) else 'links')
        for prim in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()):
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = np.array(mesh.GetPointsAttr().Get())
                counts = mesh.GetFaceVertexCountsAttr().Get()
                ids = list(mesh.GetFaceVertexIndicesAttr().Get())
                indices, cursor = [], 0
                for count in counts:
                    face = ids[cursor:cursor+count]
                    indices.extend(i for n in range(1,count-1) for i in (face[0],face[n],face[n+1]))
                    cursor += count
            elif prim.IsA(UsdGeom.Cube):
                points = cubes*UsdGeom.Cube(prim).GetSizeAttr().Get()/2
                indices = cube_faces
            else:
                continue
            matrix = cache.GetLocalToWorldTransform(prim)*inverse
            points = np.array([matrix.Transform(Gf.Vec3d(*map(float,p))) for p in points])
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get() or 'box'
            if approximation=='convexHull':
                hull = ConvexHull(points)
                triangles = []
                for triangle, plane in zip(hull.simplices,hull.equations):
                    ids = list(map(int,triangle))
                    if np.dot(np.cross(points[ids[1]]-points[ids[0]],points[ids[2]]-points[ids[0]]),plane[:3])<0:
                        ids[1],ids[2] = ids[2],ids[1]
                    triangles.extend(ids)
                indices = triangles
                used, remap_used = np.unique(indices,return_inverse=True)
                points, indices = points[used],remap_used
            unique, remap = np.unique(np.round(points,7), axis=0, return_inverse=True)
            parts.append({'body':body,'category':category,'path':str(prim.GetPath()),'approximation':approximation,
                          'positions':unique.ravel().tolist(),'indices':remap[np.array(indices)].tolist()})
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.Joint(prim)
        parent = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
        matrix = cache.GetLocalToWorldTransform(parent)*inverse
        position = matrix.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get()))
        joints.append({'name':prim.GetName(),'position':np.round(position,7).tolist(),
                       'external':bool(joint.GetExcludeFromArticulationAttr().Get())})
    if len(parts)!=21:
        raise ValueError(f'Expected 21 composed collision geometries, found {len(parts)}')
    model = {'asset':asset.name,'asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
             'num_envs':state['num_envs'],'parts':parts,'joints':joints}
    folder = ROOT/'mine_model/results/collision_simplification_20261004/visualization'
    folder.mkdir(exist_ok=True)
    data = json.dumps(model, separators=(',',':'), ensure_ascii=False)
    (folder/'collision_geometry.json').write_text(data+'\n')
    template = ROOT/'mine_model/scripts/templates/collision_viewer.html'
    output = folder/'current-simplified-collisions.html'
    output.write_text(template.read_text().replace('__MODEL_DATA__',data))
    if output.stat().st_size>1_000_000:
        raise ValueError('Viewer exceeds inline size limit')
    print(output, output.stat().st_size, 'bytes; parts',len(parts),'joints',len(joints))


if __name__=='__main__':
    main()
