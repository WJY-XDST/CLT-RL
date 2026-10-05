"""Inspect the original leg parts and collision flags without changing the asset.

Offline pxr requires Isaac's USD library path; figures use CAD mesh triangles.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(next((ROOT / 'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent))
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    source = ROOT / 'wheel_legged_isaaclab/assets/robots/mine/export_20261003_181515/mine_closed_chain_tires_96.usd'
    output = ROOT / 'mine_model/results/collision_simplification_20261004'
    output.mkdir(exist_ok=True)
    stage = Usd.Stage.Open(str(source))
    cache = UsdGeom.XformCache()
    base_inv = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/mine_robot/base_link')).GetInverse()
    report = {'source': str(source), 'parts': [], 'joints': [],
              'runtime_self_collisions': False,
              'finding': 'Asset and training config disable articulation self collisions; existing internal geometry alone does not implement a physical stop.'}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            report['joints'].append({'name': prim.GetName(), 'collision_enabled': UsdPhysics.Joint(prim).GetCollisionEnabledAttr().Get(),
                                     'body0': [str(x) for x in prim.GetRelationship('physics:body0').GetTargets()],
                                     'body1': [str(x) for x in prim.GetRelationship('physics:body1').GetTargets()]})
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            report['asset_self_collisions'] = prim.GetAttribute('physxArticulation:enabledSelfCollisions').Get()
    fig = plt.figure(figsize=(16, 15), constrained_layout=True)
    palette = ['#e0701f', '#247aaf', '#a151a1', '#329b71']
    for row, name in enumerate(('LB_link', 'LS_link', 'LL_link')):
        side = fig.add_subplot(3, 2, row * 2 + 1)
        spatial = fig.add_subplot(3, 2, row * 2 + 2, projection='3d')
        col = stage.GetPrimAtPath('/mine_robot/' + name + '/collisions')
        bounds = []
        for index, prim in enumerate(p for p in Usd.PrimRange(col, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)):
            mesh = UsdGeom.Mesh(prim)
            transform = cache.GetLocalToWorldTransform(prim) * base_inv
            points = np.array([transform.Transform(Gf.Vec3d(*v)) for v in mesh.GetPointsAttr().Get()])
            counts = list(mesh.GetFaceVertexCountsAttr().Get())
            indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
            if set(counts) != {3}:
                raise ValueError('Expected STL triangle mesh')
            triangles = points[indices.reshape(-1, 3)]
            part = prim.GetParent().GetParent().GetName()
            color = palette[index % len(palette)]
            lower, upper = points.min(axis=0), points.max(axis=0)
            bounds.append(np.array([lower, upper]))
            report['parts'].append({'body': name, 'part': part, 'mesh': str(prim.GetPath()),
                                    'vertices': len(points), 'triangles': len(triangles),
                                    'bounds_base_m': [lower.tolist(), upper.tolist()],
                                    'approximation': str(UsdPhysics.MeshCollisionAPI(prim.GetParent()).GetApproximationAttr().Get())})
            # Figures may subsample dense triangles; exported CAD stays exact.
            stride = max(1, len(triangles) // 12000)
            side.add_collection(PolyCollection(triangles[::stride][:, :, [0, 2]] * 1000, facecolor=color, edgecolor='none', alpha=.58, label=part))
            spatial.add_collection3d(Poly3DCollection(triangles[::stride] * 1000, facecolor=color, edgecolor='none', alpha=.75))
        points = np.concatenate(bounds, axis=0) * 1000
        lo, hi = points.min(axis=0) - 8, points.max(axis=0) + 8
        side.set_xlim(lo[0], hi[0]); side.set_ylim(lo[2], hi[2]); side.set_aspect('equal')
        side.set_xlabel('base X (mm)'); side.set_ylabel('base Z (mm)'); side.grid(alpha=.2)
        side.set_title(name + ': original collision parts, side view')
        side.legend(fontsize=8, loc='best')
        spatial.set_xlim(lo[0], hi[0]); spatial.set_ylim(lo[1], hi[1]); spatial.set_zlim(lo[2], hi[2]); spatial.set_box_aspect(hi - lo)
        spatial.set_xlabel('X mm'); spatial.set_ylabel('Y mm'); spatial.set_zlabel('Z mm'); spatial.view_init(20, -70)
        spatial.set_title(name + ': original CAD geometry')
    fig.savefig(output / 'left_leg_collision_parts.png', dpi=160)
    plt.close(fig)
    (output / 'leg_stop_geometry_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
