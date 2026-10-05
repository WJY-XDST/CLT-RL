"""Projected CAD contact sweep for the suspected LB/LS mechanical stop.

This is a geometric diagnosis, not a PhysX proof of a functioning hard stop.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(next((ROOT / 'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent))
from pxr import Gf, Usd, UsdGeom


def main():
    import shapely
    from shapely import affinity
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output = ROOT / 'mine_model/results/collision_simplification_20261004'
    stage = Usd.Stage.Open(str(ROOT / 'wheel_legged_isaaclab/assets/robots/mine/export_20261003_181515/mine_closed_chain_tires_96.usd'))
    cache = UsdGeom.XformCache()
    base_inv = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/mine_robot/base_link')).GetInverse()

    def silhouette(path):
        prim = stage.GetPrimAtPath(path)
        mesh = UsdGeom.Mesh(prim)
        transform = cache.GetLocalToWorldTransform(prim) * base_inv
        points = np.array([transform.Transform(Gf.Vec3d(*v)) for v in mesh.GetPointsAttr().Get()])
        triangles = points[np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)][:, :, [0, 2]] * 1000
        area = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        nonzero = triangles[abs(area) > 1e-7]
        polygons = shapely.polygons(nonzero)
        return shapely.union_all(polygons), [float(points[:, 1].min()), float(points[:, 1].max())]

    stop_path = '/mine_robot/LB_link/collisions/part_413_4_28/node_STL_BINARY_/mesh'
    shank_path = '/mine_robot/LS_link/collisions/part_410___1_15/node_STL_BINARY_/mesh'
    stop, stop_y = silhouette(stop_path)
    shank, shank_y = silhouette(shank_path)
    joint = stage.GetPrimAtPath('/mine_robot/joints/LS_joint')
    parent = stage.GetPrimAtPath(joint.GetRelationship('physics:body0').GetTargets()[0])
    transform = cache.GetLocalToWorldTransform(parent) * base_inv
    anchor = transform.Transform(Gf.Vec3d(*joint.GetAttribute('physics:localPos0').Get()))
    pivot = (anchor[0] * 1000, anchor[2] * 1000)
    axis = transform.TransformDir(Gf.Vec3d(0, 0, 1))
    angles = np.arange(-90., 90.01, .25)
    exact = [stop.intersection(affinity.rotate(shank, a, origin=pivot)).area for a in angles]
    hull = [stop.convex_hull.intersection(affinity.rotate(shank.convex_hull, a, origin=pivot)).area for a in angles]
    report = {'status': 'candidate identification; projected sweep only', 'stop_mesh': stop_path,
              'symmetric_stop_mesh': '/mine_robot/RB_link/collisions/part_192_4_28/node_STL_BINARY_/mesh',
              'shank_mesh': shank_path, 'pivot_base_mm': list(pivot), 'joint_axis_base': list(axis),
              'stop_y_bounds_m': stop_y, 'shank_y_bounds_m': shank_y,
              'sweep_angle_deg': angles.tolist(), 'CAD_overlap_area_mm2': exact,
              'convex_hull_overlap_area_mm2': hull,
              'limitations': 'Two-dimensional silhouette sweep holds LB fixed and rotates LS only. It does not enforce the other five-bar closures, simulate contact, or establish the robot leg-length limits.'}
    (output / 'stop_silhouette_sweep.json').write_text(json.dumps(report, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    def draw(ax, poly, color, label):
        geoms = poly.geoms if hasattr(poly, 'geoms') else [poly]
        for index, geom in enumerate(geoms):
            xy = np.asarray(geom.exterior.coords)
            ax.fill(xy[:, 0], xy[:, 1], color=color, alpha=.65, label=label if index == 0 else None)
            for hole in geom.interiors:
                xy = np.asarray(hole.coords); ax.fill(xy[:, 0], xy[:, 1], color='white')
    draw(axes[0], stop, '#2c9f70', 'LB stop candidate: part_413_4_28')
    draw(axes[0], shank, '#e88335', 'LS main plate: part_410')
    axes[0].plot(*pivot, 'ko', label='LS hinge'); axes[0].set_aspect('equal'); axes[0].legend(fontsize=8)
    axes[0].set_xlabel('base X (mm)'); axes[0].set_ylabel('base Z (mm)'); axes[0].set_title('Original CAD, default pose')
    axes[1].plot(angles, exact, label='Original CAD silhouette')
    axes[1].plot(angles, hull, '--', label='Single convex hull silhouette')
    axes[1].set_xlabel('Projected LS rotation from default (deg)'); axes[1].set_ylabel('Overlap area (mm²)')
    axes[1].set_title('Candidate contact sweep; not a dynamics test'); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.savefig(output / 'stop_candidate_and_sweep.png', dpi=160)
    contact = angles[np.array(exact) > .01]
    print('CAD projected overlap interval:', (contact.min(), contact.max()) if len(contact) else None)


if __name__ == '__main__':
    main()
