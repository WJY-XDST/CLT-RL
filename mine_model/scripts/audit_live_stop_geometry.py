"""Check original stop/shank CAD projections using actual GPU body poses.

Disjoint projections prove no CAD interpenetration at a sampled pose. Positive
projected overlap remains inconclusive in 3D and must never pass this gate.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'wheel_legged_isaaclab'))
from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain
sys.path.insert(0, str(next((ROOT/'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent))
from pxr import Usd, UsdGeom, Gf


def main():
    import shapely
    from scipy.spatial.transform import Rotation
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.test/'report.json').read_text())
    stage = Usd.Stage.Open(report['asset'])
    cache = UsdGeom.XformCache()
    base_inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/mine_robot/base_link')).GetInverse()
    archive = np.load(args.test/'protected_cad_poses.npz')
    poses = archive['poses']  # step, env, body, xyz+xyzw
    names = archive['body_names'].tolist()
    pairs = [('LB_link', 'part_413_4_28', 'LS_link', 'part_410___1_15'),
             ('RB_link', 'part_192_4_28', 'RS_link', 'part_156___1_15')]
    results = []
    for upper, stop, lower, shank in pairs:
        def mesh_in_body(body, part):
            body_prim = stage.GetPrimAtPath('/mine_robot/'+body)
            mesh_prim = stage.GetPrimAtPath(f'/mine_robot/{body}/collisions/{part}/node_STL_BINARY_/mesh')
            mesh = UsdGeom.Mesh(mesh_prim)
            matrix = cache.GetLocalToWorldTransform(mesh_prim)*cache.GetLocalToWorldTransform(body_prim).GetInverse()
            points = np.array([matrix.Transform(Gf.Vec3d(*p)) for p in mesh.GetPointsAttr().Get()])
            faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
            return points, faces
        stop_points, stop_faces = mesh_in_body(upper, stop)
        shank_points, shank_faces = mesh_in_body(lower, shank)
        upper_to_base = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/mine_robot/'+upper))*base_inverse
        matrix = np.asarray(upper_to_base)
        def project(points):
            transformed = points@matrix[:3, :3]+matrix[3, :3]
            return transformed[:, [0, 2]]*1000
        def polygons(points, faces):
            triangles = points[faces]
            x, y = triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]
            areas = x[:, 0]*y[:, 1]-x[:, 1]*y[:, 0]
            return shapely.polygons(triangles[abs(areas)>1e-8])
        stop_polygon = shapely.union_all(polygons(project(stop_points), stop_faces))
        maximum = 0.
        worst = None
        for step in range(len(poses)):
            for env in range(poses.shape[1]):
                u, l = poses[step, env, names.index(upper)], poses[step, env, names.index(lower)]
                upper_rotation, lower_rotation = Rotation.from_quat(u[3:]), Rotation.from_quat(l[3:])
                # Move CAD points from lower local -> live world -> upper local.
                points = upper_rotation.inv().apply(lower_rotation.apply(shank_points)+l[:3]-u[:3])
                projected = polygons(project(points), shank_faces)
                # Only triangles intersecting the stop projection need union.
                projected = projected[shapely.intersects(projected, stop_polygon)]
                area = float(shapely.area(shapely.intersection(shapely.union_all(projected), stop_polygon))) if len(projected) else 0.
                if area > maximum:
                    maximum, worst = area, [step, env]
            if step % 100 == 0:
                print(upper, step, 'max overlap mm2', maximum, flush=True)
        results.append({'upper': upper, 'lower': lower, 'maximum_projected_CAD_overlap_mm2': maximum,
                        'worst_step_env': worst, 'sampled_poses': int(len(poses)*poses.shape[1]),
                        'no_CAD_interpenetration_at_sampled_poses': maximum <= 1e-8})
    asset_chain, _ = verify_asset_chain(report['asset'])
    result = {'asset': report['asset'], 'asset_chain_sha256': asset_chain, 'pairs': results,
              'passed': all(r['no_CAD_interpenetration_at_sampled_poses'] for r in results),
              'scope': 'Original CAD triangle projections transformed by actual GPU poses, all recorded 50 Hz samples across four deliberate overstroke tests. Disjoint X/Z projections establish no CAD intersection at those samples; substep contact depth and continuous-time nonpenetration are not established.'}
    (args.test/'cad_nonpenetration.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
