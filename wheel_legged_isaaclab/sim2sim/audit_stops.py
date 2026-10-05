"""Conservative original CAD intersection check at recorded MuJoCo poses."""
import argparse
import json
from pathlib import Path

import numpy as np
import shapely
from scipy.spatial.transform import Rotation
from pxr import Usd, UsdGeom

from mine_model import mesh_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    report = json.loads((root / 'report.json').read_text())
    stage = Usd.Stage.Open(report['asset'])
    cache = UsdGeom.XformCache()
    archive = np.load(root / 'protected_cad_poses.npz')
    poses, names = archive['poses'], archive['body_names'].tolist()
    results = []
    pairs = [('LB_link', 'part_413_4_28', 'LS_link', 'part_410___1_15'),
             ('RB_link', 'part_192_4_28', 'RS_link', 'part_156___1_15')]
    for upper, stop, lower, shank in pairs:
        def get_mesh(body, part):
            body_prim = stage.GetPrimAtPath('/mine_robot/' + body)
            prim = stage.GetPrimAtPath(f'/mine_robot/{body}/collisions/{part}/node_STL_BINARY_/mesh')
            inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body_prim)).T)
            return mesh_geometry(prim, cache, inverse)
        stop_points, stop_faces = get_mesh(upper, stop)
        shank_points, shank_faces = get_mesh(lower, shank)
        projection = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/mine_robot/' + upper))).T
        def polygons(points, faces):
            points = (points @ projection[:3, :3].T + projection[:3, 3])[:, [0, 2]] * 1000
            triangles = points[faces]
            a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            areas = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            return shapely.polygons(triangles[abs(areas) > 1e-8])
        stop_polygon = shapely.union_all(polygons(stop_points, stop_faces))
        maximum, worst = 0., None
        for step, frame in enumerate(poses):
            u, l = frame[0, names.index(upper)], frame[0, names.index(lower)]
            points = Rotation.from_quat(u[3:]).inv().apply(Rotation.from_quat(l[3:]).apply(shank_points) + l[:3] - u[:3])
            triangles = polygons(points, shank_faces)
            triangles = triangles[shapely.intersects(triangles, stop_polygon)]
            overlap = float(shapely.area(shapely.intersection(shapely.union_all(triangles), stop_polygon))) if len(triangles) else 0.
            if overlap > maximum:
                maximum, worst = overlap, step
        results.append({'upper': upper, 'lower': lower, 'sampled_poses': len(poses),
                        'maximum_projected_CAD_overlap_mm2': maximum, 'worst_step': worst,
                        'sampled_CAD_nonintersection_proved': maximum <= 1e-8})
        print(results[-1], flush=True)
    result = {'passed': all(r['sampled_CAD_nonintersection_proved'] for r in results), 'pairs': results,
              'scope': 'Original CAD triangles at recorded 50 Hz MuJoCo body poses. Disjoint projections prove no 3D CAD intersection at those samples. Positive projection overlap is inconclusive in 3D. No continuous-time/substep proof.'}
    (root / 'cad_nonpenetration.json').write_text(json.dumps(result, indent=2) + '\n')
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
