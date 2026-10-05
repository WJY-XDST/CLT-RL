"""Convert the composed training USD to a closed-chain MuJoCo model.

Body inertias and all joint frames come from USD, not the old robot URDF.
Each external revolute hinge is represented by two connect constraints on
its axis. Protected CAD shapes are independently decomposed by CoACD;
MuJoCo and PhysX cooking/contact algorithms are not claimed equivalent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Gf, Usd, UsdGeom, UsdPhysics


def values(array):
    return " ".join(f"{x:.12g}" for x in np.asarray(array).ravel())


def quaternion(q):
    return np.array([q.GetReal(), *q.GetImaginary()], dtype=float)


def quat_rotation(q):
    wxyz = quaternion(q)
    return Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()


def matrix_quat(matrix):
    xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return xyzw[[3, 0, 1, 2]]


def mesh_geometry(prim, cache, body_inverse):
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
    transform = body_inverse @ matrix
    points = points @ transform[:3, :3].T + transform[:3, 3]
    counts = mesh.GetFaceVertexCountsAttr().Get()
    ids = list(mesh.GetFaceVertexIndicesAttr().Get())
    faces, cursor = [], 0
    for count in counts:
        face = ids[cursor:cursor + count]
        faces.extend((face[0], face[n], face[n + 1]) for n in range(1, count - 1))
        cursor += count
    # USD STL input duplicates every triangle vertex. Weld before decomposition.
    unique, remap = np.unique(np.round(points, 9), axis=0, return_inverse=True)
    faces = remap[np.asarray(faces, dtype=int)]
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return unique, faces[good]


def write_obj(path, points, faces):
    with path.open('w') as handle:
        for point in points:
            handle.write('v ' + values(point) + '\n')
        for face in faces:
            handle.write('f ' + ' '.join(str(int(v) + 1) for v in face) + '\n')


def build_mine_model(cfg, output_path):
    source = Path(cfg['robot']['spawn']['usd_path']).resolve()
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh_dir = output.parent / 'meshes'
    mesh_dir.mkdir(exist_ok=True)
    stage = Usd.Stage.Open(str(source))
    if not stage or UsdGeom.GetStageMetersPerUnit(stage) != 1:
        raise ValueError('Expected the composed training USD in meters')
    cache = UsdGeom.XformCache()
    body_prims = {p.GetName(): p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    poses = {name: np.asarray(cache.GetLocalToWorldTransform(p)).T for name, p in body_prims.items()}
    joints, children, loops = [], {}, []
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        joint = UsdPhysics.RevoluteJoint(prim)
        parents = [str(getattr(joint, f'GetBody{i}Rel')().GetTargets()[0]).split('/')[-1] for i in (0, 1)]
        axis = np.eye(3)['XYZ'.index(joint.GetAxisAttr().Get())]
        record = {'name': prim.GetName(), 'bodies': parents,
                  'positions': [np.array(getattr(joint, f'GetLocalPos{i}Attr')().Get()) for i in (0, 1)],
                  'axes': [quat_rotation(getattr(joint, f'GetLocalRot{i}Attr')().Get()) @ axis for i in (0, 1)],
                  'external': bool(joint.GetExcludeFromArticulationAttr().Get()),
                  'limits': [joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()]}
        joints.append(record)
        if record['external']:
            loops.append(record)
        else:
            if parents[1] in children:
                raise ValueError('USD articulation tree has two parents')
            children[parents[1]] = record
    if len(body_prims) != 15 or len(joints) != 18 or len(loops) != 4 or len(children) != 14:
        raise ValueError('Unexpected new-model body/joint topology')
    root = ET.Element('mujoco', model='mine_closed_chain_protected_stops_v2')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true', inertiafromgeom='false',
                  meshdir=str(mesh_dir), fusestatic='false')
    ET.SubElement(root, 'option', timestep=str(cfg['sim']['dt'] / 50), gravity=values(cfg['sim']['gravity']),
                  integrator='implicitfast', solver='Newton', cone='elliptic', iterations='100', tolerance='1e-10')
    asset = ET.SubElement(root, 'asset')
    ET.SubElement(asset, 'texture', name='checker', type='2d', builtin='checker',
                  rgb1='.22 .28 .30', rgb2='.46 .52 .54', width='512', height='512')
    ET.SubElement(asset, 'material', name='ground', texture='checker', texrepeat='2 2', texuniform='true')
    world = ET.SubElement(root, 'worldbody')
    # Isaac Lab ground and robot both use friction .5 and multiply combine: .25.
    friction = cfg['sim']['physics_material']['dynamic_friction'] * cfg['terrain']['physics_material']['dynamic_friction']
    ET.SubElement(world, 'geom', name='floor', type='plane', size='200 200 .1', contype='1',
                  conaffinity='14', material='ground', friction=f'{friction} .005 .0001', condim='3')
    ET.SubElement(world, 'light', pos='0 -2 4', dir='0 0 -1', directional='true')
    bodies = {}
    body_records = []

    def add_body(name):
        if name in bodies:
            return bodies[name]
        if name == 'base_link':
            transform, parent = np.eye(4), world
        else:
            record = children[name]
            parent = add_body(record['bodies'][0])
            transform = np.linalg.inv(poses[record['bodies'][0]]) @ poses[name]
        body = ET.SubElement(parent, 'body', name=name, pos=values(transform[:3, 3]), quat=values(matrix_quat(transform)))
        bodies[name] = body
        mass = UsdPhysics.MassAPI(body_prims[name])
        properties = {'name': name, 'mass': float(mass.GetMassAttr().Get()),
                      'com': list(mass.GetCenterOfMassAttr().Get()),
                      'inertia': list(mass.GetDiagonalInertiaAttr().Get()),
                      'principal_axes': quaternion(mass.GetPrincipalAxesAttr().Get()).tolist(),
                      'pose': poses[name].tolist()}
        body_records.append(properties)
        ET.SubElement(body, 'inertial', pos=values(properties['com']), mass=str(properties['mass']),
                      diaginertia=values(properties['inertia']), quat=values(properties['principal_axes']))
        if name == 'base_link':
            ET.SubElement(body, 'freejoint', name='root')
        else:
            record = children[name]
            attrs = dict(name=record['name'], type='hinge', pos=values(record['positions'][1]),
                         axis=values(record['axes'][1]), damping='0', armature='0', frictionloss='0', limited='false')
            lo, hi = record['limits']
            if lo is not None and hi is not None and np.isfinite([lo, hi]).all():
                attrs.update(limited='true', range=values(np.radians([lo, hi])))
            ET.SubElement(body, 'joint', **attrs)
        return body

    for name in body_prims:
        add_body(name)
    equality = ET.SubElement(root, 'equality')
    for loop in loops:
        # Two points on a common hinge axis constrain translation and tilt,
        # while leaving rotation around that axis free (five independent DOFs).
        for endpoint, distance in enumerate((0., .03)):
            sites = []
            for side, body_name in enumerate(loop['bodies']):
                site_name = f"{loop['name']}_{endpoint}_{side}"
                sites.append(site_name)
                ET.SubElement(bodies[body_name], 'site', name=site_name,
                              pos=values(loop['positions'][side] + distance * loop['axes'][side]), size='.001', group='4')
            ET.SubElement(equality, 'connect', name=f"{loop['name']}_{endpoint}", site1=sites[0], site2=sites[1],
                          solref='.01 1', solimp='.9999 .9999 .001')
    geometries = []

    def add_mesh(name, points, faces):
        path = mesh_dir / (name + '.obj')
        write_obj(path, points, faces)
        ET.SubElement(asset, 'mesh', name=name, file=path.name)

    for collider in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not collider.HasAPI(UsdPhysics.CollisionAPI) or UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get() is False:
            continue
        path = str(collider.GetPath())
        body_name = path.split('/')[2]
        inverse = np.linalg.inv(poses[body_name])
        category = ('stop' if any(s in path for s in ('part_413_4_28', 'part_192_4_28')) else
                    'shank' if any(s in path for s in ('part_410___1_15', 'part_156___1_15')) else 'ordinary')
        contype, affinity = {'ordinary': (2, 1), 'stop': (4, 9), 'shank': (8, 5)}[category]
        attrs = dict(contype=str(contype), conaffinity=str(affinity), group='3', condim='3',
                     friction=f'{friction} .005 .0001', rgba='.25 .55 .9 .3', margin='0', solref='.01 1')
        if category in ('stop', 'shank'):
            attrs.update(solref='.002 1', solimp='.9999 .9999 .001', margin='.001')
        approx = UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
        count = 0
        for prim in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()):
            name = f'collision_{len(geometries)}_{count}'
            if prim.IsA(UsdGeom.Cube):
                transform = inverse @ np.asarray(cache.GetLocalToWorldTransform(prim)).T
                sizes = np.linalg.norm(transform[:3, :3], axis=0)
                rigid = transform.copy()
                rigid[:3, :3] /= sizes
                ET.SubElement(bodies[body_name], 'geom', name=name, type='box', pos=values(transform[:3, 3]),
                              quat=values(matrix_quat(rigid)), size=values(sizes * UsdGeom.Cube(prim).GetSizeAttr().Get() / 2), **attrs)
                count += 1
            elif prim.IsA(UsdGeom.Mesh):
                points, faces = mesh_geometry(prim, cache, inverse)
                approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or approx
                if approximation == 'convexDecomposition':
                    import coacd
                    coacd.set_log_level('error')
                    pieces = coacd.run_coacd(coacd.Mesh(points, faces), threshold=.01, max_convex_hull=64,
                                             preprocess_mode='auto', preprocess_resolution=50,
                                             resolution=2000, mcts_iterations=100, seed=43)
                else:
                    pieces = [(points, faces)]
                for vertices, triangles in pieces:
                    piece_name = f'collision_{len(geometries)}_{count}'
                    add_mesh(piece_name, vertices, triangles)
                    ET.SubElement(bodies[body_name], 'geom', name=piece_name, type='mesh', mesh=piece_name, **attrs)
                    count += 1
        geometries.append({'source': path, 'body': body_name, 'category': category, 'approximation': approx,
                           'mujoco_convex_pieces': count})
        print(f'Converted collision {len(geometries)}/21: {category} {body_name} -> {count} pieces', flush=True)
    if len(geometries) != 21:
        raise ValueError('Expected 21 enabled source collision shapes')
    # Preserve CAD appearance independently of physics shapes and inertias.
    for body_name, prim in body_prims.items():
        vertices, triangles, offset = [], [], 0
        for mesh in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if mesh.IsA(UsdGeom.Mesh) and '/visuals/' in str(mesh.GetPath()):
                points, faces = mesh_geometry(mesh, cache, np.linalg.inv(poses[body_name]))
                vertices.append(points)
                triangles.append(faces + offset)
                offset += len(points)
        if vertices:
            name = body_name + '_visual'
            add_mesh(name, np.concatenate(vertices), np.concatenate(triangles))
            color = '.15 .55 .8 1' if body_name.startswith('L') else '.85 .55 .15 1'
            if body_name == 'base_link':
                color = '.55 .6 .65 1'
            ET.SubElement(bodies[body_name], 'geom', name=name, type='mesh', mesh=name, contype='0',
                          conaffinity='0', group='2', rgba=color)
    actuator = ET.SubElement(root, 'actuator')
    motor_order = ('LB_joint', 'LL_joint', 'LW_joint', 'RB_joint', 'RL_joint', 'RW_joint')
    for name in motor_order:
        wheel = name in cfg['wheel_joint_names']
        kp = 0. if wheel else cfg['leg_joint_stiffness']
        kd = cfg['wheel_damping'] if wheel else cfg['leg_joint_damping']
        limit = cfg['wheel_effort_limit'] if wheel else cfg['leg_effort_limit']
        ET.SubElement(actuator, 'general', name=name + '_drive', joint=name, dyntype='none', gaintype='fixed',
                      biastype='affine', gainprm=values([kd if wheel else kp]),
                      biasprm=values([0, -kp, -kd]), forcelimited='true', forcerange=values([-limit, limit]))
    ET.SubElement(root, 'statistic', center='0 0 .3', extent='1.2')
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding='utf-8', xml_declaration=True)
    model = mujoco.MjModel.from_xml_path(str(output))
    manifest = {'source_usd': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'mjcf_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'mesh_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in mesh_dir.glob('*.obj')},
                'usd_layers': {layer.realPath: hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
                               for layer in stage.GetUsedLayers() if layer.realPath},
                'bodies': body_records, 'joints': joints, 'collisions': geometries,
                'motor_order': motor_order, 'mujoco': mujoco.__version__,
                'mass_kg': float(model.body_mass.sum()), 'nq': model.nq, 'nv': model.nv,
                'tree_hinges': model.njnt - 1, 'external_hinges': len(loops), 'equality_connects': model.neq,
                'floor_friction': friction,
                'integration_dt': cfg['sim']['dt'] / 50, 'reference_control_dt': cfg['sim']['dt'],
                'policy_dt': cfg['sim']['dt'] * cfg['decimation'],
                'protected_decomposition': {'engine': 'CoACD', 'threshold': .01, 'max_convex_hull': 64,
                                            'preprocess_resolution': 50, 'resolution': 2000, 'mcts_iterations': 100, 'seed': 43},
                'contact_variant': {'protected_solref': [.002, 1], 'protected_solimp': [.9999, .9999, .001], 'protected_margin_m': .001},
                'limitations': ['MuJoCo external revolutes use two soft connect constraints on the hinge axis.',
                                'Protected CAD convex decomposition is CoACD, independently cooked from PhysX.',
                                'Contact/friction/implicit-drive solvers are different between simulators.']}
    output.with_suffix('.json').write_text(json.dumps(manifest, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
    return model


if __name__ == '__main__':
    from core import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    model = build_mine_model(load_config(args.env_config), args.output)
    print(f'Saved {args.output}: mass={model.body_mass.sum():.6f}, bodies={model.nbody-1}, hinges={model.njnt-1}, connects={model.neq}')
