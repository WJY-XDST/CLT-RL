"""Copy one identical wheel assembly's principal moments to the other side.

The right CAD wheel is the unmeasured inertia reference. Its tensor is mirrored
in the base frame, then expressed in the left link frame. Existing source USDs
remain immutable; only a new composition layer is authored. This does not
change the owner-defined MCU whole-vehicle feedforward mass.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
USD_LIBRARY = next((ROOT/'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent
if os.environ.get('CLT_USD_NATIVE_READY') != '1':
    environment = dict(os.environ, CLT_USD_NATIVE_READY='1')
    environment['LD_LIBRARY_PATH'] = str(USD_LIBRARY/'bin')+':'+environment.get('LD_LIBRARY_PATH', '')
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)
sys.path.insert(0, str(USD_LIBRARY))
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matrix_rotation(matrix):
    result = np.column_stack([np.asarray(matrix.TransformDir(Gf.Vec3d(*v)))
                              for v in ((1, 0, 0), (0, 1, 0), (0, 0, 1))])
    if not np.allclose(result.T@result, np.eye(3), atol=1.e-6):
        raise ValueError('Wheel/base rigid-body transforms must contain a pure rotation.')
    return result


def inspect_asset(path):
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    base_inverse = cache.GetLocalToWorldTransform(stage.GetPrimAtPath('/mine_robot/base_link')).GetInverse()
    result = dict(path=str(path),sha256=digest(path),wheels={},used_layers=[],total_mass_kg=0.,body_masses_kg={})
    for layer in stage.GetUsedLayers():
        if not layer.identifier.startswith('anon:'):
            result['used_layers'].append(dict(path=layer.realPath,sublayers=list(layer.subLayerPaths)))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get()
            result['body_masses_kg'][prim.GetName()] = mass
            result['total_mass_kg'] += mass
    for name in ('LW_link', 'RW_link'):
        prim = stage.GetPrimAtPath('/mine_robot/'+name)
        mass_api = UsdPhysics.MassAPI(prim)
        diagonal = np.asarray(mass_api.GetDiagonalInertiaAttr().Get())
        quaternion = mass_api.GetPrincipalAxesAttr().Get()
        principal_rotation = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
        link_rotation = matrix_rotation(cache.GetLocalToWorldTransform(prim)*base_inverse)
        tensor_link = principal_rotation@np.diag(diagonal)@principal_rotation.T
        tensor_base = link_rotation@tensor_link@link_rotation.T
        result['wheels'][name] = dict(mass_kg=mass_api.GetMassAttr().Get(),com_local=list(mass_api.GetCenterOfMassAttr().Get()),
            principal_inertia_kg_m2=diagonal.tolist(),principal_axes_xyzw=[*quaternion.GetImaginary(), quaternion.GetReal()],
            rotation_link_to_base=link_rotation.tolist(),tensor_link_frame=tensor_link.tolist(),tensor_base_frame=tensor_base.tolist(),
            spin_inertia_base_y_kg_m2=float(tensor_base[1, 1]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    before = inspect_asset(source)
    report = dict(before=before,authority='owner_instruction_identical_wheel_assemblies',inertia_measured=False)
    if args.output:
        target = args.output.resolve()
        if target.exists():
            raise FileExistsError('Keep calibrated USD versions immutable: '+str(target))
        left, right = before['wheels']['LW_link'], before['wheels']['RW_link']
        reflection = np.diag([1., -1., 1.])
        left_link = np.asarray(left['rotation_link_to_base'])
        right_link = np.asarray(right['rotation_link_to_base'])
        right_principal = Rotation.from_quat(right['principal_axes_xyzw']).as_matrix()
        left_principal = left_link.T@reflection@right_link@right_principal
        # An eigenvector sign is arbitrary; flip one column to obtain a proper
        # rotation while retaining the same physical mirrored inertia tensor.
        if np.linalg.det(left_principal) < 0:
            left_principal[:, 0] *= -1.
        quaternion = Rotation.from_matrix(left_principal).as_quat()
        layer = Sdf.Layer.CreateNew(str(target))
        layer.defaultPrim = 'mine_robot'
        layer.subLayerPaths = [os.path.relpath(source, target.parent)]
        stage = Usd.Stage.Open(layer)
        for name in ('LW_link', 'RW_link'):
            api = UsdPhysics.MassAPI(stage.OverridePrim('/mine_robot/'+name))
            api.CreateMassAttr(.665)
            api.CreateDiagonalInertiaAttr(Gf.Vec3f(*right['principal_inertia_kg_m2']))
        api = UsdPhysics.MassAPI(stage.GetPrimAtPath('/mine_robot/LW_link'))
        api.CreatePrincipalAxesAttr(Gf.Quatf(float(quaternion[3]), Gf.Vec3f(*quaternion[:3])))
        layer.Save()
        after = inspect_asset(target)
        for name in ('LW_link', 'RW_link'):
            assert after['wheels'][name]['mass_kg'] == before['wheels'][name]['mass_kg']
            assert after['wheels'][name]['com_local'] == before['wheels'][name]['com_local']
        assert after['body_masses_kg'] == before['body_masses_kg']
        assert after['wheels']['LW_link']['principal_inertia_kg_m2'] == after['wheels']['RW_link']['principal_inertia_kg_m2']
        expected = reflection@np.asarray(after['wheels']['RW_link']['tensor_base_frame'])@reflection
        mismatch = float(np.max(np.abs(np.asarray(after['wheels']['LW_link']['tensor_base_frame'])-expected)))
        assert mismatch < 1.e-9
        assert digest(source) == before['sha256']
        report.update(after=after,base_frame_mirror_tensor_max_difference_kg_m2=mismatch,
                      source_unchanged=True,all_body_masses_unchanged=True,both_wheel_com_unchanged=True,
                      same_principal_moments=True,qualification='USD composition only; native PhysX loading still required')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
