"""Check compiled MuJoCo cylinder dimensions and body-frame end points."""
import unittest
import hashlib
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from pxr import Gf, Usd, UsdGeom
from scipy.spatial.transform import Rotation

from mine_model import cylinder_geometry, training_usd_stage


class CylinderConversionTests(unittest.TestCase):
    def fixture(self, axis='Z', scales=(1., 1., 1.), shear=0.):
        stage = Usd.Stage.CreateInMemory()
        cylinder = UsdGeom.Cylinder.Define(stage, '/wheel')
        cylinder.GetAxisAttr().Set(axis)
        cylinder.GetRadiusAttr().Set(.06)
        cylinder.GetHeightAttr().Set(.035)
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_euler('xyz', [.31, -.42, .17]).as_matrix() @ np.diag(scales)
        matrix[:3, 1] += shear * matrix[:3, 0]
        matrix[:3, 3] = [.23, -.41, .12]
        UsdGeom.Xformable(cylinder).AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
        body_pose = np.eye(4)
        body_pose[:3, :3] = Rotation.from_euler('xyz', [-.13, .21, -.32]).as_matrix()
        body_pose[:3, 3] = [-.08, .14, -.03]
        return stage, cylinder, matrix, np.linalg.inv(body_pose)

    def test_compiled_cylinder_matches_usd_endpoints_and_radius(self):
        for axis in 'XYZ':
            for reflected in (False, True):
                with self.subTest(axis=axis, reflected=reflected):
                    axial = 'XYZ'.index(axis)
                    scales = np.full(3, 1.3)
                    scales[axial] = -1.7 if reflected else 1.7
                    stage, cylinder, matrix, inverse = self.fixture(axis, scales)
                    attrs = cylinder_geometry(cylinder.GetPrim(), UsdGeom.XformCache(), inverse)
                    root = ET.Element('mujoco')
                    world = ET.SubElement(root, 'worldbody')
                    ET.SubElement(world, 'geom', type='cylinder', **attrs)
                    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
                    self.assertEqual(model.geom_type[0], mujoco.mjtGeom.mjGEOM_CYLINDER)
                    np.testing.assert_allclose(model.geom_size[0, :2], [.06 * 1.3, .035 * 1.7 / 2.], atol=1.e-12)
                    w, x, y, z = model.geom_quat[0]
                    direction = Rotation.from_quat([x, y, z, w]).as_matrix()[:, 2]
                    actual = model.geom_pos[0] + np.array([-1., 1.])[:, None] * direction * model.geom_size[0, 1]
                    local = np.zeros((2, 4))
                    local[:, 3] = 1.
                    local[:, axial] = [-.035 / 2., .035 / 2.]
                    expected = (inverse @ matrix @ local.T).T[:, :3]
                    np.testing.assert_allclose(actual, expected, rtol=0., atol=1.e-10)

    def test_unrepresentable_transforms_are_rejected(self):
        for kwargs in [dict(scales=(1., 1.2, 1.)), dict(shear=.2), dict(scales=(0., 1., 1.))]:
            with self.subTest(**kwargs):
                stage, cylinder, _, inverse = self.fixture(**kwargs)
                with self.assertRaises(ValueError):
                    cylinder_geometry(cylinder.GetPrim(), UsdGeom.XformCache(), inverse)

    def test_meter_base_overlay_keeps_numeric_geometry_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            base_path = Path(temporary) / 'base.usda'
            overlay_path = Path(temporary) / 'overlay.usda'
            base = Usd.Stage.CreateNew(str(base_path))
            UsdGeom.SetStageMetersPerUnit(base, 1.)
            UsdGeom.Cylinder.Define(base, '/wheel').GetRadiusAttr().Set(.06)
            base.GetRootLayer().Save()
            overlay = Usd.Stage.CreateNew(str(overlay_path))
            overlay.GetRootLayer().subLayerPaths = [str(base_path)]
            overlay.GetRootLayer().Save()
            digest = hashlib.sha256(overlay_path.read_bytes()).hexdigest()
            self.assertFalse(UsdGeom.StageHasAuthoredMetersPerUnit(overlay))
            stage, metadata = training_usd_stage(overlay_path)
            self.assertEqual(UsdGeom.GetStageMetersPerUnit(stage), 1.)
            self.assertTrue(metadata['anonymous_meter_context'])
            self.assertFalse(metadata['geometry_rescaled'])
            self.assertEqual(UsdGeom.Cylinder(stage.GetPrimAtPath('/wheel')).GetRadiusAttr().Get(), .06)
            self.assertEqual(hashlib.sha256(overlay_path.read_bytes()).hexdigest(), digest)

    def test_non_meter_or_unknown_unit_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for name, units in [('centimeters', .01), ('unknown', None)]:
                with self.subTest(units=units):
                    path = Path(temporary) / (name + '.usda')
                    stage = Usd.Stage.CreateNew(str(path))
                    if units is not None:
                        UsdGeom.SetStageMetersPerUnit(stage, units)
                    UsdGeom.Cylinder.Define(stage, '/wheel')
                    stage.GetRootLayer().Save()
                    with self.assertRaises(ValueError):
                        training_usd_stage(path)


if __name__ == '__main__':
    unittest.main()
