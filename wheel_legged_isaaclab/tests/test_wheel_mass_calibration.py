"""Compose the calibrated USD and preserve the historical CAD physics contract.

The USD checks also run with env_mujoco's usd-core, without launching Isaac Sim.
"""
import json
from pathlib import Path
import unittest

from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST

try:
    from pxr import Usd, UsdPhysics
except ImportError:
    Usd = UsdPhysics = None


class CalibrationIntegrityTests(unittest.TestCase):
    def test_calibrated_asset_has_verified_original_dependencies(self):
        asset = ASSETS / MANIFEST['training_usd']
        hashes, manifests = verify_asset_chain(asset)
        self.assertIn(str(asset.resolve()), hashes)
        profile = json.loads(asset.with_suffix('.json').read_text())
        self.assertEqual(profile['wheel_masses_kg'], {'LW_link': .665, 'RW_link': .665})
        self.assertFalse(profile['inertia']['measured'])
        self.assertIn(str((asset.parent / profile['source']).resolve()), hashes)
        self.assertGreaterEqual(len(manifests), 5)


@unittest.skipUnless(Usd is not None, 'USD composition checks need usd-core or Isaac Sim startup')
class CalibrationCompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asset = ASSETS / MANIFEST['training_usd']
        cls.profile = json.loads(asset.with_suffix('.json').read_text())
        cls.original = Usd.Stage.Open(str(asset.parent / cls.profile['source']))
        cls.calibrated = Usd.Stage.Open(str(asset))
        if not cls.original or not cls.calibrated:
            raise RuntimeError('Calibration USD cannot be composed')

    def test_both_wheel_and_hub_motor_masses_match_owner_confirmation(self):
        for name, target in self.profile['wheel_masses_kg'].items():
            api = UsdPhysics.MassAPI(self.calibrated.GetPrimAtPath('/mine_robot/' + name))
            self.assertAlmostEqual(api.GetMassAttr().Get(), target, places=6)
        total = sum(UsdPhysics.MassAPI(p).GetMassAttr().Get()
                    for p in self.calibrated.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertAlmostEqual(total, self.profile['calibrated_composed_total_mass_kg'], places=6)

    def test_estimated_left_inertia_preserves_cad_axes_and_center(self):
        for name in ('LW_link', 'RW_link'):
            old = UsdPhysics.MassAPI(self.original.GetPrimAtPath('/mine_robot/' + name))
            new = UsdPhysics.MassAPI(self.calibrated.GetPrimAtPath('/mine_robot/' + name))
            self.assertEqual(old.GetCenterOfMassAttr().Get(), new.GetCenterOfMassAttr().Get())
            self.assertEqual(old.GetPrincipalAxesAttr().Get(), new.GetPrincipalAxesAttr().Get())
            scale = .665 / old.GetMassAttr().Get() if name == 'LW_link' else 1.
            for before, after in zip(old.GetDiagonalInertiaAttr().Get(), new.GetDiagonalInertiaAttr().Get()):
                self.assertAlmostEqual(after, before * scale, places=10)

    def test_nonwheel_dynamics_and_all_joint_frames_are_unchanged(self):
        for prim in self.original.Traverse():
            other = self.calibrated.GetPrimAtPath(prim.GetPath())
            self.assertTrue(other.IsValid())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() not in ('LW_link', 'RW_link'):
                for key in ('mass', 'centerOfMass', 'diagonalInertia', 'principalAxes'):
                    self.assertEqual(prim.GetAttribute('physics:' + key).Get(),
                                     other.GetAttribute('physics:' + key).Get())
            if prim.IsA(UsdPhysics.RevoluteJoint):
                for attr in prim.GetAttributes():
                    self.assertEqual(attr.Get(), other.GetAttribute(attr.GetName()).Get())
                for rel in prim.GetRelationships():
                    self.assertEqual(rel.GetTargets(), other.GetRelationship(rel.GetName()).GetTargets())


if __name__ == '__main__':
    unittest.main()
