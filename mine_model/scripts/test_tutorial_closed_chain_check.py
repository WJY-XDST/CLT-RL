"""CPU-only checks for the read-only tutorial helper, using anonymous USD layers."""
import unittest
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Gf
from tutorial_closed_chain_check import check_stage, REFERENCE


def stage_copy():
    layer = Sdf.Layer.CreateAnonymous()
    layer.TransferContent(Usd.Stage.Open(str(REFERENCE)).Flatten())
    return Usd.Stage.Open(layer)


class TutorialChecks(unittest.TestCase):
    def test_reference(self):
        result = check_stage(stage_copy())
        self.assertTrue(result["passed"], result["errors"])
        self.assertEqual((result["rigid_bodies"], result["hinges"], result["tree_hinges"]), (15, 18, 14))
        self.assertEqual(len(result["frame_plan"]), 4)

    def test_open_chain(self):
        stage = stage_copy()
        stage.RemovePrim("/mine_robot/loop_joints")
        self.assertTrue(check_stage(stage, require_closed=False)["passed"])
        self.assertFalse(check_stage(stage, require_closed=True)["passed"])

    def test_exchanged_tree(self):
        stage = stage_copy()
        for side in "LR":
            UsdPhysics.Joint(stage.GetPrimAtPath(f"/mine_robot/joints/{side}S_joint")).GetExcludeFromArticulationAttr().Set(True)
            UsdPhysics.Joint(stage.GetPrimAtPath(f"/mine_robot/loop_joints/{side}L3_{side}S_closure")).GetExcludeFromArticulationAttr().Set(False)
        self.assertTrue(check_stage(stage)["passed"])

    def test_cycle(self):
        stage = stage_copy()
        UsdPhysics.Joint(stage.GetPrimAtPath("/mine_robot/loop_joints/RL3_RS_closure")).GetExcludeFromArticulationAttr().Set(False)
        self.assertFalse(check_stage(stage)["passed"])

    def test_disabled(self):
        stage = stage_copy()
        UsdPhysics.Joint(stage.GetPrimAtPath("/mine_robot/loop_joints/RL3_RS_closure")).GetJointEnabledAttr().Set(False)
        self.assertFalse(check_stage(stage)["passed"])

    def test_anchor_typo(self):
        stage = stage_copy()
        joint = UsdPhysics.Joint(stage.GetPrimAtPath("/mine_robot/loop_joints/RL3_RS_closure"))
        joint.GetLocalPos0Attr().Set(joint.GetLocalPos0Attr().Get() + Gf.Vec3f(.01, 0, 0))
        self.assertFalse(check_stage(stage)["passed"])

    def test_wrong_hip_parent(self):
        stage = stage_copy()
        UsdPhysics.Joint(stage.GetPrimAtPath("/mine_robot/joints/RL_joint")).GetBody0Rel().SetTargets(["/mine_robot/RB_link"])
        self.assertFalse(check_stage(stage)["passed"])

    def test_passive_drive(self):
        stage = stage_copy()
        drive = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath("/mine_robot/joints/RS_joint"), "angular")
        drive.CreateStiffnessAttr(10.)
        drive.CreateMaxForceAttr(2.)
        self.assertFalse(check_stage(stage)["passed"])
        drive.GetStiffnessAttr().Set(0.)
        drive.CreateDampingAttr(0.)
        self.assertTrue(check_stage(stage)["passed"])

    def test_rigid_root_motion_preserves_frame_plan(self):
        stage = stage_copy()
        root = UsdGeom.Xformable(stage.GetDefaultPrim())
        root.AddTransformOp().Set(Gf.Matrix4d(1.).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), 30.))
                                  .SetTranslateOnly(Gf.Vec3d(1, 2, 3)))
        result = check_stage(stage)
        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(result["initial_assembly_pose"])
        expected = check_stage(stage_copy())["frame_plan"]
        self.assertEqual(result["frame_plan"], expected)

    def test_bad_units(self):
        stage = stage_copy()
        UsdGeom.SetStageMetersPerUnit(stage, .001)
        self.assertFalse(check_stage(stage)["passed"])

    def test_read_only(self):
        stage = stage_copy()
        before = stage.GetRootLayer().ExportToString()
        check_stage(stage)
        self.assertEqual(before, stage.GetRootLayer().ExportToString())


if __name__ == "__main__":
    unittest.main()
