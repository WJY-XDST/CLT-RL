import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

from wheel_legged_gym_isaaclab.physics_contract import (
    EXTERNAL_FORCE_FIELD, apply_force_iteration_setting,
    checkpoint_force_iteration_setting, require_matching_force_iteration_setting,
    ROBOT_ASSET_FIELD, checkpoint_robot_asset_setting, require_matching_robot_asset_setting,
)


class TestPhysicsContract(unittest.TestCase):
    def test_mass_calibration_does_not_silently_change_historical_replay(self):
        saved = {'robot': {'spawn': {'usd_path': '/tmp/original_cad.usd'}}}
        self.assertEqual(checkpoint_robot_asset_setting(saved), '/tmp/original_cad.usd')
        self.assertEqual(checkpoint_robot_asset_setting(saved, {ROBOT_ASSET_FIELD: '/tmp/calibrated.usda'}),
                         '/tmp/calibrated.usda')
        self.assertEqual(saved['robot']['spawn']['usd_path'], '/tmp/original_cad.usd')

    def test_calibrated_mass_resume_is_an_explicit_physics_experiment(self):
        saved = {'robot': {'spawn': {'usd_path': '/tmp/original_cad.usd'}}}
        require_matching_robot_asset_setting(saved, '/tmp/original_cad.usd')
        with self.assertRaisesRegex(ValueError, 'physics contract changed'):
            require_matching_robot_asset_setting(saved, '/tmp/calibrated.usda')
        require_matching_robot_asset_setting(saved, '/tmp/calibrated.usda', allow_change=True)

    def test_asset_contract_rejects_invalid_path_values(self):
        saved = {'robot': {'spawn': {'usd_path': '/tmp/original_cad.usd'}}}
        for value in (None, '', '  ', False, 0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint_robot_asset_setting(saved, {ROBOT_ASSET_FIELD: value})
            with self.subTest(value=value), self.assertRaises(ValueError):
                require_matching_robot_asset_setting(saved, value, allow_change=True)

    def test_historical_checkpoints_keep_the_false_default(self):
        self.assertFalse(checkpoint_force_iteration_setting({}))

    def test_new_checkpoint_keeps_its_true_setting(self):
        self.assertTrue(checkpoint_force_iteration_setting({EXTERNAL_FORCE_FIELD: True}))

    def test_explicit_ablation_takes_priority_without_mutating_saved_config(self):
        old = {EXTERNAL_FORCE_FIELD: False}
        self.assertTrue(checkpoint_force_iteration_setting(old, {EXTERNAL_FORCE_FIELD: True}))
        self.assertFalse(old[EXTERNAL_FORCE_FIELD])

    def test_string_booleans_and_numbers_are_not_accepted(self):
        for value in ('False', 'True', 0, 1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                checkpoint_force_iteration_setting({EXTERNAL_FORCE_FIELD: value})

    def test_resume_requires_an_explicit_physics_change(self):
        with self.assertRaisesRegex(ValueError, 'start fresh'):
            require_matching_force_iteration_setting({}, True)
        require_matching_force_iteration_setting({}, True, allow_change=True)
        require_matching_force_iteration_setting({}, False)
        require_matching_force_iteration_setting({EXTERNAL_FORCE_FIELD: True}, True)

    def test_scene_setting_is_authored_and_read_back(self):
        stage = Mock()
        api = Mock()
        api.GetEnableExternalForcesEveryIterationAttr.return_value.Get.return_value = True
        schema = SimpleNamespace(PhysxSceneAPI=SimpleNamespace(Apply=Mock(return_value=api)))
        with patch.dict(sys.modules, {'pxr': SimpleNamespace(PhysxSchema=schema)}):
            self.assertTrue(apply_force_iteration_setting(stage, '/physicsScene', True))
        stage.GetPrimAtPath.assert_called_once_with('/physicsScene')
        api.CreateEnableExternalForcesEveryIterationAttr.assert_called_once_with(True)

    def test_missing_scene_and_failed_authorship_are_rejected(self):
        stage = Mock()
        stage.GetPrimAtPath.return_value.IsValid.return_value = False
        api = Mock()
        schema = SimpleNamespace(PhysxSceneAPI=SimpleNamespace(Apply=Mock(return_value=api)))
        with patch.dict(sys.modules, {'pxr': SimpleNamespace(PhysxSchema=schema)}):
            with self.assertRaisesRegex(ValueError, 'Physics scene not found'):
                apply_force_iteration_setting(stage, '/missing', True)
            stage.GetPrimAtPath.return_value.IsValid.return_value = True
            api.GetEnableExternalForcesEveryIterationAttr.return_value.Get.return_value = False
            with self.assertRaisesRegex(RuntimeError, 'Failed to author'):
                apply_force_iteration_setting(stage, '/physicsScene', True)

    def test_new_mine_config_opts_in_and_environment_applies_before_articulation(self):
        root = Path(__file__).resolve().parents[1]/'wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat'
        config = ast.parse((root/'wheel_legged_vmc_flat_env_cfg.py').read_text())
        mine = next(n for n in config.body if isinstance(n, ast.ClassDef) and n.name == 'WheelLeggedVMCFlatEnvCfg')
        assignment = next(n for n in mine.body if isinstance(n, (ast.Assign, ast.AnnAssign)) and
                          any(isinstance(t, ast.Name) and t.id == EXTERNAL_FORCE_FIELD
                              for t in (n.targets if isinstance(n, ast.Assign) else [n.target])))
        self.assertIs(ast.literal_eval(assignment.value), True)
        env = ast.parse((root/'wheel_legged_vmc_flat_env.py').read_text())
        method = next(n for n in ast.walk(env) if isinstance(n, ast.FunctionDef) and n.name == '_setup_scene')
        source = ast.unparse(method)
        self.assertLess(source.index('apply_force_iteration_setting'), source.index('Articulation('))


if __name__ == '__main__':
    unittest.main()
