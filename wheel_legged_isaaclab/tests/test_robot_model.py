"""Imported larger-five-bar geometry and no-fallback vehicle contract."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from wheel_legged_gym_isaaclab.robot_model import MODEL, MAPPED_FIVE_BAR, mapped_five_bar_geometry, validate_robot_contract, validate_checkpoint_robot
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state, five_bar_torques
from wheel_legged_gym_isaaclab.mcu_hardware import HARDWARE_PROFILE, LEG_EFFORT_LIMIT, WHEEL_EFFORT_LIMIT


class RobotModelTests(unittest.TestCase):
    def cfg(self):
        return {"leg_model": "mine_five_bar", "five_bar_geometry": copy.deepcopy(MAPPED_FIVE_BAR),
                "hardware_profile": HARDWARE_PROFILE,
                **{key: copy.deepcopy(MODEL[key]) for key in ("leg_joint_names", "wheel_joint_names",
                   "wheel_radius", "wheel_track_width")},
                "leg_effort_limit": LEG_EFFORT_LIMIT, "wheel_effort_limit": WHEEL_EFFORT_LIMIT}

    def test_runtime_is_large_five_bar_and_mapping_is_idempotent(self):
        self.assertEqual(MAPPED_FIVE_BAR['scale'], 1.)
        for value in MAPPED_FIVE_BAR['upper']:
            self.assertAlmostEqual(value, .210, places=7)
        for value in MAPPED_FIVE_BAR['lower']:
            self.assertAlmostEqual(value, .250, places=7)
        self.assertEqual(mapped_five_bar_geometry(MAPPED_FIVE_BAR), MAPPED_FIVE_BAR)

    def test_both_geometry_forms_have_identical_virtual_work(self):
        generator = torch.Generator().manual_seed(37)
        for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 3e-5)):
            b = (torch.rand(500, 2, generator=generator, dtype=dtype) - .5) * .4
            l = (torch.rand(500, 2, generator=generator, dtype=dtype) - .5) * .4
            before = five_bar_state(b, l, **MODEL['five_bar'])
            after = five_bar_state(b, l, **MAPPED_FIVE_BAR)
            for x, y in zip(before[:3], after[:3]):
                torch.testing.assert_close(x, y, rtol=tolerance, atol=tolerance)
            self.assertTrue(torch.equal(before[3], after[3]))
            f, t = torch.full_like(b, 50.), torch.full_like(b, 2.)
            for x, y in zip(five_bar_torques(before[2], f, t), five_bar_torques(after[2], f, t)):
                torch.testing.assert_close(x, y, rtol=tolerance, atol=tolerance)

    def test_forward_state_matches_raw_cad_geometry_at_joint_samples(self):
        b, l = torch.meshgrid(torch.linspace(-.25, .25, 31, dtype=torch.float64),
                             torch.linspace(-.25, .25, 31, dtype=torch.float64), indexing='ij')
        sign = torch.tensor([[1., -1.]], dtype=torch.float64)
        b, l = (x.reshape(-1, 1) * sign for x in (b, l))
        result = five_bar_state(b, l, **MAPPED_FIVE_BAR)
        raw = five_bar_state(b, l, **MODEL['five_bar'])
        self.assertTrue(result[3].all())
        for x, y in zip(result[:3], raw[:3]):
            torch.testing.assert_close(x, y, rtol=1e-10, atol=1e-10)

    def test_checkpoint_small_form_is_normalized_once(self):
        cfg = self.cfg()
        cfg['five_bar_geometry'] = copy.deepcopy(MODEL['five_bar'])
        self.assertEqual(validate_robot_contract(cfg), MAPPED_FIVE_BAR)

    def test_missing_other_vehicle_or_wrong_geometry_is_rejected(self):
        for mutate in (lambda c: c.pop('leg_model'), lambda c: c.update(leg_model='unsupported'),
                       lambda c: c.update(wheel_radius=.1), lambda c: c.update(leg_joint_names=['wrong']*4),
                       lambda c: c['five_bar_geometry']['upper'].__setitem__(0, .4)):
            cfg = self.cfg()
            mutate(cfg)
            with self.assertRaises((ValueError, KeyError)):
                validate_robot_contract(cfg)

    def test_only_configuration_contains_the_actual_vehicle_defaults(self):
        path = Path(__file__).resolve().parents[1] / 'wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env_cfg.py'
        tree = ast.parse(path.read_text())
        base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'WheelLeggedVMCFlatEnvCfg')
        fields = {node.target.id: node.value for node in base.body if isinstance(node, ast.AnnAssign)}
        for key in ('robot', 'leg_model', 'leg_joint_names', 'wheel_radius', 'wheel_track_width',
                    'five_bar_geometry', 'feedforward_force', 'kp_l0', 'kp_theta'):
            self.assertNotEqual(ast.unparse(fields[key]), 'MISSING')
        self.assertEqual(ast.literal_eval(fields['leg_model']), 'mine_five_bar')
        self.assertEqual(ast.unparse(fields['five_bar_geometry']), 'MAPPED_FIVE_BAR')
        self.assertFalse(path.with_name('mine_env_cfg.py').exists())

    def test_checkpoint_requires_matching_saved_vehicle_configuration(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            checkpoint = Path(folder) / 'model.pt'
            with self.assertRaises(ValueError):
                validate_checkpoint_robot(checkpoint)
            cfg = self.cfg()
            (Path(folder) / 'env.yaml').write_text(json.dumps(cfg))
            self.assertEqual(validate_checkpoint_robot(checkpoint), cfg)
            cfg['wheel_radius'] *= 2
            (Path(folder) / 'env.yaml').write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):
                validate_checkpoint_robot(checkpoint)

    def test_active_tasks_cannot_select_an_example_vehicle(self):
        path = Path(__file__).resolve().parents[1] / 'wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/__init__.py'
        tree = ast.parse(path.read_text())
        task_ids = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                    and isinstance(n.value, str) and n.value.startswith('WheelLegged') and n.value.endswith('-v0')]
        self.assertEqual(set(task_ids), {
            'WheelLeggedVMC-Flat-v0', 'WheelLeggedJointPD-Flat-v0',
        })
        self.assertIn('.wheel_legged_vmc_flat_env_cfg:WheelLeggedVMCFlatEnvCfg', path.read_text())
        self.assertNotIn('mine_env_cfg', path.read_text().replace('Mine 适配层', ''))


if __name__ == '__main__':
    unittest.main()
