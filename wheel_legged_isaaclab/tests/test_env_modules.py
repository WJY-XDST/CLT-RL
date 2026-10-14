"""One environment owns every LAB callback; only command sampling is separated."""
import ast
import unittest
from types import SimpleNamespace

import torch

from env_source import ENV_DIRECTORY, ENV_FILES, environment_tree


class TestEnvironmentModules(unittest.TestCase):
    def classes(self):
        return [node for node in environment_tree().body if isinstance(node, ast.ClassDef)]

    def test_main_owns_every_lab_callback_without_mixin_hierarchy(self):
        classes = {cls.name: cls for cls in self.classes()}
        main = classes['WheelLeggedVMCFlatEnv']
        self.assertEqual([base.id for base in main.bases], ['DirectRLEnv'])
        self.assertEqual(set(classes), {'WheelLeggedVMCFlatEnv'})
        callbacks = {node.name for node in main.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({'__init__', '_setup_scene', '_pre_physics_step', '_apply_action',
                         '_get_observations', '_get_rewards', '_get_dones', '_reset_idx'} <= callbacks)

    def test_every_method_has_exactly_one_owner(self):
        names = [method.name for cls in self.classes() for method in cls.body
                 if isinstance(method, ast.FunctionDef)]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 23)

    def test_file_sizes_remain_readable(self):
        for name in ENV_FILES:
            with self.subTest(name=name):
                limit = 1250 if name == 'wheel_legged_vmc_flat_env.py' else 230
                self.assertLessEqual(len((ENV_DIRECTORY / name).read_text().splitlines()), limit)

    def test_no_retired_inverse_or_jump_task_in_environment(self):
        tree = environment_tree()
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertNotIn('five_bar_inverse', names)
        self.assertNotIn('JumpPlan', names)

    def isolated(self, name):
        method = next(n for n in ast.walk(environment_tree())
                      if isinstance(n, ast.FunctionDef) and n.name == name)
        scope = {'torch': torch}
        exec(compile(ast.Module(body=[method], type_ignores=[]), '<production_env>', 'exec'), scope)
        return scope[name]

    def test_diagnostic_vmc_hooks_remain_identity_and_zero_bias(self):
        length = torch.full((2, 2), .26)
        angle = torch.zeros_like(length)
        dummy = SimpleNamespace(_L0=length)
        returned_length, returned_angle = self.isolated('_modify_leg_references')(dummy, length, angle)
        self.assertIs(returned_length, length)
        self.assertIs(returned_angle, angle)
        torch.testing.assert_close(self.isolated('_get_leg_force_bias')(dummy), torch.zeros_like(length))

    def test_termination_timer_preserves_boundary_and_safe_step_reset(self):
        advance = self.isolated('_advance_failure_counter')
        dummy = SimpleNamespace(step_dt=.02)
        unsafe = torch.tensor([True, False])
        previous = torch.tensor([1., 9.])
        counter, exclusive = advance(dummy, previous, unsafe, .04)
        _, inclusive = advance(dummy, previous, unsafe, .04, inclusive=True)
        torch.testing.assert_close(counter, torch.tensor([2., 0.]))
        self.assertEqual(exclusive.tolist(), [False, False])
        self.assertEqual(inclusive.tolist(), [True, False])
        _, low_height = advance(dummy, torch.zeros(2), unsafe, 0., inclusive=True, minimum_steps=1.)
        self.assertEqual(low_height.tolist(), [True, False])


if __name__ == '__main__':
    unittest.main()
