import ast
from pathlib import Path
import unittest

from env_source import environment_tree

from wheel_legged_gym_isaaclab.leg_control import (
    leg_drive_gains, require_leg_control_mode, resolve_training_leg_mode,
)


class TestLegControl(unittest.TestCase):
    def test_explicit_vmc_removes_imported_or_overridden_joint_pd(self):
        self.assertEqual(leg_drive_gains('explicit_vmc', 300., 3.), (0., 0.))

    def test_implicit_mode_retains_requested_joint_pd(self):
        self.assertEqual(leg_drive_gains('implicit_joint_position', 300., 3.), (300., 3.))

    def test_invalid_implicit_gains_are_rejected(self):
        for kp,kd in ((0.,3.),(300.,-1.),(float('nan'),3.)):
            with self.assertRaises(ValueError):
                leg_drive_gains('implicit_joint_position',kp,kd)

    def test_required_mode_cannot_be_overridden_by_hydra(self):
        with self.assertRaises(ValueError):
            require_leg_control_mode('implicit_joint_position','explicit_vmc')
        require_leg_control_mode('explicit_vmc','explicit_vmc')

    def test_new_training_defaults_to_explicit_vmc(self):
        self.assertEqual(resolve_training_leg_mode({})['leg_control_mode'],'explicit_vmc')

    def test_requested_mode_overrides_initial_settings_in_a_new_experiment(self):
        old={'leg_control_mode':'implicit_joint_position','kp_l0':900.}
        result=resolve_training_leg_mode(old,'explicit_vmc')
        self.assertEqual(result['leg_control_mode'],'explicit_vmc')
        self.assertEqual(old['leg_control_mode'],'implicit_joint_position')
        self.assertEqual(result['kp_l0'],900.)

    def test_old_assessments_cannot_be_reused_after_switching_modes(self):
        with self.assertRaisesRegex(ValueError,'new --output'):
            resolve_training_leg_mode({},'explicit_vmc','implicit_joint_position')

    def test_same_mode_resume_is_allowed(self):
        self.assertEqual(resolve_training_leg_mode({'leg_control_mode':'explicit_vmc'},
                         'explicit_vmc','explicit_vmc')['leg_control_mode'],'explicit_vmc')

    def test_retired_mode_is_rejected_at_all_contract_entry_points(self):
        for callback in (lambda: require_leg_control_mode('implicit_joint_reference'),
                         lambda: leg_drive_gains('implicit_joint_reference',300.,3.),
                         lambda: resolve_training_leg_mode({'leg_control_mode':'implicit_joint_reference'})):
            with self.assertRaisesRegex(ValueError,'removed'):
                callback()

    def test_vmc_path_contains_no_leg_effort_clearing_or_inverse(self):
        path=Path(__file__).resolve().parents[1]/'wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py'
        tree=environment_tree()
        method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_apply_action')
        source=ast.unparse(method)
        self.assertNotIn('implicit_joint_reference',source)
        self.assertNotIn('five_bar_inverse',source)
        assignments=[n for n in ast.walk(method) if isinstance(n,ast.Assign) and
                     any(ast.unparse(t).startswith('torques[') for t in n.targets)]
        self.assertEqual(assignments,[])


if __name__=='__main__':
    unittest.main()
