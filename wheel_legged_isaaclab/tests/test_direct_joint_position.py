"""Direct four-hip policy contract; no simulator import needed."""
import ast
import copy
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch

from env_source import environment_tree

from wheel_legged_gym_isaaclab.leg_control import (
    DIRECT_JOINT_POSITION_MODE, LEG_ACTION_COLUMNS, joint_position_targets,
    leg_drive_gains, require_matching_policy_contract,
)
from wheel_legged_gym_isaaclab.robot_model import MAPPED_FIVE_BAR, MODEL
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state, five_bar_torques

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / 'wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py'
OFFSET = (-.0707452938, -.0336936327, .0707452938, .0336936327)
SCALE = (.30,) * 4


class TestDirectJointPosition(unittest.TestCase):
    def limits(self, n=3):
        return torch.tensor([-1., 1.]).expand(n, 4, 2).clone()

    def cfg(self):
        return dict(leg_control_mode=DIRECT_JOINT_POSITION_MODE, leg_joint_position_offset=OFFSET,
                    leg_joint_position_scale=SCALE, leg_joint_target_limit_margin=.01,
                    leg_joint_gravity_feedforward=True,
                    observation_space=27, action_space=6)

    def test_zero_policy_is_nominal_pose_from_this_robot(self):
        target, applied = joint_position_targets(torch.zeros(3, 4), OFFSET, SCALE, self.limits(), .01)
        torch.testing.assert_close(target, torch.tensor(OFFSET).expand(3, 4))
        self.assertEqual(applied.abs().max().item(), 0.)
        length, angle, _, valid = five_bar_state(target[:, ::2], target[:, 1::2], **MAPPED_FIVE_BAR)
        self.assertTrue(valid.all())
        torch.testing.assert_close(angle, torch.zeros_like(angle), atol=1e-6, rtol=0.)
        root_height = length + MODEL['wheel_radius'] - MODEL['hip_z']
        torch.testing.assert_close(root_height, torch.full_like(root_height, .30), atol=1e-6, rtol=0.)

    def test_clips_every_joint_action_and_respects_articulation_stops(self):
        a = torch.tensor([[10., -10., float('nan'), float('inf')]])
        limits = self.limits(1)
        limits[:, 0, 1] = .1
        target, applied = joint_position_targets(a, OFFSET, SCALE, limits, .01)
        self.assertAlmostEqual(target[0, 0].item(), .09, places=6)
        self.assertAlmostEqual(target[0, 1].item(), OFFSET[1] - .3, places=6)
        self.assertAlmostEqual(target[0, 2].item(), OFFSET[2], places=6)
        self.assertTrue(torch.isfinite(target).all())
        self.assertLessEqual(applied.abs().max().item(), 1.)

    def test_bad_mapping_and_limit_envelopes_fail_closed(self):
        for offset, scale, margin, limits in (
                (OFFSET[:3], SCALE, .01, self.limits()),
                (OFFSET, (.3, .3, 0., .3), .01, self.limits()),
                ((float('nan'), *OFFSET[1:]), SCALE, .01, self.limits()),
                (OFFSET, SCALE, -1., self.limits()),
                (OFFSET, SCALE, 1., self.limits()),
                ((2.,) * 4, SCALE, .01, self.limits())):
            with self.assertRaises(ValueError):
                joint_position_targets(torch.zeros(3, 4), offset, scale, limits, margin)

    def test_action_envelope_has_valid_closed_chain_geometry(self):
        a, b = torch.meshgrid(torch.linspace(-1., 1., 61), torch.linspace(-1., 1., 61), indexing='ij')
        action = torch.stack((a.flatten(), b.flatten(), a.flatten(), b.flatten()), dim=-1)
        target, _ = joint_position_targets(action, OFFSET, SCALE, self.limits(len(action)), .01)
        state = five_bar_state(target[:, ::2], target[:, 1::2], **MAPPED_FIVE_BAR)
        self.assertTrue(state[3].all())
        self.assertTrue(torch.isfinite(state[2]).all())

    def test_raw_motor_axes_do_not_add_a_second_right_leg_sign_flip(self):
        action = torch.tensor([[.1, -.2, .3, -.4]])
        target, _ = joint_position_targets(action, OFFSET, SCALE, self.limits(1), .01)
        torch.testing.assert_close(target - torch.tensor(OFFSET), action * .3)

    def test_new_mode_has_implicit_position_gains(self):
        self.assertEqual(leg_drive_gains(DIRECT_JOINT_POSITION_MODE, 300., 3.), (300., 3.))

    def test_old_virtual_leg_policy_is_rejected_even_with_same_dimensions(self):
        for old in ('explicit_vmc', 'implicit_joint_reference'):
            saved = {**self.cfg(), 'leg_control_mode': old}
            with self.assertRaisesRegex(ValueError, 'fresh policy'):
                require_matching_policy_contract(saved, self.cfg())

    def test_new_policy_can_resume_only_with_its_own_mapping(self):
        cfg = self.cfg()
        require_matching_policy_contract(cfg, copy.deepcopy(cfg))
        for name, value in (('leg_joint_position_scale', (.4,) * 4),
                            ('leg_joint_position_offset', (0.,) * 4),
                            ('leg_joint_target_limit_margin', 0.), ('leg_joint_gravity_feedforward', False), ('observation_space', 30)):
            with self.assertRaisesRegex(ValueError, name):
                require_matching_policy_contract(cfg, {**cfg, name: value})

    def test_new_execution_bypasses_virtual_pd_and_ik_but_retains_gravity(self):
        tree = environment_tree()
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_apply_action')
        branch = next(n for n in method.body if isinstance(n, ast.If) and 'DIRECT_JOINT_POSITION_MODE' in ast.unparse(n.test))
        self.assertIsInstance(branch.body[-1], ast.Return)
        direct = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_apply_direct_joint_position_action')
        text = ast.unparse(direct)
        for disallowed in ('five_bar_inverse', 'self._vmc(', 'self.cfg.kp_theta', 'self.cfg.kp_l0', 'height_feedback_gain', 'height_reference_blend'):
            self.assertNotIn(disallowed, text)
        self.assertIn('set_joint_position_target', text)
        self.assertIn('torch.zeros_like(self._actions)', text)
        self.assertIn('self._get_gravity_feedforward()', text)
        self.assertIn('torch.zeros_like(self._theta0)', text)

    def test_feedforward_is_effort_only_not_an_angle_offset(self):
        tree = environment_tree()
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_apply_direct_joint_position_action')
        scope = dict(torch=torch, LEG_ACTION_COLUMNS=LEG_ACTION_COLUMNS, joint_position_targets=joint_position_targets,
                     five_bar_state=five_bar_state, five_bar_torques=five_bar_torques)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENV), 'exec'), scope)
        wheel_method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_wheel_effort_from_reference')
        exec(compile(ast.Module(body=[wheel_method], type_ignores=[]), str(ENV), 'exec'), scope)
        n = 2
        position = torch.tensor(OFFSET).expand(n, 4).clone()
        length, angle, jac, _ = five_bar_state(position[:, ::2], position[:, 1::2], **MAPPED_FIVE_BAR)
        saved = {}
        robot = SimpleNamespace(data=SimpleNamespace(joint_pos_limits=self.limits(n)),
                                set_joint_position_target=lambda x, **k: saved.update(position=x.clone()),
                                set_joint_velocity_target=lambda x, **k: None,
                                set_joint_effort_target=lambda x, **k: saved.update(effort=x.clone()))
        cfg = SimpleNamespace(leg_joint_position_offset=OFFSET, leg_joint_position_scale=SCALE,
                              leg_joint_target_limit_margin=.01, leg_joint_gravity_feedforward=True,
                              leg_effort_limit=35.574, action_scale_vel=15., wheel_control_mode='implicit_velocity',
                              five_bar_geometry=MAPPED_FIVE_BAR)
        force = torch.full_like(length, 60.5277)
        dummy = SimpleNamespace(cfg=cfg, _robot=robot, _raw_actions=torch.zeros(n, 6), _actions=torch.zeros(n, 6),
                                _leg_joint_ids=[0, 1, 2, 3], _wheel_joint_ids=[4, 5], _torque_joint_ids=list(range(6)),
                                _torque_limits=torch.full((n, 6), cfg.leg_effort_limit), _five_bar_jacobian=jac,
                                _theta0=angle, _leg_gravity_effort_applied=torch.zeros(n, 4),
                                _get_gravity_feedforward=lambda: force)
        dummy._wheel_effort_from_reference = MethodType(scope['_wheel_effort_from_reference'], dummy)
        scope['_apply_direct_joint_position_action'](dummy)
        a, b = five_bar_torques(jac, force, torch.zeros_like(angle))
        expected = torch.stack((a[:, 0], b[:, 0], a[:, 1], b[:, 1]), dim=-1)
        torch.testing.assert_close(saved['position'], position)
        torch.testing.assert_close(saved['effort'][:, LEG_ACTION_COLUMNS], expected)
        self.assertGreater(expected.abs().max().item(), .1)
        cfg.leg_joint_gravity_feedforward = False
        scope['_apply_direct_joint_position_action'](dummy)
        torch.testing.assert_close(saved['position'], position)
        self.assertEqual(saved['effort'].abs().max().item(), 0.)

    def test_physx_total_effort_envelope_is_applied_before_cloning(self):
        tree = environment_tree()
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_setup_scene')
        text = ast.unparse(method)
        self.assertIn("prim.ApplyAPI(PERF_ENV_API, 'angular')", text)
        self.assertLess(text.index('prim.ApplyAPI'), text.index('clone_environments'))

    def test_direct_leg_observation_order_position4_velocity4(self):
        tree = environment_tree()
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_get_leg_observation')
        scope = dict(torch=torch, DIRECT_JOINT_POSITION_MODE=DIRECT_JOINT_POSITION_MODE)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(ENV), 'exec'), scope)
        data = SimpleNamespace(joint_pos=torch.tensor([OFFSET]) + .1, joint_vel=torch.tensor([[1., 2., 3., 4.]]))
        cfg = SimpleNamespace(leg_control_mode=DIRECT_JOINT_POSITION_MODE, leg_joint_position_offset=OFFSET,
                              obs_clip=SimpleNamespace(theta=3.14, theta_dot=50.),
                              obs_scales=SimpleNamespace(dof_pos=1., dof_vel=.05))
        dummy = SimpleNamespace(cfg=cfg, _robot=SimpleNamespace(data=data), _leg_joint_ids=[0, 1, 2, 3],
                                _sanitize_observation=lambda x, cap: x.clamp(-cap, cap))
        obs = scope['_get_leg_observation'](dummy)
        torch.testing.assert_close(obs, torch.tensor([[.1, .1, .1, .1, .05, .1, .15, .2]]))


if __name__ == '__main__':
    unittest.main()
