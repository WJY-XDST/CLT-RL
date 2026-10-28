"""CPU contract tests; these do not claim native PhysX robustness qualification."""
from types import SimpleNamespace as NS
import unittest

import torch

from wheel_legged_gym_isaaclab.deployment_randomization import (
    DeploymentRandomizationSession, DeploymentRandomizationSpec,
)


class View:
    def __init__(self, n):
        self.mass = torch.tensor([[9.77, 1.27, .665, 1.27, .665]]).repeat(n, 1)
        self.inertia = torch.eye(3).reshape(1, 1, 9).repeat(n, 5, 1) * .01
        self.material = torch.tensor([[[.5, .5, 0.]]]).repeat(n, 5, 1)
        self.com = torch.zeros(n, 5, 7)
        self.com[..., 6] = 1.
        self.stiffness = torch.tensor([[300., 300., 0., 300., 300., 0.]]).repeat(n, 1)
        self.damping = torch.tensor([[3., 3., 4.5, 3., 3., 4.5]]).repeat(n, 1)
        self.writes = 0

    def get_masses(self): return self.mass.clone()
    def get_inertias(self): return self.inertia.clone()
    def get_material_properties(self): return self.material.clone()
    def get_coms(self): return self.com.clone()
    def get_dof_stiffnesses(self): return self.stiffness.clone()
    def get_dof_dampings(self): return self.damping.clone()

    def _set(self, name, value, ids):
        self.writes += 1
        getattr(self, name)[ids] = value[ids]

    def set_masses(self, value, ids): self._set('mass', value, ids)
    def set_inertias(self, value, ids): self._set('inertia', value, ids)
    def set_material_properties(self, value, ids): self._set('material', value, ids)
    def set_coms(self, value, ids): self._set('com', value, ids)


def fixture(n=8, command=0.):
    view = View(n)
    robot = NS(root_physx_view=view, body_names=['base_link', 'left_leg', 'LW_link', 'right_leg', 'RW_link'],
               _ALL_INDICES=torch.arange(n), data=NS(root_ang_vel_b=torch.zeros(n, 3),
               projected_gravity_b=torch.tensor([[0., 0., -1.]]).repeat(n, 1)))
    def stiffness(value, *, joint_ids): view.stiffness[:, joint_ids] = value
    def damping(value, *, joint_ids): view.damping[:, joint_ids] = value
    robot.write_joint_stiffness_to_sim = stiffness
    robot.write_joint_damping_to_sim = damping
    scale = NS(ang_vel=.25, dof_pos=1., dof_vel=.05, lin_vel=2.)
    clip = NS(ang_vel=20., projected_gravity=1., theta=3.1416, theta_dot=50., lin_vel=5., wheel_vel=100.)
    cfg = NS(leg_control_mode='implicit_joint_position', feedforward_force=66.9042,
             wheel_action_limit=1.25, sim=NS(dt=.005), obs_scales=scale, obs_clip=clip,
             terrain=NS(physics_material=NS(static_friction=.5, dynamic_friction=.5,
                                            friction_combine_mode='multiply')))
    env = NS(cfg=cfg, num_envs=n, device='cpu', _robot=robot,
             _raw_actions=torch.zeros(n, 6), _actions=torch.zeros(n, 6),
             _leg_joint_ids=[0, 1, 3, 4], _wheel_joint_ids=[2, 5], observation_queries=0)
    env._sanitize_observation = lambda value, bound: value.clamp(-bound, bound)
    env._get_leg_observation = lambda: torch.zeros(n, 8)
    env._get_heading_frame_horizontal_velocity = lambda: torch.zeros(n, 2)
    env._get_wheel_vel_forward_positive = lambda: torch.zeros(n, 2)
    def pre(actions):
        env._raw_actions = actions.clone()
        env._actions = actions.clone()
    def apply(): env.last_applied = env._raw_actions.clone()
    def observations():
        env.observation_queries += 1
        values = torch.zeros(n, 27)
        values[:, 5] = -1.
        values[:, 6:9] = command
        return {'policy': values}
    env._pre_physics_step, env._apply_action = pre, apply
    env._get_observations = observations
    env._reset_idx = lambda ids: None
    return env, view


class DeploymentRandomizationTests(unittest.TestCase):
    def test_disabled_path_does_not_need_environment_or_draw_RNG(self):
        before = torch.random.get_rng_state().clone()
        result = DeploymentRandomizationSession(None, DeploymentRandomizationSpec())
        self.assertFalse(result.applied)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_payload_endpoints_change_total_mass_and_body_inertia_only(self):
        for delta in (-2., 2.):
            with self.subTest(delta=delta):
                env, view = fixture()
                old_mass, old_inertia = view.mass.clone(), view.inertia.clone()
                result = DeploymentRandomizationSession(env, DeploymentRandomizationSpec(
                    enabled=True, body_mass_offset_kg=(delta, delta)))
                torch.testing.assert_close(view.mass.sum(-1), torch.full((8,), 13.64 + delta))
                self.assertTrue(torch.equal(view.mass[:, 1:], old_mass[:, 1:]))
                self.assertTrue(torch.equal(view.inertia[:, 1:], old_inertia[:, 1:]))
                torch.testing.assert_close(view.inertia[:, 0], old_inertia[:, 0] * ((9.77 + delta) / 9.77))
                self.assertEqual(env.cfg.feedforward_force, 66.9042)
                self.assertTrue(result.evidence['physical_properties_readback_verified'])
                self.assertEqual(env.observation_queries, 0)

    def test_random_payload_is_bounded_and_independent_of_commands_and_global_RNG(self):
        spec = DeploymentRandomizationSpec(enabled=True, body_mass_offset_kg=(-2., 2.),
            material_static_friction=(.2, 1.5), dynamic_friction_fraction=(.7, 1.))
        before = torch.random.get_rng_state().clone()
        a, va = fixture(512, command=-100.)
        b, vb = fixture(512, command=100.)
        first, second = DeploymentRandomizationSession(a, spec), DeploymentRandomizationSession(b, spec)
        self.assertTrue(torch.equal(va.mass, vb.mass))
        self.assertTrue(torch.equal(va.material, vb.material))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        offsets = torch.tensor(first.evidence['body_mass_offset_kg'])
        self.assertGreaterEqual(float(offsets.min()), -2.)
        self.assertLessEqual(float(offsets.max()), 2.)
        self.assertGreater(float(offsets.max() - offsets.min()), 3.9)
        self.assertFalse(second.evidence['parameters_chosen_from_commands'])

    def test_material_bounds_restitution_and_actual_gains_are_read_back(self):
        env, view = fixture(32)
        result = DeploymentRandomizationSession(env, DeploymentRandomizationSpec(
            enabled=True, material_static_friction=(.2, 1.5), dynamic_friction_fraction=(.7, 1.),
            restitution=(0., .15), leg_stiffness_scale=(.8, 1.2),
            leg_damping_scale=(.8, 1.2), wheel_damping_scale=(.8, 1.2)))
        self.assertGreaterEqual(float(view.material[..., 0].min()), .2 - 1.e-6)
        self.assertLessEqual(float(view.material[..., 0].max()), 1.5)
        self.assertTrue((view.material[..., 1] <= view.material[..., 0]).all())
        self.assertTrue((view.material[..., 2] <= .15).all())
        for entry in result.evidence['implicit_PD_readback'].values():
            self.assertTrue(entry['readback_verified'])
        self.assertTrue(torch.equal(view.stiffness[:, [2, 5]], torch.zeros(32, 2)))
        self.assertEqual(env.cfg.feedforward_force, 66.9042)

    def test_com_translation_preserves_orientation_and_positive_inertia(self):
        env, view = fixture()
        quaternion = view.com[..., 3:].clone()
        DeploymentRandomizationSession(env, DeploymentRandomizationSpec(enabled=True,
            body_com_x_m=(.02, .02), body_com_y_m=(-.01, -.01), body_com_z_m=(.01, .01),
            body_inertia_scale=(.8, 1.2)))
        torch.testing.assert_close(view.com[:, 0, :3], torch.tensor([[.02, -.01, .01]]).repeat(8, 1))
        self.assertTrue(torch.equal(view.com[..., 3:], quaternion))
        self.assertTrue((torch.linalg.eigvalsh(view.inertia.reshape(8, 5, 3, 3)) > 0).all())

    def test_impossible_mass_fails_before_writing_properties_or_installing_hooks(self):
        env, view = fixture()
        original = env._get_observations
        with self.assertRaises(ValueError):
            DeploymentRandomizationSession(env, DeploymentRandomizationSpec(
                enabled=True, body_mass_offset_kg=(-20., -20.)))
        self.assertEqual(view.writes, 0)
        self.assertIs(env._get_observations, original)

    def test_readback_failure_prevents_installing_hooks(self):
        env, view = fixture()
        original = env._get_observations
        view.set_masses = lambda value, ids: None
        with self.assertRaises(RuntimeError):
            DeploymentRandomizationSession(env, DeploymentRandomizationSpec(
                enabled=True, body_mass_offset_kg=(2., 2.)))
        self.assertIs(env._get_observations, original)

    def test_delay_has_no_future_actions_and_reset_clears_history(self):
        env, view = fixture(1)
        session = DeploymentRandomizationSession(env, DeploymentRandomizationSpec(
            enabled=True, action_delay_steps=(2, 2), sensor_delay_steps=(2, 2)))
        env._pre_physics_step(torch.full((1, 6), .4))
        for expected in (0., 0., .4):
            env._apply_action()
            torch.testing.assert_close(env.last_applied, torch.full((1, 6), expected))
        self.assertEqual(env.observation_queries, 0)
        env._get_observations()
        self.assertEqual(env.observation_queries, 1)
        env._reset_idx(torch.tensor([0]))
        self.assertTrue(torch.equal(session.action_history.values, torch.zeros_like(session.action_history.values)))

    def test_ambiguous_or_nonphysical_ranges_fail_closed(self):
        for data in ({'body_mass_offset_kg': [2., -2.]},
                     {'material_static_friction': [.2, 1.], 'friction_scale': [.5, 1.5]},
                     {'dynamic_friction_fraction': [1., 1.1]}, {'restitution': [-.1, .1]},
                     {'body_inertia_scale': [0., 1.]}, {'leg_stiffness_scale': [True, 1.]},
                     {'body_com_x_m': [float('nan'), 0.]}, {'payload_body_name': ''}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                DeploymentRandomizationSpec.from_dict(data)


if __name__ == '__main__':
    unittest.main()
