"""New closed-chain deployment parity against the current Isaac Lab source."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares

from core import PROJECT, Simulation, load_actor, load_config
from test_interface import reference_class, namespace


class MineInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        root = PROJECT.parent
        checkpoint = os.environ.get('MINE_SIM2SIM_CHECKPOINT')
        if not checkpoint:
            checkpoint = json.loads((root / 'mine_model/results/latest_live_replay.json').read_text())['checkpoint']
        cls.checkpoint = Path(checkpoint)
        cls.cfg = load_config(cls.checkpoint.parent / 'params/env.yaml')
        cls.model_path = Path(os.environ.get('MINE_SIM2SIM_MJCF', root / 'mine_model/exports/mujoco_protected_stops_v2_20261004/scene_protected_contact.xml'))
        cls.model = mujoco.MjModel.from_xml_path(str(cls.model_path))
        cls.audit = json.loads(cls.model_path.with_suffix('.json').read_text())
        if Path(cls.audit['source_usd']) != Path(cls.cfg['robot']['spawn']['usd_path']).resolve():
            raise ValueError('Test fixture differs from the checkpoint training asset')
        cls.reference = reference_class()

    def test_body_inertias_joint_frames_and_collision_filter(self):
        model = self.model
        self.assertEqual((model.nbody - 1, model.njnt - 1, model.nu, model.neq), (15, 14, 6, 8))
        self.assertEqual(self.cfg['leg_model'], 'mine_five_bar')
        for body in self.audit['bodies']:
            bid = model.body(body['name']).id
            np.testing.assert_allclose(model.body_mass[bid], body['mass'], atol=1e-12)
            np.testing.assert_allclose(model.body_ipos[bid], body['com'], atol=1e-12)
            np.testing.assert_allclose(model.body_inertia[bid], body['inertia'], atol=1e-12)
            np.testing.assert_allclose(abs(np.dot(model.body_iquat[bid], body['principal_axes'])), 1, atol=1e-7)
        for joint in self.audit['joints']:
            if joint['external']:
                continue
            jid = model.joint(joint['name']).id
            np.testing.assert_allclose(model.jnt_pos[jid], joint['positions'][1], atol=1e-10)
            np.testing.assert_allclose(model.jnt_axis[jid], joint['axes'][1], atol=1e-7)
        for a in range(model.ngeom):
            for b in range(a + 1, model.ngeom):
                if not model.geom_bodyid[a] or not model.geom_bodyid[b]:
                    continue
                contact = ((model.geom_contype[a] & model.geom_conaffinity[b]) or
                           (model.geom_contype[b] & model.geom_conaffinity[a]))
                if contact:
                    self.assertEqual({model.geom_contype[a], model.geom_contype[b]}, {4, 8})
        data = Simulation(self.cfg, model).data
        gaps = [np.linalg.norm(data.site_xpos[model.eq_obj1id[i]] - data.site_xpos[model.eq_obj2id[i]]) for i in range(model.neq)]
        self.assertLess(max(gaps), 1e-6)
        np.testing.assert_allclose(model.actuator_forcerange, [[-10, 10]] * 6)
        np.testing.assert_allclose(model.actuator_gainprm[:, 0], [300, 300, 2, 300, 300, 2])
        np.testing.assert_allclose(model.actuator_biasprm[:, :3], [[0, -300, -3], [0, -300, -3], [0, 0, -2],
                                                                 [0, -300, -3], [0, -300, -3], [0, 0, -2]])

    def test_observation_applied_action_and_drive_targets_match_isaaclab(self):
        rng = np.random.default_rng(42)
        sim = Simulation(self.cfg, self.model)
        for _ in range(40):
            sim.data.qpos[sim.qids] = rng.uniform(-.25, .25, 6)
            sim.data.qpos[2] = rng.uniform(.28, .32)
            quaternion = rng.normal(size=4)
            sim.data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
            sim.data.qvel[:] = rng.normal(size=self.model.nv) * 2
            mujoco.mj_forward(sim.model, sim.data)
            sim.command = np.array([rng.uniform(-.6, .6), rng.uniform(-.3, .3), rng.uniform(.28, .32)])
            sim.set_action(torch.tensor(rng.normal(size=(1, 6)), dtype=torch.float32))
            sim.apply_control()
            ref = self.reference()
            ref.cfg = namespace(self.cfg)
            ref.cfg.five_bar_geometry = self.cfg['five_bar_geometry']
            ref.num_envs, ref.device, ref.pi = 1, 'cpu', torch.pi
            ref.step_dt = self.cfg['sim']['dt'] * self.cfg['decimation']
            ref._leg_joint_ids, ref._wheel_joint_ids = [0, 1, 3, 4], [2, 5]
            ref._torque_joint_ids = list(range(6))
            ref._raw_actions, ref._actions = sim.raw.clone(), sim.raw.clamp(-1, 1).clone()
            ref._commands = torch.tensor([list(sim.command) + [0]], dtype=torch.float32)
            scales = self.cfg['obs_scales']
            ref._commands_scale = torch.tensor([scales['lin_vel'], scales['ang_vel'], scales['height_measurements']])
            ref._target_lin_vel_x = ref._commands[:, 0].clone()
            ref._last_actions = torch.zeros(1, 6, 2)
            ref._torque_limits = torch.full((1, 6), 10.)
            ref._update_commands = lambda: None
            for name in ('_theta1', '_theta2', '_L0', '_theta0', '_L0_dot', '_theta0_dot'):
                setattr(ref, name, torch.zeros(1, 2))
            ref._five_bar_jacobian = torch.zeros(1, 2, 2, 2)
            ref._five_bar_valid = torch.ones(1, 2, dtype=torch.bool)
            angular, gravity, *_ = sim.body_state()
            rotation = sim.data.xmat[sim.base_id].reshape(3, 3)
            com_velocity = sim.data.qvel[:3] + np.cross(rotation @ sim.data.qvel[3:6], rotation @ sim.model.body_ipos[sim.base_id])
            data = SimpleNamespace(
                joint_pos=torch.tensor(sim.data.qpos[sim.qids][None], dtype=torch.float32),
                joint_vel=torch.tensor(sim.data.qvel[sim.vids][None], dtype=torch.float32),
                root_pos_w=torch.tensor(sim.data.qpos[:3][None], dtype=torch.float32),
                root_quat_w=torch.tensor(sim.data.qpos[3:7][None], dtype=torch.float32),
                root_ang_vel_b=torch.tensor(angular[None], dtype=torch.float32),
                projected_gravity_b=torch.tensor(gravity[None], dtype=torch.float32),
                root_lin_vel_w=torch.tensor(com_velocity[None], dtype=torch.float32))
            ref._robot = SimpleNamespace(data=data,
                set_joint_effort_target=lambda value, joint_ids: setattr(ref, 'effort', value),
                set_joint_position_target=lambda value, joint_ids: setattr(ref, 'position_ref', value),
                set_joint_velocity_target=lambda value, joint_ids: setattr(ref, 'velocity_ref', value))
            ref._apply_action()
            np.testing.assert_allclose(sim.joint_ref, ref.position_ref.numpy()[0], atol=2e-6)
            np.testing.assert_allclose(sim.data.ctrl[[2, 5]], ref.velocity_ref.numpy()[0], atol=2e-6)
            np.testing.assert_allclose(ref.effort.numpy(), 0, atol=0)
            np.testing.assert_allclose(sim.actions.numpy(), ref._actions.numpy(), atol=2e-6)
            # Isaac computes quaternion/frame math in float32; MuJoCo uses float64.
            np.testing.assert_allclose(sim.observation().numpy(), ref._get_observations()['policy'].numpy(), atol=2e-6, rtol=2e-6)

    def test_physical_closed_chain_matches_five_bar_forward_kinematics(self):
        sim = Simulation(self.cfg, self.model)
        active = set(sim.qids)
        passive = np.array([self.model.jnt_qposadr[i] for i in range(1, self.model.njnt)
                            if self.model.jnt_qposadr[i] not in active])
        rng = np.random.default_rng(43)
        for _ in range(12):
            sim.data.qpos[sim.qids[[0, 1, 3, 4]]] = rng.uniform(-.15, .15, 4)
            def residual(q):
                sim.data.qpos[passive] = q
                mujoco.mj_forward(self.model, sim.data)
                return np.concatenate([sim.data.site_xpos[self.model.eq_obj1id[i]] - sim.data.site_xpos[self.model.eq_obj2id[i]] for i in range(self.model.neq)])
            solution = least_squares(residual, sim.data.qpos[passive], xtol=1e-12, ftol=1e-12, gtol=1e-12)
            self.assertLess(np.max(np.abs(residual(solution.x))), 1e-6)
            _, _, length, angle, *_ = sim.coordinates()
            for side, hip, wheel in ((0, 'LB_link', 'LW_link'), (1, 'RB_link', 'RW_link')):
                offset = sim.data.xmat[sim.base_id].reshape(3, 3).T @ (sim.data.xpos[self.model.body(wheel).id] - sim.data.xpos[self.model.body(hip).id])
                actual_length = np.linalg.norm(offset[[0, 2]])
                actual_angle = np.arctan2(-offset[0], -offset[2])
                self.assertAlmostEqual(length[0, side].item(), actual_length, delta=2e-6)
                self.assertAlmostEqual(angle[0, side].item(), actual_angle, delta=2e-5)

    def test_latest_actor_matches_checkpoint_layers(self):
        actor = load_actor(self.checkpoint, self.checkpoint.parent / 'params/agent.yaml')
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)['model_state_dict']
        inputs = torch.randn(16, 27, generator=torch.Generator().manual_seed(43))
        expected = inputs.clone()
        for index in (0, 2, 4, 6):
            expected = torch.nn.functional.linear(expected, state[f'actor.{index}.weight'], state[f'actor.{index}.bias'])
            if index != 6:
                expected = torch.nn.functional.elu(expected)
        torch.testing.assert_close(actor(inputs), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
