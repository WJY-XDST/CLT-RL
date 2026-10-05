"""Compare the deployment against actual Isaac Lab methods on identical states."""

import ast
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from core import DEFAULT_BUNDLE, PROJECT, VMC, FIVE_BAR, Simulation, build_model, load_actor, load_config


def namespace(value):
    return SimpleNamespace(**{key: namespace(item) for key, item in value.items()}) if isinstance(value, dict) else value


def reference_class():
    # Extract simulator-independent methods from the real source, avoiding Kit startup.
    source = PROJECT / "wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py"
    original = next(node for node in ast.parse(source.read_text()).body if isinstance(node, ast.ClassDef))
    methods = {"_get_wheel_vel_forward_positive", "_apply_action", "_update_forward_kinematics",
               "_forward_kinematics", "_vmc", "_sanitize_observation", "_get_observations",
               "_get_heading_frame_horizontal_velocity"}
    original.body = [node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    original.bases, original.decorator_list = [], []

    def quat_apply(q, v):
        xyz = q[:, 1:]
        cross = 2 * torch.cross(xyz, v, dim=-1)
        return v + q[:, :1] * cross + torch.cross(xyz, cross, dim=-1)

    scope = {"torch": torch, "leg_coordinates": VMC.leg_coordinates,
             "virtual_leg_torques": VMC.virtual_leg_torques, "quat_apply": quat_apply,
             "five_bar_state": FIVE_BAR.five_bar_state, "five_bar_inverse": FIVE_BAR.five_bar_inverse,
             "five_bar_torques": FIVE_BAR.five_bar_torques}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[original], type_ignores=[])), str(source), "exec"), scope)
    return scope[original.name]


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cfg = load_config(DEFAULT_BUNDLE / "env.yaml")
        cls.directory = tempfile.TemporaryDirectory()
        cls.model = build_model(cls.cfg, Path(cls.directory.name) / "scene.xml")
        cls.reference = reference_class()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_urdf_mass_joint_order_and_collision_masks(self):
        self.assertAlmostEqual(self.model.body_mass.sum(), 12.28, places=6)
        sim = Simulation(self.cfg, self.model)
        self.assertEqual(self.model.nq, 13)
        self.assertEqual(self.model.nv, 12)
        self.assertEqual(self.model.nu, 6)
        np.testing.assert_allclose(sim.data.qpos[sim.qids], [.5, .35, 0, -.5, -.35, 0])
        for first in range(self.model.ngeom):
            for second in range(first + 1, self.model.ngeom):
                if self.model.geom_bodyid[first] and self.model.geom_bodyid[second]:
                    self.assertFalse((self.model.geom_contype[first] & self.model.geom_conaffinity[second]) or
                                     (self.model.geom_contype[second] & self.model.geom_conaffinity[first]))

    def test_observations_and_torques_match_isaaclab_source(self):
        rng = np.random.default_rng(42)
        for legacy in (False, True):
            cfg = copy.deepcopy(self.cfg)
            cfg["vmc_legacy_angular_mapping"] = legacy
            sim = Simulation(cfg, self.model)
            for _ in range(40):
                sim.data.qpos[sim.qids] = rng.uniform([-.2, -.8, -3, -1, -.8, -3], [1, .8, 3, .2, .8, 3])
                sim.data.qpos[2] = rng.uniform(.13, .24)
                quaternion = rng.normal(size=4)
                sim.data.qpos[3:7] = quaternion / np.linalg.norm(quaternion)
                sim.data.qvel[:] = rng.normal(size=12) * 3
                mujoco.mj_forward(sim.model, sim.data)
                sim.command = np.array([rng.uniform(-.8, .8), .2, rng.uniform(.16, .2)])
                action = torch.from_numpy(rng.normal(size=(1, 6)).astype(np.float32) * 2)
                sim.set_action(action)
                ref = self.reference()
                ref.cfg = namespace(cfg)
                ref.num_envs, ref.device, ref.pi = 1, "cpu", torch.pi
                ref.step_dt = cfg["sim"]["dt"] * cfg["decimation"]
                ref._leg_joint_ids, ref._wheel_joint_ids = [0, 1, 3, 4], [2, 5]
                ref._torque_joint_ids = list(range(6))
                ref._raw_actions, ref._actions = sim.raw.clone(), sim.actions.clone()
                ref._commands = torch.tensor([list(sim.command) + [0]], dtype=torch.float32)
                ref._commands_scale = torch.tensor([2., .25, 5.])
                ref._target_lin_vel_x = ref._commands[:, 0].clone()
                ref._last_actions = torch.zeros(1, 6, 2)
                ref._torque_limits = torch.tensor([[30, 30, 5, 30, 30, 5]])
                ref._update_commands = lambda: None
                angular, gravity, *_ = sim.body_state()
                rotation = sim.data.xmat[sim.base_id].reshape(3, 3)
                com_offset = rotation @ sim.model.body_ipos[sim.base_id]
                body_velocity = sim.data.qvel[:3] + np.cross(rotation @ sim.data.qvel[3:6], com_offset)
                data = SimpleNamespace(
                    joint_pos=torch.tensor(sim.data.qpos[sim.qids][None], dtype=torch.float32),
                    joint_vel=torch.tensor(sim.data.qvel[sim.vids][None], dtype=torch.float32),
                    root_pos_w=torch.tensor(sim.data.qpos[:3][None], dtype=torch.float32),
                    root_quat_w=torch.tensor(sim.data.qpos[3:7][None], dtype=torch.float32),
                    root_ang_vel_b=torch.tensor(angular[None], dtype=torch.float32),
                    projected_gravity_b=torch.tensor(gravity[None], dtype=torch.float32),
                    root_lin_vel_w=torch.tensor(body_velocity[None], dtype=torch.float32))
                ref._robot = SimpleNamespace(data=data, set_joint_effort_target=lambda value, joint_ids: setattr(ref, "torques", value))
                ref._apply_action()
                sim.apply_control()
                np.testing.assert_allclose(sim.torques, ref.torques.numpy()[0], atol=2e-5, rtol=2e-5)
                np.testing.assert_allclose(sim.actions.numpy(), ref._actions.numpy(), atol=2e-6)
                np.testing.assert_allclose(sim.observation().numpy(), ref._get_observations()["policy"].numpy(), atol=2e-6)

    def test_velocity_is_at_com_and_in_correct_frames(self):
        sim = Simulation(self.cfg, self.model)
        sim.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        sim.data.qvel[:6] = [1, 2, 3, .4, .5, .6]
        mujoco.mj_forward(sim.model, sim.data)
        angular, gravity, heading, *_ = sim.body_state()
        np.testing.assert_allclose(angular, [.4, .5, .6], atol=1e-12)
        rotation = sim.data.xmat[sim.base_id].reshape(3, 3)
        com_velocity = sim.data.qvel[:3] + np.cross(rotation @ sim.data.qvel[3:6], rotation @ sim.model.body_ipos[sim.base_id])
        np.testing.assert_allclose(heading, [com_velocity[1], -com_velocity[0]], atol=1e-12)
        np.testing.assert_allclose(gravity, [0, 0, -1], atol=1e-12)

    def test_actor_matches_checkpoint_linear_layers(self):
        actor = load_actor(DEFAULT_BUNDLE / "model.pt", DEFAULT_BUNDLE / "agent.yaml")
        state = torch.load(DEFAULT_BUNDLE / "model.pt", map_location="cpu", weights_only=True)["model_state_dict"]
        inputs = torch.randn(16, 27, generator=torch.Generator().manual_seed(43))
        expected = inputs.clone()
        for index in (0, 2, 4, 6):
            expected = torch.nn.functional.linear(expected, state[f"actor.{index}.weight"], state[f"actor.{index}.bias"])
            if index != 6:
                expected = torch.nn.functional.elu(expected)
        torch.testing.assert_close(actor(inputs), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
