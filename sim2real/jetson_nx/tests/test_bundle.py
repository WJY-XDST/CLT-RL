"""Optional integration checks against the explicitly prepared local checkpoint."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

from wheelleg_deploy.policy import BundlePolicy, load_yaml, prepare_bundle
from wheelleg_deploy.targets import action_to_targets

PROJECT = Path(__file__).resolve().parents[1]
BUNDLE = Path(os.environ.get("SIM2REAL_TEST_BUNDLE", PROJECT / "bundles/direct_joint_interface_check"))


@unittest.skipUnless(BUNDLE.is_dir(), "Prepare a local trusted checkpoint bundle to run export integration checks")
class ExportIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.policy = BundlePolicy(BUNDLE)
        cls.obs = json.loads((BUNDLE / "synthetic_obs.json").read_text())["obs"]

    def test_frozen_actor_matches_existing_sim2sim_loader(self):
        import torch
        try:
            import mujoco
        except ImportError:
            self.skipTest("Existing sim2sim loader needs MuJoCo; host itself does not")
        source = PROJECT.parents[1] / "wheel_legged_isaaclab/sim2sim/core.py"
        spec = importlib.util.spec_from_file_location("sim2real_existing_sim2sim", source)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        checkpoint = Path(self.policy.metadata["checkpoint"])
        actor = baseline.load_actor(checkpoint, checkpoint.parent / "params/agent.yaml")
        for speed, yaw, height in ((0., 0., .3), (.3, 0., .28), (-.6, 0., .32), (0., .3, .3)):
            obs = self.obs.copy()
            obs[6:9] = [speed * 2, yaw * .25, height * 5]
            with torch.inference_mode():
                expected = actor(torch.tensor([obs], dtype=torch.float32))[0]
            result = torch.tensor(self.policy(obs))
            torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_output_matches_selected_config_and_never_approves_hardware(self):
        action = self.policy(self.obs)
        target = action_to_targets(action, self.obs, self.policy.cfg)
        self.assertEqual(len(action), 6)
        self.assertEqual(len(target["joint_position_ref_rad"]), 4)
        self.assertEqual(len(target["wheel_speed_ref_rad_s"]), 2)
        self.assertFalse(self.policy.metadata["hardware_approved"])
        self.assertFalse(self.policy.metadata["policy_precision_accepted"])
        self.assertAlmostEqual(self.policy.metadata["policy_period_s"], .02)

    def test_exported_wheel_bound_and_targets_match_the_checkpoint(self):
        checkpoint = Path(self.policy.metadata['checkpoint'])
        source = load_yaml(checkpoint.parent/'params/env.yaml')
        bound = source.get('wheel_action_limit', 1.)
        self.assertEqual(self.policy.cfg.get('wheel_action_limit', 1.), bound)
        target = action_to_targets([0., 0., 3., 0., 0., -3.], self.obs, self.policy.cfg)
        self.assertEqual(target['wheel_speed_ref_rad_s'],
                         [bound*source['action_scale_vel'], -bound*source['action_scale_vel']])

    def test_changed_policy_or_control_configuration_is_rejected(self):
        for name in ("policy.pt", "runtime_config.json", "obs_schema.json"):
            with self.subTest(file=name), tempfile.TemporaryDirectory() as root:
                clone = Path(root) / "bundle"
                shutil.copytree(BUNDLE, clone)
                if name == "policy.pt":
                    with (clone / name).open("ab") as stream:
                        stream.write(b"invalid")
                else:
                    data = json.loads((clone / name).read_text())
                    if name == "runtime_config.json":
                        data["leg_joint_position_scale"][0] *= 2.
                    else:
                        data["fields"][0]["scale"] = 100.
                    (clone / name).write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    BundlePolicy(clone)

    def test_prepare_refuses_overwrite_or_missing_paired_configs(self):
        checkpoint = Path(self.policy.metadata["checkpoint"])
        with self.assertRaises(ValueError):
            prepare_bundle(checkpoint, BUNDLE)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(FileNotFoundError):
                prepare_bundle(Path(root) / "model.pt", Path(root) / "output")


if __name__ == "__main__":
    unittest.main()
