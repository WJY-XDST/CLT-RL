"""Admission and launch-contract tests; no GPU or training process is started."""
import copy
import importlib.util
from pathlib import Path
import sys
import unittest

DIRECTORY = Path(__file__).resolve().parents[2] / "mine_model/scripts"
sys.path.insert(0, str(DIRECTORY))
spec = importlib.util.spec_from_file_location("solver_guarded_resume", DIRECTORY / "resume_after_solver_check_20261011.py")
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def fixtures():
    qualification = dict(
        training_permitted=True,
        candidate=dict(physics_hz=200, control_hz=200, policy_hz=50,
                       position_iterations=128, velocity_iterations=8,
                       external_forces_every_iteration=True),
        checks={"robot": dict(finite=True, survived=True, fixed_controller_clock=True)},
        gravity_accumulation={"probe": dict(passed=True, fp32_model_max_difference_m_s=0.)},
    )
    strict = dict(comparisons=[dict(passed=True)], per_case={
        "true_static_800hz_p128": dict(checks=dict(gravity_velocity=False, survived=True))})
    metrics = dict(upright_support=True, body_p95_deg=.5, speed_mae_m_s=.001, yaw_mae_rad_s=.001)
    standing = dict(name="stand", trials=[dict(survived=True, metrics=metrics)])
    reports = [dict(physics_hz=hz, position_iterations=p, checkpoint_sha256="abc",
                    survivors=27, total_trials=27, standing=copy.deepcopy(standing),
                    all_tracking_passed=False)
               for hz, p in ((200, 64), (200, 128), (400, 128), (800, 128))]
    regression = dict(completed=True, checkpoint_unchanged=True, reports=reports)
    return qualification, strict, regression


class SolverGuardedResumeTests(unittest.TestCase):
    def test_documented_fp32_residual_does_not_mask_other_checks(self):
        self.assertIsNone(resume.qualified(*fixtures(), "abc"))

    def test_unexplained_numerical_failure_refuses_training(self):
        q, s, r = fixtures()
        s["per_case"]["true_static_800hz_p128"]["checks"]["hinge_positions"] = False
        self.assertEqual(resume.qualified(q, s, r, "abc"), "unexplained_strict_numerical_failure")

    def test_gravity_model_mismatch_refuses_training(self):
        q, s, r = fixtures()
        q["gravity_accumulation"]["probe"]["fp32_model_max_difference_m_s"] = .001
        self.assertEqual(resume.qualified(q, s, r, "abc"), "gravity_accumulation_diagnosis_not_verified")

    def test_missing_solver_case_refuses_training(self):
        q, s, r = fixtures()
        r["reports"].pop()
        self.assertEqual(resume.qualified(q, s, r, "abc"), "frozen_policy_regression_cases_missing")

    def test_modified_checkpoint_refuses_training(self):
        q, s, r = fixtures()
        r["reports"][0]["checkpoint_sha256"] = "changed"
        self.assertEqual(resume.qualified(q, s, r, "abc"), "frozen_policy_checkpoint_mismatch")

    def test_standing_drift_refuses_training(self):
        q, s, r = fixtures()
        r["reports"][0]["standing"]["trials"][0]["metrics"]["speed_mae_m_s"] = .15
        self.assertEqual(resume.qualified(q, s, r, "abc"), "frozen_policy_support_regression")

    def test_incomplete_regression_does_not_resume(self):
        q, s, r = fixtures()
        r["completed"] = False
        self.assertEqual(resume.qualified(q, s, r, "abc"), "frozen_policy_regression_incomplete")

    def test_motion_accuracy_is_not_misrepresented_as_support(self):
        # An iteration-800 policy has not practiced motion. It must still pass
        # support checks but is not required to pass final tracking yet.
        q, s, r = fixtures()
        self.assertFalse(r["reports"][0]["all_tracking_passed"])
        self.assertIsNone(resume.qualified(q, s, r, "abc"))

    def test_admission_preserves_both_controller_clocks(self):
        q, s, r = fixtures()
        q["candidate"]["control_hz"] = 400
        self.assertEqual(resume.qualified(q, s, r, "abc"), "fixed_controller_qualification_not_passed")

    def test_launch_preserves_optimizer_and_all_action_heads(self):
        command = resume.training_command(Path("/tmp/source.pt"), 1000, "test", {
            "commands.fixed_standing_groups": True,
            "commands.standing_env_fraction": .25,
            "robot.spawn.articulation_props.solver_position_iteration_count": 128,
        })
        self.assertEqual(command[command.index("--target_iteration") + 1], "1001")
        self.assertEqual(command[command.index("--num_envs") + 1], "8192")
        self.assertEqual(command[command.index("--require_leg_control_mode") + 1], "explicit_vmc")
        for forbidden in ("--wheel_head_only_finetune", "--reset_optimizer", "--reset_exploration_std"):
            self.assertNotIn(forbidden, command)
        self.assertIn("env.commands.fixed_standing_groups=true", command)
        self.assertIn("env.commands.standing_env_fraction=0.25", command)


if __name__ == "__main__":
    unittest.main()
