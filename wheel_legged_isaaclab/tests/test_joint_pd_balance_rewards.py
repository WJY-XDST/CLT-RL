"""Execute production reward equations without importing the simulator SDK."""
import ast
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from env_source import environment_tree

from wheel_legged_gym_isaaclab.leg_control import DIRECT_JOINT_POSITION_MODE, LEG_ACTION_COLUMNS
from wheel_legged_gym_isaaclab.reward_signals import (
    horizontal_velocity_error_squared, action_second_difference_penalty, height_tracking_kernel, wheel_action_overshoot_penalty,
    joint_target_length_error_squared,
    guard_tracking_bonuses,
)
from wheel_legged_gym_isaaclab.yaw_training import spin_center_velocity_penalty

DIRECTORY = Path(__file__).resolve().parents[1] / 'wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat'


def current_reward_config():
    tree = ast.parse((DIRECTORY / 'wheel_legged_vmc_flat_env_cfg.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RewardsCfg')
    values = {}
    for n in cls.body:
        if isinstance(n, ast.Assign):
            values[n.targets[0].id] = ast.literal_eval(n.value)
    values.update(base_height_target=.30, leg_length_target_height_offset=-.03534645531,
                  leg_length_target_cap=.32)
    return SimpleNamespace(**values)


def reward_fixture(n=3, source_text=None):
    rewards = current_reward_config()
    data = SimpleNamespace(
        root_lin_vel_b=torch.zeros(n, 3), root_ang_vel_b=torch.zeros(n, 3),
        projected_gravity_b=torch.tensor([0., 0., -1.]).expand(n, 3).clone(),
        root_pos_w=torch.tensor([0., 0., .30]).expand(n, 3).clone(),
        joint_pos=torch.zeros(n, 6), joint_vel=torch.zeros(n, 6), joint_acc=torch.zeros(n, 6),
        applied_torque=torch.zeros(n, 6), joint_pos_limits=torch.tensor([-1., 1.]).expand(n, 6, 2),
    )
    heading = torch.zeros(n, 2)
    commands = torch.zeros(n, 4)
    commands[:, 2] = .30
    obj = SimpleNamespace(
        cfg=SimpleNamespace(rewards=rewards, leg_control_mode=DIRECT_JOINT_POSITION_MODE,
                            leg_model='mine_five_bar', action_scale_vel=15., wheel_radius=.06,
                            action_scale_l0=.09, l0_offset=.27, collision_force_threshold=1.),
        _robot=SimpleNamespace(data=data), _commands=commands, _target_lin_vel_x=torch.zeros(n),
        _contact_sensor=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(n, 3, 3))),
        _penalised_contact_ids=[0, 1, 2], _leg_joint_ids=[0, 1, 3, 4],
        _theta0=torch.zeros(n, 2), _actions=torch.zeros(n, 6), _raw_actions=torch.zeros(n, 6),
        _last_actions=torch.zeros(n, 6, 2), _update_forward_kinematics=lambda: None,
        _get_heading_frame_horizontal_velocity=lambda: heading,
        _get_wheel_vel_forward_positive=lambda: torch.zeros(n, 2),
        _tracking_diagnostics={'fixture': 1}, _tracking_goal_criteria=None, common_step_counter=1, step_dt=.02,
        reset_terminated=torch.zeros(n, dtype=torch.bool),
        _episode_sums=defaultdict(lambda: torch.zeros(n)),
    )
    tree = ast.parse(source_text) if source_text is not None else environment_tree()
    method = next(x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == '_get_rewards')
    scale_fields = next(
        x for x in tree.body if isinstance(x, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == 'REWARD_SCALE_FIELDS' for target in x.targets)
    )
    scope = dict(torch=torch, horizontal_velocity_error_squared=horizontal_velocity_error_squared,
                 spin_center_velocity_penalty=spin_center_velocity_penalty,
                 action_second_difference_penalty=action_second_difference_penalty,
                 height_tracking_kernel=height_tracking_kernel,
                 wheel_action_overshoot_penalty=wheel_action_overshoot_penalty,
                 joint_target_length_error_squared=joint_target_length_error_squared,
                 guard_tracking_bonuses=guard_tracking_bonuses,
                 DIRECT_JOINT_POSITION_MODE=DIRECT_JOINT_POSITION_MODE, LEG_ACTION_COLUMNS=LEG_ACTION_COLUMNS)
    exec(compile(ast.Module(body=[scale_fields, method], type_ignores=[]), '<production_reward>', 'exec'), scope)
    return obj, heading, lambda: scope['_get_rewards'](obj)


class TestJointPDBalanceRewards(unittest.TestCase):
    def test_perfect_tracking_is_positive_without_kernel_clipping(self):
        env, _, calculate = reward_fixture()
        torch.testing.assert_close(calculate(), torch.full((3,), .10))  # 5 reward/s * 20 ms
        for name, expected in [('track_lin_vel', .04), ('track_ang_vel', .02), ('base_height', .04)]:
            torch.testing.assert_close(env._last_reward_terms[name], torch.full((3,), expected))

    def test_height_errors_remain_distinguishable_near_target(self):
        env, _, calculate = reward_fixture()
        env._robot.data.root_pos_w[:, 2] += torch.tensor([0., .005, .010])
        calculate()
        r = env._last_reward_terms['base_height']
        self.assertGreater(r[0].item(), r[1].item())
        self.assertGreater(r[1].item(), r[2].item())

    def test_legacy_height_kernel_is_exact_and_new_band_is_inside_acceptance(self):
        target = torch.full((1001,), .30)
        height = torch.linspace(.2, .4, 1001)
        original = torch.exp(-(height-target).square()/.0004)
        torch.testing.assert_close(height_tracking_kernel(height, target, .0004), original, rtol=0, atol=0)
        value = height_tracking_kernel(torch.tensor([.30, .31, .32, .33]), torch.full((4,), .30), .0004, .05)
        self.assertEqual(value[0], value[1])
        self.assertGreater(value[1], value[2])
        self.assertGreater(value[2], value[3])

    def test_measured_height_vs_yaw_conflict_is_not_rewarded_inside_band(self):
        env, heading, calculate = reward_fixture(2)
        env.cfg.rewards.yaw_rate_error_sq = -10.
        # Values from paired frozen-policy stand/LB+0.03 diagnostic, same target.
        env._robot.data.root_pos_w[:, 2] = torch.tensor([.309139, .304606])
        env._robot.data.root_ang_vel_b[:, 2] = torch.tensor([.108544, .175464])
        heading[:, 0] = torch.tensor([.0215, .03587])
        old = calculate().clone()
        self.assertGreater(old[1], old[0])  # Height polishing rewards worse drift.
        env.cfg.rewards.base_height_relative_deadband = .05
        corrected = calculate()
        self.assertLess(corrected[1], corrected[0])
        torch.testing.assert_close(env._last_reward_terms['base_height'], torch.full((2,), .04))

    def test_yaw_error_does_not_flatten_at_old_threshold(self):
        env, _, calculate = reward_fixture()
        env._robot.data.root_ang_vel_b[:, 2] = torch.tensor([.3, .6, 1.])
        calculate()
        torch.testing.assert_close(env._last_reward_terms['yaw_rate_error_sq'],
                                   -torch.tensor([.09, .36, 1.]) * .02)

    def test_lateral_drift_cost_is_command_independent(self):
        env, heading, calculate = reward_fixture()
        env._commands[:, 0] = torch.tensor([0., .5, -.5])
        heading[:, 0] = env._commands[:, 0]
        heading[:, 1] = .2
        calculate()
        torch.testing.assert_close(env._last_reward_terms['lin_vel_error_sq'], torch.full((3,), -.0016))
        self.assertEqual(env._last_reward_terms['standing_velocity'].abs().sum().item(), 0.)

    def test_legacy_vx_only_formula_is_unchanged(self):
        v = torch.tensor([[.3, .8], [-.4, -.5]])
        c = torch.tensor([.2, -.2])
        torch.testing.assert_close(horizontal_velocity_error_squared(v, c, 0.), (c-v[:, 0]).square())

    def test_no_fixed_leg_symmetry_or_virtual_length_action_cost(self):
        env, _, calculate = reward_fixture()
        env._theta0[:, 0], env._theta0[:, 1] = .15, -.1
        env._raw_actions[:, 1], env._raw_actions[:, 4] = .5, -.5
        calculate()
        for name in ('nominal_state', 'standing_leg_angle', 'leg_length_action_saturation',
                     'leg_length_action_difference', 'leg_length_target_underreach'):
            self.assertEqual(env._last_reward_terms[name].abs().sum().item(), 0.)

    def test_nonzero_balancing_torque_is_not_a_stationary_zero_action_objective(self):
        env, _, calculate = reward_fixture()
        env._robot.data.applied_torque[:] = 5.
        env._actions[:, (2, 5)] = .1
        env._last_actions[:, :, 0] = env._actions
        calculate()
        for name in ('torques', 'standing_wheel_tracking', 'action_rate'):
            self.assertEqual(env._last_reward_terms[name].abs().sum().item(), 0.)

    def test_raw_joint_soft_saturation_has_monotonic_cost(self):
        env, _, calculate = reward_fixture()
        env._raw_actions[:, 4] = torch.tensor([.85, 1.1, 1.5])
        calculate()
        r = env._last_reward_terms['joint_position_action_saturation']
        self.assertEqual(r[0].item(), 0.)
        self.assertGreater(r[0].item(), r[1].item())
        self.assertGreater(r[1].item(), r[2].item())

    def test_terminal_cost_is_one_time_and_timeouts_are_not_unsafe(self):
        env, _, calculate = reward_fixture()
        env.reset_terminated[0] = True
        calculate()
        torch.testing.assert_close(env._last_reward_terms['termination'], torch.tensor([-20., 0., 0.]))

    def test_reward_baseline_is_defined_once_in_the_environment_configuration(self):
        original = current_reward_config()
        self.assertEqual(original.tracking_lin_vel_enhance, 0.)
        self.assertEqual(original.tracking_lin_vel, 2.)
        self.assertEqual(original.termination, -20.)
        text = (DIRECTORY / '__init__.py').read_text()
        self.assertIn('"WheelLeggedJointPD-Flat-v0"', text)
        self.assertIn('"WheelLeggedVMC-Flat-v0"', text)
        self.assertNotIn('configure_direct_joint_balance_rewards', text)

    def test_registration_does_not_mutate_or_overlay_reward_values(self):
        text = (DIRECTORY / '__init__.py').read_text()
        self.assertNotIn('__post_init__', text)
        self.assertNotIn('mine_env_cfg', text)

    def test_invalid_velocity_shapes_and_multiplier_are_refused(self):
        for weight in (-1., float('nan')):
            with self.assertRaises(ValueError):
                horizontal_velocity_error_squared(torch.zeros(2, 2), torch.zeros(2), weight)
        with self.assertRaises(ValueError):
            horizontal_velocity_error_squared(torch.zeros(2, 3), torch.zeros(2))


if __name__ == '__main__':
    unittest.main()
