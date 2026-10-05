import unittest
from types import SimpleNamespace
import torch
from wheel_legged_gym_isaaclab.exploration import set_wheel_exploration_std

class TestWheelExploration(unittest.TestCase):
    def test_leg_distribution_and_parameter_identity_are_preserved(self):
        value=torch.nn.Parameter(torch.tensor([.01,.02,.03,.04,.05,.06]))
        policy=SimpleNamespace(std=value)
        set_wheel_exploration_std(policy,.08)
        self.assertIs(policy.std,value)
        torch.testing.assert_close(value.detach()[[0,1,3,4]],torch.tensor([.01,.02,.04,.05]))
        torch.testing.assert_close(value.detach()[[2,5]],torch.tensor([.08,.08]))
        value.square().sum().backward()
        self.assertIsNotNone(value.grad)

    def test_log_distribution_uses_same_physical_wheel_std(self):
        value=torch.nn.Parameter(torch.tensor([.01,.02,.03,.04,.05,.06]).log())
        set_wheel_exploration_std(SimpleNamespace(log_std=value),.08)
        torch.testing.assert_close(value.detach().exp(),torch.tensor([.01,.02,.08,.04,.05,.08]))

    def test_invalid_request_cannot_change_any_action(self):
        for bad in (0.,-1.,float('nan'),float('inf')):
            value=torch.nn.Parameter(torch.ones(6))
            with self.assertRaises(ValueError):set_wheel_exploration_std(SimpleNamespace(std=value),bad)
            torch.testing.assert_close(value.detach(),torch.ones(6))
