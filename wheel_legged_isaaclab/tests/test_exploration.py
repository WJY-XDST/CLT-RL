import unittest
from types import SimpleNamespace
import torch
from wheel_legged_gym_isaaclab.exploration import enforce_positive_exploration, set_wheel_exploration_std

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

    def test_optimizer_cannot_make_scalar_noise_negative(self):
        value=torch.nn.Parameter(torch.full((6,),.0007))
        optimizer=torch.optim.SGD([value],lr=.01)
        handle=enforce_positive_exploration(SimpleNamespace(std=value),optimizer)
        before=value.detach().clone()
        torch.testing.assert_close(value.detach(),before,rtol=0,atol=0)
        value.sum().backward();optimizer.step()
        torch.testing.assert_close(value.detach(),torch.full((6,),1.e-4))
        torch.distributions.Normal(torch.zeros(6),value).sample()
        self.assertTrue(value.requires_grad)
        # A subsequent gradient can increase every entry away from the floor.
        optimizer.zero_grad();(-value.sum()).backward();optimizer.step()
        self.assertTrue((value>1.e-4).all())
        handle.remove()

    def test_log_noise_has_the_same_physical_floor(self):
        value=torch.nn.Parameter(torch.full((6,),.01).log())
        optimizer=torch.optim.SGD([value],lr=100.)
        handle=enforce_positive_exploration(SimpleNamespace(log_std=value),optimizer)
        value.sum().backward();optimizer.step()
        torch.testing.assert_close(value.detach().exp(),torch.full((6,),1.e-4))
        handle.remove()

    def test_invalid_floor_leaves_parameter_unchanged(self):
        for floor in (0.,-1.,float('nan'),float('inf')):
            value=torch.nn.Parameter(torch.ones(6))
            optimizer=torch.optim.SGD([value],lr=.01)
            with self.assertRaises(ValueError):
                enforce_positive_exploration(SimpleNamespace(std=value),optimizer,floor)
            torch.testing.assert_close(value.detach(),torch.ones(6))

    def test_nonfinite_noise_is_rejected_not_repaired(self):
        for bad in (float('nan'),float('inf'),float('-inf')):
            value=torch.nn.Parameter(torch.full((6,),bad))
            optimizer=torch.optim.SGD([value],lr=.01)
            with self.assertRaisesRegex(ValueError,'non-finite'):
                enforce_positive_exploration(SimpleNamespace(std=value),optimizer)

    def test_noise_projection_does_not_touch_actor_or_optimizer_history(self):
        policy=torch.nn.Module()
        policy.actor=torch.nn.Linear(27,6)
        policy.std=torch.nn.Parameter(torch.full((6,),.0007))
        optimizer=torch.optim.Adam(policy.parameters(),lr=.00002)
        policy.std.square().sum().backward();optimizer.step()
        actor_before={k:v.detach().clone() for k,v in policy.actor.state_dict().items()}
        moments={k:v.clone() for k,v in optimizer.state[policy.std].items()}
        handle=enforce_positive_exploration(policy,optimizer)
        for k,v in policy.actor.state_dict().items():
            torch.testing.assert_close(v,actor_before[k],rtol=0,atol=0)
        for k,v in optimizer.state[policy.std].items():
            torch.testing.assert_close(v,moments[k],rtol=0,atol=0)
        handle.remove()

    def test_wheel_floor_preserves_leg_noise_and_all_adam_moments(self):
        for use_log in (False, True):
            with self.subTest(log_std=use_log):
                policy=torch.nn.Module()
                policy.actor=torch.nn.Linear(27,6)
                policy.critic=torch.nn.Linear(27,1)
                sigma=torch.tensor([.015,.012,.0104,.0113,.01,.013])
                value=torch.nn.Parameter(sigma.log() if use_log else sigma)
                policy.register_parameter('log_std' if use_log else 'std',value)
                optimizer=torch.optim.Adam(policy.parameters(),lr=.00002)
                inputs=torch.randn(3,27)
                loss=(policy.actor(inputs).square().mean()+policy.critic(inputs).square().mean()+value.square().sum())
                loss.backward();optimizer.step()
                original_guard=enforce_positive_exploration(policy,optimizer,.01)
                original_guard.remove()
                before=value.detach().clone()
                weights={k:v.detach().clone() for k,v in policy.state_dict().items() if k not in ('std','log_std')}
                moments={p:{k:v.clone() for k,v in state.items()} for p,state in optimizer.state.items()}
                handle=enforce_positive_exploration(policy,optimizer,.01,minimum_wheel_std=.03)
                self.assertIs(getattr(policy,'log_std' if use_log else 'std'),value)
                torch.testing.assert_close(value.detach()[[0,1,3,4]],before[[0,1,3,4]],rtol=0,atol=0)
                physical=value.detach().exp() if use_log else value.detach()
                torch.testing.assert_close(physical[[2,5]],torch.full((2,),.03))
                for k,v in weights.items():torch.testing.assert_close(policy.state_dict()[k],v,rtol=0,atol=0)
                for p,state in moments.items():
                    for k,v in state.items():torch.testing.assert_close(optimizer.state[p][k],v,rtol=0,atol=0)
                # All six entries can still learn above their individual floors.
                initial=value.detach().clone();optimizer.zero_grad();(-value.sum()).backward();optimizer.step()
                self.assertTrue((value>initial).all())
                self.assertTrue(all(p.requires_grad for p in policy.parameters()))
                handle.remove()

    def test_wheel_floor_rejects_invalid_values_before_mutating_noise(self):
        for bad in (0.,-1.,float('nan'),float('inf'),.005):
            value=torch.nn.Parameter(torch.full((6,),.02))
            optimizer=torch.optim.Adam([value],lr=.00002)
            before=value.detach().clone()
            with self.assertRaises(ValueError):
                enforce_positive_exploration(SimpleNamespace(std=value),optimizer,.01,minimum_wheel_std=bad)
            torch.testing.assert_close(value.detach(),before,rtol=0,atol=0)
