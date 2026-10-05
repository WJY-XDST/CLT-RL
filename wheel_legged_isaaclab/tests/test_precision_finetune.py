import unittest
import torch
from wheel_legged_gym_isaaclab.precision_finetune import enable_wheel_head_finetuning


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor=torch.nn.Sequential(torch.nn.Linear(27,8),torch.nn.ELU(),torch.nn.Linear(8,6))
        self.critic=torch.nn.Linear(27,1)
        self.std=torch.nn.Parameter(torch.full((6,),.1))


class TestPrecisionFinetune(unittest.TestCase):
    def test_adam_momentum_cannot_move_frozen_features_leg_rows_or_exploration(self):
        torch.manual_seed(43)
        p=Policy();optimizer=torch.optim.Adam(p.parameters(),lr=.001)
        obs=torch.randn(16,27)
        def update():
            optimizer.zero_grad()
            loss=p.actor(obs).square().mean()+p.critic(obs).square().mean()+p.std.square().mean()
            loss.backward();optimizer.step()
        for _ in range(3):update()
        before={k:v.detach().clone() for k,v in p.named_parameters()}
        wheel_moment=optimizer.state[p.actor[-1].weight]['exp_avg'][[2,5]].clone()
        mean=p.actor(obs).detach().clone()
        handles=enable_wheel_head_finetuning(p,optimizer)
        torch.testing.assert_close(p.actor(obs),mean,rtol=0,atol=0)
        torch.testing.assert_close(optimizer.state[p.actor[-1].weight]['exp_avg'][[2,5]],wheel_moment)
        for _ in range(5):update()
        for key in ('actor.0.weight','actor.0.bias'):
            torch.testing.assert_close(dict(p.named_parameters())[key],before[key],rtol=0,atol=0)
        for key in ('actor.2.weight','actor.2.bias','std'):
            torch.testing.assert_close(dict(p.named_parameters())[key][[0,1,3,4]],before[key][[0,1,3,4]],rtol=0,atol=0)
            self.assertFalse(torch.equal(dict(p.named_parameters())[key][[2,5]],before[key][[2,5]]))
        self.assertFalse(torch.equal(p.critic.weight,before['critic.weight']))
        for h in handles:h.remove()

    def test_weight_decay_cannot_silently_move_masked_rows(self):
        p=Policy();opt=torch.optim.Adam(p.parameters(),weight_decay=.01)
        with self.assertRaisesRegex(ValueError,'weight decay'):enable_wheel_head_finetuning(p,opt)


if __name__=='__main__':unittest.main()
