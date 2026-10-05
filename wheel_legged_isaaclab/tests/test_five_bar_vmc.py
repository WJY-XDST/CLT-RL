"""Closed-chain virtual-work and velocity checks in raw motor coordinates."""
import json
from pathlib import Path
import unittest
import torch
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state, five_bar_torques, five_bar_inverse

ROOT = Path(__file__).resolve().parents[1] / "assets/robots/mine"
MODEL = json.loads((ROOT / json.loads((ROOT / "manifest.json").read_text())["metadata"]).read_text())
GEOMETRY = MODEL["five_bar"]


class TestFiveBar(unittest.TestCase):
    def test_inverse_reconstructs_entire_action_reference_range(self):
        length,angle=torch.meshgrid(torch.linspace(.18,.36,41,dtype=torch.float64),torch.linspace(-.2,.2,31,dtype=torch.float64),indexing='ij')
        length=length.reshape(-1,1).expand(-1,2)
        angle=angle.reshape(-1,1).expand(-1,2)
        b,l,valid=five_bar_inverse(length,angle,**GEOMETRY)
        self.assertTrue(valid.all())
        actual_length,actual_angle,_,actual_valid=five_bar_state(b,l,**GEOMETRY)
        self.assertTrue(actual_valid.all())
        self.assertTrue((b.abs()<1.57).all() and (l.abs()<1.57).all())
        torch.testing.assert_close(length,actual_length,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(angle,actual_angle,rtol=1e-10,atol=1e-10)

    def test_inverse_rejects_unreachable_length(self):
        length=torch.tensor([[.8,.8]],dtype=torch.float64)
        _,_,valid=five_bar_inverse(length,torch.zeros_like(length),**GEOMETRY)
        self.assertFalse(valid.any())

    def test_virtual_work_and_velocity(self):
        g = torch.Generator().manual_seed(37)
        b = ((torch.rand(64, 2, generator=g, dtype=torch.float64)-.5)*.6).requires_grad_()
        l = ((torch.rand(64, 2, generator=g, dtype=torch.float64)-.5)*.6).requires_grad_()
        length, angle, jac, valid = five_bar_state(b,l,**GEOMETRY)
        self.assertTrue(valid.all())
        f = torch.randn(64,2,generator=g,dtype=torch.float64)*50
        t = torch.randn(64,2,generator=g,dtype=torch.float64)*5
        expected = torch.autograd.grad((f*length+t*angle).sum(),(b,l))
        actual = five_bar_torques(jac,f,t)
        for x,y in zip(actual,expected):
            torch.testing.assert_close(x,y,rtol=1e-10,atol=1e-10)
        dq = torch.randn(64,2,2,generator=g,dtype=torch.float64)
        eps = 1e-6
        plus = five_bar_state(b+eps*dq[...,0],l+eps*dq[...,1],**GEOMETRY)
        minus = five_bar_state(b-eps*dq[...,0],l-eps*dq[...,1],**GEOMETRY)
        fd = torch.stack([(plus[i]-minus[i])/(2*eps) for i in (0,1)],-1)
        torch.testing.assert_close((jac@dq[...,None]).squeeze(-1),fd,rtol=1e-7,atol=1e-9)

    def test_cad_zero_and_mirror(self):
        q = torch.zeros(1,2,dtype=torch.float64)
        length,angle,jac,valid = five_bar_state(q,q,**GEOMETRY)
        self.assertTrue(valid.all())
        self.assertAlmostEqual(length[0,0].item(),MODEL["nominal_leg_length"],places=10)
        self.assertAlmostEqual((-length*angle.sin())[0,0].item(),-.013528852,places=7)
        self.assertAlmostEqual((-length*angle.cos())[0,0].item(),-.258841279,places=7)
        torch.testing.assert_close(jac[:,0],-jac[:,1])

    def test_singularity_is_rejected(self):
        b = torch.zeros(1,2,dtype=torch.float64)
        l = (GEOMETRY["offsets"][0]-GEOMETRY["offsets"][1])*torch.tensor([[1.,-1.]],dtype=torch.float64)
        _,_,_,valid = five_bar_state(b,l,**GEOMETRY)
        self.assertFalse(valid.any())


if __name__ == "__main__":
    unittest.main()
