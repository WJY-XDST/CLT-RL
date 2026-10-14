"""Batched closed-five-bar VMC; both motor angles are relative to base_link."""
import torch


def five_bar_state(q_b, q_l, *, upper, lower, offsets, scale, branch):
    """Return length, backward-positive leg angle, raw-motor Jacobian, validity.

    Inputs [..., 2] use leg order [left, right], radians at the exported zero.
    J[..., 0, :] = d(length)/d(q_B,q_L), J[..., 1, :] = d(angle)/d(q_B,q_L).
    Standard base axes are X forward, Y left, Z up; theta=atan2(-X,-Z).
    """
    sign = q_b.new_tensor([1., -1.])
    theta_b = offsets[0]+sign*q_b
    theta_l = offsets[1]+sign*q_l
    c = upper[0]*torch.stack((theta_b.cos(),theta_b.sin()),dim=-1)
    a = upper[1]*torch.stack((theta_l.cos(),theta_l.sin()),dim=-1)
    delta = a-c
    distance = torch.linalg.vector_norm(delta,dim=-1)
    d = distance.clamp_min(1e-7)
    along = (lower[0]**2-lower[1]**2+d.square())/(2*d)
    height2 = lower[0]**2-along.square()
    normal = torch.stack((-delta[...,1],delta[...,0]),dim=-1)/d[...,None]
    p = c+along[...,None]*delta/d[...,None]+branch*height2.clamp_min(0).sqrt()[...,None]*normal
    wheel = scale*p  # CAD (Y,Z), i.e. standard (Z,X), relative to hip
    length = torch.linalg.vector_norm(wheel,dim=-1).clamp_min(1e-6)
    angle = torch.atan2(-wheel[...,1],-wheel[...,0])
    u,v = p-c,p-a
    dc = sign[...,None]*upper[0]*torch.stack((-theta_b.sin(),theta_b.cos()),dim=-1)
    da = sign[...,None]*upper[1]*torch.stack((-theta_l.sin(),theta_l.cos()),dim=-1)
    rhs_b=(u*dc).sum(-1)
    rhs_l=(v*da).sum(-1)
    determinant=u[...,0]*v[...,1]-u[...,1]*v[...,0]
    valid=(distance>1e-5)&(height2>1e-10)&(determinant.abs()>1e-7)
    det=torch.where(determinant<0,-determinant.abs().clamp_min(1e-7),determinant.abs().clamp_min(1e-7))
    j_cart=scale*torch.stack((
        torch.stack((v[...,1]*rhs_b,-u[...,1]*rhs_l),dim=-1),
        torch.stack((-v[...,0]*rhs_b,u[...,0]*rhs_l),dim=-1)),dim=-2)/det[...,None,None]
    polar=torch.stack((wheel/length[...,None],
                       torch.stack((-wheel[...,1],wheel[...,0]),dim=-1)/length.square()[...,None]),dim=-2)
    jacobian=polar@j_cart
    return length,angle,jacobian,valid


def five_bar_torques(jacobian, force, torque):
    """Virtual work in raw motor coordinates, including each side's axis sign."""
    effort=jacobian.transpose(-1,-2)@torch.stack((force,torque),dim=-1)[...,None]
    return effort[...,0,0],effort[...,1,0]
