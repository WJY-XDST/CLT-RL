"""Virtual-leg coordinates and force mapping, independent of the simulator."""

import torch


def leg_coordinates(theta1, theta2, *, l1, l2, offset):
    x = offset + l1 * torch.cos(theta1) + l2 * torch.cos(theta1 + theta2)
    y = l1 * torch.sin(theta1) + l2 * torch.sin(theta1 + theta2)
    length = torch.sqrt(x.square() + y.square())
    angle = torch.atan2(y, x) - torch.pi / 2.0
    return length, angle


def virtual_leg_torques(
    theta1, theta2, length, angle, force, torque, *, l1, l2, legacy_angular_mapping=False
):
    """Apply J.T @ [force, torque], with J = d[length, angle]/d[theta1, theta2].

    Angles use the canonical left-leg coordinates. The environment converts
    the mirrored right-leg torques back to its raw joint coordinates.
    """
    radial_angle = angle + torch.pi / 2.0
    delta1 = radial_angle - theta1
    delta2 = radial_angle - theta1 - theta2
    dlength_dtheta1 = l1 * torch.sin(delta1) + l2 * torch.sin(delta2)
    dlength_dtheta2 = l2 * torch.sin(delta2)
    dangle_dtheta1 = (l1 * torch.cos(delta1) + l2 * torch.cos(delta2)) / length
    dangle_dtheta2 = l2 * torch.cos(delta2) / length
    if legacy_angular_mapping:
        # Compatibility only for archived policies trained with the old sign error.
        dangle_dtheta1 = (-l1 * torch.cos(delta1) + l2 * torch.cos(delta2)) / length
    hip_torque = dlength_dtheta1 * force + dangle_dtheta1 * torque
    knee_torque = dlength_dtheta2 * force + dangle_dtheta2 * torque
    return hip_torque, knee_torque
