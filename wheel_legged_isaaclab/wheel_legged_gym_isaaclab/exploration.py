"""Selective exploration for the six-action virtual-leg/wheel policy."""
import math
import torch


def set_wheel_exploration_std(policy, value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Wheel exploration std must be positive and finite.')
    parameter = getattr(policy, 'std', None)
    use_log = False
    if not isinstance(parameter, torch.Tensor):
        parameter = getattr(policy, 'log_std', None)
        use_log = True
    if not isinstance(parameter, torch.Tensor) or parameter.shape != (6,):
        raise ValueError('Unsupported six-action policy exploration parameter.')
    with torch.no_grad():
        parameter[[2, 5]] = math.log(value) if use_log else value
