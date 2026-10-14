"""Selective exploration for the six-action virtual-leg/wheel policy."""
import math
import torch


def enforce_positive_exploration(policy, optimizer, minimum_std=1.e-4, *, minimum_wheel_std=None):
    """Project learned noise onto its valid domain after every optimizer step.

    RSL-RL's scalar std is an unconstrained parameter and can cross zero when
    previously frozen outputs are reopened. Preserve parameter identity, the
    actor/critic weights and optimizer state; do not freeze any noise entry.
    Non-finite values are errors rather than silently repaired checkpoints.
    """
    if not math.isfinite(minimum_std) or minimum_std <= 0:
        raise ValueError('Exploration std floor must be positive and finite.')
    if minimum_wheel_std is not None and (
            not math.isfinite(minimum_wheel_std) or minimum_wheel_std < minimum_std):
        raise ValueError('Wheel exploration floor must be finite and at least the global floor.')
    parameter = getattr(policy, 'std', None)
    use_log = False
    if not isinstance(parameter, torch.nn.Parameter):
        parameter = getattr(policy, 'log_std', None)
        use_log = True
    if not isinstance(parameter, torch.nn.Parameter) or parameter.shape != (6,):
        raise ValueError('Unsupported six-action policy exploration parameter.')
    floor = math.log(minimum_std) if use_log else minimum_std
    if minimum_wheel_std is not None:
        floor = parameter.new_full((6,), floor)
        floor[[2, 5]] = math.log(minimum_wheel_std) if use_log else minimum_wheel_std

    def project_noise(*_args, **_kwargs):
        with torch.no_grad():
            if not torch.isfinite(parameter).all():
                raise ValueError('Policy exploration contains non-finite values.')
            parameter.clamp_(min=floor)

    project_noise()
    return optimizer.register_step_post_hook(project_noise)


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
