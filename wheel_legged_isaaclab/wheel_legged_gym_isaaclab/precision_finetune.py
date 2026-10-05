"""Training-only wheel-head precision mode; inference architecture is unchanged."""
import torch


def enable_wheel_head_finetuning(policy, optimizer):
    """Freeze shared actor features and leg outputs; train wheel rows/std and critic.

    Masked rows also need zero Adam moments to avoid momentum drift with a
    zero gradient. All trained wheel moments and critic state are preserved.
    """
    if not isinstance(policy.actor, torch.nn.Sequential):
        raise ValueError('Expected feed-forward actor')
    head = policy.actor[-1]
    if not isinstance(head, torch.nn.Linear) or head.out_features != 6:
        raise ValueError('Expected six-output actor head')
    noise = getattr(policy, 'std', None)
    if noise is None:
        noise = getattr(policy, 'log_std', None)
    if not isinstance(noise, torch.nn.Parameter) or noise.shape != (6,):
        raise ValueError('Expected six exploration parameters')
    if any(group.get('weight_decay', 0.) != 0. for group in optimizer.param_groups):
        raise ValueError('Masked-row precision mode requires zero weight decay')
    for param in policy.actor.parameters():
        param.requires_grad_(False)
        param.grad = None
    handles = []
    for param in (head.weight, head.bias, noise):
        param.requires_grad_(True)
        mask = torch.zeros_like(param)
        mask[[2, 5]] = 1.
        handles.append(param.register_hook(lambda grad, mask=mask: grad * mask))
        moments = optimizer.state.get(param, {})
        for key in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
            if key in moments:
                if moments[key].shape != param.shape:
                    raise ValueError('Optimizer moment shape mismatch')
                moments[key][[0, 1, 3, 4]] = 0.
    return handles
