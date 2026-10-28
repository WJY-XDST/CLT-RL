"""Explicit Gaussian std minima after a verified full-state resume.

The vector is fixed by actuator role, never by commands or evaluation cases.
The same trainable Parameter and the complete Adam state are retained. The
initial adjustment is separate from full restore and runs before sampling;
the future projection runs only after an ordinary optimizer step.
"""
import copy
import math
from pathlib import Path
import weakref

import torch

from resume_std_pulse import digest, equal


def minima_tensor(values, parameter):
    if (not isinstance(values, (list, tuple)) or len(values) != 6
            or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                   for v in values)):
        raise ValueError('Require six positive finite actuator std minima')
    if not isinstance(parameter, torch.nn.Parameter) or parameter.shape != (6,) or not parameter.requires_grad:
        raise ValueError('Retain the original six trainable std parameters')
    return torch.as_tensor(values, dtype=parameter.dtype, device=parameter.device)


class RestoredVectorStdPulse:
    def __init__(self, *, checkpoint, checkpoint_sha256, actual_updates, minimum_std_by_head):
        self.checkpoint = Path(checkpoint).resolve()
        if type(actual_updates) is not int or actual_updates <= 0 or digest(self.checkpoint) != checkpoint_sha256:
            raise ValueError('Require an exact preserved source and its real update count')
        self.sha, self.updates = checkpoint_sha256, actual_updates
        self.saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        std = self.saved['model_state_dict'].get('std')
        state = self.saved['optimizer_state_dict']['state']
        if (not isinstance(std, torch.Tensor) or std.shape != (6,) or not torch.isfinite(std).all()
                or (std <= 0).any() or len(state) != 17
                or {float(v['step']) for v in state.values()} != {20. * actual_updates}):
            raise ValueError('Require the real six learned std and all17 Adam states')
        minima_tensor(minimum_std_by_head, torch.nn.Parameter(std.clone()))
        self.minima = list(minimum_std_by_head)
        self._applied = weakref.WeakSet()

    def apply_once(self, policy, optimizer, *, loaded_checkpoint_sha256, actual_learning_rate):
        if loaded_checkpoint_sha256 != self.sha:
            return dict(applied=False, reason='later_or_different_checkpoint_not_reseeded',
                        requested_source_sha256=self.sha, loaded_checkpoint_sha256=loaded_checkpoint_sha256)
        if policy in self._applied:
            return dict(applied=False, reason='already_applied_to_this_policy_instance',
                        loaded_checkpoint_sha256=self.sha)
        parameter = policy.std
        minima = minima_tensor(self.minima, parameter)
        if sum(v is parameter for g in optimizer.param_groups for v in g['params']) != 1:
            raise ValueError('The same std Parameter must occur once in the existing optimizer')
        if getattr(policy, 'distribution', None) is not None:
            raise ValueError('The explicit adjustment must precede the first stochastic rollout')
        equal(self.saved['model_state_dict'], policy.state_dict(), 'full_source_restore')
        equal(self.saved['optimizer_state_dict'], optimizer.state_dict(), 'full17_Adam_restore')
        rate = self.saved['optimizer_state_dict']['param_groups'][0]['lr']
        if actual_learning_rate != rate or any(g['lr'] != rate for g in optimizer.param_groups):
            raise ValueError('Retain the actual saved adaptive learning rate')
        old_policy, old_optimizer = copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict())
        rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(parameter.device).clone() if parameter.is_cuda else None
        identity, before = id(parameter), parameter.detach().clone()
        with torch.no_grad():
            parameter.copy_(torch.maximum(parameter, minima))
        equal(torch.maximum(before, minima), parameter, 'declared_vector_std_minimum_only')
        for name, value in old_policy.items():
            if name != 'std':
                equal(value, policy.state_dict()[name], 'unchanged_mean_value/' + name)
        equal(old_optimizer, optimizer.state_dict(), 'unchanged_all17_Adam')
        if id(policy.std) != identity or not torch.equal(rng, torch.random.get_rng_state()):
            raise RuntimeError('Adjustment changed Parameter identity or CPU RNG')
        if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(parameter.device)):
            raise RuntimeError('Adjustment advanced CUDA RNG')
        if digest(self.checkpoint) != self.sha:
            raise RuntimeError('The preserved source file changed')
        self._applied.add(policy)
        return dict(applied=True,
            phase='after_complete_restore_and_base_future_floor_before_first_stochastic_rollout',
            source_checkpoint=str(self.checkpoint), loaded_checkpoint_sha256=self.sha,
            source_actual_updates=self.updates, all17_Adam_step=self.updates * 20,
            declared_minimum_std_by_head=self.minima,
            before_std=before.cpu().tolist(), after_std=parameter.detach().cpu().tolist(),
            actually_changed_heads=torch.nonzero(parameter.detach() != before).flatten().cpu().tolist(),
            only_std_tensor_intentionally_changed=True,
            all_other_policy_state_tensors_bitwise_preserved=True,
            all17_Adam_state_and_parameter_groups_bitwise_preserved=True,
            saved_actual_learning_rate=actual_learning_rate,
            parameter_identity_preserved=True, all_six_std_entries_trainable=True,
            RNG_unchanged=True, additional_action_samples=0, additional_observation_queries=0,
            optimizer_steps_executed=0, added_loss_or_backward=False,
            source_file_unchanged=True, policy_accepted=False)


def install_future_vector_floor(policy, optimizer, minimum_std_by_head):
    """Register a projection without modifying the already initialized policy."""
    parameter = policy.std
    minima = minima_tensor(minimum_std_by_head, parameter)
    if sum(v is parameter for g in optimizer.param_groups for v in g['params']) != 1:
        raise ValueError('Retain std identity in the existing optimizer')
    if not torch.isfinite(parameter).all() or (parameter.detach() < minima).any():
        raise ValueError('Declare and apply any initial std change before installing future floors')

    def project(*_args, **_kwargs):
        if policy.std is not parameter or not torch.isfinite(parameter).all():
            raise RuntimeError('Invalid or replaced std parameter')
        with torch.no_grad():
            parameter.copy_(torch.maximum(parameter, minima))

    handle = optimizer.register_step_post_hook(project)
    return handle, project
