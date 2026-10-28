"""Explicit once-only Gaussian-std adjustment after a complete source restore.

This helper is opt-in. It does not load a policy into an environment, sample
an action, call an observation function, step an optimizer, install hooks or
change a source checkpoint. Callers must invoke it after the existing resume
and future-floor checks and before the first stochastic rollout. A separate
receipt must describe this intentional std change; an unchanged-std receipt
from the earlier restore phase must never stand in for that receipt.
"""
from pathlib import Path
import copy
import hashlib
import math
import weakref

import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def equal(left, right, label):
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype or not torch.equal(
                left.detach().cpu(), right.detach().cpu()):
            raise ValueError('Changed tensor: ' + label)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise ValueError('Changed mapping: ' + label)
        for key in left:
            equal(left[key], right[key], label + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise ValueError('Changed sequence: ' + label)
        for index, (a, b) in enumerate(zip(left, right)):
            equal(a, b, label + '/' + str(index))
    elif left != right:
        raise ValueError('Changed scalar: ' + label)


class RestoredStdPulse:
    def __init__(self, *, checkpoint, checkpoint_sha256, actual_updates, minimum_std):
        self.checkpoint = Path(checkpoint).resolve()
        if (type(minimum_std) not in (int, float) or not math.isfinite(minimum_std)
                or minimum_std <= 0 or type(actual_updates) is not int or actual_updates <= 0):
            raise ValueError('Require a positive finite std and real positive source update count')
        if digest(self.checkpoint) != checkpoint_sha256:
            raise ValueError('The pulse source must be the exact preserved checkpoint')
        self.sha, self.updates, self.minimum = checkpoint_sha256, actual_updates, float(minimum_std)
        self.saved = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        std = self.saved['model_state_dict'].get('std')
        states = self.saved['optimizer_state_dict']['state']
        if (not isinstance(std, torch.Tensor) or std.shape != (6,) or not torch.isfinite(std).all()
                or (std <= 0).any() or len(states) != 17
                or {float(value['step']) for value in states.values()} != {20. * actual_updates}):
            raise ValueError('Require this task\'s six learned std and full17 native Adam states')
        self._applied = weakref.WeakSet()

    def apply_once(self, policy, optimizer, *, loaded_checkpoint_sha256, actual_learning_rate):
        if loaded_checkpoint_sha256 != self.sha:
            return dict(applied=False, reason='later_or_different_checkpoint_not_reseeded',
                        requested_source_sha256=self.sha, loaded_checkpoint_sha256=loaded_checkpoint_sha256)
        if policy in self._applied:
            return dict(applied=False, reason='already_applied_to_this_policy_instance',
                        loaded_checkpoint_sha256=self.sha)
        parameter = getattr(policy, 'std', None)
        if (not isinstance(parameter, torch.nn.Parameter) or parameter.shape != (6,)
                or not parameter.requires_grad):
            raise ValueError('The original six learned std parameters must remain trainable')
        if sum(item is parameter for group in optimizer.param_groups for item in group['params']) != 1:
            raise ValueError('The same std parameter must occur once in the existing optimizer')
        if getattr(policy, 'distribution', None) is not None:
            raise ValueError('Apply the declared pulse before the first stochastic rollout')
        equal(self.saved['model_state_dict'], policy.state_dict(), 'restored_policy_before_pulse')
        equal(self.saved['optimizer_state_dict'], optimizer.state_dict(), 'restored_all17_Adam_before_pulse')
        expected_rate = self.saved['optimizer_state_dict']['param_groups'][0]['lr']
        if actual_learning_rate != expected_rate or any(group['lr'] != expected_rate for group in optimizer.param_groups):
            raise ValueError('Preserve the actual saved adaptive learning rate')
        before_policy, before_optimizer = copy.deepcopy(policy.state_dict()), copy.deepcopy(optimizer.state_dict())
        before_rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(parameter.device).clone() if parameter.is_cuda else None
        identity = id(parameter)
        before_std = parameter.detach().clone()
        with torch.no_grad():
            parameter.clamp_(min=self.minimum)
        expected_std = before_std.clamp_min(self.minimum)
        equal(expected_std, policy.std, 'only_declared_std_minimum')
        for name, value in before_policy.items():
            if name != 'std':
                equal(value, policy.state_dict()[name], 'mean_value_and_buffers/' + name)
        equal(before_optimizer, optimizer.state_dict(), 'all17_Adam_after_pulse')
        if id(policy.std) != identity or not torch.equal(before_rng, torch.random.get_rng_state()):
            raise RuntimeError('The pulse replaced a parameter or advanced the CPU RNG')
        if cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(parameter.device)):
            raise RuntimeError('The pulse advanced the CUDA RNG')
        if digest(self.checkpoint) != self.sha:
            raise RuntimeError('The preserved source checkpoint changed')
        self._applied.add(policy)
        return dict(applied=True,
            phase='after_complete_restore_and_future_floor_before_first_stochastic_rollout',
            source_checkpoint=str(self.checkpoint), loaded_checkpoint_sha256=self.sha,
            source_actual_updates=self.updates, all17_Adam_step=self.updates * 20,
            declared_common_minimum_std=self.minimum,
            before_std=before_std.detach().cpu().tolist(), after_std=policy.std.detach().cpu().tolist(),
            only_std_tensor_intentionally_changed=True,
            all_other_policy_state_tensors_bitwise_preserved=True,
            all17_Adam_state_and_parameter_groups_bitwise_preserved=True,
            saved_actual_learning_rate=actual_learning_rate,
            parameter_identity_preserved=True, all_six_std_entries_trainable=True,
            RNG_unchanged=True, additional_action_samples=0, additional_observation_queries=0,
            optimizer_steps_executed=0, added_loss_or_backward=False,
            source_file_unchanged=True, policy_accepted=False)
