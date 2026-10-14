# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Optional PPO mean-bound loss for the four directly commanded leg joints.

The action distribution, log probabilities, entropy and KL stay unchanged.
This loss penalizes only raw means outside the environment's executed [-1, 1]
interval; feasible means and both wheel outputs receive zero direct gradient.
Weights, Adam state and the inference graph retain their existing schema.

The positive-coefficient update is RSL-RL's BSD-3 update with one extra loss.
Its upstream source is pinned so a library upgrade cannot silently diverge.
A zero coefficient delegates to upstream PPO.update without any extra work.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import math
from pathlib import Path

import torch
from torch import nn
import yaml
from rsl_rl.algorithms import PPO

JOINT_ACTION_INDICES = (0, 1, 3, 4)
UPSTREAM_UPDATE_SHA256 = '9303ef999b55523476bcdf9efdb0d7888c96275afd8869906568597516c17ec0'


def validate_coefficient(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0. <= value <= 10.:
        raise ValueError('Joint mean guard coefficient must be finite and within [0, 10].')
    return float(value)


def joint_mean_overshoot_loss(means):
    if means.ndim != 2 or means.shape[-1] != 6:
        raise ValueError('Joint mean guard requires a batch of six-action means.')
    return (means[:, JOINT_ACTION_INDICES].abs() - 1.).clamp_min(0.).square().mean()


def resolve_coefficient(checkpoint=None, requested=None):
    if requested is not None:
        return validate_coefficient(requested)
    if checkpoint is None:
        return 0.
    path = Path(checkpoint).parent / 'params/agent.yaml'
    if not path.is_file():
        return 0.
    algorithm = yaml.safe_load(path.read_text())['algorithm']
    name = algorithm.get('class_name', 'PPO')
    value = validate_coefficient(algorithm.get('joint_mean_guard_coef', 0.))
    if name not in ('PPO', 'JointMeanGuardPPO') or (name == 'PPO' and value != 0.):
        raise ValueError('Saved algorithm has an incompatible joint mean guard configuration.')
    return value


def configured_runner_dict(config, coefficient, height_coefficient=0., height_model=None):
    result = copy.deepcopy(config)
    coefficient = validate_coefficient(coefficient)
    from wheel_legged_gym_isaaclab.joint_height_feedback import validate_coefficient as validate_height
    height_coefficient=validate_height(height_coefficient)
    algorithm = result['algorithm']
    if algorithm.get('class_name', 'PPO') not in ('PPO', 'JointMeanGuardPPO'):
        raise ValueError('Joint mean guard supports the original PPO runner only.')
    if coefficient == 0. and height_coefficient == 0.:
        algorithm['class_name'] = 'PPO'
        algorithm.pop('joint_mean_guard_coef', None)
    else:
        algorithm['class_name'] = 'JointMeanGuardPPO'
        algorithm['joint_mean_guard_coef'] = coefficient
    if height_coefficient:
        if height_model is None:
            raise ValueError('Positive height feedback needs the current forward geometry model')
        algorithm['joint_height_feedback_coef']=height_coefficient
        algorithm['joint_height_feedback_model']=copy.deepcopy(height_model)
    else:
        algorithm.pop('joint_height_feedback_coef',None)
        algorithm.pop('joint_height_feedback_model',None)
    return result


def register_runner_algorithm():
    # OnPolicyRunner resolves the class in this module; register an additional
    # name without replacing PPO or editing the installed RSL-RL package.
    import rsl_rl.runners.on_policy_runner as runner_module
    runner_module.JointMeanGuardPPO = JointMeanGuardPPO


class JointMeanGuardPPO(PPO):
    def __init__(self, policy, *, joint_mean_guard_coef=0., joint_height_feedback_coef=0., joint_height_feedback_model=None, **kwargs):
        self.joint_mean_guard_coef = validate_coefficient(joint_mean_guard_coef)
        from wheel_legged_gym_isaaclab.joint_height_feedback import validate_coefficient as validate_height
        self.joint_height_feedback_coef=validate_height(joint_height_feedback_coef)
        self.joint_height_feedback_model=copy.deepcopy(joint_height_feedback_model)
        if self.joint_height_feedback_coef and self.joint_height_feedback_model is None:
            raise ValueError('Height feedback requires an explicit forward geometry model')
        if self.joint_mean_guard_coef or self.joint_height_feedback_coef:
            if hashlib.sha256(inspect.getsource(PPO.update).encode()).hexdigest() != UPSTREAM_UPDATE_SHA256:
                raise RuntimeError('Installed PPO update changed; requalify the joint mean guard before training.')
            if policy.actor[-1].out_features != 6:
                raise ValueError('Joint mean guard requires the six-action joint-PD policy.')
        super().__init__(policy, **kwargs)

    def update(self):  # noqa: C901
        if self.joint_mean_guard_coef == 0. and self.joint_height_feedback_coef == 0.:
            return super().update()
        mean_joint_height_feedback_loss = 0.
        mean_joint_mean_guard_loss = 0.
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # iterate over batches
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            # we assume policy group is always there and needs augmentation
            original_batch_size = obs_batch.batch_size[0]

            # check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch,
                    actions=actions_batch,
                    env=self.symmetry["_env"],
                )
                # compute number of augmentations per sample
                # we assume policy group is always there and needs augmentation
                num_aug = int(obs_batch.batch_size[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the policy with the new parameters
            # -- actor
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate
                    # Perform this adaptation only on the main process
                    # TODO: Is this needed? If KL-divergence is the "same" across all GPUs,
                    #       then the learning rate should be the same across all GPUs.
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            joint_mean_guard_loss = joint_mean_overshoot_loss(mu_batch)
            loss = loss + self.joint_mean_guard_coef * joint_mean_guard_loss
            if self.joint_height_feedback_coef:
                from wheel_legged_gym_isaaclab.joint_height_feedback import height_proxy_loss
                joint_height_feedback_loss=height_proxy_loss(mu_batch,obs_batch['policy'][:original_batch_size],
                                                            self.joint_height_feedback_model)
                loss=loss+self.joint_height_feedback_coef*joint_height_feedback_loss
                mean_joint_height_feedback_loss+=joint_height_feedback_loss.item()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=self.symmetry["_env"])
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.policy.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # Random Network Distillation loss
            # TODO: Move this processing to inside RND module.
            if self.rnd:
                # extract the rnd_state
                # TODO: Check if we still need torch no grad. It is just an affine transformation.
                with torch.no_grad():
                    rnd_state_batch = self.rnd.get_rnd_state(obs_batch[:original_batch_size])
                    rnd_state_batch = self.rnd.state_normalizer(rnd_state_batch)
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch).detach()
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # Compute the gradients
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            # -- For RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()  # type: ignore
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients
            # -- For PPO
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_joint_mean_guard_loss += joint_mean_guard_loss.item()
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage
        self.storage.clear()

        # construct the loss dictionary
        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        loss_dict["joint_mean_guard"] = mean_joint_mean_guard_loss / num_updates
        if self.joint_height_feedback_coef:
            loss_dict['joint_height_feedback']=mean_joint_height_feedback_loss/num_updates
        return loss_dict
