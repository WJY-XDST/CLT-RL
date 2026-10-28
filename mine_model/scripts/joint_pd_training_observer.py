"""Observation-only instrumentation of ordinary PPO and native transitions.

No loss, backward pass, action, reward, parameter or RNG mutation is added.
Motor-target FK probes describe network updates on fixed inputs, not actual
future body motion, a dynamics Jacobian, stability or tracking acceptance.
"""
from pathlib import Path

import numpy as np
import torch
from rsl_rl.algorithms import PPO

from auto_train_mine import write_json
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.joint_height_feedback import predict_support_heights


def maximum_error(a, b):
    return float((a-b).abs().max()) if a.numel() else 0.


def gradient_norm(parameters):
    terms = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


class TrainingObserver:
    def __init__(self, algorithm, model, *, env=None, cfg=None, output=None):
        if type(algorithm) is not PPO or algorithm.policy.is_recurrent:
            raise ValueError('Observe ordinary feedforward PPO without replacing its update')
        if not isinstance(algorithm.policy.actor_obs_normalizer, torch.nn.Identity):
            raise ValueError('Fixed-input observer requires the unchanged unnormalized actor')
        if any(isinstance(m, (torch.nn.Dropout, torch.nn.modules.batchnorm._BatchNorm))
               for m in algorithm.policy.actor.modules()):
            raise ValueError('Diagnostic forward passes must not change RNG or running state')
        self.algorithm, self.model, self.env, self.cfg = algorithm, model, env, cfg
        self.output = Path(output) if output is not None else None
        self.updating = False
        self.capture = True
        self.minibatch = None
        self.gradients = []
        self.alignment = []
        self.native_rows = []
        self.updates = []
        self.pending = None
        self.snapshot = None
        self.sample_ids = None
        self.original = {name:getattr(algorithm,name) for name in ('act','process_env_step','update')}
        self.handles = [algorithm.policy.actor.register_forward_pre_hook(self._capture_actor),
                        algorithm.optimizer.register_step_pre_hook(self._before_step),
                        algorithm.optimizer.register_step_post_hook(self._after_step)]
        algorithm.act = self._act
        algorithm.process_env_step = self._process
        algorithm.update = self._update
        if env is not None:
            self.original_reward = env._get_rewards
            env._get_rewards = self._reward

    def _ids(self, value):
        if self.sample_ids is None:
            self.sample_ids = torch.linspace(0,len(value)-1,min(32,len(value)),device=value.device).long()
        return self.sample_ids

    def _act(self, observations):
        if self.env is not None:
            self.pending = dict(observations=observations['policy'].detach().clone())
            scales = observations['policy'].new_tensor([self.cfg['obs_scales'][key]
                for key in ('lin_vel','ang_vel','height_measurements')])
            self.pending['command_before_act_error'] = maximum_error(
                observations['policy'][:,6:9],self.env._commands[:,:3]*scales)
        actions = self.original['act'](observations)
        if self.env is not None:
            self.pending.update(actions=actions.detach().clone(),
                mean=self.algorithm.transition.action_mean.detach().clone(),
                sigma=self.algorithm.transition.action_sigma.detach().clone())
        return actions

    def _reward(self):
        reward = self.original_reward()
        env = self.env
        data = env._robot.data
        self.snapshot = {key:value.detach().clone() for key,value in dict(
            reward=reward, commands=env._commands[:,:3], height=data.root_pos_w[:,2],
            gravity=data.projected_gravity_b, speed=env._get_heading_frame_horizontal_velocity()[:,0],
            yaw=data.root_ang_vel_b[:,2], raw_actions=env._raw_actions,
            leg_targets=env._joint_position_ref_applied,
            joint_positions=data.joint_pos[:,env._leg_joint_ids],
            leg_lengths=env._L0, leg_references=env._l0_ref_applied).items()}
        return reward

    def _process(self, observations, rewards, dones, extras):
        if self.env is None:
            return self.original['process_env_step'](observations,rewards,dones,extras)
        if self.pending is None or self.snapshot is None:
            raise RuntimeError('Native reward/actor sample is missing')
        delivered = rewards.detach().clone()
        timeouts = extras.get('time_outs',torch.zeros_like(dones)).detach().clone()
        step = self.algorithm.storage.step
        result = self.original['process_env_step'](observations,rewards,dones,extras)
        storage = self.algorithm.storage
        if storage.step != step+1:
            raise RuntimeError('PPO did not store exactly one transition')
        p, s = self.pending, self.snapshot
        scales = p['observations'].new_tensor([self.cfg['obs_scales'][key]
            for key in ('lin_vel','ang_vel','height_measurements')])
        expected_reward = delivered+self.algorithm.gamma*(storage.values[step,:,0]*timeouts)
        errors = dict(command_before_act_max_error=p['command_before_act_error'],
            command_at_reward_max_error=maximum_error(p['observations'][:,6:9],s['commands']*scales),
            actor_observation_storage_max_error=maximum_error(p['observations'],storage.observations['policy'][step]),
            action_storage_max_error=maximum_error(p['actions'],storage.actions[step]),
            action_native_max_error=maximum_error(p['actions'],s['raw_actions']),
            old_mean_storage_max_error=maximum_error(p['mean'],storage.mu[step]),
            old_sigma_storage_max_error=maximum_error(p['sigma'],storage.sigma[step]),
            native_reward_delivery_max_error=maximum_error(s['reward'],delivered),
            timeout_bootstrap_storage_max_error=maximum_error(expected_reward,storage.rewards[step,:,0]))
        self.alignment.append(errors)
        ids = self._ids(delivered)
        row = {key:np.array(value[ids].cpu(),copy=True) for key,value in s.items()}
        row['timeouts'] = np.array(timeouts[ids].cpu(),copy=True)
        self.native_rows.append(row)
        self.pending = self.snapshot = None
        return result

    def _capture_actor(self, module, inputs):
        if self.updating and self.capture:
            self.minibatch_size = len(inputs[0])
            self.minibatch = inputs[0][:128].detach().clone()

    def _probe(self):
        inputs = self.minibatch
        if inputs is None:
            raise RuntimeError('The PPO actor minibatch was not observed')
        cpu_rng = torch.random.get_rng_state().clone()
        gpu_rng = torch.cuda.get_rng_state(inputs.device).clone() if inputs.device.type == 'cuda' else None
        self.capture = False
        try:
            with torch.no_grad():
                means = self.algorithm.policy.actor(inputs)
                height,_,valid = predict_support_heights(means,inputs,self.model)
                command = inputs[:,8]/self.model['height_scale']
                normalized = (height-command[:,None])/(.1*command.abs().clamp_min(.001)[:,None])
                changed = inputs.repeat(2,1).clone()
                changed[:,8] = changed.new_tensor(SAFE_BASE_HEIGHT_RANGE_M).repeat_interleave(len(inputs))*self.model['height_scale']
                predicted = self.algorithm.policy.actor(changed)
                _,length,valid_changed = predict_support_heights(predicted,changed,self.model)
                if (not valid.all() or not valid_changed.all() or not torch.isfinite(height).all()
                        or not torch.isfinite(length).all()):
                    raise ValueError('Invalid FK in the read-only fixed-state probe')
                response = length.reshape(2,len(inputs),2).mean(1)
                result = dict(height_reference_normalized_squared_error=float(normalized.square().mean()),
                    height_reference_MAE_m=float((height-command[:,None]).abs().mean()),
                    left_right_height_command_reference_span_m=(response[1]-response[0]).tolist())
        finally:
            self.capture = True
        if not torch.equal(cpu_rng,torch.random.get_rng_state()):
            raise RuntimeError('The read-only probe changed CPU RNG state')
        if gpu_rng is not None and not torch.equal(gpu_rng,torch.cuda.get_rng_state(inputs.device)):
            raise RuntimeError('The read-only probe changed CUDA RNG state')
        return result

    def _before_step(self, *_):
        if self.updating:
            alg = self.algorithm
            self.current = dict(minibatch_size=self.minibatch_size,
                actor_gradient_norm_after_global_clip=gradient_norm(alg.policy.actor.parameters()),
                critic_gradient_norm_after_global_clip=gradient_norm(alg.policy.critic.parameters()),
                all_parameter_gradient_norm_after_global_clip=gradient_norm(alg.policy.parameters()),
                adaptive_learning_rate=alg.optimizer.param_groups[0]['lr'],before=self._probe())

    def _after_step(self, *_):
        if self.updating:
            self.current['after'] = self._probe()
            self.current['height_reference_loss_change'] = (
                self.current['after']['height_reference_normalized_squared_error']
                -self.current['before']['height_reference_normalized_squared_error'])
            self.gradients.append(self.current)

    def _update(self):
        storage = self.algorithm.storage
        ids = self._ids(storage.actions[0])
        arrays = {'observations':np.array(storage.observations['policy'][:,ids].cpu(),copy=True)}
        for key in ('actions','rewards','dones','values','mu','sigma','returns','advantages'):
            arrays[key] = np.array(getattr(storage,key)[:,ids].cpu(),copy=True)
        if self.native_rows:
            if len(self.native_rows) != storage.num_transitions_per_env:
                raise RuntimeError('Native transition record is incomplete')
            for key in self.native_rows[0]:
                arrays['native_'+key] = np.stack([row[key] for row in self.native_rows])
        self.last_arrays = arrays
        start = len(self.gradients)
        self.updating = True
        try:
            result = self.original['update']()
        finally:
            self.updating = False
        errors = {key:max(row[key] for row in self.alignment[-storage.num_transitions_per_env:])
                  for key in self.alignment[-1]} if self.alignment else {}
        record = dict(update=len(self.updates)+1,all_environment_alignment_max_errors=errors,
            native_alignment_recorded=self.env is not None,
            original_loss_metrics=result,optimizer_steps=self.gradients[start:],
            sampled_environment_ids=ids.cpu().tolist(),total_environments=storage.num_envs,
            read_only_hooks=True,added_loss_or_backward=False,probe_RNG_unchanged=True,
            policy_accepted=False,limitation=__doc__)
        self.updates.append(record)
        if self.output is not None:
            number = len(self.updates)
            np.savez_compressed(self.output/f'rollout_{number:04d}.npz',**arrays)
            write_json(self.output/f'audit_update_{number:04d}.json',record)
        self.native_rows.clear()
        return result

    def close(self):
        for handle in self.handles:
            handle.remove()
        for name,method in self.original.items():
            setattr(self.algorithm,name,method)
        if self.env is not None:
            self.env._get_rewards = self.original_reward
