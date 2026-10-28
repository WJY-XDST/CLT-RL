"""Opt-in physical-parameter, sensor and latency perturbations for frozen tests.

Distributions do not inspect commands, task IDs, heights or policy errors.
The disabled path installs no hooks, writes no properties and draws no RNG.
Physical properties are sampled once per virtual robot; noise is sampled per
observation. Sensor and target delays have physics-step resolution.
"""
from dataclasses import asdict, dataclass, fields
import math

import torch


@dataclass(frozen=True)
class RobustnessSpec:
    enabled: bool = False
    seed: int = 20261026
    mass_scale: tuple = (1., 1.)
    inertia_scale: tuple = (1., 1.)
    friction_scale: tuple = (1., 1.)
    sensor_delay_steps: tuple = (0, 0)
    action_delay_steps: tuple = (0, 0)
    gyro_noise_rad_s: float = 0.
    gravity_noise: float = 0.
    joint_position_noise_rad: float = 0.
    joint_velocity_noise_rad_s: float = 0.
    linear_velocity_noise_m_s: float = 0.
    wheel_velocity_noise_rad_s: float = 0.
    gyro_bias_rad_s: float = 0.
    joint_zero_bias_rad: float = 0.

    @classmethod
    def from_dict(cls, value):
        unknown = set(value) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError('Unknown robustness fields: ' + str(sorted(unknown)))
        result = cls(**value)
        result.validate()
        return result

    def validate(self):
        if type(self.enabled) is not bool or type(self.seed) is not int or self.seed < 0:
            raise ValueError('Require a boolean enabled flag and a nonnegative integer seed')
        for name in ('mass_scale', 'inertia_scale', 'friction_scale'):
            pair = getattr(self, name)
            if (len(pair) != 2 or any(type(v) not in (float, int) or not math.isfinite(v) or v <= 0 for v in pair)
                    or pair[0] > pair[1]):
                raise ValueError('Invalid positive scale interval: ' + name)
        for name in ('sensor_delay_steps', 'action_delay_steps'):
            pair = getattr(self, name)
            if len(pair) != 2 or any(type(v) is not int or not 0 <= v <= 20 for v in pair) or pair[0] > pair[1]:
                raise ValueError('Delay intervals require integers in [0,20]: ' + name)
        for field in fields(self):
            if field.name.endswith(('_rad_s', '_rad', '_m_s')) or field.name == 'gravity_noise':
                value = getattr(self, field.name)
                if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                    raise ValueError('Noise and bias amplitudes must be finite and nonnegative: ' + field.name)


class DelayHistory:
    """Per-environment integer age lookup, with explicit reset isolation."""
    def __init__(self, count, width, maximum_age, *, device, dtype=torch.float32):
        self.values = torch.zeros(maximum_age + 1, count, width, device=device, dtype=dtype)
        self.cursor = 0
        self.ids = torch.arange(count, device=device)

    def push(self, value):
        self.values[self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % len(self.values)

    def read(self, ages):
        if ages.shape != self.ids.shape or ages.dtype != torch.long:
            raise ValueError('One integer age per environment is required')
        return self.values[(self.cursor - 1 - ages) % len(self.values), self.ids].clone()

    def reset(self, env_ids, value):
        self.values[:, env_ids] = value


class RobustnessSession:
    sensor_columns = (0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)

    def __init__(self, env, spec):
        spec.validate()
        self.env, self.spec = env, spec
        self.applied = False
        if not spec.enabled:
            self.evidence = dict(enabled=False, hooks_installed=0, RNG_draws=0,
                                 physics_properties_written=False)
            return
        if env.cfg.leg_control_mode != 'implicit_joint_position':
            raise ValueError('Robustness preparation currently requires the direct-joint PD contract')
        self.device_generator = torch.Generator(device=env.device).manual_seed(spec.seed)
        self.cpu_generator = torch.Generator(device='cpu').manual_seed(spec.seed)
        self.sensor_ages = self._ages(spec.sensor_delay_steps)
        self.action_ages = self._ages(spec.action_delay_steps)
        self.sensor_history = DelayHistory(env.num_envs, 18, max(spec.sensor_delay_steps), device=env.device)
        self.action_history = DelayHistory(env.num_envs, 6, max(spec.action_delay_steps), device=env.device)
        self.requested_actions = torch.zeros_like(env._raw_actions)
        self.gyro_bias = self._uniform((-spec.gyro_bias_rad_s, spec.gyro_bias_rad_s), (env.num_envs, 3), env.device)
        self.joint_bias = self._uniform((-spec.joint_zero_bias_rad, spec.joint_zero_bias_rad), (env.num_envs, 4), env.device)
        self._apply_properties()
        self._install()
        self.applied = True

    def _uniform(self, interval, shape, device):
        low, high = interval
        if low == high:
            return torch.full(shape, low, device=device)
        generator = self.cpu_generator if str(device) == 'cpu' else self.device_generator
        return low + (high - low) * torch.rand(shape, device=device, generator=generator)

    def _ages(self, interval):
        lo, hi = interval
        if lo == hi:
            return torch.full((self.env.num_envs,), lo, device=self.env.device, dtype=torch.long)
        return torch.randint(lo, hi + 1, (self.env.num_envs,), device=self.env.device,
                             generator=self.device_generator)

    def _apply_properties(self):
        env, spec = self.env, self.spec
        view = env._robot.root_physx_view
        before_mass, before_inertia = view.get_masses().clone(), view.get_inertias().clone()
        masses, inertias = before_mass.clone(), before_inertia.clone()
        ids = torch.arange(env.num_envs, device='cpu')
        scale = self._uniform(spec.mass_scale, (env.num_envs, 1), 'cpu')
        # Wheel mass was measured. Preserve it; vary the less certain body/links.
        moving = [i for i, name in enumerate(env._robot.body_names) if name not in ('LW_link', 'RW_link')]
        mass_ratio = torch.ones_like(masses)
        mass_ratio[:, moving] = scale
        shape_ratio = self._uniform(spec.inertia_scale, (env.num_envs, 1), 'cpu')
        masses *= mass_ratio
        inertias *= (mass_ratio * shape_ratio)[..., None]
        if tuple(spec.mass_scale) != (1., 1.):
            view.set_masses(masses, ids)
        if tuple(spec.mass_scale) != (1., 1.) or tuple(spec.inertia_scale) != (1., 1.):
            view.set_inertias(inertias, ids)
        materials = view.get_material_properties().clone()
        before_materials = materials.clone()
        friction = self._uniform(spec.friction_scale, (env.num_envs, 1), 'cpu')
        materials[..., :2] *= friction[..., None]
        if tuple(spec.friction_scale) != (1., 1.):
            view.set_material_properties(materials, ids)
        actual_mass, actual_inertia = view.get_masses(), view.get_inertias()
        actual_materials = view.get_material_properties()
        if not (torch.allclose(actual_mass, masses, atol=1.e-6, rtol=1.e-6)
                and torch.allclose(actual_inertia, inertias, atol=1.e-7, rtol=1.e-6)
                and torch.allclose(actual_materials, materials, atol=1.e-6, rtol=1.e-6)):
            raise RuntimeError('Readback of randomized PhysX parameters differs from requested values')
        self.evidence = dict(enabled=True, configuration=asdict(spec),
            nominal_mass_kg=before_mass.sum(-1).tolist(), actual_mass_kg=actual_mass.sum(-1).tolist(),
            nonwheel_mass_scale=scale[:, 0].tolist(), uniform_inertia_shape_scale=shape_ratio[:, 0].tolist(),
            friction_scale=friction[:, 0].tolist(),
            nominal_material_coefficients=before_materials[0].tolist(),
            actual_material_coefficients=actual_materials.tolist(),
            sensor_delay_ms=(self.sensor_ages * env.cfg.sim.dt * 1000.).tolist(),
            action_delay_ms=(self.action_ages * env.cfg.sim.dt * 1000.).tolist(),
            physical_properties_readback_verified=True, measured_wheel_masses_preserved=True,
            inertia_scaled_with_mass_and_positive_shape_scalar=True,
            gravity_feedforward_retains_nominal_mass=True,
            parameters_chosen_from_commands=False, optimizer_executed=False,
            limitations=['Noise affects actor sensors; the implicit joint servo and local gravity feedforward use simulator state.',
                         'Latencies are integer physics steps; no substep inference-time claim is made.',
                         'Distributions are engineering stress ranges, not measured hardware uncertainty.'])

    def _sensors(self):
        env, cfg = self.env, self.env.cfg
        return torch.cat((
            env._sanitize_observation(env._robot.data.root_ang_vel_b, cfg.obs_clip.ang_vel) * cfg.obs_scales.ang_vel,
            env._sanitize_observation(env._robot.data.projected_gravity_b, cfg.obs_clip.projected_gravity),
            env._get_leg_observation(),
            env._sanitize_observation(env._get_heading_frame_horizontal_velocity(), cfg.obs_clip.lin_vel) * cfg.obs_scales.lin_vel,
            env._sanitize_observation(env._get_wheel_vel_forward_positive(), cfg.obs_clip.wheel_vel) * cfg.obs_scales.dof_vel,
        ), -1)

    def _noise(self, obs):
        spec, cfg = self.spec, self.env.cfg
        result = obs.clone()
        blocks = ((slice(0, 3), spec.gyro_noise_rad_s, cfg.obs_scales.ang_vel, cfg.obs_clip.ang_vel),
                  (slice(9, 13), spec.joint_position_noise_rad, cfg.obs_scales.dof_pos, cfg.obs_clip.theta),
                  (slice(13, 17), spec.joint_velocity_noise_rad_s, cfg.obs_scales.dof_vel, cfg.obs_clip.theta_dot),
                  (slice(17, 19), spec.linear_velocity_noise_m_s, cfg.obs_scales.lin_vel, cfg.obs_clip.lin_vel),
                  (slice(19, 21), spec.wheel_velocity_noise_rad_s, cfg.obs_scales.dof_vel, cfg.obs_clip.wheel_vel))
        for selection, std, scale, bound in blocks:
            value = result[:, selection]
            if std:
                value = value + torch.randn(value.shape, device=value.device, generator=self.device_generator) * std * scale
            if selection.start == 0:
                value = value + self.gyro_bias * scale
            if selection.start == 9:
                value = value + self.joint_bias * scale
            result[:, selection] = value.clamp(-bound * scale, bound * scale)
        if spec.gravity_noise:
            g = result[:, 3:6]
            g = g + torch.randn(g.shape, device=g.device, generator=self.device_generator) * spec.gravity_noise
            result[:, 3:6] = g / torch.linalg.vector_norm(g, dim=-1, keepdim=True).clamp_min(1.e-8)
        return result

    def _install(self):
        from wheel_legged_gym_isaaclab.wheel_control import apply_wheel_action_limit
        env = self.env
        original_pre, original_apply = env._pre_physics_step, env._apply_action
        original_obs, original_reset = env._get_observations, env._reset_idx

        def pre(actions):
            original_pre(actions)
            self.requested_actions.copy_(env._raw_actions)

        def apply():
            self.sensor_history.push(self._sensors())
            self.action_history.push(self.requested_actions)
            if self.spec.action_delay_steps[1]:
                env._raw_actions = self.action_history.read(self.action_ages)
                env._actions = apply_wheel_action_limit(env._raw_actions.clamp(-1., 1.),
                                                       env._raw_actions, env.cfg.wheel_action_limit)
            original_apply()

        def observations():
            value = original_obs()
            obs = value['policy']
            if self.spec.sensor_delay_steps[1]:
                delayed = self.sensor_history.read((self.sensor_ages - 1).clamp_min(0))
                obs = obs.clone()
                obs[:, self.sensor_columns] = torch.where((self.sensor_ages > 0)[:, None],
                                                          delayed, obs[:, self.sensor_columns])
            value['policy'] = self._noise(obs)
            return value

        def reset(env_ids):
            original_reset(env_ids)
            ids = env._robot._ALL_INDICES if env_ids is None else env_ids
            self.requested_actions[ids] = 0.
            self.action_history.reset(ids, torch.zeros((len(ids), 6), device=env.device))
            self.sensor_history.reset(ids, self._sensors()[ids])

        env._pre_physics_step, env._apply_action = pre, apply
        env._get_observations, env._reset_idx = observations, reset
        all_ids = env._robot._ALL_INDICES
        self.sensor_history.reset(all_ids, self._sensors())
        self.evidence.update(hooks_installed=4, action_history_uses_actual_applied_targets=True,
                             command_and_previous_action_observation_columns_delayed=False,
                             sensor_noise_uses_separate_generator=True,
                             geometry_rewards_gains_limits_and_command_sampler_changed=False)
