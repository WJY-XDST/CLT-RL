"""Offline candidate: encoder/IMU height-error features for ordinary PPO.

The external observation/action contract remains 27 inputs and six direct
joint/wheel actions. Forward kinematics enriches the first actor/critic layer;
it never changes an action or solves IK. This module is not imported by the
active trainer. A native trial requires separate migration/export review.
"""
import copy
from functools import partial
import math

import torch
from torch import nn
from torch.nn import functional as F
from rsl_rl.modules import ActorCritic


def feature_model_from_config(cfg, hip_z):
    if (cfg['leg_control_mode'] != 'implicit_joint_position'
            or cfg['observation_space'] != 27 or cfg['action_space'] != 6):
        raise ValueError('Keep the original direct-joint observation/action contract')
    result = dict(profile='encoder_imu_support_height_error_v1',
        geometry=copy.deepcopy(cfg['five_bar_geometry']),
        joint_position_offset=list(cfg['leg_joint_position_offset']),
        joint_position_observation_scale=cfg['obs_scales']['dof_pos'],
        command_height_observation_scale=cfg['obs_scales']['height_measurements'],
        wheel_radius_m=cfg['wheel_radius'], hip_z_m=hip_z,
        wheel_track_width_m=cfg['wheel_track_width'], error_scale_per_m=10.)
    if (result['geometry']['branch'] not in (-1, 1)
            or any(not math.isfinite(float(v)) or v <= 0 for v in (
                result['joint_position_observation_scale'], result['command_height_observation_scale'],
                result['wheel_radius_m'], result['wheel_track_width_m'], result['error_scale_per_m']))):
        raise ValueError('Invalid paired geometry or observation scales')
    return result


class EncoderImuHeightError(nn.Module):
    """Two measured support-plane errors; no physical command generation."""
    def __init__(self, model):
        super().__init__()
        if model['profile'] != 'encoder_imu_support_height_error_v1':
            raise ValueError('Unknown height feature definition')
        geometry = model['geometry']
        # Geometry is immutable architecture metadata, stored with migration.
        self.register_buffer('joint_offset', torch.tensor(model['joint_position_offset']), persistent=False)
        self.register_buffer('mirror', torch.tensor([1., -1.]), persistent=False)
        self.register_buffer('lateral', torch.tensor([
            model['wheel_track_width_m'] / 2, -model['wheel_track_width_m'] / 2]), persistent=False)
        self.register_buffer('upper', torch.tensor(geometry['upper']), persistent=False)
        self.register_buffer('lower', torch.tensor(geometry['lower']), persistent=False)
        self.register_buffer('offsets', torch.tensor(geometry['offsets']), persistent=False)
        self.geometry_scale = float(geometry['scale'])
        self.branch = float(geometry['branch'])
        self.hip_z = float(model['hip_z_m'])
        self.wheel_radius = float(model['wheel_radius_m'])
        self.joint_scale = float(model['joint_position_observation_scale'])
        self.height_scale = float(model['command_height_observation_scale'])
        self.error_scale = float(model['error_scale_per_m'])

    def support_heights(self, observations):
        q = observations[:, 9:13] / self.joint_scale + self.joint_offset
        theta_b = self.offsets[0] + q[:, 0::2] * self.mirror
        theta_l = self.offsets[1] + q[:, 1::2] * self.mirror
        c = self.upper[0] * torch.stack((theta_b.cos(), theta_b.sin()), dim=-1)
        a = self.upper[1] * torch.stack((theta_l.cos(), theta_l.sin()), dim=-1)
        delta = a - c
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1.e-7)
        along = (self.lower[0].square() - self.lower[1].square() + distance.square()) / (2 * distance)
        height2 = self.lower[0].square() - along.square()
        normal = torch.stack((-delta[..., 1], delta[..., 0]), dim=-1) / distance[..., None]
        p = (c + along[..., None] * delta / distance[..., None]
             + self.branch * height2.clamp_min(0.).sqrt()[..., None] * normal)
        wheel = self.geometry_scale * p
        gravity = observations[:, 3:6]
        # CAD planar coordinates (Z, X) map directly to chassis coordinates.
        return (self.wheel_radius + gravity[:, 0, None] * wheel[..., 1]
                + gravity[:, 1, None] * self.lateral
                + gravity[:, 2, None] * (self.hip_z + wheel[..., 0]))

    def forward(self, observations):
        command = observations[:, 8:9] / self.height_scale
        return (command - self.support_heights(observations)) * self.error_scale


class HeightFeatureLinear(nn.Module):
    """Preserve the old first affine map exactly when the new columns are zero."""
    def __init__(self, original, model):
        super().__init__()
        if original.in_features != 27:
            raise ValueError('Require the original first-layer input width')
        self.in_features = 27  # exporter and external policy input stay 27-D
        self.out_features = original.out_features
        old = original.weight.detach()
        self.weight = nn.Parameter(torch.cat((old.clone(), old.new_zeros((len(old), 2))), dim=1))
        self.bias = original.bias
        self.height_error = EncoderImuHeightError(model).to(device=old.device, dtype=old.dtype)

    def forward(self, observations):
        # Separate contractions keep the old GEMM shape and reduction order.
        base = F.linear(observations, self.weight[:, :27].contiguous(), self.bias)
        correction = F.linear(self.height_error(observations), self.weight[:, 27:].contiguous())
        return base + correction


class JointPdHeightFeatureActorCritic(ActorCritic):
    def __init__(self, *args, height_feature_model, **kwargs):
        if kwargs.get('actor_obs_normalization', False) or kwargs.get('critic_obs_normalization', False):
            raise ValueError('Keep the qualified unnormalized policy')
        super().__init__(*args, **kwargs)
        if self.actor[-1].out_features != 6:
            raise ValueError('Keep all six original action heads')
        self.actor[0] = HeightFeatureLinear(self.actor[0], height_feature_model)
        self.critic[0] = HeightFeatureLinear(self.critic[0], height_feature_model)


def migrate_saved_state(saved, parameter_names, model):
    """Pad new columns and their moments; never reset any existing Adam state."""
    saved = copy.deepcopy(saved)
    names = list(parameter_names)
    if len(names) != 17 or set(names) != set(saved['model_state_dict']):
        raise ValueError('Require the paired 17-parameter ordinary actor/critic')
    groups = saved['optimizer_state_dict']['param_groups']
    ids = [identifier for group in groups for identifier in group['params']]
    if len(ids) != 17 or len(set(ids)) != 17:
        raise ValueError('Require all 17 optimizer parameter identifiers exactly once')
    optimizer = saved['optimizer_state_dict']['state']
    if set(ids) != set(optimizer):
        raise ValueError('Require complete existing Adam moments')
    for name in ('actor.0.weight', 'critic.0.weight'):
        value = saved['model_state_dict'][name]
        if value.ndim != 2 or value.shape[1] != 27:
            raise ValueError('Migrate only once from the original first-layer width')
        saved['model_state_dict'][name] = F.pad(value, (0, 2))
        state = optimizer[ids[names.index(name)]]
        for key, moment in list(state.items()):
            if torch.is_tensor(moment) and moment.ndim:
                if moment.shape != value.shape:
                    raise ValueError('Unexpected Adam moment shape for ' + name)
                state[key] = F.pad(moment, (0, 2))
    saved['joint_pd_height_features'] = copy.deepcopy(model)
    return saved


def register_runner_policy(model):
    import rsl_rl.runners.on_policy_runner as runner
    runner.JointPdHeightFeatureActorCritic = partial(
        JointPdHeightFeatureActorCritic, height_feature_model=copy.deepcopy(model))
