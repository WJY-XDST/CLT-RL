"""Explicit plain-policy source200 fork; retain the complete upstream PPO state."""
import copy
from pathlib import Path

import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha, tensor_finite, optimizer_step
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
from wheel_legged_gym_isaaclab.training_state import restore_adaptive_learning_rate
import joint_pd_analytic_tire_runtime as tires

KEY = 'source200_stationary_height'
SOURCE_SHA = '556ad6b47c108e3cb8522a700c8b0fec143666e3568d4d6de8b4afc6a85349d4'
SOURCE = ROOT / 'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat/2026-10-15_standing_seed_recovery_source_snapshot/model_200.pt'
HEIGHT_SIGMA = .0025
LEG_STD_FLOOR = .02
WHEEL_STD_FLOOR = .03


def profile():
    return dict(profile='plain_source200_stationary_broad_height_v1',
        source_anchor_sha256=SOURCE_SHA, internal_observations=27, actions=6,
        policy_class='ActorCritic', ordinary_PPO=True, auxiliary_losses=False,
        minimum_leg_std=LEG_STD_FLOOR, minimum_wheel_std=WHEEL_STD_FLOOR,
        base_height_sigma=HEIGHT_SIGMA, height_reward_coefficient=2.,
        tire_physics=tires.profile(), native_runtime_sha256=sha(__file__))


def require_source(source, *, initial_transfer=False):
    source = Path(source).resolve()
    saved = torch.load(source, map_location='cpu', weights_only=True)
    cfg = validate_checkpoint_robot(source)
    infos = saved.get('infos')
    declared = infos.get(KEY) if isinstance(infos, dict) else None
    if declared is None:
        if (not initial_transfer or sha(source) != SOURCE_SHA
                or saved_updates(source) != 201 or optimizer_step(saved) != 4020.):
            raise ValueError('Only the exact owner-selected model200 may begin this explicit fork')
    elif (declared != profile() or cfg['rewards']['base_height_sigma'] != HEIGHT_SIGMA
            or cfg['commands']['training_motion_scale'] != 0.
            or Path(cfg['robot']['spawn']['usd_path']).resolve() != tires.ASSET):
        raise ValueError('Continue only the paired plain stationary branch')
    if (not tensor_finite(saved) or cfg['leg_control_mode'] != 'implicit_joint_position'
            or saved['model_state_dict']['actor.0.weight'].shape[1] != 27
            or saved['model_state_dict']['critic.0.weight'].shape[1] != 27
            or len(saved['optimizer_state_dict']['state']) != 17
            or optimizer_step(saved) != 20 * saved_updates(source)):
        raise ValueError('Require the plain27 direct-joint policy and all17 finite continuous Adam states')
    return saved, cfg


def equal_state(left, right, label):
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or not torch.equal(left.cpu(), right.detach().cpu()):
            raise ValueError('Loaded state changed: ' + label)
    elif isinstance(left, dict):
        if set(left) != set(right):
            raise ValueError('Loaded state keys changed: ' + label)
        for key in left:
            equal_state(left[key], right[key], label + '/' + str(key))
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise ValueError('Loaded state size changed: ' + label)
        for i, (a, b) in enumerate(zip(left, right)):
            equal_state(a, b, label + '/' + str(i))
    elif left != right:
        raise ValueError('Loaded scalar changed: ' + label)


def install_runner(output):
    import rsl_rl.runners as runners
    upstream = runners.OnPolicyRunner

    class RetainedSource200Runner(upstream):
        def load(self, path, load_optimizer=True, map_location=None):
            if not load_optimizer:
                raise ValueError('Retain all17 Adam states in the source200 fork')
            source = Path(path).resolve()
            saved, _ = require_source(source, initial_transfer=sha(source) == SOURCE_SHA)
            result = super().load(str(source), load_optimizer=True, map_location=map_location)
            equal_state(saved['model_state_dict'], self.alg.policy.state_dict(), 'policy')
            equal_state(saved['optimizer_state_dict'], self.alg.optimizer.state_dict(), 'Adam')
            lr = restore_adaptive_learning_rate(self.alg)
            if any(g['lr'] != lr for g in saved['optimizer_state_dict']['param_groups']):
                raise ValueError('Retain the actual saved adaptive learning rate')
            write_json(Path(output) / 'source200_runtime_load.json', dict(
                checkpoint=str(source), checkpoint_sha256=sha(source),
                source_actual_updates=saved_updates(source), all17_Adam_step=optimizer_step(saved),
                all_model_and_optimizer_tensors_bitwise_preserved=True,
                initial_std=saved['model_state_dict']['std'].tolist(), optimizer_LR=lr,
                architecture_migrated=False, optimizer_reset=False, profile=profile()))
            return result

        def save(self, path, infos=None):
            metadata = copy.deepcopy(infos) if isinstance(infos, dict) else {'upstream_infos': infos}
            if metadata.get(KEY, profile()) != profile():
                raise ValueError('Do not silently change the source200 branch definition')
            metadata[KEY] = profile()
            return super().save(path, infos=metadata)

    runners.OnPolicyRunner = RetainedSource200Runner
    return upstream, RetainedSource200Runner


def observe_environment(env, output):
    from carb.settings import get_settings
    from pxr import UsdGeom, UsdPhysics
    env = env.unwrapped
    cfg = env.cfg
    if (cfg.leg_control_mode != 'implicit_joint_position'
            or cfg.observation_space != 27 or cfg.action_space != 6
            or cfg.commands.training_motion_scale != 0.
            or cfg.rewards.base_height_sigma != HEIGHT_SIGMA
            or cfg.rewards.base_height != 2.
            or cfg.rewards.base_height_error_sq != 0.
            or cfg.rewards.joint_target_leg_length_error_sq != 0.
            or Path(cfg.robot.spawn.usd_path).resolve() != tires.ASSET
            or get_settings().get('/physics/collisionApproximateCylinders') is not False):
        raise ValueError('Actual environment differs from the explicit plain-source200 recipe')
    for side in ('LW_link', 'RW_link'):
        root = '/World/envs/env_0/Robot/' + side
        prim = env.sim.stage.GetPrimAtPath(root + '/analytic_tire_collision')
        old = env.sim.stage.GetPrimAtPath(root + '/smooth_tire_collision')
        if (not prim.IsA(UsdGeom.Cylinder)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
                or UsdPhysics.CollisionAPI(old).GetCollisionEnabledAttr().Get() is not False):
            raise ValueError('Keep the qualified actual analytic tire contact geometry')
    masses = env._robot.root_physx_view.get_masses().sum(-1)
    if not torch.allclose(masses, torch.full_like(masses, 12.34), atol=1.e-4, rtol=0.):
        raise ValueError('Keep the measured12.34kg vehicle')
    proof = dict(profile=profile(), native_environments=env.num_envs,
        actual_mass_range_kg=[float(masses.min()), float(masses.max())],
        native_environment_verified=True, observer_changes_actions_or_rewards=False)
    Path(output).mkdir(parents=True, exist_ok=True)
    write_json(Path(output) / 'source200_environment.json', proof)


def install_environment_observer(output):
    import gymnasium as gym
    original = gym.make

    def make(*args, **kwargs):
        result = original(*args, **kwargs)
        if (args[0] if args else kwargs.get('id')) == 'WheelLeggedJointPD-Flat-v0':
            observe_environment(result, output)
        return result

    gym.make = make
