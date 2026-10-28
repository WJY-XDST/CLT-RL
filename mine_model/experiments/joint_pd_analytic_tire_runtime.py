"""Pair an isolated analytic-tire physics profile with unchanged feature PPO.

No PPO update, policy parameter, optimizer loading or action method is
replaced. The only runner override adds an explicit checkpoint metadata key.
"""
import copy
import json
from pathlib import Path

import torch

from joint_pd_continuation import ROOT, sha
from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot

ASSET = ROOT / 'wheel_legged_isaaclab/assets/robots/mine/export_20261003_181515/mine_closed_chain_training_measured_mass_analytic_tires_20261017.usda'
ASSET_SHA = '48cbb96dffb5b8df95dd5dba16d34c7a4d2e1dcce44b06309bec6a974a6d2ddf'
LEGACY_SOURCE_SHA = 'e7dff1c3b4b951ef34a03cab9011581ae54ab99971a3a6fa673cf976da11aefd'
KEY = 'joint_pd_tire_collision'
KIT_ARGUMENT = '--kit_args=--/physics/collisionApproximateCylinders=false'


def profile():
    if sha(ASSET) != ASSET_SHA:
        raise ValueError('Qualified analytic tire asset changed')
    chain, _ = verify_asset_chain(ASSET)
    return dict(profile='measured_mass_analytic_tires_v1', asset=str(ASSET), asset_sha256=ASSET_SHA,
        asset_chain_sha256=chain, required_process_setting={'/physics/collisionApproximateCylinders': False},
        tire_radius_m=.06, tire_width_m=.02, vehicle_mass_kg=12.34)


def arguments(values):
    values = list(values)
    if any(v in values for v in ('--reset_optimizer', '--reset_exploration_std', '--save_initial_checkpoint')):
        raise ValueError('Analytic-tire adaptation must retain the complete source')
    existing = [v for v in values if v.startswith('--kit_args')]
    if existing and existing != [KIT_ARGUMENT]:
        raise ValueError('Use the single qualified cylinder startup setting')
    return values if existing else [*values, KIT_ARGUMENT]


def require_source(source, *, initial_transfer=False):
    source = Path(source).resolve()
    saved = torch.load(source, map_location='cpu', weights_only=True)
    cfg = validate_checkpoint_robot(source)
    metadata = saved.get('infos')
    declared = metadata.get(KEY) if isinstance(metadata, dict) else None
    if declared is None:
        if not initial_transfer or sha(source) != LEGACY_SOURCE_SHA:
            raise ValueError('Only the exact retained stable526 may start the explicit physics transfer')
    elif declared != profile() or Path(cfg['robot']['spawn']['usd_path']).resolve() != ASSET:
        raise ValueError('Checkpoint physics metadata and paired asset differ')
    if (saved['model_state_dict']['actor.0.weight'].shape[1] != 29
            or len(saved['optimizer_state_dict']['state']) != 17):
        raise ValueError('Keep the complete retained feature model and all17 Adam states')
    return saved, cfg


def add_metadata(infos):
    result = copy.deepcopy(infos) if isinstance(infos, dict) else {'upstream_infos': infos}
    expected = profile()
    if result.get(KEY, expected) != expected:
        raise ValueError('Cannot silently replace the declared tire physics profile')
    result[KEY] = expected
    return result


def install_runner_metadata():
    import rsl_rl.runners as runners
    upstream = runners.OnPolicyRunner

    class AnalyticTireRunner(upstream):
        def save(self, path, infos=None):
            return super().save(path, infos=add_metadata(infos))

    runners.OnPolicyRunner = AnalyticTireRunner
    return upstream, AnalyticTireRunner


def install_environment_observer(observation_output):
    import gymnasium as gym
    original = gym.make

    def make(*args, **kwargs):
        result = original(*args, **kwargs)
        identifier = args[0] if args else kwargs.get('id')
        if identifier != 'WheelLeggedJointPD-Flat-v0':
            return result
        from carb.settings import get_settings
        from pxr import UsdGeom, UsdPhysics
        from joint_pd_height_feature_runtime import write
        env = result.unwrapped
        before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        if Path(env.cfg.robot.spawn.usd_path).resolve() != ASSET:
            raise ValueError('Native adaptation did not load the qualified tire asset')
        if get_settings().get('/physics/collisionApproximateCylinders') is not False:
            raise ValueError('Native cylinder approximation setting differs from the saved profile')
        for name in ('LW_link', 'RW_link'):
            prim = env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/' + name + '/analytic_tire_collision')
            old = env.sim.stage.GetPrimAtPath('/World/envs/env_0/Robot/' + name + '/smooth_tire_collision')
            if (not prim.IsA(UsdGeom.Cylinder)
                    or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
                    or UsdPhysics.CollisionAPI(old).GetCollisionEnabledAttr().Get() is not False):
                raise ValueError('Actual source articulation has the wrong active tire geometry')
        masses = env._robot.root_physx_view.get_masses().sum(-1)
        if not torch.allclose(masses, torch.full_like(masses, 12.34), atol=1.e-4, rtol=0.):
            raise ValueError('Native tire profile changed measured total mass')
        if (not torch.equal(before_cpu, torch.get_rng_state()) or any(
                not torch.equal(a, b) for a, b in zip(before_cuda, torch.cuda.get_rng_state_all()))):
            raise ValueError('Read-only startup observation changed policy random state')
        proof = dict(profile=profile(), native_environments=env.num_envs,
            native_mass_range_kg=[float(masses.min()), float(masses.max())],
            source_articulation_analytic_tires_verified=True, policy_RNG_unchanged=True,
            policy_actions_or_rewards_modified=False, policy_accepted=False)
        write(Path(env.cfg.log_dir) / 'params/tire_collision_contract.json', proof)
        write(Path(observation_output) / 'tire_geometry_runtime.json', proof)
        return result

    gym.make = make

