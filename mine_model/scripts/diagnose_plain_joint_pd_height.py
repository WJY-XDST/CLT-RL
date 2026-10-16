"""Read-only height-command response for ordinary direct-joint PPO.

Build geometry from the paired configuration and composed USD, without
requiring or enabling the retired optional auxiliary training model.
Frozen-state target geometry is not actual closed-loop height acceptance.
"""
import argparse
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import torch
import yaml

from auto_train_mine import write_json
import diagnose_height_response as original
from joint_pd_continuation import optimizer_step, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain
from wheel_legged_gym_isaaclab.leg_control import require_matching_policy_contract
from wheel_legged_gym_isaaclab.robot_model import MODEL, validate_checkpoint_robot


def cpu_usd_environment():
    """Expose the installed SDK's CPU USD libraries without starting Kit."""
    cache = Path(__file__).resolve().parents[2]/'IsaacLab/_isaac_sim/extscache'
    libraries = [p for p in cache.glob('omni.usd.libs-*') if (p/'pxr/Usd').is_dir()]
    if len(libraries) != 1:
        raise RuntimeError('Select one installed CPU USD library before offline diagnosis')
    environment = dict(os.environ, CLT_CPU_USD_DIAG_READY='1')
    for key, path in [('PYTHONPATH', libraries[0]), ('LD_LIBRARY_PATH', libraries[0]/'bin')]:
        environment[key] = str(path)+os.pathsep+environment.get(key, '')
    return environment


def usd_joint_limits(asset, names):
    from pxr import Usd, UsdPhysics
    stage = Usd.Stage.Open(str(asset))
    if stage is None:
        raise ValueError('Cannot read the paired robot USD')
    found = {name: [] for name in names}
    for prim in stage.Traverse():
        if prim.GetName() in found and prim.IsA(UsdPhysics.RevoluteJoint):
            joint = UsdPhysics.RevoluteJoint(prim)
            found[prim.GetName()].append([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])
    limits = []
    for name in names:
        values = found[name]
        if (len(values) != 1 or any(v is None or not math.isfinite(v) for v in values[0])
                or values[0][0] >= values[0][1]):
            raise ValueError('Require one finite authored hard limit for '+name)
        limits.append([math.radians(v) for v in values[0]])
    return limits


def geometry(cfg, limits):
    """Only fields consumed by the existing no-gradient forward geometry."""
    if cfg['leg_control_mode'] != 'implicit_joint_position' or cfg['observation_space'] != 27:
        raise ValueError('Require the paired 27-D direct-joint configuration')
    return dict(geometry=copy.deepcopy(cfg['five_bar_geometry']),
        offset=list(cfg['leg_joint_position_offset']), scale=list(cfg['leg_joint_position_scale']),
        margin=cfg['leg_joint_target_limit_margin'], joint_limits=limits,
        wheel_radius=cfg['wheel_radius'], hip_z=MODEL['hip_z'],
        wheel_track_width=cfg['wheel_track_width'], height_scale=cfg['obs_scales']['height_measurements'])


def require_plain_ppo(agent):
    algorithm, policy = agent['algorithm'], agent['policy']
    if (algorithm['class_name'] != 'PPO' or algorithm.get('joint_mean_guard_coef', 0.)
            or algorithm.get('joint_height_feedback_coef', 0.)
            or policy['activation'] != 'elu' or policy.get('actor_obs_normalization', False)):
        raise ValueError('Read ordinary unnormalized ELU PPO without auxiliary losses')


def diagnose(replay, checkpoints):
    replay = Path(replay).resolve()
    manifest = json.loads((replay/'manifest.json').read_text())
    completed = json.loads((replay/'completed.json').read_text())
    if not completed['completed'] or completed['resets']:
        raise ValueError('Require a completed surviving fixed-input replay')
    baseline = Path(manifest['checkpoint']).resolve()
    source_cfg = validate_checkpoint_robot(baseline)
    expected_samples = round(manifest['seconds_per_case']/manifest['policy_dt'])*len(manifest['cases'])*len(manifest['seeds'])
    if completed['samples'] != expected_samples:
        raise ValueError('Incomplete fixed-input sample record')
    frozen_assets = manifest['asset_chain_sha256']
    audits = {}

    def load(path):
        path = Path(path)
        cfg = validate_checkpoint_robot(path)
        require_matching_policy_contract(source_cfg, cfg)
        hashes, _ = verify_asset_chain(cfg['robot']['spawn']['usd_path'])
        if hashes != frozen_assets:
            raise ValueError('Geometry asset chain differs from the fixed-input replay')
        agent = yaml.safe_load((path.parent/'params/agent.yaml').read_text())
        require_plain_ppo(agent)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        actual = saved_updates(path)
        if (not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
                or optimizer_step(saved) != 20*actual):
            raise ValueError('Require complete finite weights and their 17 Adam states')
        limits = usd_joint_limits(cfg['robot']['spawn']['usd_path'], cfg['leg_joint_names'])
        audits[str(path.resolve())] = dict(actual_ppo_updates=actual, Adam_step=optimizer_step(saved),
            optimizer_states=17, learned_std=saved['model_state_dict']['std'].tolist(),
            USD_joint_limits_rad=limits, geometry_source='Paired env.yaml and composed USD hard limits',
            auxiliary_losses_enabled=False)
        return saved['model_state_dict'], geometry(cfg, limits), int(saved['iter'])

    # A private module instance keeps the running controller's pinned diagnostic
    # file and all imported module globals unchanged.
    spec = importlib.util.spec_from_file_location('_plain_joint_pd_height_read_only', original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_checkpoint = load
    result = module.diagnose(replay, checkpoints)
    for row in result['checkpoints']:
        row.update(audits[row['checkpoint']])
    result.update(auxiliary_losses_enabled=False, native_training_modified=False,
        paired_asset_chain_sha256=frozen_assets,
        diagnostic_sources_sha256={str(Path(p).resolve()): sha(p)
                                  for p in (__file__, original.__file__)})
    return result


if __name__ == '__main__':
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        if os.environ.get('CLT_CPU_USD_DIAG_READY'):
            raise RuntimeError('Installed CPU USD libraries did not load; no simulator was started')
        os.execve(sys.executable, [sys.executable, *sys.argv], cpu_usd_environment())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, action='append', default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh report path; preserve previous evidence')
    result = diagnose(args.replay, args.checkpoint)
    write_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), checkpoints=[dict(
        checkpoint=row['checkpoint'], actual_updates=row['actual_ppo_updates'],
        spans_m={v['fixed_state_case']: v['height_command_response_span_m'] for v in row['conditions']})
        for row in result['checkpoints']], closed_loop_acceptance=False), indent=2))
