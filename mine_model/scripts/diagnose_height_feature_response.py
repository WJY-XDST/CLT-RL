"""Read-only fixed-state response of legacy and encoder/IMU-feature policies.

External observations remain 27-D. The feature policy has 29 internal first
layer columns, which must be evaluated rather than silently ignored.
"""
import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

import torch
from tensordict import TensorDict
import yaml

from auto_train_mine import write_json
import diagnose_height_response as original
import diagnose_plain_joint_pd_height as plain
from joint_pd_continuation import ROOT, optimizer_step, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain
from wheel_legged_gym_isaaclab.leg_control import require_matching_policy_contract
from wheel_legged_gym_isaaclab.robot_model import MODEL, validate_checkpoint_robot

sys.path.insert(0, str(ROOT / 'mine_model/experiments'))
from joint_pd_height_features_candidate import EncoderImuHeightError, JointPdHeightFeatureActorCritic
from joint_pd_height_feature_runtime import require_paired_feature_model


def actor(value, state):
    if state['actor.0.weight'].shape[1] == 27:
        return original.actor(value, state)
    shape = value.shape
    flat = value.reshape(-1, 27)
    features = EncoderImuHeightError(state['_encoder_imu_model']).to(device=value.device, dtype=value.dtype)
    weight, bias = state['actor.0.weight'], state['actor.0.bias']
    hidden = (torch.nn.functional.linear(flat, weight[:, :27].contiguous(), bias)
              + torch.nn.functional.linear(features(flat), weight[:, 27:].contiguous()))
    hidden = torch.nn.functional.elu(hidden)
    for index in (2, 4, 6):
        hidden = torch.nn.functional.linear(hidden, state[f'actor.{index}.weight'], state[f'actor.{index}.bias'])
        if index != 6:
            hidden = torch.nn.functional.elu(hidden)
    return hidden.reshape(*shape[:-1], 6)


def diagnose(replay, checkpoints):
    replay = Path(replay).resolve()
    manifest = json.loads((replay / 'manifest.json').read_text())
    completed = json.loads((replay / 'completed.json').read_text())
    expected = round(manifest['seconds_per_case'] / manifest['policy_dt']) * len(manifest['cases']) * len(manifest['seeds'])
    if not completed['completed'] or completed['resets'] or completed['samples'] != expected:
        raise ValueError('Use a complete surviving canonical fixed-input replay')
    baseline = Path(manifest['checkpoint']).resolve()
    source_cfg = validate_checkpoint_robot(baseline)
    frozen_assets = manifest['asset_chain_sha256']
    audits = {}

    def load(path):
        path = Path(path).resolve()
        cfg = validate_checkpoint_robot(path)
        require_matching_policy_contract(source_cfg, cfg)
        hashes, _ = verify_asset_chain(cfg['robot']['spawn']['usd_path'])
        if hashes != frozen_assets:
            raise ValueError('Keep the fixed replay physical asset chain')
        agent = yaml.safe_load((path.parent / 'params/agent.yaml').read_text())
        plain.require_plain_ppo(agent)
        saved = torch.load(path, map_location='cpu', weights_only=True)
        actual = saved_updates(path)
        if (not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
                or optimizer_step(saved) != 20 * actual):
            raise ValueError('Read only finite complete actor/critic/std/17-Adam checkpoints')
        width = saved['model_state_dict']['actor.0.weight'].shape[1]
        state = dict(saved['model_state_dict'])
        parity = None
        if width == 29:
            model = require_paired_feature_model(saved, cfg, MODEL['hip_z'])
            if agent['policy']['class_name'] != 'JointPdHeightFeatureActorCritic':
                raise ValueError('Feature metadata and declared policy class differ')
            state['_encoder_imu_model'] = model
            # Check the diagnostic callback against the actual declared policy,
            # including nonzero learned columns and non-nominal height commands.
            inputs = torch.zeros(32, 27)
            inputs[:, 5] = -1.
            inputs[:, 8] = torch.linspace(.22534645531, .37534645531, 32) * cfg['obs_scales']['height_measurements']
            inputs[:, 9:13] = .02
            batch = TensorDict({'policy': inputs}, batch_size=[32])
            kwargs = dict(agent['policy'])
            kwargs.pop('class_name')
            with torch.random.fork_rng(devices=[]), contextlib.redirect_stdout(io.StringIO()):
                network = JointPdHeightFeatureActorCritic(batch, agent['obs_groups'], 6,
                    height_feature_model=model, **kwargs)
            network.load_state_dict(saved['model_state_dict'], strict=True)
            with torch.no_grad():
                parity = float((network.act_inference(batch) - actor(inputs, state)).abs().max())
            if parity > 1.e-7:
                raise ValueError('Feature diagnostic does not reproduce its declared policy')
        elif width != 27 or agent['policy']['class_name'] != 'ActorCritic':
            raise ValueError('Unsupported or misdeclared first-layer architecture')
        limits = plain.usd_joint_limits(cfg['robot']['spawn']['usd_path'], cfg['leg_joint_names'])
        audits[str(path)] = dict(actual_ppo_updates=actual, Adam_step=optimizer_step(saved),
            optimizer_states=17, learned_std=saved['model_state_dict']['std'].tolist(),
            first_layer_internal_columns=width, external_observation_size=27,
            actual_declared_policy_callback_max_error=parity,
            USD_joint_limits_rad=limits, geometry_source='Paired env and composed USD',
            auxiliary_losses_enabled=False)
        return state, plain.geometry(cfg, limits), int(saved['iter'])

    spec = importlib.util.spec_from_file_location('_feature_height_read_only', original.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_checkpoint, module.actor = load, actor
    result = module.diagnose(replay, checkpoints)
    for row in result['checkpoints']:
        row.update(audits[row['checkpoint']])
    sources = [Path(__file__), Path(original.__file__), Path(plain.__file__),
               ROOT / 'mine_model/experiments/joint_pd_height_features_candidate.py',
               ROOT / 'mine_model/experiments/joint_pd_height_feature_runtime.py']
    result.update(auxiliary_losses_enabled=False, native_training_modified=False,
        paired_asset_chain_sha256=frozen_assets,
        diagnostic_sources_sha256={str(p.resolve()): sha(p) for p in sources})
    return result


if __name__ == '__main__':
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        if os.environ.get('CLT_CPU_USD_DIAG_READY'):
            raise RuntimeError('CPU USD libraries unavailable; no simulator was started')
        os.execve(sys.executable, [sys.executable, *sys.argv], plain.cpu_usd_environment())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, action='append', default=[])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh path and preserve prior evidence')
    result = diagnose(args.replay, args.checkpoint)
    write_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), checkpoints=[dict(
        checkpoint=row['checkpoint'], actual_updates=row['actual_ppo_updates'],
        internal_columns=row['first_layer_internal_columns'],
        spans_mm={v['fixed_state_case']: 1000 * v['height_command_response_span_m'] for v in row['conditions']})
        for row in result['checkpoints']], closed_loop_acceptance=False), indent=2))
