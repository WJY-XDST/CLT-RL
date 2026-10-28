"""Measure executed motor-target and FK length exploration from saved native samples.

USD is read offline. The original direct-joint map and FK are telemetry only:
no simulator, policy, optimizer, new random samples or controller is run.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
USD_LIBRARY = next((ROOT / 'IsaacLab/_isaac_sim/extscache').glob('omni.usd.libs-*/pxr')).parent
if os.environ.get('CLT_USD_NATIVE_READY') != '1':
    environment = dict(os.environ, CLT_USD_NATIVE_READY='1')
    environment['LD_LIBRARY_PATH'] = str(USD_LIBRARY / 'bin') + ':' + environment.get('LD_LIBRARY_PATH', '')
    os.execvpe(sys.executable, [sys.executable, *sys.argv], environment)
sys.path.insert(0, str(USD_LIBRARY))

import numpy as np
import torch
from pxr import Usd, UsdPhysics
import wheel_legged_gym_isaaclab.leg_control as mapping
import wheel_legged_gym_isaaclab.five_bar_vmc as geometry

SCRIPT = Path(__file__).resolve()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(directory, through, destination=None):
    directory = Path(directory).resolve()
    destination = (Path(destination).resolve() if destination is not None else
        directory / f'joint_target_geometric_exploration_prefix_{through:04d}_readonly.json')
    if destination.exists() or not 1 <= through <= 200:
        raise ValueError('Use a fresh diagnostic of a committed original200-update prefix')
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    cfg = plan['training_config']
    if (cfg['leg_control_mode'] != mapping.DIRECT_JOINT_POSITION_MODE
            or cfg['observation_space'] != 27 or plan['native_num_envs'] != 4096
            or cfg['leg_joint_names'] != ['LB_joint', 'LL_joint', 'RB_joint', 'RL_joint']):
        raise ValueError('Require the original plain27 direct-joint action contract')
    pins = {str(p): sha(p) for p in (SCRIPT, plan_path, Path(mapping.__file__), Path(geometry.__file__))}
    asset = Path(cfg['robot']['spawn']['usd_path'])
    stage = Usd.Stage.Open(str(asset))
    limits, joint_paths = [], []
    for name in cfg['leg_joint_names']:
        matches = [p for p in stage.Traverse() if p.GetName() == name and p.IsA(UsdPhysics.RevoluteJoint)]
        if len(matches) != 1:
            raise ValueError('Require one composed original motor joint: ' + name)
        joint = UsdPhysics.RevoluteJoint(matches[0])
        limits.append([math.radians(joint.GetLowerLimitAttr().Get()),
                       math.radians(joint.GetUpperLimitAttr().Get())])
        joint_paths.append(str(matches[0].GetPath()))
    for layer in stage.GetUsedLayers():
        if not layer.identifier.startswith('anon:'):
            pins[str(Path(layer.realPath))] = sha(layer.realPath)
    fields = ('native_raw_actions', 'mu', 'native_leg_targets', 'native_leg_references', 'native_commands', 'dones')
    chunks, ids = {key: [] for key in fields}, None
    for index in range(1, through + 1):
        proof_path = directory / f'native_learning_signal/audit_update_{index:04d}.json'
        data_path = proof_path.with_name(f'rollout_{index:04d}.npz')
        proof = json.loads(proof_path.read_text())
        if (proof['update'] != index or proof['total_environments'] != 4096
                or len(proof['all_environment_alignment_max_errors']) != 9
                or any(proof['all_environment_alignment_max_errors'].values())
                or not proof['probe_RNG_unchanged'] or proof['added_loss_or_backward']):
            raise ValueError('Resolve native4096 alignment before interpreting geometry exploration')
        if ids is None:
            ids = proof['sampled_environment_ids']
        if ids != proof['sampled_environment_ids']:
            raise ValueError('Native sample identities changed')
        with np.load(data_path, allow_pickle=False) as data:
            for key in fields:
                value = data[key]
                if value.shape[:2] != (48, len(ids)) or not np.isfinite(value).all():
                    raise ValueError('Require finite, complete original native telemetry: ' + key)
                chunks[key].append(value.copy())
        pins.update({str(p): sha(p) for p in (proof_path, data_path)})
    data = {key: np.concatenate(value) for key, value in chunks.items()}
    shape = data['native_commands'].shape[:2]
    rng = torch.random.get_rng_state().clone()
    raw, means = [torch.from_numpy(data[key][..., mapping.LEG_ACTION_COLUMNS].reshape(-1, 4))
                  for key in ('native_raw_actions', 'mu')]
    boundaries = torch.tensor(limits, dtype=raw.dtype).expand(raw.shape[0], -1, -1)

    def project(actions):
        return mapping.joint_position_targets(actions, cfg['leg_joint_position_offset'],
            cfg['leg_joint_position_scale'], boundaries, cfg['leg_joint_target_limit_margin'])[0]

    target, deterministic = project(raw), project(means)
    length, _, _, valid = geometry.five_bar_state(target[:, ::2], target[:, 1::2], **cfg['five_bar_geometry'])
    mean_length, _, _, mean_valid = geometry.five_bar_state(
        deterministic[:, ::2], deterministic[:, 1::2], **cfg['five_bar_geometry'])
    target_parity = float((target - torch.from_numpy(data['native_leg_targets'].reshape(-1, 4))).abs().max())
    reference_parity = float((length - torch.from_numpy(data['native_leg_references'].reshape(-1, 2))).abs().max())
    if target_parity > 1.e-7 or reference_parity > 2.e-7:
        raise ValueError('Original USD/motor mapping/FK does not reproduce saved native targets/references')
    raw_q_noise = ((raw - means) * raw.new_tensor(cfg['leg_joint_position_scale'])).numpy().reshape(*shape, 4)
    q_noise = (target - deterministic).numpy().reshape(*shape, 4)
    length_noise = (length - mean_length).numpy().reshape(*shape, 2)
    validity = (valid & mean_valid).numpy().reshape(*shape, 2)
    commands, dones = data['native_commands'], data['dones'][..., 0].astype(bool)
    dt = cfg['sim']['dt'] * cfg['decimation']
    age, command_age = np.zeros(shape), np.zeros(shape)
    age[0] = dt
    for row in range(1, shape[0]):
        age[row] = np.where(dones[row - 1], dt, age[row - 1] + dt)
        changed = np.any(abs(commands[row] - commands[row - 1]) > 1.e-7, axis=-1)
        command_age[row] = np.where(changed | dones[row - 1], 0., command_age[row - 1] + dt)
    settled = (age > 2.) & (command_age >= 1.) & ~dones
    static = np.broadcast_to(np.array(ids)[None] < plan['fixed_standing_environments'], shape)
    low, high = cfg['commands']['ranges_height']
    populations = dict(all_settled=settled, dedicated_static=settled & static,
        static_low=settled & static & np.isclose(commands[..., 2], low, atol=1.e-6, rtol=0.),
        static_high=settled & static & np.isclose(commands[..., 2], high, atol=1.e-6, rtol=0.),
        motion_forward=settled & ~static & (commands[..., 0] > .05),
        motion_reverse=settled & ~static & (commands[..., 0] < -.05))
    windows = dict(entire_prefix=np.ones(shape, bool))
    if through >= 40:
        rows = np.arange(shape[0])[:, None]
        windows.update(first20=np.broadcast_to(rows < 20 * 48, shape),
            last20=np.broadcast_to(rows >= (through - 20) * 48, shape))
    groups = {}
    for window, selection in windows.items():
        groups[window] = {}
        for name, population in populations.items():
            mask = population & selection
            if not mask.any():
                continue
            q, unbounded_q, delta = [array[mask].astype(np.float64) for array in (q_noise, raw_q_noise, length_noise)]
            groups[window][name] = dict(samples=int(mask.sum()),
                raw_motor_target_noise_RMS_rad=np.sqrt((unbounded_q**2).mean(0)).tolist(),
                executed_motor_target_noise_RMS_rad=np.sqrt((q**2).mean(0)).tolist(),
                exactly_zero_executed_motor_noise_fraction=(abs(q) < 1.e-8).mean(0).tolist(),
                FK_length_departure_from_executed_mean_RMS_mm=(1000 * np.sqrt((delta**2).mean(0))).tolist(),
                FK_length_departure_mean_mm=(1000 * delta.mean(0)).tolist(),
                FK_length_departure_centered_std_mm=(1000 * delta.std(0)).tolist(),
                FK_common_length_departure_RMS_mm=float(1000 * np.sqrt((delta.mean(-1)**2).mean())),
                FK_differential_length_departure_RMS_mm=float(1000 * np.sqrt(((delta[:, 1] - delta[:, 0])**2).mean())),
                exactly_zero_FK_length_departure_fraction=(abs(delta) < 1.e-8).mean(0).tolist(),
                sampled_or_mean_invalid_FK_fraction=(~validity[mask]).mean(0).tolist())
    if not torch.equal(rng, torch.random.get_rng_state()) or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Read-only geometric audit changed RNG or immutable native sources')
    report = dict(observed_at=datetime.now().astimezone().isoformat(), committed_updates=through,
        training_output=str(directory), native_num_envs=4096, sampled_environment_ids=ids,
        motor_order=cfg['leg_joint_names'], length_order=['left', 'right'],
        USD_joint_paths=joint_paths, USD_composed_joint_limits_rad=limits,
        original_native_motor_target_parity_max_error_rad=target_parity,
        original_native_FK_reference_parity_max_error_m=reference_parity,
        original_mapper_and_FK_reused_without_changes=True, groups=groups, source_pins=pins,
        RNG_draws_added=0, RNG_state_unchanged=True, simulator_started=False,
        policy_or_optimizer_loaded=False, optimizer_executed=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False, limitations=[
            '32 retained native environments from evolving policies; not the entire4096 physical-state distribution or independent acceptance.',
            'Motor and virtual-length target departures are measured relative to the executed deterministic mean, not centered policy moments or measured base-height motion.',
            'USD hard limits, raw action clipping and the exact original FK are applied. No IK or virtual-leg feedback is added to control.',
            'Invalid FK samples are counted without exclusion. Actual dynamics, support contact, pitch and PD response can further change physical exploration.',
            'No noise magnitude or local optimum is judged solely from these target-space statistics.'])
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return destination, report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    cli.add_argument('--destination', type=Path)
    args = cli.parse_args()
    path, result = audit(args.directory, args.through, args.destination)
    print(json.dumps(dict(report=str(path), motor_target_parity=result['original_native_motor_target_parity_max_error_rad'],
        FK_reference_parity=result['original_native_FK_reference_parity_max_error_m'],
        latest_window=result['groups'].get('last20', result['groups']['entire_prefix']), training_modified=False)))
