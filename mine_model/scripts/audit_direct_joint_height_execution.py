"""Separate native direct-angle intent, motor following and contact projection.

Read immutable sampled native records only. No actor, Adam, simulator or RNG
is run. FK is a diagnostic projection and is never installed as a controller.
"""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from diagnose_plain_joint_pd_height import cpu_usd_environment, usd_joint_limits
from joint_pd_continuation import sha
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state
from wheel_legged_gym_isaaclab.leg_control import joint_position_targets
from wheel_legged_gym_isaaclab.robot_model import MODEL


def audit(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'direct_joint_height_execution_prefix_{through:04d}_readonly.json'
    if destination.exists():
        raise ValueError('Preserve an existing immutable diagnostic')
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    cfg = plan['training_config']
    if cfg['leg_control_mode'] != 'implicit_joint_position' or cfg['observation_space'] != 27:
        raise ValueError('Require the real plain27 direct joint-angle execution chain')
    limits = usd_joint_limits(cfg['robot']['spawn']['usd_path'], cfg['leg_joint_names'])
    signal = directory / 'native_learning_signal'
    fields = ('native_commands', 'native_height', 'native_gravity', 'native_speed', 'dones',
        'native_leg_targets', 'native_joint_positions', 'native_leg_lengths',
        'native_leg_references', 'native_raw_actions', 'mu', 'sigma', 'advantages')
    chunks = {key: [] for key in fields}
    pins = {str(plan_path): sha(plan_path), str(Path(__file__).resolve()): sha(__file__)}
    ids = None
    for index in range(1, through + 1):
        proof_path = signal / f'audit_update_{index:04d}.json'
        data_path = signal / f'rollout_{index:04d}.npz'
        proof = json.loads(proof_path.read_text())
        errors = proof['all_environment_alignment_max_errors']
        if (proof['update'] != index or proof['total_environments'] != plan['native_num_envs']
                or len(errors) != 9 or any(errors.values()) or proof['added_loss_or_backward']
                or not proof['probe_RNG_unchanged']):
            raise ValueError('Resolve native alignment before interpreting height')
        if ids is None:
            ids = proof['sampled_environment_ids']
        if ids != proof['sampled_environment_ids']:
            raise ValueError('Sample identity changed within this physical segment')
        with np.load(data_path, allow_pickle=False) as arrays:
            for key in fields:
                value = arrays[key]
                if value.shape[:2] != (48, len(ids)) or not np.isfinite(value).all():
                    raise ValueError('Invalid immutable native field: ' + key)
                chunks[key].append(value.copy())
        pins.update({str(p): sha(p) for p in (proof_path, data_path)})
    data = {key: np.concatenate(value) for key, value in chunks.items()}
    steps, count = data['native_height'].shape
    shape = (steps, count)
    gravity = torch.from_numpy(data['native_gravity'].reshape(-1, 3))
    raw = torch.from_numpy(data['native_raw_actions'].reshape(-1, 6))
    targets = torch.from_numpy(data['native_leg_targets'].reshape(-1, 4))
    joints = torch.from_numpy(data['native_joint_positions'].reshape(-1, 4))

    def projection(q):
        length, angle, _, valid = five_bar_state(q[:, ::2], q[:, 1::2], **cfg['five_bar_geometry'])
        x = -length * angle.sin()
        y = length.new_tensor([cfg['wheel_track_width'] / 2, -cfg['wheel_track_width'] / 2])
        z = MODEL['hip_z'] - length * angle.cos()
        height = cfg['wheel_radius'] + gravity[:, 0, None] * x + gravity[:, 1, None] * y + gravity[:, 2, None] * z
        if not valid.all() or not torch.isfinite(height).all():
            raise ValueError('Invalid diagnostic target or actual-joint FK')
        return height.numpy().reshape(*shape, 2), length.numpy().reshape(*shape, 2)

    reconstructed, _ = joint_position_targets(raw[:, (0, 1, 3, 4)],
        cfg['leg_joint_position_offset'], cfg['leg_joint_position_scale'],
        raw.new_tensor(limits).expand(len(raw), -1, -1), cfg['leg_joint_target_limit_margin'])
    target_height, target_length = projection(targets)
    actual_joint_height, actual_length = projection(joints)
    target_error = float((reconstructed - targets).abs().max())
    actual_fk_error = float(np.max(np.abs(actual_length - data['native_leg_lengths'])))
    target_fk_error = float(np.max(np.abs(target_length - data['native_leg_references'])))
    if target_error > 1.e-6 or max(actual_fk_error, target_fk_error) > 1.e-5:
        raise ValueError('CPU recomputation differs beyond the declared numerical tolerances')
    # A mean-policy target differs from the actual Gaussian action. Keep it
    # separate instead of attributing exploration displacement to motor PD.
    means = torch.from_numpy(data['mu'].reshape(-1, 6))
    mean_targets, _ = joint_position_targets(means[:, (0, 1, 3, 4)],
        cfg['leg_joint_position_offset'], cfg['leg_joint_position_scale'],
        raw.new_tensor(limits).expand(len(raw), -1, -1), cfg['leg_joint_target_limit_margin'])
    mean_height, _ = projection(mean_targets)
    target_mean, actual_mean, mean_policy = [h.mean(-1) for h in (target_height, actual_joint_height, mean_height)]
    observed = data['native_height']
    command = data['native_commands'][..., 2]
    motor = (targets - joints).numpy().reshape(*shape, 4)
    done = data['dones'][..., 0].astype(bool)
    age, command_age = np.zeros(shape), np.zeros(shape)
    dt = cfg['sim']['dt'] * cfg['decimation']
    age[0] = dt
    for t in range(1, steps):
        age[t] = np.where(done[t - 1], dt, age[t - 1] + dt)
        changed = np.any(np.abs(data['native_commands'][t] - data['native_commands'][t - 1]) > 1.e-7, axis=-1)
        command_age[t] = np.where(changed | done[t - 1], 0., command_age[t - 1] + dt)
    settled = (age > 2.) & (command_age >= 1.) & ~done
    static = np.broadcast_to(np.array(ids)[None, :] < plan['fixed_standing_environments'], shape)
    advantage = data['advantages'][..., 0]
    low, high = cfg['commands']['ranges_height']
    angle = np.degrees(np.arcsin(np.clip(data['native_gravity'][..., 0], -1., 1.)))

    def metrics(mask):
        n = int(mask.sum())
        if not n:
            return dict(samples=0)
        target_bias = target_mean[mask] - command[mask]
        following = actual_mean[mask] - target_mean[mask]
        projection_residual = observed[mask] - actual_mean[mask]
        if float(np.max(np.abs((target_bias + following + projection_residual)
                               - (observed[mask] - command[mask])))) > 2.e-7:
            raise ValueError('Signed height decomposition does not close')
        def describe(value):
            return dict(signed_mean=float(np.mean(value)), MAE=float(np.mean(np.abs(value))),
                        abs_P95=float(np.percentile(np.abs(value), 95)))
        noise = target_mean[mask] - mean_policy[mask]
        adv = advantage[mask]
        correlation = float(np.corrcoef(noise, adv)[0, 1]) if np.std(noise) and np.std(adv) else None
        by_env = []
        for column, env_id in enumerate(ids):
            selected = mask[:, column]
            if selected.sum() < 20:
                continue
            delta = target_mean[selected, column] - mean_policy[selected, column]
            a = advantage[selected, column]
            by_env.append(dict(environment_id=env_id, samples=int(selected.sum()),
                height_noise_advantage_correlation=float(np.corrcoef(delta, a)[0, 1])
                    if np.std(delta) and np.std(a) else None))
        return dict(samples=n, actual_height_error_m=describe(observed[mask] - command[mask]),
            applied_policy_target_height_error_m=describe(target_bias),
            mean_policy_target_height_error_m=describe(mean_policy[mask] - command[mask]),
            actual_joint_minus_target_height_m=describe(following),
            body_height_minus_actual_joint_projection_m=describe(projection_residual),
            active_joint_position_error_rad=describe(motor[mask]),
            mean_policy_height_noise_std_m=float(np.std(noise)),
            sampled_height_noise_advantage_correlation=correlation,
            sampled_height_noise_advantage_by_environment=by_env,
            raw_leg_action_clipping_fraction=float(np.any(np.abs(data['native_raw_actions'][..., (0, 1, 3, 4)]) > 1., axis=-1)[mask].mean()),
            mean_abs_pitch_deg=float(np.abs(angle[mask]).mean()))

    groups = {}
    for label, population in (('static', static), ('motion', ~static)):
        for height_name, height_mask in (('all_heights', np.ones(shape, dtype=bool)),
                ('low', np.isclose(command, low, atol=1.e-6, rtol=0.)),
                ('high', np.isclose(command, high, atol=1.e-6, rtol=0.))):
            groups[label + '_' + height_name] = metrics(settled & population & height_mask)
    result = dict(observed_at=datetime.now().astimezone().isoformat(), output=str(directory),
        committed_updates=through, sampled_environment_ids=ids, sampled_transitions=steps * count,
        native_num_envs=plan['native_num_envs'], USD_active_joint_limits_rad=limits,
        source_actual_updates=plan['source_actual_ppo_updates'],
        physical_segment_reset_age_reconstructed=True, source_pins=pins,
        raw_action_to_applied_motor_target_reconstruction_max_error_rad=target_error,
        CPU_actual_joint_FK_vs_recorded_native_FK_max_error_m=actual_fk_error,
        CPU_target_FK_vs_recorded_native_reference_max_error_m=target_fk_error,
        groups=groups, weights_updated=False, optimizer_loaded=False, simulator_started=False,
        training_modified=False, auxiliary_losses_enabled=False, policy_accepted=False,
        limitations=['32 environments with changing policies and Gaussian training actions; not independent acceptance.',
            'Target projection uses recorded post-physics gravity. No forward dynamics or stable contact proof.',
            'Projection residual includes contact height, wheel tilt and geometric approximation; it is not a torque estimate.',
            'Advantage correlations are observational, serially correlated and sampled; no causal or significance claim.',
            'Physical reset age is reconstructed separately inside this simulator segment, never across the reboot.'])
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return destination, result


if __name__ == '__main__':
    try:
        from pxr import Usd
    except ImportError:
        if os.environ.get('CLT_CPU_USD_DIAG_READY'):
            raise RuntimeError('CPU USD libraries did not load; no Kit fallback')
        os.execve(sys.executable, [sys.executable, *sys.argv], cpu_usd_environment())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--through', type=int, required=True)
    arguments = parser.parse_args()
    path, report = audit(arguments.directory, arguments.through)
    print(json.dumps(dict(report=str(path), groups={k: {field: value for field, value in row.items()
        if field != 'sampled_height_noise_advantage_by_environment'} for k, row in report['groups'].items()}), ensure_ascii=False))
