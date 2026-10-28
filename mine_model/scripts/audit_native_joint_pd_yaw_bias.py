"""Read signed yaw errors and mean/sample references from committed rollouts."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output, through):
    output = Path(output).resolve()
    destination = output / f'native_yaw_bias_prefix_{through:04d}_readonly.json'
    if destination.exists():
        raise ValueError('Preserve the existing immutable yaw diagnostic')
    plan_path = output / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    cfg = plan['training_config']
    if not plan['ordinary_full_policy_PPO'] or cfg['leg_control_mode'] != 'implicit_joint_position':
        raise ValueError('Require the original direct-joint PPO execution contract')
    directory = output / 'native_learning_signal'
    if not ((directory / f'audit_update_{through + 1:04d}.json').exists()
            or through == 200 and (directory / 'completed_external.json').exists()):
        raise ValueError('Wait for the complete committed prefix')
    fields = ('native_commands', 'native_yaw', 'mu', 'native_raw_actions', 'dones')
    chunks, ids, pins = {key: [] for key in fields}, None, {str(plan_path): sha(plan_path)}
    for index in range(1, through + 1):
        proof_path = directory / f'audit_update_{index:04d}.json'
        data_path = proof_path.with_name(f'rollout_{index:04d}.npz')
        proof = json.loads(proof_path.read_text())
        if (proof['update'] != index or proof['total_environments'] != plan['native_num_envs']
                or len(proof['all_environment_alignment_max_errors']) != 9
                or any(proof['all_environment_alignment_max_errors'].values())
                or not proof['probe_RNG_unchanged'] or proof['added_loss_or_backward']):
            raise ValueError('Resolve native observation/action/reward alignment first')
        if ids is None:
            ids = proof['sampled_environment_ids']
        if proof['sampled_environment_ids'] != ids:
            raise ValueError('Recorded sample identities changed')
        with np.load(data_path, allow_pickle=False) as arrays:
            for key in fields:
                value = arrays[key]
                if value.shape[:2] != (48, len(ids)) or not np.isfinite(value).all():
                    raise ValueError('Invalid original native array: ' + key)
                chunks[key].append(value.copy())
        pins.update({str(p): sha(p) for p in (proof_path, data_path)})
    data = {key: np.concatenate(value) for key, value in chunks.items()}
    commands, body, done = data['native_commands'], data['native_yaw'], data['dones'][..., 0].astype(bool)
    dt = cfg['sim']['dt'] * cfg['decimation']
    age, command_age = np.zeros(body.shape), np.zeros(body.shape)
    age[0] = dt
    for step in range(1, len(body)):
        age[step] = np.where(done[step - 1], dt, age[step - 1] + dt)
        changed = np.any(abs(commands[step] - commands[step - 1]) > 1.e-7, axis=-1)
        command_age[step] = np.where(changed | done[step - 1], 0., command_age[step - 1] + dt)
    settled = (age > 2.) & (command_age >= 1.) & ~done
    static = np.broadcast_to(np.array(ids)[None] < plan['fixed_standing_environments'], body.shape)
    vx, yaw, height = [commands[..., index].astype(np.float64) for index in range(3)]
    populations = dict(dedicated_static=static,
        pure_turn_left=(abs(vx) < .01) & (yaw > .02),
        pure_turn_right=(abs(vx) < .01) & (yaw < -.02),
        forward_no_yaw=(vx > .05) & (abs(yaw) < .01),
        reverse_no_yaw=(vx < -.05) & (abs(yaw) < .01),
        translation_with_yaw=(abs(vx) > .05) & (abs(yaw) > .02))
    low, high = cfg['commands']['ranges_height']
    low_mask, high_mask = [np.isclose(height, endpoint, atol=1.e-6, rtol=0.) for endpoint in (low, high)]
    heights = dict(all=np.ones(body.shape, bool), low=low_mask,
                   interior=~low_mask & ~high_mask, high=high_mask)
    row_index = np.arange(len(body))[:, None]
    windows = dict(entire_prefix=np.ones(body.shape, bool))
    if through >= 40:
        windows.update(first20=np.broadcast_to(row_index < 20 * 48, body.shape),
            last20=np.broadcast_to(row_index >= (through - 20) * 48, body.shape))
    scale = cfg['action_scale_vel'] * cfg['wheel_radius'] / cfg['wheel_track_width']
    mean, sampled = [np.clip(data[key][..., (2, 5)].astype(np.float64), -1., 1.)
                     for key in ('mu', 'native_raw_actions')]
    mean_proxy = scale * (mean[..., 1] - mean[..., 0])
    sample_proxy = scale * (sampled[..., 1] - sampled[..., 0])
    error = body.astype(np.float64) - yaw
    components = (mean_proxy - yaw, sample_proxy - mean_proxy, body - sample_proxy)
    closure = float(abs(sum(components) - error).max())
    if closure > 1.e-12:
        raise ValueError('Signed policy/sample/body decomposition does not close')
    groups = {}
    for window, window_mask in windows.items():
        groups[window] = {}
        for label, population in populations.items():
            groups[window][label] = {}
            for height_label, height_mask in heights.items():
                mask = window_mask & settled & population & height_mask
                if not mask.any():
                    continue
                e = error[mask]
                groups[window][label][height_label] = dict(samples=int(mask.sum()),
                    yaw_command_signed_mean=float(yaw[mask].mean()),
                    actual_yaw_signed_mean=float(body[mask].mean()),
                    yaw_signed_error_mean=float(e.mean()), yaw_MAE=float(abs(e).mean()),
                    yaw_error_std=float(e.std()),
                    policy_mean_reference_proxy_minus_command=float(components[0][mask].mean()),
                    sampled_reference_proxy_minus_mean_proxy=float(components[1][mask].mean()),
                    measured_body_minus_sampled_reference_proxy=float(components[2][mask].mean()))
    pins[str(Path(__file__).resolve())] = sha(__file__)
    if any(sha(path) != digest for path, digest in pins.items()):
        raise ValueError('Immutable input changed while diagnosing yaw')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_updates=through,
        native_num_envs=plan['native_num_envs'], sampled_environment_ids=ids,
        sampled_transitions=int(np.prod(body.shape)), groups=groups,
        signed_decomposition_closure_max_error=closure, source_pins=pins,
        RNG_draws_added=0, training_modified=False, optimizer_executed=False,
        simulator_started=False, policy_accepted=False, local_optimum_proven=False,
        limitations=[
            '32 selected native environments and evolving sampled-Gaussian policies; not a fixed-model independent replay.',
            'Groups and height mixtures differ across windows and can contain correlated samples. No causal improvement or statistical guarantee is established.',
            'Mean and sampled wheel-reference rolling proxies are not measured wheel velocities.',
            'Body-minus-sampled-reference includes wheel servo dynamics, filtering, contact, geometry and leg motion; it does not isolate a PD bug or slip.',
            'The physical-age/command-age filter is diagnostic only. Every unsafe training event remains in the separate all4096 terminal trace.'])
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return destination, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--through', type=int, required=True)
    args = parser.parse_args()
    path, result = audit(args.output, args.through)
    print(json.dumps(dict(report=str(path), committed_updates=result['committed_updates'],
        closure_error=result['signed_decomposition_closure_max_error'], training_modified=False)))
