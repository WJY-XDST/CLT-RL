"""Read-only summaries of committed native PPO observations, never acceptance.

The all-environment alignment checks and the 32-environment physical sample
have deliberately separate scopes. No simulator, policy or optimizer is run.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output, through):
    output = Path(output).resolve()
    signal = output / 'native_learning_signal'
    plan_path = output / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    records, arrays, pins = [], {}, {str(plan_path): sha(plan_path)}
    env_ids = None
    alignment = {}
    fields = ('native_commands', 'native_height', 'native_gravity', 'native_speed',
              'native_yaw', 'dones', 'native_timeouts', 'actions', 'mu', 'sigma')
    for update in range(1, through + 1):
        data_path = signal / f'rollout_{update:04d}.npz'
        proof_path = signal / f'audit_update_{update:04d}.json'
        if not proof_path.exists():
            raise ValueError(f'Update {update} is not committed')
        proof = json.loads(proof_path.read_text())
        ids = proof['sampled_environment_ids']
        if env_ids is None:
            env_ids = ids
        if ids != env_ids or proof['total_environments'] != plan['native_num_envs']:
            raise ValueError('Sample identity or native population changed')
        if proof['added_loss_or_backward'] or not proof['probe_RNG_unchanged']:
            raise ValueError('Require unchanged observation-only native instrumentation')
        errors = proof['all_environment_alignment_max_errors']
        if len(errors) != 9 or any(v != 0. for v in errors.values()):
            raise ValueError('Resolve native alignment errors before interpreting tracking')
        for key, value in errors.items():
            alignment[key] = max(alignment.get(key, 0.), value)
        with np.load(data_path, allow_pickle=False) as data:
            for key in fields:
                value = data[key]
                if value.shape[:2] != (48, len(ids)) or not np.isfinite(value).all():
                    raise ValueError('Invalid physical sample: ' + key)
                arrays.setdefault(key, []).append(value.copy())
        records.append(update)
        pins.update({str(data_path): sha(data_path), str(proof_path): sha(proof_path)})
    if not records:
        raise ValueError('No committed records')
    data = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    dt = plan['training_config']['sim']['dt'] * plan['training_config']['decimation']
    steps, count = data['native_height'].shape
    done = data['dones'][..., 0].astype(bool)
    timeout = data['native_timeouts'].astype(bool)
    age = np.empty((steps, count), dtype=np.float64)
    stable_command_age = np.empty_like(age)
    # PPO may randomize episode-length counters before learning. Real physical
    # age is instead reconstructed from this continuous pre-reset trace.
    age[0] = dt
    stable_command_age[0] = 0.
    for t in range(1, steps):
        age[t] = np.where(done[t - 1], dt, age[t - 1] + dt)
        changed = np.any(np.abs(data['native_commands'][t] - data['native_commands'][t - 1]) > 1.e-7, axis=-1)
        stable_command_age[t] = np.where(changed | done[t - 1], 0., stable_command_age[t - 1] + dt)
    command = data['native_commands']
    gravity = data['native_gravity']
    pitch = np.degrees(np.arcsin(np.clip(gravity[..., 0], -1., 1.)))
    roll = np.degrees(np.arctan2(-gravity[..., 1], -gravity[..., 2]))
    error = np.abs(data['native_height'] - command[..., 2])
    speed_error = np.abs(data['native_speed'] - command[..., 0])
    yaw_error = np.abs(data['native_yaw'] - command[..., 1])
    static = np.broadcast_to(np.array(env_ids)[None, :] < plan['fixed_standing_environments'], (steps, count))
    settled = (age > 2.) & (stable_command_age >= 1.) & ~done
    sigma = data['sigma']
    leg_floor = plan['exploration_guard']['minimum_leg_std']
    wheel_floor = plan['exploration_guard']['minimum_wheel_std']
    if (sigma[..., [0, 1, 3, 4]].min() < leg_floor - 1.e-7
            or sigma[..., [2, 5]].min() < wheel_floor - 1.e-7):
        raise ValueError('Native exploration fell below the declared floors')

    def metrics(mask):
        n = int(mask.sum())
        if not n:
            return {'samples': 0}
        return dict(samples=n, height_MAE_m=float(error[mask].mean()),
                    height_relative_MAE=float((error / command[..., 2])[mask].mean()),
                    speed_MAE_m_s=float(speed_error[mask].mean()),
                    yaw_MAE_rad_s=float(yaw_error[mask].mean()),
                    abs_pitch_P95_deg=float(np.percentile(np.abs(pitch[mask]), 95)),
                    abs_roll_P95_deg=float(np.percentile(np.abs(roll[mask]), 95)),
                    body_max_angle_P95_deg=float(np.percentile(np.maximum(np.abs(pitch), np.abs(roll))[mask], 95)),
                    below90pct_command_height_fraction=float((data['native_height'][mask] < .9 * command[..., 2][mask]).mean()))

    groups = {}
    low, high = plan['full_height_range']
    windows = {'entire_prefix': np.ones((steps, count), dtype=bool)}
    if through >= 40:
        first, last = np.zeros_like(settled), np.zeros_like(settled)
        first[:20 * 48] = True
        last[-20 * 48:] = True
        windows.update(first20_updates=first, last20_updates=last)
    for name, window in windows.items():
        groups[name] = {}
        for group, group_mask in (('dedicated_static', static), ('motion_population', ~static)):
            mask = window & group_mask
            groups[name][group] = dict(all_sampled_transitions=metrics(mask),
                settled_sampled_transitions=metrics(mask & settled),
                low_height_settled=metrics(mask & settled & np.isclose(command[..., 2], low, rtol=0., atol=1.e-6)),
                high_height_settled=metrics(mask & settled & np.isclose(command[..., 2], high, rtol=0., atol=1.e-6)),
                sampled_ended_episodes=int((done & mask).sum()),
                sampled_non_timeout_terminals=int((done & ~timeout & mask).sum()),
                clipped_wheel_action_fraction=float(np.any(np.abs(data['actions'][..., [2, 5]]) > 1., axis=-1)[mask].mean()))
    report = dict(observed_at=datetime.now().astimezone().isoformat(),
        committed_updates=through, native_environment_count=plan['native_num_envs'],
        all_environment_alignment_max_errors=alignment,
        observer_RNG_unchanged=True, added_loss_or_backward=False,
        sampled_environment_ids=env_ids, sampled_physical_transitions=steps * count,
        reconstructed_physical_age_instead_of_randomized_initial_episode_counter=True,
        settled_filter='Physical reset age >2s, applied command unchanged >=1s, nonterminal',
        initial_clock_warmup_offset_possible=True,
        native_leg_std_range=[float(sigma[..., [0, 1, 3, 4]].min()), float(sigma[..., [0, 1, 3, 4]].max())],
        native_wheel_std_range=[float(sigma[..., [2, 5]].min()), float(sigma[..., [2, 5]].max())],
        groups=groups, source_sha256=pins, script_sha256=sha(__file__),
        training_modified=False, simulator_started=False, optimizer_loaded=False,
        policy_accepted=False,
        limitations=['Physical tracking summaries cover32 sampled environments, not all4096.',
                    'Training rollouts include Gaussian exploration and changing policies/commands. These are not independent deterministic replay or acceptance results.',
                    'First/last windows need not have matched command distributions. Their differences do not by themselves prove improvement or regression.',
                    'Non-timeout terminal counts exclude overlapping timeout/failure flags and are not a substitute for the full-environment terminal event trace.'])
    target = output / f'native_motion_training_prefix_{through:04d}_readonly.json'
    if target.exists():
        raise ValueError('Preserve the existing immutable prefix report')
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    return target, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--through', type=int, required=True)
    args = parser.parse_args()
    path, result = audit(args.output, args.through)
    print(json.dumps(dict(report=str(path), committed_updates=result['committed_updates'],
        sampled_physical_transitions=result['sampled_physical_transitions'],
        all_environment_alignment_max_errors=result['all_environment_alignment_max_errors'],
        latest_window=result['groups'].get('last20_updates', result['groups']['entire_prefix'])), ensure_ascii=False))
