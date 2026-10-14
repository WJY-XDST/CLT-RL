"""Diagnose exact frozen-policy feedback from NPZ inputs, without a simulator.

These are pre-step policy inputs, not independent post-physics measurements.
Final tracking acceptance remains the separate multi-seed trace assessment.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def wheel_spectrum(reference, dt):
    """High-frequency energy alone is not an alert without real amplitude."""
    reference = np.asarray(reference, dtype=float)
    centered = reference - reference.mean()
    power = abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(reference), dt)
    power[0] = 0.
    total = power.sum()
    high = frequencies >= .8 * (.5 / dt)
    return dict(
        std_rad_s=float(centered.std()),
        delta_rms_rad_s=float(np.sqrt(np.mean(np.diff(reference) ** 2))),
        second_difference_rms_rad_s=float(np.sqrt(np.mean(np.diff(reference, n=2) ** 2))),
        peak_hz=float(frequencies[power.argmax()]) if total > 1.e-16 else 0.,
        high_frequency_power_fraction=float(power[high].sum() / total) if total > 1.e-16 else 0.,
    )


def joint_pd_feedback(observations, means, manifest, cfg, state):
    """Exact actor parity, measured bias and fixed-state command sensitivity.

    A counterfactual changes one input while holding all other observed state
    fixed. It is not a closed-loop plant test or proof of stability.
    """
    import torch
    if cfg.get('leg_control_mode') != 'implicit_joint_position':
        raise ValueError('Joint-PD diagnosis requires its paired motor-angle configuration')
    o, mu = np.asarray(observations), np.asarray(means)
    if o.shape[:-1] != mu.shape[:-1] or o.shape[-1] != 27 or mu.shape[-1] != 6:
        raise ValueError('Expected paired 27-D exact observations and 6-D actor means')
    if not np.isfinite(o).all() or not np.isfinite(mu).all():
        raise ValueError('Nonfinite exact actor record')
    def actor(value):
        for index in (0, 2, 4, 6):
            value = torch.nn.functional.linear(value, state[f'actor.{index}.weight'],
                                              state[f'actor.{index}.bias'])
            if index != 6:
                value = torch.nn.functional.elu(value)
        return value
    tail = o[-round(3. / manifest['policy_dt']):].astype(np.float32)
    raw = mu[-len(tail):]
    with torch.no_grad():
        parity = float((actor(torch.from_numpy(tail)) - torch.from_numpy(raw)).abs().max())
    if parity > 1.e-4:
        raise ValueError('Saved actor does not reproduce exact inference means')
    scale, rewards = cfg['obs_scales'], cfg['rewards']
    rows = []
    for case_id, case in enumerate(manifest['cases']):
        selection = slice(case_id * len(manifest['seeds']), (case_id + 1) * len(manifest['seeds']))
        inputs = tail[:, selection].reshape(-1, 27)
        actions = raw[:, selection].reshape(-1, 6)
        yaw = inputs[:, 2] / scale['ang_vel']
        error = inputs[:, 7] / scale['ang_vel'] - yaw
        orientation = inputs[:, 3]**2 + rewards['orientation_roll_multiplier'] * inputs[:, 4]**2
        wheels = actions[:, (2, 5)].clip(-1., 1.) * cfg['action_scale_vel']
        counterfactual = []
        for command in (-.3, 0., .3):
            changed = torch.from_numpy(inputs.copy())
            changed[:, 7] = command * scale['ang_vel']
            with torch.no_grad():
                reference = actor(changed)[:, (2, 5)].clamp(-1., 1.) * cfg['action_scale_vel']
            counterfactual.append(dict(yaw_command_rad_s=command,
                left_right_reference_rad_s=reference.mean(0).tolist(),
                right_minus_left_reference_rad_s=float((reference[:, 1] - reference[:, 0]).mean())))
        rows.append(dict(name=case['name'], seed_average=True,
            yaw_command_rad_s=case['yaw'], yaw_rate_mean_rad_s=float(yaw.mean()),
            speed_mean_m_s=float((inputs[:, 17] / scale['lin_vel']).mean()),
            pitch_mean_deg=float(np.rad2deg(np.arcsin(inputs[:, 3].clip(-1., 1.))).mean()),
            roll_mean_deg=float(np.rad2deg(np.arctan2(-inputs[:, 4], -inputs[:, 5])).mean()),
            raw_action_clipped_fraction=(np.abs(actions) >= 1.).mean(0).tolist(),
            left_right_wheel_reference_rad_s=wheels.mean(0).tolist(),
            left_right_measured_wheel_rad_s=(inputs[:, 19:21] / scale['dof_vel']).mean(0).tolist(),
            weighted_yaw_square_cost_per_s=float(abs(rewards['yaw_rate_error_sq']) * (error**2).mean()),
            yaw_kernel_lost_reward_per_s=float(rewards['tracking_ang_vel'] *
                (1. - np.exp(-error**2 / rewards['tracking_sigma_ang'])).mean()),
            orientation_cost_per_s=float(abs(rewards['orientation']) * orientation.mean()),
            orientation_cost_clipped_fraction=float((abs(rewards['orientation']) * orientation >=
                rewards['orientation_penalty_clip']).mean()),
            command_counterfactual=counterfactual))
    return dict(checkpoint=manifest['checkpoint'], checkpoint_sha256=manifest['checkpoint_sha256'],
                iteration=manifest['checkpoint_iteration'], actor_inference_parity_max=parity,
                physical_contract_changed=False, weights_updated=False, conditions=rows,
                limitation='Exact frozen actor with ELU; fixed-state counterfactual is not a plant dynamics or stability Jacobian.')


def diagnose_joint_pd(directory, output):
    import torch
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    cfg = json.loads((directory / 'resolved_env.json').read_text())
    import hashlib
    checkpoint = Path(manifest['checkpoint'])
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != manifest['checkpoint_sha256']:
        raise ValueError('Diagnosed checkpoint has changed')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)['model_state_dict']
    with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as data:
        report = joint_pd_feedback(data['observations'], data['means'], manifest, cfg, state)
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    return report


def diagnose_arrays(observations, means, manifest, cfg, completed):
    observations = np.asarray(observations)
    means = np.asarray(means)
    dt = float(manifest['policy_dt'])
    seeds = manifest['seeds']
    cases = manifest['cases']
    steps = round(manifest['seconds_per_case'] / dt)
    count = len(cases) * len(seeds)
    if dt <= 0 or observations.shape != (steps, count, 27) or means.shape != (steps, count, 6):
        raise ValueError('Unexpected policy input/action shapes or timestep')
    if not np.isfinite(observations).all() or not np.isfinite(means).all():
        raise ValueError('Nonfinite policy feedback')
    if not completed.get('completed') or completed['samples'] != steps * count:
        raise ValueError('Replay incomplete')
    if steps < round(10 / dt):
        raise ValueError('At least 10 seconds required for feedback diagnosis')
    scales, rewards = cfg['obs_scales'], cfg['rewards']
    tail = slice(-round(3 / dt), None)
    spectral_tail = slice(-round(10 / dt), None)
    trials = []
    for case_id, case in enumerate(cases):
        for seed_id, seed in enumerate(seeds):
            index = case_id * len(seeds) + seed_id
            o = observations[tail, index].astype(float)
            actions = np.clip(means[:, index], -1, 1)
            command = o[:, 6] / scales['lin_vel']
            speed = o[:, 17] / scales['lin_vel']
            speed_cost = abs(rewards['lin_vel_error_sq']) * (command - speed) ** 2
            posture_cost = abs(rewards['orientation']) * (
                o[:, 3] ** 2 + rewards['orientation_roll_multiplier'] * o[:, 4] ** 2)
            pitch = np.rad2deg(np.arcsin(o[:, 3].clip(-1, 1)))
            roll = np.rad2deg(np.arctan2(-o[:, 4], -o[:, 5]))
            wheels = [wheel_spectrum(actions[spectral_tail, column] * cfg['action_scale_vel'], dt)
                      for column in (2, 5)]
            opposite = (command * speed < 0) & (abs(speed) > .025)
            command_reached = bool(np.max(abs(command - case['speed'])) < .005)
            metrics = dict(
                command_m_s=float(command.mean()), speed_m_s=float(speed.mean()),
                speed_mae_m_s=float(np.mean(abs(command - speed))),
                opposite_direction_fraction=float(opposite.mean()), command_reached=command_reached,
                pitch_mean_deg=float(pitch.mean()), roll_mean_deg=float(roll.mean()),
                speed_penalty_saturated_fraction=float(np.mean(speed_cost >= rewards['lin_vel_penalty_clip'])),
                posture_penalty_saturated_fraction=float(np.mean(posture_cost >= rewards['orientation_penalty_clip'])),
                angle_head_clipping_fraction=float(np.mean(abs(means[tail, index][:, (0, 3)]) >= 1)),
                wheel_spectra=wheels,
            )
            alerts = []
            if command_reached and abs(case['speed']) >= .03 and opposite.mean() >= .6:
                alerts.append('opposite_motion')
            if any(w['std_rad_s'] >= 1. and w['high_frequency_power_fraction'] >= .6
                   and w['second_difference_rms_rad_s'] >= 3. for w in wheels):
                alerts.append('wheel_reference_chatter')
            if metrics['speed_penalty_saturated_fraction'] >= .8:
                alerts.append('speed_penalty_saturated')
            if metrics['posture_penalty_saturated_fraction'] >= .8:
                alerts.append('posture_penalty_saturated')
            if metrics['angle_head_clipping_fraction'] >= .95:
                alerts.append('angle_reference_saturated')
            trials.append(dict(case=case['name'], seed=seed, metrics=metrics, alerts=alerts))
    # A repeated independent-seed anomaly gates training, not one tiny ripple.
    repeated = {}
    for case in cases:
        group = [t for t in trials if t['case'] == case['name']]
        for alert in {a for t in group for a in t['alerts']}:
            if sum(alert in t['alerts'] for t in group) >= 2:
                repeated.setdefault(alert, []).append(case['name'])
    reasons = ['opposite_motion', 'wheel_reference_chatter', 'speed_penalty_saturated',
               'posture_penalty_saturated']
    reason = next((r for r in reasons if r in repeated), None)
    return dict(
        checkpoint=manifest.get('checkpoint'), checkpoint_sha256=manifest.get('checkpoint_sha256'),
        iteration=manifest.get('checkpoint_iteration'), trials=trials, repeated_alerts=repeated,
        stop_reason=reason, exact_input_diagnosis=True,
        steady_diagnosis_valid=completed.get('resets', 0) == 0,
        any_episode_reset=completed.get('resets', 0) != 0,
        notes='Frozen deterministic means; high-frequency chatter is not inference sampling noise. '
              'Resetting episodes require the separate survival assessment before steady metrics are trusted.',
    )


def diagnose(directory, output=None):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    manifest, cfg, completed = read('manifest.json'), read('resolved_env.json'), read('completed.json')
    with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as saved:
        if list(saved['seeds']) != manifest['seeds'] or abs(float(saved['policy_dt']) - manifest['policy_dt']) > 1.e-12:
            raise ValueError('Policy input metadata disagrees with replay manifest')
        report = diagnose_arrays(saved['observations'], saved['means'], manifest, cfg, completed)
    target = Path(output) if output else directory / 'feedback_diagnosis.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--joint-pd', action='store_true', help='Exact motor-angle policy bias and command response analysis')
    args = parser.parse_args()
    if args.joint_pd:
        if not args.output:
            parser.error('--joint-pd requires a separate --output; preserve existing diagnoses')
        result = diagnose_joint_pd(args.directory, args.output)
        print(json.dumps(dict(iteration=result['iteration'], parity_max=result['actor_inference_parity_max']), indent=2))
        raise SystemExit(0)
    result = diagnose(args.directory, args.output)
    print(json.dumps({k: result[k] for k in ('iteration', 'repeated_alerts', 'stop_reason', 'steady_diagnosis_valid')}, indent=2))
