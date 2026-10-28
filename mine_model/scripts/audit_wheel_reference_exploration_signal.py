"""Analytic wheel-reference noise versus commands on every original frozen51 trial.

This is a conditional action-space diagnostic, not body-yaw noise, extra
sampling, a learned improvement, a selected std change or acceptance.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch

from joint_pd_continuation import sha

SCRIPT = Path(__file__).resolve()


def audit(directory, destination):
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Preserve the immutable exploration diagnostic')
    manifest, complete, cfg = [json.loads((directory / n).read_text())
        for n in ('manifest.json', 'completed.json', 'resolved_env.json')]
    checkpoint = Path(manifest['checkpoint'])
    if (not complete['completed'] or complete['samples'] != 51000 or complete['resets']
            or len(manifest['cases']) != 17 or manifest['seeds'] != [483, 484, 485]
            or manifest['sampled_policy_actions'] or manifest['weights_updated']
            or manifest['policy_input_remapping'] or manifest['checkpoint_sha256'] != sha(checkpoint)
            or manifest['wheel_reference_limit_probe'] is not None):
        raise ValueError('Require the intact original deterministic51 replay')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    sigma = saved['model_state_dict']['std'].numpy().astype(np.float64)
    if sigma.shape != (6,) or np.any(sigma <= 0.):
        raise ValueError('Require the actual six positive saved action scales')
    with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as data:
        means = data['means'].copy().astype(np.float64)
    if means.shape != (1000, 51, 6):
        raise ValueError('Keep all original time steps, cases and seeds')
    trace = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    if len(trace) != 51000 or any(np.any(trace[k]) for k in ('episode', 'terminated', 'timeout')):
        raise ValueError('Do not discard any trajectory')
    k = cfg['action_scale_vel'] * cfg['wheel_radius'] / cfg['wheel_track_width']
    variants = {'saved': sigma[[2, 5]], 'hypothetical_wheel025': np.array([.025, .025]),
        'hypothetical_wheel020': np.array([.02, .02])}
    rows, parity = [], 0.
    for cid, case in enumerate(manifest['cases']):
        for sid, seed in enumerate(manifest['seeds']):
            part = trace[(trace['case_id'] == cid) & (trace['seed'] == seed)]
            mu = means[:, cid * 3 + sid, :][:, [2, 5]]
            reference = cfg['action_scale_vel'] * np.clip(mu, -1., 1.)
            actual_reference = np.stack((part['left_wheel_ref_rad_s'], part['right_wheel_ref_rad_s']), -1)
            parity = max(parity, float(abs(reference - actual_reference).max()))
            mask = part['time_s'] >= 17.
            if len(part) != 1000 or mask.sum() != 151:
                raise ValueError('Retain the original inclusive final3s window')
            mu = mu[mask]
            mean_yaw_proxy = k * (mu[:, 1] - mu[:, 0])
            results = {}
            for label, std in variants.items():
                tail = ndtr((-1. - mu) / std) + ndtr((mu - 1.) / std)
                bound = float(tail.sum(-1).max())
                if bound > 1.e-12:
                    raise ValueError('Wheel clipping is material; use the clipped distribution instead')
                yaw_std = float(k * np.linalg.norm(std))
                command = case['yaw']
                # Same fixed mean, changed wheel Gaussian scales only. This
                # quantifies a hypothetical abrupt std reset, not a floor edit.
                raw_kl = float(np.sum(np.log(std / variants['saved'])
                    + (variants['saved'] ** 2) / (2. * std ** 2) - .5))
                results[label] = dict(wheel_action_std=std.tolist(), wheel_reference_yaw_proxy_std_rad_s=yaw_std,
                    noise_std_over_abs_yaw_command=None if command == 0 else yaw_std / abs(command),
                    reference_yaw_proxy_opposite_command_probability=None if command == 0 else
                        float(ndtr(-np.sign(command) * mean_yaw_proxy / yaw_std).mean()),
                    maximum_wheel_clipping_probability_union_bound=bound,
                    hypothetical_same_mean_raw_KL_from_saved=raw_kl)
            rows.append(dict(case=case, seed=seed, samples=151,
                deterministic_wheel_reference_yaw_proxy_mean_rad_s=float(mean_yaw_proxy.mean()), variants=results))
    if parity > 2.e-6:
        raise ValueError('Recorded raw means do not reproduce the original wheel references')
    files = [SCRIPT, checkpoint, *(directory / n for n in
        ('manifest.json', 'completed.json', 'resolved_env.json', 'policy_inputs.npz', 'trace.csv'))]
    result = dict(observed_at=datetime.now().astimezone().isoformat(), checkpoint_sha256=sha(checkpoint),
        saved_std=sigma.tolist(), all51_trials_retained=True, original_final3s_samples_per_trial=151,
        target_from_raw_mean_max_error_rad_s=parity, trials=rows,
        formula='yaw_reference_proxy=(action_scale_vel*wheel_radius/track)*(a_right-a_left); independent Gaussian variance=k^2*(sigma_left^2+sigma_right^2)',
        source_pins={str(p): sha(p) for p in files}, RNG_draws_added=0, simulator_started=False,
        optimizer_executed=False, saved_std_updated=False, training_modified=False,
        proposed_parameter_change_selected=False, acceptance_changed=False, policy_accepted=False,
        limitations=['Wheel-reference noise is not actual wheel or body-yaw noise; the servo, dynamics, friction and contact intervene.',
            'Opposite-reference probabilities are conditional on old fixed observations and analytical Gaussian noise, not observed robot direction-reversal frequencies.',
            'Hypothetical smaller noise does not prove learned improvement. Abrupt std resets can exceed the original PPO KL scale; lowering a future guard is distinct from resetting a saved std.',
            'All time states are correlated. This analysis adds no sampling and establishes no local optimum or final acceptance.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.directory, args.output)
    focus = [x for x in result['trials'] if x['case']['name'] == 'stage025_high_spin_left']
    print(json.dumps(dict(report=str(args.output), high_spin_left=focus, training_modified=False)))
