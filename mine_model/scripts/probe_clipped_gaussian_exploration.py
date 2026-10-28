"""Exact clipped-Gaussian moments on frozen states, never a control change."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
import torch

from diagnose_height_response import actor
from joint_pd_continuation import sha, tensor_finite, optimizer_step
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


def moments(mean, std):
    mean, std = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0.):
        raise ValueError('Require finite means and positive Gaussian std')
    lower, upper = (-1. - mean) / std, (1. - mean) / std
    fl, fu = ndtr(lower), ndtr(upper)
    pl, pu = [np.exp(-.5 * x**2) / np.sqrt(2. * np.pi) for x in (lower, upper)]
    inside = fu - fl
    first = -fl + mean * inside + std * (pl - pu) + 1. - fu
    second = (fl + (mean**2 + std**2) * inside + 2. * mean * std * (pl - pu)
        + std**2 * (lower * pl - upper * pu) + 1. - fu)
    clipped = np.clip(mean, -1., 1.)
    variance = np.maximum(second - first**2, 0.)
    noise_mse = np.maximum(second - 2. * clipped * first + clipped**2, 0.)
    zero_noise = np.where(mean >= 1., 1. - fu, np.where(mean <= -1., fl, 0.))
    return dict(mean=first, variance=variance, noise_mse=noise_mse,
        outside_probability=fl + 1. - fu, zero_applied_noise_probability=zero_noise)


def numerical_check():
    errors = []
    for mean, std in [(0., .03), (.98, .05), (1.08, .05), (-1.08, .07), (1.2, .1), (-.4, .5)]:
        actual = moments(mean, std)
        points = [x for x in ((-1. - mean) / std, (1. - mean) / std) if -12. < x < 12.]
        values = []
        for power in (1, 2):
            value, _ = quad(lambda z: np.clip(mean + std * z, -1., 1.)**power
                * np.exp(-z*z/2.) / np.sqrt(2. * np.pi), -12., 12., points=points,
                epsabs=1.e-12, epsrel=1.e-12)
            values.append(value)
        errors.extend([abs(float(actual['mean']) - values[0]),
            abs(float(actual['variance']) - (values[1] - values[0]**2))])
    if max(errors) > 1.e-10:
        raise ValueError('Closed-form clipped moments differ from independent quadrature')
    return max(errors)


def probe(replay, output):
    replay, output = Path(replay).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError('Preserve previous exploration evidence')
    manifest = json.loads((replay / 'manifest.json').read_text())
    complete = json.loads((replay / 'completed.json').read_text())
    if (not complete['completed'] or complete['samples'] != 51000 or complete['resets']
            or len(manifest['cases']) != 17 or manifest['seeds'] != [483, 484, 485]
            or manifest['seconds_per_case'] != 20. or manifest['policy_dt'] != .02
            or manifest['weights_updated'] or manifest['sampled_policy_actions']
            or manifest['policy_input_remapping'] or manifest['command_scale'] != 1.):
        raise ValueError('Require the intact deterministic full51 frozen input source')
    source = Path(manifest['checkpoint'])
    if sha(source) != manifest['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    cfg = validate_checkpoint_robot(source)
    saved = torch.load(source, map_location='cpu', weights_only=True)
    if (not tensor_finite(saved) or cfg['leg_control_mode'] != 'implicit_joint_position'
            or len(saved['optimizer_state_dict']['state']) != 17
            or optimizer_step(saved) != 20 * saved_updates(source)):
        raise ValueError('Require complete finite original direct-joint PPO and all17 Adam states')
    with np.load(replay / 'policy_inputs.npz', allow_pickle=False) as data:
        observations, recorded = data['observations'], data['means']
        if observations.shape != (1000, 51, 27) or recorded.shape != (1000, 51, 6):
            raise ValueError('Incomplete fixed input record')
        inputs = torch.tensor(observations[-150:].copy())
        means = recorded[-150:].copy()
    with torch.no_grad():
        parity = float((actor(inputs, saved['model_state_dict']) - torch.tensor(means)).abs().max())
    if parity > 1.e-4:
        raise ValueError('Frozen source actor does not reproduce recorded means')
    original = saved['model_state_dict']['std'].numpy().astype(np.float64)
    proposals = dict(saved=original.copy())
    for label, heads, floor in [('leg_floor07', [0, 1, 3, 4], .07),
            ('leg_floor10', [0, 1, 3, 4], .1), ('wheel_floor05', [2, 5], .05)]:
        std = original.copy()
        std[heads] = np.maximum(std[heads], floor)
        proposals[label] = std
    rows = []
    for index, case in enumerate(manifest['cases']):
        mu = means[:, index*3:(index+1)*3].reshape(-1, 6)
        candidates = []
        for label, std in proposals.items():
            result = moments(mu, std)
            candidates.append(dict(label=label, hypothetical_std=std.tolist(),
                applied_noise_RMS_by_head=np.sqrt(result['noise_mse'].mean(0)).tolist(),
                applied_noise_std_by_head=np.sqrt(result['variance'].mean(0)).tolist(),
                zero_applied_noise_fraction_by_head=result['zero_applied_noise_probability'].mean(0).tolist(),
                clipping_mean_shift_by_head=(result['mean'] - np.clip(mu, -1., 1.)).mean(0).tolist(),
                raw_outside_action_envelope_fraction=result['outside_probability'].mean(0).tolist()))
        rows.append(dict(actual_manifest_case=case, frozen_samples=len(mu), proposals=candidates))
    files = [source, replay / 'manifest.json', replay / 'completed.json', replay / 'policy_inputs.npz',
        Path(__file__), source.parent / 'params/env.yaml', source.parent / 'params/agent.yaml']
    result = dict(observed_at=datetime.now().astimezone().isoformat(), source_checkpoint=str(source),
        source_checkpoint_sha256=sha(source), source_actual_updates=saved_updates(source),
        independent_quadrature_max_error=numerical_check(), source_actor_parity_max_error=parity,
        rows=rows, source_pins={str(p): sha(p) for p in files},
        RNG_draws_added=0, simulator_started=False, optimizer_executed=False,
        checkpoint_or_training_modified=False, proposals_applied=False, policy_accepted=False,
        limitations=['These are exact moments of clipping each independent Gaussian action to[-1,1] on frozen source states.',
            'Additional hard joint limits and nonlinear kinematics can further change motor and height exploration.',
            'A larger applied variance or inward mean shift is not evidence of improved return, stable contacts, convergence or independent acceptance.',
            'No proposal changes the live200-update trial. Any later exploration trial needs separate full-state qualification and matched replay.'])
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = probe(args.replay, args.output)
    print(json.dumps(dict(output=str(args.output), quadrature_max_error=result['independent_quadrature_max_error'],
        source_actor_parity=result['source_actor_parity_max_error'], proposals_applied=False), ensure_ascii=False))
