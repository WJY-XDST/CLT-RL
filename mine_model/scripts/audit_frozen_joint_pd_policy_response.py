"""Compare two actors on every recorded state of a frozen command matrix.

This diagnostic does not predict candidate closed-loop behavior. It identifies
how ordinary PPO changed wheel references and local yaw responses on identical
observations. Cases only label results; they never select a control rule.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = saved['model_state_dict']
    states = saved['optimizer_state_dict']['state']
    assert len(states) == 17
    steps = {int(value['step']) for value in states.values()}
    assert len(steps) == 1
    step = steps.pop()
    assert step % 20 == 0 and step // 20 == saved['iter'] + 1
    assert model['actor.0.weight'].shape == (256, 27)
    assert model['critic.0.weight'].shape == (256, 27)
    assert model['std'].shape == (6,) and (model['std'] > 0).all()
    assert all(torch.isfinite(t).all() for t in model.values())
    return model, step // 20


def actor(inputs, state):
    for index in (0, 2, 4, 6):
        inputs = F.linear(inputs, state[f'actor.{index}.weight'], state[f'actor.{index}.bias'])
        if index != 6:
            inputs = F.elu(inputs)
    return inputs


def response(inputs, state, *, limit, scale, radius, track, angular_scale):
    values = torch.from_numpy(inputs).requires_grad_()
    means = actor(values, state)
    # Reproduce the checkpoint's physical wheel bounds, including saved1.25.
    wheel_targets = means[:, (2, 5)].clamp(-limit, limit) * scale
    proxy = radius / track * (wheel_targets[:, 1] - wheel_targets[:, 0])
    jacobian = torch.autograd.grad(proxy.sum(), values)[0]
    return dict(means=means.detach().numpy(), targets=wheel_targets.detach().numpy(),
        proxy=proxy.detach().numpy(), command_slope=(jacobian[:, 7] * angular_scale).numpy(),
        measured_slope=(jacobian[:, 2] * angular_scale).numpy())


def audit(source, candidate, matrix, output):
    assert not output.exists(), 'Preserve earlier diagnostics'
    torch.set_num_threads(1)
    rng = torch.random.get_rng_state().clone()
    source_state, source_updates = load(source)
    candidate_state, candidate_updates = load(candidate)
    assert source_state.keys() == candidate_state.keys()
    assert all(source_state[k].shape == candidate_state[k].shape for k in source_state)
    specification = read(matrix / 'qualification.json')
    completed = read(matrix / 'complete_frozen_command_matrix.json')
    assert completed['checkpoint_sha256'] == sha(source)
    pins = {str(p): sha(p) for p in (SCRIPT, source, candidate,
        matrix / 'qualification.json', matrix / 'complete_frozen_command_matrix.json')}
    rows, errors = [], dict(recorded_source_actor=0., recorded_source_wheel_targets=0.,
                            command_observation=0., measured_observation=0., finite_difference=0.)
    for job in specification['batches']:
        directory = Path(job['replay'])
        manifest, cfg, done = [read(directory / name) for name in
            ('manifest.json', 'resolved_env.json', 'completed.json')]
        assert manifest['checkpoint_sha256'] == sha(source) and done['completed']
        assert not done['resets'] and not manifest['sampled_policy_actions']
        assert not manifest['weights_updated'] and not manifest['policy_input_remapping']
        assert cfg['leg_control_mode'] == 'implicit_joint_position'
        for name in ('manifest.json', 'resolved_env.json', 'completed.json', 'trace.csv', 'policy_inputs.npz'):
            path = directory / name
            pins[str(path)] = sha(path)
        with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as archive:
            observations, recorded = archive['observations'], archive['means']
        steps = round(manifest['seconds_per_case'] / manifest['policy_dt'])
        width = len(manifest['cases']) * len(manifest['seeds'])
        assert observations.shape == (steps, width, 27) and recorded.shape == (steps, width, 6)
        fields = ['time_s', 'yaw_cmd_rad_s', 'yaw_rate_rad_s',
                  'left_wheel_ref_rad_s', 'right_wheel_ref_rad_s']
        with (directory / 'trace.csv').open() as stream:
            header = stream.readline().strip().split(',')
        trace = np.loadtxt(directory / 'trace.csv', delimiter=',', skiprows=1,
            usecols=[header.index(k) for k in fields]).reshape(steps, width, len(fields))
        angular_scale = cfg['obs_scales']['ang_vel']
        errors['command_observation'] = max(errors['command_observation'],
            float(abs(observations[:, :, 7] / angular_scale - trace[:, :, 1]).max()))
        errors['measured_observation'] = max(errors['measured_observation'],
            float(abs(observations[1:, :, 2] / angular_scale - trace[:-1, :, 2]).max()))
        tail = np.flatnonzero(trace[:, 0, 0] >= manifest['seconds_per_case'] - 3.)
        assert len(tail) == round(3. / manifest['policy_dt']) + 1
        params = dict(limit=cfg['wheel_action_limit'], scale=cfg['action_scale_vel'],
            radius=cfg['wheel_radius'], track=cfg['wheel_track_width'], angular_scale=angular_scale)
        for ci, case in enumerate(manifest['cases']):
            for si, seed in enumerate(manifest['seeds']):
                index = ci * len(manifest['seeds']) + si
                values = observations[tail, index].copy()
                before = response(values, source_state, **params)
                after = response(values, candidate_state, **params)
                errors['recorded_source_actor'] = max(errors['recorded_source_actor'],
                    float(abs(before['means'] - recorded[tail, index]).max()))
                errors['recorded_source_wheel_targets'] = max(errors['recorded_source_wheel_targets'],
                    float(abs(before['targets'] - trace[tail, index, 3:5]).max()))
                # Independent finite differences check both input derivatives.
                for state, result in ((source_state, before), (candidate_state, after)):
                    for column, key in ((7, 'command_slope'), (2, 'measured_slope')):
                        plus, minus = values.copy(), values.copy()
                        plus[:, column] += .005 * angular_scale
                        minus[:, column] -= .005 * angular_scale
                        with torch.no_grad():
                            a = actor(torch.from_numpy(plus), state)[:, (2, 5)].clamp(-params['limit'], params['limit'])
                            b = actor(torch.from_numpy(minus), state)[:, (2, 5)].clamp(-params['limit'], params['limit'])
                        numerical = (params['scale'] * params['radius'] / params['track']
                            * ((a[:, 1] - a[:, 0]) - (b[:, 1] - b[:, 0])) / .01).numpy()
                        errors['finite_difference'] = max(errors['finite_difference'],
                            float(abs(numerical - result[key]).max()))
                sigma0, sigma1 = source_state['std'].numpy(), candidate_state['std'].numpy()
                kl = (np.log(sigma1 / sigma0) + (sigma0 ** 2
                    + (before['means'] - after['means']) ** 2) / (2. * sigma1 ** 2) - .5).sum(-1)
                compact = lambda x: dict(wheel_yaw_proxy_mean=float(x['proxy'].mean()),
                    command_yaw_local_slope_mean=float(x['command_slope'].mean()),
                    measured_yaw_local_slope_mean=float(x['measured_slope'].mean()),
                    mean_raw_actions=x['means'].mean(0).tolist())
                rows.append(dict(case=case, seed=seed, steady_frames=len(tail), source=compact(before),
                    candidate=compact(after), wheel_yaw_proxy_change=float((after['proxy'] - before['proxy']).mean()),
                    fixed_state_raw_Gaussian_KL_mean=float(kl.mean())))
    assert len(rows) == len(completed['all_trials'])
    assert errors['recorded_source_actor'] < 5.e-6 and errors['recorded_source_wheel_targets'] < 3.e-5
    assert errors['command_observation'] < 1.e-6 and errors['measured_observation'] < 1.e-6
    # ELU and clipping finite differences have a finite step; do not describe
    # crossing a clipping corner as a smooth local derivative.
    assert errors['finite_difference'] < .01, errors
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(sha(path) == digest for path, digest in pins.items())
    report = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        source_checkpoint=str(source), candidate_checkpoint=str(candidate),
        source_actual_updates=source_updates, candidate_actual_updates=candidate_updates,
        source_std=source_state['std'].tolist(), candidate_std=candidate_state['std'].tolist(),
        all_trials_retained=len(rows), input_and_numeric_verification_max_errors=errors,
        trials=rows, source_pins=pins, RNG_unchanged=True, optimizer_executed=False,
        simulator_started=False, training_modified=False, policy_accepted=False,
        limitation='Identical old observations are used for both actors. Local network responses, '
                   'wheel rolling proxies and raw Gaussian KL do not establish candidate closed-loop '
                   'error, stability, improvement, causality of the learned bias or acceptance.')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'candidate', 'matrix', 'output'):
        cli.add_argument('--' + name, type=Path, required=True)
    args = cli.parse_args()
    result = audit(*(getattr(args, name).resolve() for name in ('source', 'candidate', 'matrix', 'output')))
    print(json.dumps(dict(report=str(args.output), trials=result['all_trials_retained'],
        errors=result['input_and_numeric_verification_max_errors'], policy_accepted=False)))
