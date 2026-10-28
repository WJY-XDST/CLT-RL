"""Cross two preserved actors with matched frozen replay observations.

This is read-only neural output accounting on every matching numeric command
and seed, including passed cases. Input integration is optional and describes
an algebraic output difference, not independently causal physical effects.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diagnose_height_response import actor
from joint_pd_continuation import sha


GROUPS = dict(body_gyro=[0, 1, 2], projected_gravity=[3, 4, 5],
    commands=[6, 7, 8], leg_joint_position=[9, 10, 11, 12],
    leg_joint_velocity=[13, 14, 15, 16], heading_linear_velocity=[17, 18],
    measured_wheel_velocity=[19, 20], previous_leg_actions=[21, 22, 24, 25],
    previous_wheel_actions=[23, 26])


def key(case, seed):
    return tuple(round(float(case[k]), 9) for k in ('height', 'speed', 'yaw')) + (int(seed),)


def read(path):
    return json.loads(Path(path).read_text())


def load_recordings(directories, pins):
    trials, checkpoint, cfg = {}, None, None
    for folder in directories:
        paths = [folder / n for n in ('manifest.json', 'completed.json',
                                     'resolved_env.json', 'policy_inputs.npz')]
        pins.update({str(p): sha(p) for p in paths})
        manifest, complete, current = [read(p) for p in paths[:3]]
        assert complete['completed'] and not complete['resets']
        assert manifest['policy_dt'] == .02 and manifest['seconds_per_case'] == 20.
        assert not manifest['sampled_policy_actions'] and not manifest['weights_updated']
        assert manifest['action_bias_probes'] is None and not manifest['wheel_probe']
        assert not manifest['policy_input_remapping']
        source = Path(manifest['checkpoint'])
        assert sha(source) == manifest['checkpoint_sha256']
        checkpoint = checkpoint or source
        cfg = cfg or current
        assert source == checkpoint
        for k in ('wheel_radius', 'wheel_track_width', 'wheel_damping', 'wheel_action_limit',
                  'action_scale_vel', 'leg_control_mode', 'obs_scales'):
            assert current[k] == cfg[k], k
        count = len(manifest['cases']) * len(manifest['seeds'])
        with np.load(paths[3], allow_pickle=False) as z:
            observations, means = z['observations'], z['means']
            assert observations.shape == (1000, count, 27) and means.shape == (1000, count, 6)
            assert np.isfinite(observations).all() and np.isfinite(means).all()
            for cid, case in enumerate(manifest['cases']):
                for sid, seed in enumerate(manifest['seeds']):
                    index = cid * len(manifest['seeds']) + sid
                    k = key(case, seed)
                    assert k not in trials, k
                    trials[k] = dict(case=case, seed=seed, directory=str(folder),
                        observations=observations[-151:, index].copy(),
                        initial_observation=observations[0, index].copy(),
                        means=means[-151:, index].copy())
    return trials, checkpoint, cfg


def load_actor(path, pins):
    agent_path = path.parent / 'params/agent.yaml'
    policy = yaml.safe_load(agent_path.read_text())['policy']
    assert policy['class_name'] == 'ActorCritic'
    assert policy['actor_hidden_dims'] == [256, 128, 64]
    assert policy['activation'] == 'elu' and not policy.get('actor_obs_normalization', False)
    pins.update({str(p): sha(p) for p in (path, agent_path)})
    saved = torch.load(path, map_location='cpu', weights_only=True)
    states = saved['optimizer_state_dict']['state']
    assert len(states) == 17
    steps = [float(s['step']) for s in states.values()]
    assert len(set(steps)) == 1 and steps[0] == 20 * (int(saved['iter']) + 1)
    model = {k: v.to(torch.float64) for k, v in saved['model_state_dict'].items()
             if k.startswith('actor.')}
    assert all(not v.requires_grad and torch.isfinite(v).all() for v in model.values())
    assert model['actor.0.weight'].shape == (256, 27)
    assert model['actor.6.weight'].shape == (6, 64)
    return model, dict(checkpoint=str(path), sha256=sha(path),
        actual_updates=int(saved['iter']) + 1, all17_Adam_step=steps[0],
        saved_std=saved['model_state_dict']['std'].tolist())


def audit(args):
    assert not args.output.exists()
    assert sorted(i for cols in GROUPS.values() for i in cols) == list(range(27))
    torch.set_num_threads(1)
    rng = torch.random.get_rng_state().clone()
    pins = {str(Path(__file__).resolve()): sha(__file__),
            str(Path(actor.__code__.co_filename)): sha(actor.__code__.co_filename)}
    before, path_a, cfg_a = load_recordings(args.before_replay, pins)
    after, path_b, cfg_b = load_recordings(args.after_replay, pins)
    assert path_a != path_b and set(after).issubset(before)
    for name in ('wheel_radius', 'wheel_track_width', 'wheel_damping', 'wheel_action_limit',
                 'action_scale_vel', 'leg_control_mode', 'obs_scales'):
        assert cfg_a[name] == cfg_b[name], name
    assert cfg_a['leg_control_mode'] == 'implicit_joint_position'
    states, clocks = {}, {}
    for label, path in [('before', path_a), ('after', path_b)]:
        states[label], clocks[label] = load_actor(path, pins)
    indexed = {}
    for label, path in [('before', args.before_assessment), ('after', args.after_assessment)]:
        pins[str(path)] = sha(path)
        assessment = read(path)
        assert assessment['verified']
        indexed[label] = {key(r['case'], r['seed']): r
            for group in assessment['results'] for r in group['all_trials']}
        assert set(after).issubset(indexed[label])
    order = list(after)
    arrays = {label: np.stack([data[k]['observations'] for k in order])
              for label, data in [('before', before), ('after', after)]}
    x, z = [torch.tensor(arrays[label].reshape(-1, 27), dtype=torch.float64)
            for label in ('before', 'after')]
    factor = cfg_a['wheel_radius'] / cfg_a['wheel_track_width'] * cfg_a['action_scale_vel']
    limit = cfg_a['wheel_action_limit']

    def proxy(inputs, state):
        mu = actor(inputs, state)
        return factor * (mu[:, 5].clamp(-limit, limit) - mu[:, 2].clamp(-limit, limit))

    raw, outputs, parity = {}, {}, {}
    with torch.inference_mode():
        for policy, state in states.items():
            for origin, value in [('before', x), ('after', z)]:
                raw[policy, origin] = actor(value, state)
                outputs[policy, origin] = proxy(value, state)
                if policy == origin:
                    recorded = np.stack([({'before': before, 'after': after}[origin])[k]['means']
                                         for k in order]).reshape(-1, 6)
                    parity[origin] = float(abs(raw[policy, origin] - torch.tensor(recorded)).max())
                    assert parity[origin] < 2.e-5, parity
    aa, ba = outputs['before', 'before'], outputs['after', 'before']
    ab, bb = outputs['before', 'after'], outputs['after', 'after']
    parameter = .5 * (ba - aa + bb - ab)
    state_change = .5 * (ab - aa + bb - ba)
    closure = float(abs(parameter + state_change - (bb - aa)).max())
    assert closure < 1.e-12
    integration, attempts = None, []
    if args.input_attribution:
        delta = z - x
        for points in (64, 128, 256):
            integration = torch.zeros_like(x)
            for node, weight in zip(*[(v + 1.) / 2. if i == 0 else v / 2.
                                     for i, v in enumerate(np.polynomial.legendre.leggauss(points))]):
                for state in states.values():
                    value = (x + float(node) * delta).requires_grad_(True)
                    gradient, = torch.autograd.grad(proxy(value, state).sum(), value)
                    integration += .5 * float(weight) * gradient.detach() * delta
            error = float(abs(integration.sum(-1) - state_change).max())
            attempts.append(dict(points=points, maximum_frame_closure_error_rad_s=error))
            if error < 1.e-7:
                break
        assert error < 1.e-7, attempts
    rows = []
    for index, k in enumerate(order):
        sl = slice(index * 151, (index + 1) * 151)
        prior, current = indexed['before'][k], indexed['after'][k]
        rec = {}
        for label, native in [('before', aa), ('after', bb)]:
            expected = after[k]['case']['yaw'] - indexed[label][k]['signed_command_minus_reference_rad_s']
            rec[label] = float(native[sl].mean())
            assert abs(rec[label] - expected) < 2.e-6
        rows.append(dict(case=after[k]['case'], seed=k[-1],
            before_original_passed=prior['original_passed'], after_original_passed=current['original_passed'],
            before_strict_passed=prior['original_AND_literal10_passed'],
            after_strict_passed=current['original_AND_literal10_passed'],
            before_relative_percent=prior['nonzero_relative_percent'],
            after_relative_percent=current['nonzero_relative_percent'],
            actual_signed_body_yaw_bias_rad_s={label: indexed[label][k]['body_yaw_signed_bias_rad_s']
                                             for label in indexed},
            native_reference_yaw_proxy_rad_s=rec,
            same_old_inputs_network_change_rad_s=float((ba - aa)[sl].mean()),
            same_new_inputs_network_change_rad_s=float((bb - ab)[sl].mean()),
            symmetric_parameter_contribution_rad_s=float(parameter[sl].mean()),
            symmetric_recorded_state_contribution_rad_s=float(state_change[sl].mean()),
            total_native_reference_change_rad_s=float((bb - aa)[sl].mean()),
            grouped_symmetric_state_contributions_rad_s=None if integration is None else
                {name: float(integration[sl, cols].sum(-1).mean()) for name, cols in GROUPS.items()},
            mean_observation_change_by_group={name: (arrays['after'][index, :, cols].mean(-1)
                - arrays['before'][index, :, cols].mean(-1)).tolist() for name, cols in GROUPS.items()},
            initial_observation_difference=(after[k]['initial_observation'] - before[k]['initial_observation']).tolist()))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(not v.requires_grad and v.grad is None for state in states.values() for v in state.values())
    assert all(sha(p) == digest for p, digest in pins.items())
    report = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        paired_trials=len(order), inclusive_tail_frames=151, checkpoints=clocks,
        original_ELU_actor_reader_reused=True, observation_groups=GROUPS,
        native_actor_mean_parity_max_error=parity,
        symmetric_decomposition_max_frame_error_rad_s=closure,
        input_integration_attempts=attempts, rows=rows, source_pins=pins,
        RNG_unchanged=True, input_gradients_only=bool(args.input_attribution),
        model_parameter_gradients_created=False, optimizer_constructed=False, optimizer_executed=False,
        simulator_started=False, training_modified=False, case_specific_training_logic_added=False,
        acceptance_changed=False, next_parameter_or_source_selected=False, policy_accepted=False,
        limitations=['State/parameter contributions account algebraically for actor outputs, not independent physical causes.',
            'Interpolated inputs need not lie on valid physical trajectories.',
            'Different replay batch layouts may change numerical physics despite identical command/seed.',
            'Partial intersections and proxy accounting cannot replace complete independent acceptance.'])
    with args.output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(verified=True, report=str(args.output), paired_trials=len(order),
        native_mean_parity=parity, integration_attempts=attempts, policy_accepted=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before-replay', nargs='+', type=Path, required=True)
    parser.add_argument('--after-replay', nargs='+', type=Path, required=True)
    parser.add_argument('--before-assessment', type=Path, required=True)
    parser.add_argument('--after-assessment', type=Path, required=True)
    parser.add_argument('--input-attribution', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    audit(parser.parse_args())
