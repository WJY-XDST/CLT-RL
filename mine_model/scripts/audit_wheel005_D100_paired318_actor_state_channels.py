"""Account for the recorded-state part of318 fixed-actor output changes.

Integrate input gradients along each paired observation segment. Groups sum
to the actor output difference within an explicit quadrature tolerance. This
is neural-input accounting, not a causal mechanics or closed-loop experiment.
"""
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import torch

from diagnose_height_response import actor
import audit_wheel005_D100_frozen7201_vs_trained7401_318 as fixtures
import audit_wheel005_D100_paired318_fixed_actor_outputs as fixed
import audit_paired_yaw_kernel_trace_budget as original


SCRIPT = Path(__file__).resolve()
OUTPUT = fixtures.TRAINING / 'sameD100_all318_actor_recorded_state_channel_accounting_readonly_20261024.json'
GROUPS = dict(body_gyro=[0, 1, 2], projected_gravity=[3, 4, 5], commands=[6, 7, 8],
    leg_position=[9, 10, 11, 12], leg_velocity=[13, 14, 15, 16],
    heading_linear_velocity=[17, 18], measured_forward_wheel_velocity=[19, 20],
    previous_leg_actions=[21, 22, 24, 25], previous_wheel_actions=[23, 26])


def run():
    assert not OUTPUT.exists()
    assert sorted(i for value in GROUPS.values() for i in value) == list(range(27))
    read, sha = original.read, original.sha
    baseline = read(fixed.OUTPUT)
    assert baseline['verified'] and baseline['retained_trials'] == 318
    final = read(fixtures.TRAINING / 'final_checkpoint_verification_readonly_20261019.json')
    torch.set_num_threads(1)
    rng = torch.random.get_rng_state().clone()
    pins = {str(p): sha(p) for p in (SCRIPT, fixtures.SCRIPT, fixed.SCRIPT, fixed.OUTPUT,
        fixtures.OUTPUT, Path(actor.__code__.co_filename), original.GATE)}
    policies = {}
    for label, field, digest in [('frozen', 'source_checkpoint', fixtures.SOURCE_SHA),
                                  ('trained', 'checkpoint', fixtures.TARGET_SHA)]:
        path = Path(final[field])
        assert sha(path) == digest
        saved = torch.load(path, map_location='cpu', weights_only=True)
        policies[label] = {k: v.to(torch.float64) for k, v in saved['model_state_dict'].items()
                           if k.startswith('actor.')}
        pins[str(path)] = digest
    source, candidate, order, shape, cfg = [], [], [], [], None
    for scope, _, old, new, count in fixtures.SCOPES:
        for storage, folder in [(source, fixtures.FROZEN / old), (candidate, fixtures.REVIEW / new)]:
            manifest = read(folder / 'manifest.json')
            current = read(folder / 'resolved_env.json')
            cfg = cfg or current
            assert current['wheel_radius'] == cfg['wheel_radius']
            assert current['wheel_track_width'] == cfg['wheel_track_width']
            assert current['wheel_action_limit'] == 1.25 and current['action_scale_vel'] == 15.
            with np.load(folder / 'policy_inputs.npz', allow_pickle=False) as z:
                values = z['observations'][-151:].copy()
            assert values.shape == (151, count, 27) and np.isfinite(values).all()
            storage.append(values.reshape(-1, 27))
            for name in ('manifest.json', 'resolved_env.json', 'policy_inputs.npz'):
                pins[str(folder / name)] = sha(folder / name)
        shape.append((scope, count, manifest))
    x = torch.tensor(np.concatenate(source), dtype=torch.float64)
    z = torch.tensor(np.concatenate(candidate), dtype=torch.float64)
    delta = z - x
    ratio = cfg['wheel_radius'] / cfg['wheel_track_width']

    def proxy(value, state):
        means = actor(value, state)
        return ratio * 15. * (means[:, 5].clamp(-1.25, 1.25) - means[:, 2].clamp(-1.25, 1.25))

    with torch.inference_mode():
        expected = {label: proxy(z, state) - proxy(x, state) for label, state in policies.items()}
    attempts = []
    # The preserved first attempt through32 nodes missed the unchanged1e-7
    # internal arithmetic tolerance. Increase integration accuracy only.
    for points in (64, 128):
        nodes, weights = np.polynomial.legendre.leggauss(points)
        integrated = {label: torch.zeros_like(x) for label in policies}
        for node, weight in zip((nodes + 1.) / 2., weights / 2.):
            for label, state in policies.items():
                observation = (x + float(node) * delta).requires_grad_(True)
                gradient, = torch.autograd.grad(proxy(observation, state).sum(), observation)
                integrated[label] += float(weight) * gradient.detach() * delta
        errors = {label: float(abs(integrated[label].sum(-1) - value).max())
                  for label, value in expected.items()}
        attempts.append(dict(points=points, maximum_per_frame_closure_error_rad_s=errors))
        if max(errors.values()) < 1.e-7:
            break
    assert max(errors.values()) < 1.e-7, attempts
    symmetric = .5 * (integrated['frozen'] + integrated['trained'])
    symmetric_error = float(abs(symmetric.sum(-1) - .5 * (expected['frozen'] + expected['trained'])).max())
    assert symmetric_error < 1.e-7
    index = {(r['scope'], r['case']['name'], r['seed']): r for r in baseline['paired_trials']}
    rows, offset, baseline_error = [], 0, 0.
    for scope, count, manifest in shape:
        section = symmetric[offset:offset + count * 151].numpy().reshape(151, count, 27)
        offset += count * 151
        for cid, case in enumerate(manifest['cases']):
            for sid, seed in enumerate(manifest['seeds']):
                eid = cid * len(manifest['seeds']) + sid
                prior = index[scope, case['name'], seed]
                values = {name: float(section[:, eid, columns].sum(-1).mean())
                          for name, columns in GROUPS.items()}
                accounted = sum(values.values())
                error = abs(accounted - prior['symmetric_recorded_state_contribution_rad_s'])
                baseline_error = max(baseline_error, error)
                assert error < 1.e-7
                rows.append(dict(scope=scope, case=case, seed=seed,
                    frozen_original_AND_literal10=prior['frozen_original_AND_literal10'],
                    trained_original_AND_literal10=prior['trained_original_AND_literal10'],
                    symmetric_parameter_contribution_rad_s=prior['symmetric_parameter_contribution_rad_s'],
                    symmetric_recorded_state_contribution_rad_s=prior['symmetric_recorded_state_contribution_rad_s'],
                    grouped_symmetric_state_channel_contributions_rad_s=values,
                    channel_sum_minus_recorded_state_component_rad_s=accounted - prior['symmetric_recorded_state_contribution_rad_s']))
    assert len(rows) == 318 and offset == len(x)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(sha(Path(p)) == digest for p, digest in pins.items())
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        retained_trials=318, tail_frames_per_trial=151, observation_groups=GROUPS,
        method='Gauss-Legendre integration of ordinary ELU actor input gradients along paired observations; averaged symmetrically over both actors',
        quadrature_attempts=attempts, chosen_points=points,
        prior_lower_order_quadrature_failure_preserved=str(fixtures.BASE /
            'joint_pd_wheel005_source7201_D100_qualification_20261024/preserved_readonly318_input_integral_quadrature32_failure_20261024'),
        symmetric_per_frame_closure_max_error_rad_s=symmetric_error,
        previous_fixed_actor_state_component_reproduction_max_error_rad_s=baseline_error,
        paired_trials=rows, source_pins=pins, RNG_unchanged=True,
        optimizer_constructed=False, optimizer_executed=False,
        functional_input_gradients_only=True, model_parameter_gradients_created=False,
        simulator_started=False, training_modified=False, controller_changed=False,
        acceptance_changed=False, next_parameter_or_source_selected=False, policy_accepted=False,
        limitations=['Groups describe which recorded neural-input changes account algebraically for actor output changes. They do not establish independently causal physical effects.',
            'Intermediate interpolated observations need not be valid physical states; this is not a trajectory experiment.',
            'No optimizer or model parameter gradients are constructed; only input derivatives are evaluated.',
            'No acceptance gate or candidate selection can be replaced by this arithmetic diagnosis.'])
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(verified=True, report=str(OUTPUT), trials=318, chosen_points=points,
        per_frame_closure_max_error=symmetric_error, original_state_component_max_error=baseline_error,
        RNG_unchanged=True, optimizer_executed=False, policy_accepted=False)))


if __name__ == '__main__':
    run()
