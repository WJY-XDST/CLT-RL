"""Separate fixed-input actor changes from different recorded D1 states.

Read all318 paired tails; use the original ELU actor arithmetic. Local input
sensitivities and symmetric output differences are algebraic descriptions,
not closed-loop stability gains or independently causal plant factors.
"""
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import torch
import yaml

from diagnose_height_response import actor
import audit_paired_yaw_kernel_trace_budget as trace_reader
import audit_wheel005_D100_frozen7201_vs_trained7401_318 as paired_reader


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
OUTPUT = paired_reader.TRAINING / 'sameD100_all318_fixed_actor_parameter_state_outputs_readonly_20261024.json'


def run():
    assert not OUTPUT.exists()
    read, sha = trace_reader.read, trace_reader.sha
    comparison = read(paired_reader.OUTPUT)
    assert comparison['verified'] and len(comparison['paired_trials']) == 318
    final = read(paired_reader.TRAINING / 'final_checkpoint_verification_readonly_20261019.json')
    torch.set_num_threads(1)
    rng = torch.random.get_rng_state().clone()
    states, clocks = {}, {}
    pins = {str(p): sha(p) for p in (SCRIPT, paired_reader.SCRIPT,
        paired_reader.OUTPUT, Path(actor.__code__.co_filename), trace_reader.GATE)}
    assert sha(trace_reader.GATE) == trace_reader.GATE_SHA
    for label, key, expected_sha, updates in (
            ('frozen', 'source_checkpoint', paired_reader.SOURCE_SHA, 7201),
            ('trained', 'checkpoint', paired_reader.TARGET_SHA, 7401)):
        checkpoint = Path(final[key])
        assert sha(checkpoint) == expected_sha
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        states[label] = {k: v.to(torch.float64) for k, v in saved['model_state_dict'].items()
                         if k.startswith('actor.')}
        optimizer = saved['optimizer_state_dict']['state']
        assert len(optimizer) == 17 and all(float(s['step']) == updates * 20 for s in optimizer.values())
        config = checkpoint.parent / 'params/agent.yaml'
        policy = yaml.safe_load(config.read_text())['policy']
        assert policy['actor_hidden_dims'] == [256, 128, 64] and policy['activation'] == 'elu'
        assert not policy.get('actor_obs_normalization', False)
        assert states[label]['actor.0.weight'].shape == (256, 27)
        assert states[label]['actor.6.weight'].shape == (6, 64)
        clocks[label] = dict(actual_updates=updates, all17_Adam_step=updates * 20,
            saved_std=saved['model_state_dict']['std'].tolist())
        pins.update({str(checkpoint): sha(checkpoint), str(config): sha(config)})
    trials, parity, closures, gains = [], {}, [], {}
    for scope, _, old_name, new_name, count in paired_reader.SCOPES:
        fixtures = {}
        for origin, folder in (('frozen', paired_reader.FROZEN / old_name),
                               ('trained', paired_reader.REVIEW / new_name)):
            manifest, complete, cfg = map(read, [folder / p for p in
                ('manifest.json', 'completed.json', 'resolved_env.json')])
            assert complete['completed'] and complete['samples'] == 1000 * count and not complete['resets']
            assert manifest['seconds_per_case'] == 20. and manifest['policy_dt'] == .02
            assert not manifest['sampled_policy_actions'] and not manifest['weights_updated']
            assert cfg['wheel_damping'] == 1. and cfg['obs_scales']['ang_vel'] == .25
            assert cfg['wheel_action_limit'] == 1.25 and cfg['action_scale_vel'] == 15.
            with np.load(folder / 'policy_inputs.npz', allow_pickle=False) as archive:
                observations = archive['observations'][-151:].copy()
                means = archive['means'][-151:].copy()
            assert observations.shape == (151, count, 27) and means.shape == (151, count, 6)
            assert np.isfinite(observations).all() and np.isfinite(means).all()
            fixtures[origin] = dict(manifest=manifest, cfg=cfg,
                x=torch.tensor(observations.reshape(-1, 27), dtype=torch.float64), recorded=means)
            for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'policy_inputs.npz'):
                pins[str(folder / name)] = sha(folder / name)
        assert fixtures['frozen']['manifest']['cases'] == fixtures['trained']['manifest']['cases']
        assert fixtures['frozen']['manifest']['seeds'] == fixtures['trained']['manifest']['seeds']
        cfg = fixtures['frozen']['cfg']
        ratio = cfg['wheel_radius'] / cfg['wheel_track_width']

        def proxy(raw):
            wheels = raw[:, [2, 5]].clamp(-1.25, 1.25) * 15.
            return ratio * (wheels[:, 1] - wheels[:, 0])

        outputs, sensitivities, means = {}, {}, {}
        with torch.inference_mode():
            for policy, state in states.items():
                for origin, fixture in fixtures.items():
                    x = fixture['x']
                    mean = actor(x, state)
                    means[policy, origin] = mean.numpy().reshape(151, count, 6)
                    outputs[policy, origin] = proxy(mean).numpy().reshape(151, count)
                    if origin == policy:
                        parity[scope + '_' + origin] = float(abs(means[policy, origin] - fixture['recorded']).max())
                        assert parity[scope + '_' + origin] < 2.e-5
                    local = {}
                    for column, name in ((2, 'physical_measured_yaw'), (7, 'physical_commanded_yaw')):
                        plus, minus = x.clone(), x.clone()
                        plus[:, column] += .001 * .25
                        minus[:, column] -= .001 * .25
                        local[name] = ((proxy(actor(plus, state)) - proxy(actor(minus, state))) / .002
                                       ).numpy().reshape(151, count)
                    sensitivities[policy, origin] = local
        ff, tf = outputs['frozen', 'frozen'], outputs['trained', 'frozen']
        ft, tt = outputs['frozen', 'trained'], outputs['trained', 'trained']
        parameter, state_change = .5 * ((tf - ff) + (tt - ft)), .5 * ((ft - ff) + (tt - tf))
        closure = float(abs(parameter + state_change - (tt - ff)).max())
        assert closure < 1.e-12
        closures.append(closure)
        manifest = fixtures['frozen']['manifest']
        paired = {(r['case']['name'], r['seed']): r for r in comparison['paired_trials'] if r['scope'] == scope}
        for cid, case in enumerate(manifest['cases']):
            for sid, seed in enumerate(manifest['seeds']):
                eid = cid * len(manifest['seeds']) + sid
                original = paired[case['name'], seed]
                trials.append(dict(scope=scope, case=case, seed=seed,
                    frozen_original_AND_literal10=original['frozen_original_AND_literal10'],
                    trained_original_AND_literal10=original['trained_original_AND_literal10'],
                    actual_signed_body_yaw_bias_rad_s={label: original[label]['metrics']['signed_bias_rad_s']
                        for label in ('frozen', 'trained')},
                    same_state_trained_minus_frozen_yaw_reference_proxy_rad_s=dict(
                        on_frozen_inputs=float((tf - ff)[:, eid].mean()),
                        on_trained_inputs=float((tt - ft)[:, eid].mean())),
                    symmetric_parameter_contribution_rad_s=float(parameter[:, eid].mean()),
                    symmetric_recorded_state_contribution_rad_s=float(state_change[:, eid].mean()),
                    total_native_reference_proxy_change_rad_s=float((tt - ff)[:, eid].mean()),
                    same_frozen_inputs_raw_action_changes_by_head=(means['trained', 'frozen'][:, eid]
                        - means['frozen', 'frozen'][:, eid]).mean(0).tolist(),
                    finite_difference_actor_proxy_sensitivities={policy + '_on_' + origin:
                        {name: float(value[:, eid].mean()) for name, value in v.items()}
                        for (policy, origin), v in sensitivities.items()}))
    assert len(trials) == 318 and torch.equal(rng, torch.random.get_rng_state())
    assert all(sha(Path(p)) == digest for p, digest in pins.items())
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        retained_trials=318, inclusive_tail_frames_per_trial=151, checkpoints=clocks,
        original_ELU_actor_reader_reused=True, own_actor_native_mean_roundoff_max_errors=parity,
        symmetric_output_decomposition_closure_max_error=max(closures),
        input_scaling='gyro2 and yaw-command7 are physical_rad_s*.25; yaw proxy=(wheel_radius/track)*15*(clipped head5-clipped head2)',
        physical_finite_difference_step_rad_s=.001, paired_trials=trials, source_pins=pins,
        RNG_unchanged=True, optimizer_constructed=False, optimizer_executed=False,
        simulator_started=False, training_modified=False, controller_changed=False,
        acceptance_changed=False, next_parameter_or_source_selected=False, policy_accepted=False,
        limitations=['Each actor is evaluated on both sets of recorded inputs; only own actor/own states reproduce the native trajectory.',
            'The parameter/state split is symmetric algebraic output accounting, not independent physical causes.',
            'These are local actor input sensitivities, not closed-loop feedback gains or proof of stability.',
            'This does not simulate the cross-policy trajectories or replace full735, dynamic and sustained acceptance.'])
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(verified=True, report=str(OUTPUT), trials=318,
        actor_native_roundoff_max_error=max(parity.values()),
        symmetric_closure_max_error=max(closures), RNG_unchanged=True, policy_accepted=False)))


if __name__ == '__main__':
    run()
