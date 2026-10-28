"""Read-only, identical-input comparison of the two same-source D1 trials.

Use the original ELU actor on both sets of the already validated318 replay
tails.  These are output/sensitivity comparisons, never new closed-loop
trajectories or an acceptance decision.  No simulator, optimizer or backward.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diagnose_height_response import actor
import audit_paired_yaw_kernel_trace_budget as trace
import audit_wheel005_D100_frozen7201_vs_trained7401_318 as recorded
import joint_pd_wheel005_source7201_D100_runtime as control_runtime
import joint_pd_wheel005_source7201_D100_yaw400_runtime as trial_runtime


SCRIPT = Path(__file__).resolve()
ROOT = recorded.ROOT
TRIAL = ROOT / 'mine_model/results/joint_pd_optimization_20261024_065638_819339_c26a87bd_wheel005_source7201_D100_yaw400_200'
PROOF = trial_runtime.QUALIFICATION / 'same_source_D1_frozen27_reward400_application_readonly_20261024.json'


def run(through):
    assert through in (50, 100, 200)
    output = TRIAL / f'controlled_C200_C400_same_input318_prefix_{through:04d}_readonly_20261024.json'
    assert not output.exists()
    read, sha = trace.read, trace.sha
    pins = {}

    def consume(path):
        path = Path(path)
        pins[str(path)] = sha(path)
        return read(path)

    for path in (SCRIPT, recorded.SCRIPT, trace.SCRIPT,
                 Path(actor.__code__.co_filename), control_runtime.SCRIPT,
                 trial_runtime.SCRIPT, trace.GATE):
        pins[str(path)] = sha(path)
    assert sha(trace.GATE) == trace.GATE_SHA
    proof = consume(PROOF)
    assert proof['verified'] and proof['same_exact_source_NN_and_D1']
    assert proof['trace_physics_fields_bitwise_equal']
    assert proof['declared_environment_configuration_differences'] == [
        dict(path='/rewards/yaw_rate_error_sq', before=-200., after=-400.)]
    previous = consume(recorded.OUTPUT)
    assert previous['verified'] and len(previous['paired_trials']) == 318
    full = consume(recorded.TRAINING / 'final_checkpoint_verification_readonly_20261019.json')
    assert full['verified'] and full['checkpoint_sha256'] == recorded.TARGET_SHA
    source, initial, _ = trial_runtime.baseline()
    checkpoints = {'source7201': (source, initial, 7201),
                   'prior_C200_final7401': (Path(full['checkpoint']), None, 7401)}
    for label, folder, runtime, healthname in (
            ('C200_prefix', recorded.TRAINING, control_runtime, 'wheel005_D100'),
            ('C400_prefix', TRIAL, trial_runtime, 'wheel005_D100_yaw400')):
        health = consume(folder / f'learning_health_{healthname}_prefix_{through:04d}_readonly.json')
        assert health['qualified'] and health['committed_prefix'] == through
        assert health['all9_native_alignment_errors_zero']
        update = consume(folder / f'policy_update_health_prefix_{through:04d}_readonly.json')['snapshots'][-1]
        assert update['committed_update'] == through
        checkpoint = Path(update['checkpoint_after'])
        assert sha(checkpoint) == update['checkpoint_after_sha256']
        saved, _ = runtime.require_source(checkpoint)
        checkpoints[label] = checkpoint, saved, 7201 + through
    prior, _ = control_runtime.require_source(checkpoints['prior_C200_final7401'][0])
    checkpoints['prior_C200_final7401'] = checkpoints['prior_C200_final7401'][0], prior, 7401

    torch.set_num_threads(1)
    rng = torch.random.get_rng_state().clone()
    states, clocks = {}, {}
    for label, (checkpoint, saved, updates) in checkpoints.items():
        assert len(saved['optimizer_state_dict']['state']) == 17
        assert all(float(s['step']) == updates * 20 for s in saved['optimizer_state_dict']['state'].values())
        config = checkpoint.parent / 'params/agent.yaml'
        policy = yaml.safe_load(config.read_text())['policy']
        assert policy['actor_hidden_dims'] == [256, 128, 64] and policy['activation'] == 'elu'
        assert not policy.get('actor_obs_normalization', False)
        states[label] = {k: v.to(torch.float64) for k, v in saved['model_state_dict'].items()
                         if k.startswith('actor.')}
        assert states[label]['actor.0.weight'].shape == (256, 27)
        assert states[label]['actor.6.weight'].shape == (6, 64)
        clocks[label] = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            actual_updates=updates, all17_Adam_step=updates * 20,
            saved_std=saved['model_state_dict']['std'].tolist(),
            actual_learning_rates=[g['lr'] for g in saved['optimizer_state_dict']['param_groups']])
        pins.update({str(checkpoint): sha(checkpoint), str(config): sha(config),
                     str(checkpoint.parent / 'params/env.yaml'): sha(checkpoint.parent / 'params/env.yaml')})

    trials, parity = [], {}
    for scope, _, old_name, new_name, count in recorded.SCOPES:
        fixtures = {}
        for origin, folder in (('frozen_source7201_D1', recorded.FROZEN / old_name),
                               ('prior_C200_final7401_D1', recorded.REVIEW / new_name)):
            manifest, complete, cfg = [consume(folder / p) for p in
                ('manifest.json', 'completed.json', 'resolved_env.json')]
            assert complete['completed'] and complete['samples'] == 1000 * count and not complete['resets']
            assert manifest['seconds_per_case'] == 20. and manifest['policy_dt'] == .02
            assert not manifest['sampled_policy_actions'] and not manifest['weights_updated']
            assert cfg['wheel_damping'] == 1. and cfg['obs_scales']['ang_vel'] == .25
            assert cfg['wheel_action_limit'] == 1.25 and cfg['action_scale_vel'] == 15.
            path = folder / 'policy_inputs.npz'
            pins[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as archive:
                observations = archive['observations'][-151:].copy()
                means = archive['means'][-151:].copy()
            assert observations.shape == (151, count, 27) and means.shape == (151, count, 6)
            assert np.isfinite(observations).all() and np.isfinite(means).all()
            fixtures[origin] = dict(manifest=manifest, cfg=cfg,
                x=torch.tensor(observations.reshape(-1, 27), dtype=torch.float64), means=means)
        origins = list(fixtures)
        left, right = (fixtures[k]['manifest'] for k in origins)
        assert left['cases'] == right['cases'] and left['seeds'] == right['seeds']
        cfg = fixtures[origins[0]]['cfg']
        ratio = cfg['wheel_radius'] / cfg['wheel_track_width']

        def proxies(raw):
            wheels = raw[:, [2, 5]].clamp(-1.25, 1.25) * 15.
            return ratio * (wheels[:, 1] - wheels[:, 0]), .5 * cfg['wheel_radius'] * wheels.sum(-1)

        outputs = {}
        with torch.inference_mode():
            for origin, fixture in fixtures.items():
                x = fixture['x']
                for label, state in states.items():
                    mean = actor(x, state)
                    yaw, common = proxies(mean)
                    v = dict(raw_mean=mean.numpy().reshape(151, count, 6),
                        yaw_proxy=yaw.numpy().reshape(151, count),
                        common_proxy=common.numpy().reshape(151, count))
                    for col, key in ((2, 'measured_yaw_sensitivity'), (7, 'command_yaw_sensitivity')):
                        plus, minus = x.clone(), x.clone()
                        plus[:, col] += .001 * .25
                        minus[:, col] -= .001 * .25
                        yp, _ = proxies(actor(plus, state))
                        ym, _ = proxies(actor(minus, state))
                        v[key] = ((yp - ym) / .002).numpy().reshape(151, count)
                    outputs[origin, label] = v
                    if (origin == origins[0] and label == 'source7201') or (
                            origin == origins[1] and label == 'prior_C200_final7401'):
                        parity[scope + '_' + origin] = float(abs(v['raw_mean'] - fixture['means']).max())
                        assert parity[scope + '_' + origin] < 2.e-5
        for cid, case in enumerate(left['cases']):
            for sid, seed in enumerate(left['seeds']):
                eid = cid * len(left['seeds']) + sid
                row = dict(scope=scope, case=case, seed=seed, recorded_inputs={})
                for origin in origins:
                    policies = {}
                    for label in states:
                        v = outputs[origin, label]
                        policies[label] = {key: float(v[key][:, eid].mean()) for key in
                            ('yaw_proxy', 'common_proxy', 'measured_yaw_sensitivity', 'command_yaw_sensitivity')}
                        policies[label]['raw_action_mean'] = v['raw_mean'][:, eid].mean(0).tolist()
                    a, b = (outputs[origin, k] for k in ('C200_prefix', 'C400_prefix'))
                    row['recorded_inputs'][origin] = dict(policies=policies,
                        C400_minus_C200_yaw_reference_proxy_rad_s=float((b['yaw_proxy'] - a['yaw_proxy'])[:, eid].mean()),
                        C400_minus_C200_common_reference_proxy_m_s=float((b['common_proxy'] - a['common_proxy'])[:, eid].mean()),
                        C400_minus_C200_raw_action_mean_by_head=(b['raw_mean'] - a['raw_mean'])[:, eid].mean(0).tolist())
                trials.append(row)
    assert len(trials) == 318 and torch.equal(rng, torch.random.get_rng_state())
    assert all(sha(Path(p)) == s for p, s in pins.items())
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        committed_controlled_prefix=through, fixed_input_trials=318,
        retained_tail_frames_per_trial=151, recorded_state_origins=origins,
        original_ELU_actor_reader_reused=True, checkpoints=clocks,
        native_own_actor_parity_max_errors=parity,
        body_yaw_reference_proxy_formula='(wheel_radius/track)*15*(clipped_head5-clipped_head2)',
        common_reference_proxy_formula='.5*wheel_radius*15*(clipped_head2+clipped_head5)',
        wheel_action_clamp=1.25, finite_difference_physical_yaw_step_rad_s=.001,
        gyro_and_yaw_command_observation_scale=.25, paired_trials=trials, source_pins=pins,
        RNG_unchanged=True, optimizer_constructed=False, optimizer_executed=False,
        backward_executed=False, simulator_started=False, training_modified=False,
        new_C400_physical_trajectory_generated=False, closed_loop_acceptance=False,
        acceptance_changed=False, next_parameter_or_source_selected=False, policy_accepted=False,
        limitations=['Both prefix policies see identical inputs; source and prior-final policies verify the two recorded native origins.',
            'C400 prefix has no new frozen closed-loop trajectory in this report.',
            'Wheel-reference proxies and finite-difference input sensitivities are not achieved body motion or closed-loop feedback gains.',
            'The original735, static810, motion810, signed540, original120 and small120 independent reviews remain required.'])
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(verified=True, report=str(output), fixed_trials=318,
        prefix=through, native_parity_max_error=max(parity.values()),
        optimizer_executed=False, simulator_started=False, policy_accepted=False)))


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--through', type=int, choices=(50, 100, 200), required=True)
    run(cli.parse_args().through)
