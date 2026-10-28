"""Account for command, policy wheel targets, measured wheels and body motion.

All matrix trials remain in the report. Wheel rolling proxies are kinematic
diagnostics, not measurements of contact slip or actual PhysX drive torque.
This utility cannot change a policy, training, controller or acceptance gate.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = ('time_s', 'case_id', 'seed', 'episode', 'terminated', 'timeout',
          'speed_cmd_m_s', 'yaw_cmd_rad_s', 'height_cmd_m', 'speed_m_s',
          'yaw_rate_rad_s', 'height_m', 'left_wheel_ref_rad_s', 'right_wheel_ref_rad_s',
          'left_wheel_rad_s', 'right_wheel_rad_s', 'pitch_rad', 'roll_rad')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def audit(matrix, output):
    assert not output.exists(), 'Preserve previous diagnostics'
    plan = read(matrix / 'qualification.json')
    completed = read(matrix / 'complete_frozen_command_matrix.json')
    assert completed['complete']
    scored = {(r['case']['name'], r['seed']): r for r in completed['all_trials']}
    pins = {str(p): sha(p) for p in (Path(__file__).resolve(),
            matrix / 'qualification.json', matrix / 'complete_frozen_command_matrix.json')}
    rows = []
    errors = dict(mean_to_wheel_reference=0., yaw_command_observation=0.,
                  yaw_measurement_observation=0., original_MAE=0., signed_chain_closure=0.)
    for job in plan['batches']:
        directory = Path(job['replay'])
        manifest, cfg, done = [read(directory / name) for name in
                              ('manifest.json', 'resolved_env.json', 'completed.json')]
        assert manifest['checkpoint_sha256'] == completed['checkpoint_sha256']
        assert done['completed'] and not manifest['weights_updated']
        assert not manifest['sampled_policy_actions'] and not manifest['policy_input_remapping']
        assert cfg['leg_control_mode'] == 'implicit_joint_position'
        steps = round(manifest['seconds_per_case'] / manifest['policy_dt'])
        width = len(manifest['cases']) * len(manifest['seeds'])
        trace = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True, usecols=FIELDS)
        assert len(trace) == steps * width == done['samples']
        assert all(np.isfinite(trace[name]).all() for name in FIELDS)
        values = {name: trace[name].reshape(steps, width) for name in FIELDS}
        expected_ids = np.repeat(np.arange(len(manifest['cases'])), len(manifest['seeds']))
        expected_seeds = np.tile(manifest['seeds'], len(manifest['cases']))
        assert (values['case_id'] == expected_ids).all() and (values['seed'] == expected_seeds).all()
        with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as archive:
            observations, means = archive['observations'], archive['means']
        assert observations.shape == (steps, width, 27) and means.shape == (steps, width, 6)
        limit, scale = cfg['wheel_action_limit'], cfg['action_scale_vel']
        reference = means[..., (2, 5)].clip(-limit, limit) * scale
        actual = np.stack([values[name] for name in ('left_wheel_ref_rad_s', 'right_wheel_ref_rad_s')], -1)
        errors['mean_to_wheel_reference'] = max(errors['mean_to_wheel_reference'], float(abs(reference - actual).max()))
        angular_scale = cfg['obs_scales']['ang_vel']
        errors['yaw_command_observation'] = max(errors['yaw_command_observation'],
            float(abs(observations[..., 7] / angular_scale - values['yaw_cmd_rad_s']).max()))
        # Across an automatic reset the next observation belongs to a new state.
        continuous = ~(values['terminated'][:-1].astype(bool) | values['timeout'][:-1].astype(bool))
        measured_error = abs(observations[1:, :, 2] / angular_scale - values['yaw_rate_rad_s'][:-1])
        errors['yaw_measurement_observation'] = max(errors['yaw_measurement_observation'],
            float(measured_error[continuous].max(initial=0.)))
        radius, track = cfg['wheel_radius'], cfg['wheel_track_width']
        tail = values['time_s'][:, 0] >= manifest['seconds_per_case'] - 3.
        for ci, case in enumerate(manifest['cases']):
            for si, seed in enumerate(manifest['seeds']):
                index = ci * len(manifest['seeds']) + si
                original = scored[(case['name'], seed)]
                if not original['survived']:
                    rows.append(dict(case=case, seed=seed, survived=False,
                                     steady_diagnostic=None, acceptance_metrics=original['metrics']))
                    continue
                sample = {name: value[tail, index] for name, value in values.items()}
                axes = {}
                for axis, body_field, command_field, metric in (
                    ('speed', 'speed_m_s', 'speed_cmd_m_s', 'speed_mae_m_s'),
                    ('yaw', 'yaw_rate_rad_s', 'yaw_cmd_rad_s', 'yaw_mae_rad_s')):
                    if axis == 'yaw':
                        ref = radius / track * (sample['right_wheel_ref_rad_s'] - sample['left_wheel_ref_rad_s'])
                        measured = radius / track * (sample['right_wheel_rad_s'] - sample['left_wheel_rad_s'])
                    else:
                        ref = radius / 2. * (sample['right_wheel_ref_rad_s'] + sample['left_wheel_ref_rad_s'])
                        measured = radius / 2. * (sample['right_wheel_rad_s'] + sample['left_wheel_rad_s'])
                    command, body = sample[command_field], sample[body_field]
                    chain = (float((command - ref).mean()), float((ref - measured).mean()),
                             float((measured - body).mean()))
                    errors['signed_chain_closure'] = max(errors['signed_chain_closure'],
                        abs(sum(chain) - float((command - body).mean())))
                    mae = float(abs(command - body).mean())
                    errors['original_MAE'] = max(errors['original_MAE'], abs(mae - original['metrics'][metric]))
                    axes[axis] = dict(command_mean=float(command.mean()), body_mean=float(body.mean()),
                        body_MAE=mae, relative_percent=original['nonzero_relative_percent'][axis],
                        wheel_reference_rolling_proxy_mean=float(ref.mean()),
                        measured_wheel_rolling_proxy_mean=float(measured.mean()),
                        signed_command_to_reference_gap=chain[0],
                        signed_reference_to_measured_gap=chain[1],
                        signed_measured_to_body_gap=chain[2])
                rows.append(dict(case=case, seed=seed, survived=True, steady_diagnostic=axes,
                    wheel_only_raw_clipping_fraction=float((abs(means[tail, index][:, (2, 5)]) > limit).mean()),
                    minimum_wheel_target_headroom_rad_s=float(limit * scale - abs(reference[tail, index]).max()),
                    original_passed=original['original_passed'],
                    original_AND_literal10_passed=original['original_AND_literal10_passed']))
        for name in ('manifest.json', 'resolved_env.json', 'completed.json', 'trace.csv', 'policy_inputs.npz'):
            pins[str(directory / name)] = sha(directory / name)
    assert len(rows) == len(scored)
    assert errors['mean_to_wheel_reference'] < 1.e-5
    assert errors['yaw_command_observation'] < 1.e-6 and errors['yaw_measurement_observation'] < 1.e-6
    assert errors['original_MAE'] < 1.e-12 and errors['signed_chain_closure'] < 1.e-12
    assert all(sha(path) == digest for path, digest in pins.items())
    report = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  checkpoint_sha256=completed['checkpoint_sha256'], trials=len(rows),
                  verification_max_errors=errors, all_trials=rows, source_pins=pins,
                  failed_trials_retained=True, simulator_started=False, optimizer_executed=False,
                  controller_or_training_modified=False, acceptance_changed=False, policy_accepted=False,
                  limitation='Signed rolling-proxy gaps are accounting identities. They do not measure tire slip, '
                  'actual implicit-drive torque, causal feedback gains or the PPO term that learned a bias.')
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--matrix', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.matrix.resolve(), args.output.resolve())
    print(json.dumps(dict(verified=result['verified'], trials=result['trials'],
                          errors=result['verification_max_errors'], output=str(args.output))))
