"""Compare actual weighted rewards and tracking on every frozen case/seed pair.

This is a read-only accounting audit. It never selects training cases or
changes a reward, controller, policy, optimizer, sampler or acceptance gate.
Recorded reward differences do not by themselves identify a causal PPO term.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_frozen_joint_pd_matrix import load, identity, numeric_command


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def samples(matrix):
    complete, scored = load(matrix / 'complete_frozen_command_matrix.json')
    plan = read(matrix / 'qualification.json')
    rows = {identity(row): row for row in scored}
    result, pins, contract = {}, {}, None
    for job in plan['batches']:
        directory = Path(job['replay'])
        manifest, cfg = [read(directory / n) for n in ('manifest.json', 'resolved_env.json')]
        current = {key: cfg[key] for key in ('rewards', 'leg_control_mode', 'wheel_control_mode',
            'leg_joint_stiffness', 'leg_joint_damping', 'wheel_damping', 'feedforward_force',
            'feedforward_mode', 'feedforward_angle_bias', 'feedforward_force_limit',
            'wheel_action_limit', 'action_scale_vel')}
        current['asset'] = cfg['robot']['spawn']['usd_path']
        current['reset_velocity_uniform_amplitude'] = manifest['reset_velocity_uniform_amplitude']
        current['policy_dt'] = manifest['policy_dt']
        if contract is None:
            contract = current
        assert current == contract
        assert manifest['checkpoint_sha256'] == complete['checkpoint_sha256']
        assert not manifest['weights_updated'] and not manifest['sampled_policy_actions']
        assert not manifest['policy_input_remapping']
        header = (directory / 'trace.csv').open().readline().strip().split(',')
        terms = [name for name in header if name.startswith('reward_') and name != 'reward_total']
        fields = ['time_s', 'case_id', 'seed', 'terminated', 'timeout',
                  'speed_cmd_m_s', 'yaw_cmd_rad_s', 'height_cmd_m',
                  'speed_m_s', 'yaw_rate_rad_s', 'height_m', 'reward_total', *terms]
        data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True, usecols=fields)
        steps = round(manifest['seconds_per_case'] / manifest['policy_dt'])
        width = len(manifest['cases']) * len(manifest['seeds'])
        assert len(data) == steps * width
        arrays = {name: data[name].reshape(steps, width) for name in fields}
        assert all(np.isfinite(value).all() for value in arrays.values())
        expected_cases = np.repeat(np.arange(len(manifest['cases'])), len(manifest['seeds']))
        expected_seeds = np.tile(manifest['seeds'], len(manifest['cases']))
        assert (arrays['case_id'] == expected_cases).all()
        assert (arrays['seed'] == expected_seeds).all()
        total_error = abs(sum(arrays[name] for name in terms) - arrays['reward_total']).max()
        assert total_error < 1.e-6, total_error
        seconds = manifest['seconds_per_case']
        masks = dict(full=np.ones(steps, dtype=bool), tail3=arrays['time_s'][:, 0] >= seconds - 3.)
        for ci, case in enumerate(manifest['cases']):
            for si, seed in enumerate(manifest['seeds']):
                column = ci * len(manifest['seeds']) + si
                key = case['name'], seed
                assert key in rows and key not in result
                row = rows[key]
                entry = dict(case=case, seed=seed, survived=row['survived'],
                             original_passed=row['original_passed'],
                             strict10_passed=row['original_AND_literal10_passed'],
                             nonzero_relative_percent=row['nonzero_relative_percent'], windows={})
                for window, mask in masks.items():
                    rewards = {name: float(arrays[name][mask, column].mean() / manifest['policy_dt'])
                               for name in ['reward_total', *terms]}
                    entry['windows'][window] = dict(
                        weighted_reward_rate_per_s=rewards,
                        speed_MAE_m_s=float(abs(arrays['speed_cmd_m_s'][mask, column] - arrays['speed_m_s'][mask, column]).mean()),
                        yaw_MAE_rad_s=float(abs(arrays['yaw_cmd_rad_s'][mask, column] - arrays['yaw_rate_rad_s'][mask, column]).mean()),
                        height_MAE_m=float(abs(arrays['height_cmd_m'][mask, column] - arrays['height_m'][mask, column]).mean()))
                if row['survived']:
                    assert abs(entry['windows']['tail3']['yaw_MAE_rad_s'] - row['metrics']['yaw_mae_rad_s']) < 1.e-12
                # Check the actually applied angular squared penalty from the
                # original reward equation, rather than a new acceptance rule.
                yaw_error2 = (arrays['yaw_cmd_rad_s'][:, column] - arrays['yaw_rate_rad_s'][:, column]) ** 2
                expected = np.clip(yaw_error2 * cfg['rewards']['yaw_rate_error_sq'],
                    -cfg['rewards']['yaw_rate_penalty_clip'], cfg['rewards']['clip_single_reward'])
                expected *= manifest['policy_dt']
                assert abs(expected - arrays['reward_yaw_rate_error_sq'][:, column]).max() < 2.e-6
                result[key] = entry
        for name in ('trace.csv', 'manifest.json', 'resolved_env.json', 'completed.json'):
            pins[str(directory / name)] = sha(directory / name)
    assert result.keys() == rows.keys()
    pins[str(matrix / 'complete_frozen_command_matrix.json')] = sha(matrix / 'complete_frozen_command_matrix.json')
    return complete, rows, result, pins, contract


def run(before, after, output):
    assert not output.exists(), 'Preserve earlier diagnostic reports'
    left_data, left_scores, left, pins, left_contract = samples(before)
    right_data, right_scores, right, more_pins, right_contract = samples(after)
    pins.update(more_pins)
    assert left_contract == right_contract, 'Isolate policy differences under identical physical/reward settings'
    assert left.keys() == right.keys()
    assert all(numeric_command(left_scores[k]) == numeric_command(right_scores[k]) for k in left)
    pairs = []
    for key in left:
        a, b = left[key], right[key]
        deltas = {}
        for window in ('full', 'tail3'):
            aa, bb = a['windows'][window], b['windows'][window]
            assert aa['weighted_reward_rate_per_s'].keys() == bb['weighted_reward_rate_per_s'].keys()
            deltas[window] = dict(
                reward_rate_per_s={name: bb['weighted_reward_rate_per_s'][name] - value
                                   for name, value in aa['weighted_reward_rate_per_s'].items()},
                yaw_MAE_rad_s=bb['yaw_MAE_rad_s'] - aa['yaw_MAE_rad_s'],
                speed_MAE_m_s=bb['speed_MAE_m_s'] - aa['speed_MAE_m_s'],
                height_MAE_m=bb['height_MAE_m'] - aa['height_MAE_m'])
        pairs.append(dict(before=a, after=b, deltas=deltas))
    summaries = {}
    for window in ('full', 'tail3'):
        # Every pair stays in the report. A mean reward from a reset trajectory
        # cannot establish success; survival and original gates remain separate.
        valid = [p for p in pairs if p['before']['survived'] and p['after']['survived']]
        def delta(p):
            return p['deltas'][window]
        names = pairs[0]['deltas'][window]['reward_rate_per_s']
        summaries[window] = dict(
            paired_survivors=len(valid), failed_pair_count=len(pairs) - len(valid),
            mean_reward_rate_delta_per_s={name: float(np.mean([delta(p)['reward_rate_per_s'][name] for p in valid]))
                                          for name in names},
            reward_increased_pairs=sum(delta(p)['reward_rate_per_s']['reward_total'] > 1.e-8 for p in valid),
            yaw_MAE_increased_pairs=sum(delta(p)['yaw_MAE_rad_s'] > 1.e-8 for p in valid),
            reward_increased_while_yaw_MAE_increased_pairs=sum(
                delta(p)['reward_rate_per_s']['reward_total'] > 1.e-8 and delta(p)['yaw_MAE_rad_s'] > 1.e-8
                for p in valid),
            newly_strict10_failed_pairs=sum(p['before']['strict10_passed'] and not p['after']['strict10_passed'] for p in pairs))
    pins[str(Path(__file__).resolve())] = sha(__file__)
    assert all(sha(path) == digest for path, digest in pins.items())
    report = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        before_checkpoint_sha256=left_data['checkpoint_sha256'],
        after_checkpoint_sha256=right_data['checkpoint_sha256'],
        paired_trials=len(pairs), all_numeric_commands_and_seeds_matched=True,
        matched_physics_control_rewards_and_reset_contract=left_contract,
        summary=summaries, all_pairs=pairs, source_pins=pins,
        all_original_failure_rows_retained=True, recorded_reward_sum_and_yaw_MSE_reproduced=True,
        training_modified=False, optimizer_executed=False, simulator_started=False,
        acceptance_changed=False, case_specific_training_or_control_created=False,
        limitation='Frozen paired reward accounting can expose objective tradeoffs. It cannot by itself identify '
                   'a causal PPO gradient, prove a local optimum, or replace the independent acceptance gates.',
        policy_accepted=False)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.before.resolve(), args.after.resolve(), args.output.resolve())
    print(json.dumps(dict(verified=True, paired_trials=result['paired_trials'], summary=result['summary']), ensure_ascii=False))
