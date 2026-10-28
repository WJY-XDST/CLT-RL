"""Read frozen telemetry and audit global yaw reward sensitivity.

No policy, optimizer, simulator, command sampler or acceptance mutation occurs.
State-reward derivatives are diagnostic; they are not PPO parameter gradients.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2 ** 20), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def yaw_terms(error, quality, rewards, dt):
    """The environment's two yaw terms, including individual term bounds."""
    coefficient = rewards['tracking_ang_vel']
    sigma = rewards['tracking_sigma_ang']
    cost = rewards['yaw_rate_error_sq']
    bound = rewards['clip_single_reward'] * dt
    cost_bound = rewards['yaw_rate_penalty_clip'] * dt
    bonus_raw = coefficient * dt * quality * math.exp(-error * error / sigma)
    cost_raw = cost * dt * error * error
    bonus = max(-bound, min(bound, bonus_raw))
    penalty = max(-cost_bound, min(bound, cost_raw))
    bonus_derivative = (-2 * error * bonus_raw / sigma
                        if -bound < bonus_raw < bound else 0.)
    cost_derivative = 2 * cost * dt * error if -cost_bound < cost_raw < bound else 0.
    return bonus, penalty, bonus_derivative + cost_derivative


def quality_from_row(row, criteria, floor):
    height_ratio = abs(row['height_m'] - row['height_cmd_m']) / (
        criteria['relative_error'] * max(abs(row['height_cmd_m']), .001))
    body_ratio = max(abs(row['pitch_rad']), abs(row['roll_rad'])) / math.radians(criteria['body_p95_deg'])
    height_excess, body_excess = max(0., height_ratio - .5), max(0., body_ratio - .5)
    return floor + (1 - floor) / ((1 + height_excess ** 2) * (1 + body_excess ** 2))


def summarize(values):
    if not values:
        return None
    values = sorted(values)
    return dict(samples=len(values), mean=statistics.mean(values), median=statistics.median(values),
                minimum=values[0], maximum=values[-1])


def audit(sources, plan_path, output, sigma_factor):
    assert not output.exists(), 'Do not overwrite earlier diagnostics'
    assert math.isfinite(sigma_factor) and sigma_factor > 0.
    plan = read(plan_path)
    current = plan['training_config']['rewards']
    assert plan['ordinary_full_policy_PPO'] and not plan['action_map_or_controller_changed']
    assert not plan['acceptance_revision'].get('acceptance_changed', False)
    pins = {str(p.resolve()): sha(p) for p in (SCRIPT, plan_path,
        ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/reward_signals.py',
        ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py')}
    rows, scopes = [], []
    replay_max_error = dict(angular_bonus=0., squared_yaw_cost=0.)
    numerical_derivative_max_error = 0.
    for source in sources:
        files = [source / name for name in ('manifest.json', 'completed.json', 'resolved_env.json',
                 'trace.csv', 'original_and_literal10_conditional_assessment.json')]
        pins.update({str(p.resolve()): sha(p) for p in files})
        manifest, complete, config, _, assessed = [read(p) if p.suffix == '.json' else None for p in files]
        assert complete['completed'] and not manifest['sampled_policy_actions'] and not manifest['weights_updated']
        assert manifest['checkpoint_sha256'] == plan['source_sha256']
        assert assessed['verified'] and not assessed['acceptance_changed']
        assert all(sha(p) == digest for p, digest in assessed['source_pins'].items())
        original = config['rewards']
        dt = manifest['policy_dt']
        assert dt == .02 and config['leg_control_mode'] == 'implicit_joint_position'
        evidence = manifest['tracking_goal_feedback_criteria']
        assert sha(evidence['path']) == evidence['sha256']
        criteria = evidence['criteria']
        assert criteria['relative_error'] == .1
        relevant = ['tracking_sigma_ang', 'yaw_rate_error_sq', 'clip_single_reward',
                    'yaw_rate_penalty_clip', 'tracking_task_quality_gate_floor']
        assert all(original[key] == current[key] for key in relevant)
        hypothetical = dict(current, tracking_sigma_ang=current['tracking_sigma_ang'] * sigma_factor)
        variants = dict(recorded_source=original, current_global_bonus=current, diagnostic_width=hypothetical)
        for rewards in variants.values():
            assert rewards['tracking_ang_vel'] > 0 and rewards['tracking_sigma_ang'] > 0
            assert rewards['yaw_rate_error_sq'] <= 0
            for error in (-.15, -.05, -.01, 0., .01, .05, .15):
                analytic = yaw_terms(error, .9, rewards, dt)[2]
                plus = sum(yaw_terms(error + 1.e-6, .9, rewards, dt)[:2])
                minus = sum(yaw_terms(error - 1.e-6, .9, rewards, dt)[:2])
                numerical_derivative_max_error = max(numerical_derivative_max_error,
                    abs(analytic - (plus - minus) / 2.e-6))
        count = defaultdict(int)
        per_trial = defaultdict(lambda: defaultdict(list))
        floor = original['tracking_task_quality_gate_floor']
        with files[3].open(newline='') as stream:
            for raw in csv.DictReader(stream):
                case_id, seed = int(float(raw['case_id'])), int(float(raw['seed']))
                assert 0 <= case_id < len(manifest['cases']) and seed in manifest['seeds']
                key = case_id, seed
                count[key] += 1
                if float(raw['time_s']) < manifest['seconds_per_case'] - 3.:
                    continue
                row = {name: float(raw[name]) for name in ['time_s', 'episode', 'terminated', 'timeout',
                    'height_m', 'height_cmd_m', 'pitch_rad', 'roll_rad', 'yaw_rate_rad_s',
                    'yaw_cmd_rad_s', 'reward_track_ang_vel', 'reward_yaw_rate_error_sq']}
                assert all(math.isfinite(value) for value in row.values())
                assert not any(row[name] for name in ('episode', 'terminated', 'timeout'))
                error = row['yaw_rate_rad_s'] - row['yaw_cmd_rad_s']
                quality = quality_from_row(row, criteria, floor)
                reproduction = yaw_terms(error, quality, original, dt)
                replay_max_error['angular_bonus'] = max(replay_max_error['angular_bonus'],
                    abs(reproduction[0] - row['reward_track_ang_vel']))
                replay_max_error['squared_yaw_cost'] = max(replay_max_error['squared_yaw_cost'],
                    abs(reproduction[1] - row['reward_yaw_rate_error_sq']))
                per_trial[key]['abs_error_rad_s'].append(abs(error))
                per_trial[key]['quality'].append(quality)
                for name, rewards in variants.items():
                    bonus, penalty, derivative = yaw_terms(error, quality, rewards, dt)
                    ideal = sum(yaw_terms(0., quality, rewards, dt)[:2])
                    per_trial[key][name + '_reward_deficit'].append(ideal - bonus - penalty)
                    per_trial[key][name + '_abs_state_derivative'].append(abs(derivative))
                    per_trial[key][name + '_bonus_at_upper_bound'].append(
                        int(bonus == rewards['clip_single_reward'] * dt))
        expected = round(manifest['seconds_per_case'] / dt)
        assert len(count) == len(manifest['cases']) * len(manifest['seeds'])
        assert sum(count.values()) == complete['samples'] and all(value == expected for value in count.values())
        by_key = {(t['case']['name'], t['seed']): t for t in assessed['trials']}
        for (case_id, seed), values in per_trial.items():
            case = manifest['cases'][case_id]
            original_trial = by_key[case['name'], seed]
            assert original_trial['survived']
            assert len(values['quality']) == 151
            rows.append(dict(scope=str(source.resolve()), case=case, seed=seed,
                original_passed=original_trial['original_passed'],
                strict_and_original_passed=original_trial['original_AND_literal10_passed'],
                yaw_nonzero_relative_percent=original_trial['nonzero_relative_percent']['yaw'],
                metrics={key: summarize(value) for key, value in values.items()}))
        scopes.append(dict(source=str(source.resolve()), trials=len(per_trial), source_checkpoint_sha256=manifest['checkpoint_sha256'],
                           recorded_bonus_coefficient=original['tracking_ang_vel']))
    assert replay_max_error['angular_bonus'] < 1.e-7 and replay_max_error['squared_yaw_cost'] < 1.e-7
    assert numerical_derivative_max_error < 1.e-7
    assert all(sha(path) == digest for path, digest in pins.items())
    metric_keys = list(rows[0]['metrics'])
    groups = {}
    # These labels organize a read-only report; none enters training or sampling.
    for name, members in [('all_trials', rows),
                          ('strict_passed', [row for row in rows if row['strict_and_original_passed']]),
                          ('strict_failed', [row for row in rows if not row['strict_and_original_passed']])]:
        groups[name] = dict(trials=len(members),
            trial_mean_metrics={key: summarize([row['metrics'][key]['mean'] for row in members]) for key in metric_keys})
    report = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(), scopes=scopes,
        scoped_trials=len(rows), current_bonus_coefficient=current['tracking_ang_vel'],
        current_sigma=current['tracking_sigma_ang'], diagnostic_sigma_factor=sigma_factor,
        source_reward_reconstruction_max_absolute_error=replay_max_error,
        analytic_vs_central_difference_max_error=numerical_derivative_max_error,
        groups=groups, all_trials=rows, source_pins=pins,
        original_assessments_unchanged=True, acceptance_changed=False, training_modified=False,
        simulator_started=False, optimizer_constructed=False, optimizer_executed=False,
        parameter_change_selected=False, policy_accepted=False,
        limitations=['State derivatives are not PPO policy gradients, advantages, or demonstrated learning benefit.',
                     'All variants are algebraic rescoring of the same source states; no hypothetical trajectory is generated.',
                     'Only yaw terms are rescored; the full reward, policy distribution, entropy and shared gradients remain relevant.',
                     'Passing and failing labels only partition diagnostic output; no case values enter training.',
                     'Current independent candidate replay is still required before choosing any subsequent configuration.'])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return {key: report[key] for key in ['verified', 'scoped_trials', 'source_reward_reconstruction_max_absolute_error',
                                        'analytic_vs_central_difference_max_error', 'groups']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, nargs='+', required=True)
    parser.add_argument('--current-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--diagnostic-sigma-factor', type=float, default=1.)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.current_plan, args.output, args.diagnostic_sigma_factor),
                     ensure_ascii=False, indent=2))
