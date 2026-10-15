"""Audit actual frozen replay rewards; local state derivatives are diagnostic.

Requires --record-reward-terms telemetry and complete pre-reset traces. It never
changes rewards, policy actions, checkpoints, acceptance or training state.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

METRICS = ('speed', 'yaw', 'height', 'leg_length', 'leg_angle', 'wheel_speed', 'body')


def huber_excess(ratios):
    excess = np.maximum(ratios - 1., 0.)
    return .5 * np.minimum(excess, 1.)**2 + np.maximum(excess - 1., 0.)


def audit(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    completed = json.loads((directory/'completed.json').read_text())
    cfg = json.loads((directory/'resolved_env.json').read_text())
    rewards = cfg['rewards']
    data = np.genfromtxt(directory/'trace.csv', delimiter=',', names=True)
    dt = manifest['policy_dt']
    expected = round(manifest['seconds_per_case']/dt)*len(manifest['cases'])*len(manifest['seeds'])
    fields = [key for key in data.dtype.names if key.startswith('reward_') and key != 'reward_total']
    if (not completed.get('completed') or len(data) != expected or not fields
            or not all(np.isfinite(data[key]).all() for key in data.dtype.names)):
        raise ValueError('Need complete finite actual reward telemetry')
    sum_error = float(np.max(np.abs(sum(data[key] for key in fields)-data['reward_total'])))
    if sum_error > 1.e-5:
        raise ValueError('Actual reward total differs from recorded components')
    criteria = manifest['tracking_goal_feedback_criteria']['criteria']
    fraction = criteria['relative_error']
    length_ref = np.stack([data[s+'_length_ref_m'] for s in ('left', 'right')], -1)
    wheel_ref = np.stack([data[s+'_wheel_ref_rad_s'] for s in ('left', 'right')], -1)
    length_error = np.maximum(*[abs(data[s+'_length_m']-data[s+'_length_ref_m']) for s in ('left', 'right')])
    angle_error = np.maximum(*[abs(data[s+'_angle_rad']-data[s+'_angle_ref_rad']) for s in ('left', 'right')])
    wheel_error = np.maximum(*[abs(data[s+'_wheel_rad_s']-data[s+'_wheel_ref_rad_s']) for s in ('left', 'right')])
    ratios = np.stack([
        abs(data['speed_m_s']-data['speed_cmd_m_s'])/np.maximum(fraction*abs(data['speed_cmd_m_s']), criteria['speed_absolute_m_s']),
        abs(data['yaw_rate_rad_s']-data['yaw_cmd_rad_s'])/np.maximum(fraction*abs(data['yaw_cmd_rad_s']), criteria['yaw_absolute_rad_s']),
        abs(data['height_m']-data['height_cmd_m'])/(fraction*np.maximum(abs(data['height_cmd_m']), .001)),
        length_error/(fraction*np.maximum(np.min(abs(length_ref), -1), .001)),
        angle_error/(fraction*criteria['leg_angle_design_range_rad']),
        wheel_error/np.maximum(fraction*np.min(abs(wheel_ref), -1), criteria['wheel_absolute_rad_s']),
        np.maximum(abs(data['pitch_rad']), abs(data['roll_rad']))*180./np.pi/criteria['body_p95_deg'],
    ], -1)

    def height_reward(height):
        excess = np.maximum(abs(height-data['height_cmd_m'])
                            - rewards['base_height_relative_deadband']*abs(data['height_cmd_m']), 0.)
        return rewards['base_height']*np.exp(-excess**2/rewards['base_height_sigma'])

    def goal_rewards(values):
        worst = np.maximum(rewards['tracking_goal_violation']*huber_excess(np.max(values, -1)),
                           -rewards['tracking_goal_penalty_clip'])
        components = np.maximum(rewards['tracking_goal_component_violation']*huber_excess(values).sum(-1),
                                -rewards['tracking_goal_component_penalty_clip'])
        return worst, components

    worst, components = goal_rewards(ratios)
    def recorded_reward(name):
        field='reward_'+name
        if field in data.dtype.names:
            return data[field]
        # The environment omits opt-in terms whose coefficient is zero.
        # Only a disabled term may be reconstructed as zero; never hide
        # missing telemetry for an active training signal.
        if rewards[name] != 0.:
            raise ValueError('Missing active reward telemetry: '+name)
        return np.zeros(len(data))

    discrepancies = dict(base_height=float(np.max(abs(dt*height_reward(data['height_m'])-recorded_reward('base_height')))),
        tracking_goal_violation=float(np.max(abs(dt*worst-recorded_reward('tracking_goal_violation')))),
        tracking_goal_component_violation=float(np.max(abs(dt*components-recorded_reward('tracking_goal_component_violation')))))
    if max(discrepancies.values()) > 1.e-5:
        raise ValueError('Independent reward formulas disagree with actual environment telemetry: '+str(discrepancies))

    epsilon = 1.e-4
    plus, minus = ratios.copy(), ratios.copy()
    plus[:, 2] = abs(data['height_m']+epsilon-data['height_cmd_m'])/(fraction*abs(data['height_cmd_m']))
    minus[:, 2] = abs(data['height_m']-epsilon-data['height_cmd_m'])/(fraction*abs(data['height_cmd_m']))
    plus_worst, plus_components = goal_rewards(plus)
    minus_worst, minus_components = goal_rewards(minus)
    derivative = ((height_reward(data['height_m']+epsilon)-height_reward(data['height_m']-epsilon))
                  + plus_worst-minus_worst + plus_components-minus_components)/(2.*epsilon)
    report = dict(checkpoint=manifest['checkpoint'], checkpoint_sha256=manifest['checkpoint_sha256'],
        asset_sha256=manifest['asset_sha256'], trace_sha256=hashlib.sha256((directory/'trace.csv').read_bytes()).hexdigest(),
        complete_finite_rows=len(data), sum_of_recorded_terms_max_abs_error=sum_error,
        independent_formulas_max_abs_errors=discrepancies,
        reward_profile=rewards, active_reward_terms=[key[7:] for key in fields if np.any(abs(data[key])>1.e-12)],
        pointwise_ratio_metric_order=list(METRICS), conditions=[],
        interpretation='Recorded reward magnitudes and a local height-state partial derivative; '
            'neither is a policy gradient or causal proof that a competing term causes tracking error. '
            'Instantaneous normalized penalties are not the independent MAE/P95 acceptance gate.')
    for cid, case in enumerate(manifest['cases']):
        mask = (data['case_id']==cid)&(data['episode']==0)&(data['time_s']>=manifest['seconds_per_case']-3.)
        if not mask.any():
            report['conditions'].append(dict(name=case['name'], steady_first_episode_samples=0))
            continue
        rates = {key[7:]:float(np.mean(data[key][mask])/dt) for key in fields}
        height_error = data['height_m'][mask]-data['height_cmd_m'][mask]
        contribution = rewards['tracking_goal_component_violation']*huber_excess(ratios[mask])
        report['conditions'].append(dict(name=case['name'], steady_first_episode_samples=int(mask.sum()),
            height_mae_m=float(np.mean(abs(height_error))), height_mean_signed_error_m=float(np.mean(height_error)),
            actual_reward_rates_per_s=rates,
            largest_actual_costs_per_s=sorted([(key, -rate) for key, rate in rates.items() if rate < 0.],
                                             key=lambda item:item[1], reverse=True),
            goal_component_costs_before_sum_clip_per_s={name:float(-contribution[:, i].mean()) for i,name in enumerate(METRICS)},
            worst_metric_fraction={name:float(np.mean(np.argmax(ratios[mask], -1)==i)) for i,name in enumerate(METRICS)},
            height_state_derivative_mean_per_m_s=float(derivative[mask].mean()),
            height_derivative_points_toward_command_fraction=float(np.mean(
                derivative[mask]*height_error < -1.e-8)),
            height_state_derivative_zero_fraction=float(np.mean(abs(derivative[mask])<1.e-8))))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    report=audit(args.directory)
    destination=args.directory/'reward_balance_audit.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('complete_finite_rows','sum_of_recorded_terms_max_abs_error',
                                                'independent_formulas_max_abs_errors','active_reward_terms')},indent=2))
