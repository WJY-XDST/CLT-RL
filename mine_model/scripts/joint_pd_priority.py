"""Reward review importance, never exclusive training phases.

All command groups train together. Original per-trial limits remain the only
acceptance criteria; better height or speed cannot average away lost stability.
"""
import copy
import math

STATIC_CASES = ('stand', 'height_low', 'height_high')
CASE_NAMES = (*STATIC_CASES, 'forward_0p3', 'forward_0p6', 'reverse_0p3',
              'reverse_0p6', 'spin_left', 'spin_right')
METRICS = ('body_p95_deg', 'height_mae_m', 'speed_mae_m_s', 'yaw_mae_rad_s')
REWARD_KEY = 'joint_target_leg_length_error_sq'
SHAPING_COSTS = (-500., -250., -100., 0.)


def trials(report):
    if (len(report['conditions']) != 9 or report['total_trials'] != 27
            or {c['name'] for c in report['conditions']} != set(CASE_NAMES)):
        raise ValueError('Require all original cases and all 27 trials')
    result = {}
    for case in report['conditions']:
        if len(case['trials']) != 3 or len({t['seed'] for t in case['trials']}) != 3:
            raise ValueError('Require three distinct seeds per condition')
        for trial in case['trials']:
            metrics = trial.get('metrics')
            if trial['survived']:
                if not metrics:
                    raise ValueError('A survivor needs actual steady metrics')
                for key in METRICS:
                    value, limit = metrics[key], trial['limits'][key]
                    if not math.isfinite(value) or value < 0 or not math.isfinite(limit) or limit <= 0:
                        raise ValueError('Metrics and original tolerances must be finite')
            elif metrics is not None:
                raise ValueError('Failed trials cannot receive steady tracking errors')
            if type(trial['resets']) is not int or trial['resets'] < 0:
                raise ValueError('Invalid reset count')
            result[case['name'], trial['seed']] = (case, trial)
    if sum(t['survived'] for _, t in result.values()) != report['survivors']:
        raise ValueError('Survivor count differs from per-trial evidence')
    return result


def priority_summary(report):
    rows = trials(report)
    static = [(case, t) for (name, _), (case, t) in rows.items() if name in STATIC_CASES]
    surviving = [(case, t) for case, t in static if t['survived']]
    height = [t['metrics']['height_mae_m']/case['height'] for case, t in static
              if case['name'] != 'stand' and t['survived']]
    return dict(survivors=report['survivors'], resets=report['resets'],
        stationary_survivors=len(surviving), missing_steady_trials=27-report['survivors'],
        stationary_pose_passed=len(surviving) == 9 and all(
            t['metrics'][k] <= t['limits'][k] for _, t in static
            for k in ('body_p95_deg', 'speed_mae_m_s', 'yaw_mae_rad_s')),
        worst_endpoint_height_relative_mae=max(height) if len(height) == 6 else None,
        full_original_gate_passed=bool(report['passed']),
        importance=['standing_posture_and_survival', 'height_tracking', 'motion_tracking'],
        tasks_trained_simultaneously=True)


def compare_priority(after, before):
    new, old = trials(after), trials(before)
    if new.keys() != old.keys():
        raise ValueError('Compare the same complete conditions and seeds')
    regressions = {'stability': [], 'height': [], 'motion': []}
    improvements = {'stability': [], 'height': [], 'motion': []}
    for key, (case, trial) in new.items():
        old_case, prior = old[key]
        # Wheel/leg reference limits legitimately depend on this policy's
        # applied references. The four task limits here depend only on the
        # unchanged command and the original acceptance specification.
        if (any(case[k] != old_case[k] for k in ('speed', 'yaw', 'height'))
                or any(trial['limits'][k] != prior['limits'][k] for k in METRICS)):
            raise ValueError('A command or original acceptance tolerance changed')
        label = case['name']+'/'+str(trial['seed'])
        if prior['survived'] and not trial['survived']:
            regressions['stability'].append(label+'/lost_survival')
        elif trial['survived'] and not prior['survived']:
            improvements['stability'].append(label+'/recovered_survival')
        if trial['resets'] != prior['resets']:
            (regressions if trial['resets'] > prior['resets'] else improvements)['stability'].append(label+'/resets')
        if not trial['survived'] or not prior['survived']:
            a, b = trial.get('first_failure_s'), prior.get('first_failure_s')
            if a is not None and b is not None and a != b:
                (regressions if a < b else improvements)['stability'].append(label+'/first_failure_s')
            continue  # Failed steady metrics remain missing, never zero.
        for metric in METRICS:
            group = ('stability' if metric == 'body_p95_deg' or
                     (case['name'] in STATIC_CASES and metric in ('speed_mae_m_s', 'yaw_mae_rad_s'))
                     else 'height' if metric == 'height_mae_m' else 'motion')
            a, b = trial['metrics'][metric], prior['metrics'][metric]
            if a > b and a > trial['limits'][metric]:
                regressions[group].append(label+'/'+metric)
            elif a < b and b > prior['limits'][metric]:
                improvements[group].append(label+'/'+metric)
    decisive = next((g for g in regressions if regressions[g] or improvements[g]), None)
    verdict = ('no_measured_priority_change' if decisive is None else
               decisive+'_regression' if regressions[decisive] else decisive+'_improvement')
    return dict(verdict=verdict, regressions=regressions, improvements=improvements,
        stability_retained=not regressions['stability'],
        lower_priority_improvement_cannot_cancel_stability_loss=True,
        full_original_gate_passed=bool(after['passed']), policy_accepted=False)


def revised_rewards(saved, coefficient):
    if coefficient not in SHAPING_COSTS or isinstance(coefficient, bool):
        raise ValueError('Use the bounded single-term shaping ablation')
    if (saved['orientation'] != -400. or saved['base_height_error_sq'] != -500.
            or saved[REWARD_KEY] not in SHAPING_COSTS):
        raise ValueError('Require the diagnosed posture and actual-height costs')
    result = copy.deepcopy(saved)
    result[REWARD_KEY] = float(coefficient)
    return result


def next_shaping_cost(current, comparisons):
    """Reduce target-FK shaping on measured stability loss, without task gates.

    This is a bounded hypothesis, not causal proof. Actual height, pose and
    motion rewards and every command group continue training simultaneously.
    """
    if current not in SHAPING_COSTS:
        raise ValueError('Unknown shaping cost')
    if comparisons and not comparisons[-1].get('latest_stability_retained', comparisons[-1]['stability_retained']):
        return SHAPING_COSTS[min(SHAPING_COSTS.index(current)+1, len(SHAPING_COSTS)-1)]
    return current
