"""Stationary training decisions; original full-policy acceptance stays separate."""
import math

from joint_pd_priority import STATIC_CASES, compare_priority, trials

POSE = ('body_p95_deg', 'speed_mae_m_s', 'yaw_mae_rad_s')


def summary(report):
    rows = [(case, trial) for (name, _), (case, trial) in trials(report).items()
            if name in STATIC_CASES]
    alive = [(case, trial) for case, trial in rows if trial['survived']]
    pose = [trial['metrics'][key] / trial['limits'][key]
            for _, trial in alive for key in POSE]
    height = [trial['metrics']['height_mae_m'] / case['height'] for case, trial in alive]
    strict = len(alive) == 9
    for _, trial in rows:
        metrics = trial.get('metrics')
        strict = bool(strict and trial['survived'] and trial['resets'] == 0 and metrics
            and metrics.get('command_reached') is True and metrics.get('upright_support') is True
            and all(key in metrics and math.isfinite(metrics[key])
                    and 0. <= metrics[key] < limit - 1.e-12
                    for key, limit in trial['limits'].items()))
    return dict(stationary_trials=9, stationary_survivors=len(alive),
        stationary_resets=sum(trial['resets'] for _, trial in rows),
        stationary_pose_passed=len(alive) == 9 and max(pose) < 1. - 1.e-12,
        stationary_all_original_metrics_passed=strict,
        worst_stationary_pose_ratio=max(pose) if pose else None,
        worst_stationary_height_relative_mae=max(height) if len(height) == 9 else None,
        full_original_gate_passed=bool(report['passed']), policy_accepted=False)


def quality(report):
    s = summary(report)
    rows = [(case, trial) for (name, _), (case, trial) in trials(report).items()
            if name in STATIC_CASES and trial['survived']]
    pose = [trial['metrics'][key]/trial['limits'][key] for _, trial in rows for key in POSE]
    all_ratios = [trial['metrics'][key]/limit for _, trial in rows
                  for key, limit in trial['limits'].items()]
    return (9-s['stationary_survivors'], s['stationary_resets'],
            sum(v >= 1. for v in pose), max([1., *pose]),
            max(.1, s['worst_stationary_height_relative_mae']) if len(rows) == 9 else 1.e9,
            sum(v >= 1. for v in all_ratios), sum(all_ratios))


def review(after, before):
    comparison = compare_priority(after, before)
    static = [label for label in comparison['regressions']['stability']
              if label.split('/')[0] in STATIC_CASES]
    s = summary(after)
    promote = (s['stationary_pose_passed'] and not static and quality(after) < quality(before))
    return dict(promote=bool(promote), stationary_stability_regressions=static,
        candidate_stationary=summary(after), source_stationary=summary(before),
        candidate_quality=list(quality(after)), source_quality=list(quality(before)),
        all_27_original_cases_still_reported=True, stage_training_selection_only=True,
        full_scope_comparison=comparison, policy_accepted=False, original_acceptance_unchanged=True)


def next_reward(cfg, evidence, history, used):
    """Isolate the command change first; then one measured existing cost."""
    signature = ()
    if not history:
        return {}
    rows = [(case, trial) for (name, _), (case, trial) in trials(evidence).items()
            if name in STATIC_CASES]
    failed = lambda key: any(not trial['survived'] or
        trial['metrics'][key] >= trial['limits'][key] - 1.e-12 for _, trial in rows)
    if any(not trial['survived'] for _, trial in rows) or failed('body_p95_deg'):
        key, levels = 'orientation', (40., 80., 160., 320., 400.)
    elif failed('speed_mae_m_s'):
        key, levels = 'standing_velocity', (25., 50., 100., 200.)
    elif failed('yaw_mae_rad_s'):
        key, levels = 'yaw_rate_error_sq', (2., 4., 8., 16.)
    else:
        a, b = quality(evidence), history[-1]['source_quality']
        substantial = history[-1]['promote'] and a[4] < .99*b[4]
        if substantial:
            return {}
        if failed('height_mae_m'):
            key, levels = 'base_height_error_sq', (250., 500., 1000.)
        elif () not in used:
            return {}
        else:
            raise RuntimeError('Stationary height/pose passed; diagnose remaining original reference metrics before another identical block')
    for value in levels:
        delta = {key: -value}
        if -value < cfg['rewards'][key] and tuple(delta.items()) not in used:
            return delta
    if history[-1]['promote'] and signature not in used:
        return {}
    raise RuntimeError('Bounded stationary repair costs exhausted; diagnose native data before further training')
