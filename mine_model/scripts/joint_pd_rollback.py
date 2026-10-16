"""Stability-first branch selection; checkpoints and their Adam stay coupled."""
import math
import copy

from joint_pd_priority import STATIC_CASES, compare_priority, priority_summary, trials


def normalize_expanded_height(cfg,step):
    result=copy.deepcopy(cfg);commands=result['commands']
    if 'full_height_range_from_start' not in commands:
        end=commands.get('height_expansion_start_steps',0)+commands.get('height_expansion_ramp_steps',48000)
        if step<end:raise ValueError('Do not silently advance an unexpanded historical height curriculum')
        commands['full_height_range_from_start']=True
    return result


def quality(report):
    rows=trials(report)
    static=[];moving_body=[];height=[];motion=[]
    for (name,_),(case,t) in rows.items():
        if not t['survived']:continue
        m,limits=t['metrics'],t['limits']
        height.append(m['height_mae_m']/case['height'])
        if name in STATIC_CASES:
            static.extend(m[k]/limits[k] for k in ('body_p95_deg','speed_mae_m_s','yaw_mae_rad_s'))
        else:
            moving_body.append(m['body_p95_deg']/limits['body_p95_deg'])
            motion.extend(m[k]/limits[k] for k in ('speed_mae_m_s','yaw_mae_rad_s'))
    # Inside the original bounds, do not demand further polishing before
    # learning height/velocity. Failed steady metrics never become zero.
    return (27-report['survivors'],report['resets'],sum(x>1. for x in static),
        max([1.,*static]),sum(x>1. for x in moving_body),max([1.,*moving_body]),
        max(height) if len(height)==27 else 1.e12,max([1.,*motion]))


def restoration_ready(report):
    return report['survivors']==27 and report['resets']==0 and priority_summary(report)['stationary_pose_passed']


def admit(candidate,incumbent,anchor):
    comparison=compare_priority(candidate,incumbent)
    retained=compare_priority(candidate,anchor)
    accepted=(comparison['stability_retained'] and retained['stability_retained']
              and quality(candidate)<quality(incumbent))
    return dict(promote=bool(accepted),comparison=comparison,against_restored_anchor=retained,
        candidate_quality=quality(candidate),incumbent_quality=quality(incumbent),
        decision='promote_measured_improvement' if accepted else 'restore_incumbent_weights_rewards_and_Adam',
        policy_accepted=False)


def proposal(cfg,report,used):
    """Never replay an identical failed source/profile/seed experiment."""
    r=cfg['rewards'];options=[{}]
    body_bad=any(t['metrics'] and t['metrics']['body_p95_deg']>t['limits']['body_p95_deg']
                 for c in report['conditions'] for t in c['trials'])
    if body_bad:
        options.extend({'orientation':v} for v in (-120.,-240.,-400.) if v<r['orientation'])
    options.extend({'base_height_error_sq':v} for v in (-50.,-100.,-250.,-500.)
                   if v<r.get('base_height_error_sq',0.))
    options.extend({'joint_target_leg_length_error_sq':v} for v in (-25.,-50.,-100.)
                   if v!=r.get('joint_target_leg_length_error_sq',0.))
    for seed in range(43,100000):
        for delta in options:
            signature=(seed,tuple(sorted(delta.items())))
            if signature not in used:return seed,delta,signature
    raise RuntimeError('No untried bounded experiment remains')
