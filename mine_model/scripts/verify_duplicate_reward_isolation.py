"""Verify a frozen zero-worst-term probe changes rewards only, never controls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from summarize_mine_trial import assess


def verify(baseline, candidate):
    baseline,candidate=Path(baseline),Path(candidate)
    manifests=[json.loads((path/'manifest.json').read_text()) for path in (baseline,candidate)]
    completions=[json.loads((path/'completed.json').read_text()) for path in (baseline,candidate)]
    for key in ('checkpoint_sha256','asset_sha256','asset_chain_sha256','cases','seeds',
                'seconds_per_case','physics_dt','policy_dt','leg_control_mode',
                'enable_external_forces_every_iteration'):
        if manifests[0][key] != manifests[1][key]:
            raise ValueError('Frozen paired conditions differ: '+key)
    if manifests[1]['policy_acceptance_test'] or manifests[1]['tracking_goal_reward_probe'] != 0.:
        raise ValueError('Reward-removal probe must be explicitly diagnostic')
    if not all(c['completed'] and c['samples']==27000 and c['resets']==0 for c in completions):
        raise ValueError('Frozen comparison is incomplete or lost survival')
    configs=[json.loads((path/'resolved_env.json').read_text()) for path in (baseline,candidate)]
    expected=json.loads(json.dumps(configs[0]))
    expected['rewards']['tracking_goal_violation']=0.
    for config in (expected,configs[1]):
        config.pop('log_dir',None)
    if expected != configs[1]:
        raise ValueError('Changes outside the one reward coefficient')
    old,new=[np.genfromtxt(path/'trace.csv',delimiter=',',names=True) for path in (baseline,candidate)]
    if len(old) != 27000 or len(new) != 27000:
        raise ValueError('Wrong trace row count')
    for values in (old,new):
        if not all(np.isfinite(values[key]).all() for key in values.dtype.names):
            raise ValueError('Nonfinite telemetry')
    physical=[name for name in old.dtype.names if not name.startswith('reward_')]
    physical_error=max(float(np.max(abs(old[key]-new[key]))) for key in physical)
    other_rewards=[name for name in old.dtype.names if name.startswith('reward_')
                   and name not in ('reward_total','reward_tracking_goal_violation')]
    reward_error=max(float(np.max(abs(old[key]-new[key]))) for key in other_rewards)
    delta_error=float(np.max(abs(new['reward_total']-old['reward_total']+old['reward_tracking_goal_violation'])))
    with np.load(baseline/'policy_inputs.npz') as left,np.load(candidate/'policy_inputs.npz') as right:
        observations_error=float(np.max(abs(left['observations']-right['observations'])))
        means_error=float(np.max(abs(left['means']-right['means'])))
    if max(physical_error,reward_error,observations_error,means_error) > 1.e-6 or delta_error > 1.e-6:
        raise ValueError('Frozen reward change affects execution or the total does not reconcile')
    try:
        assess(candidate)
    except ValueError as error:
        if 'Diagnostic action overrides cannot count as policy acceptance' not in str(error):
            raise
    else:
        raise ValueError('Diagnostic probe incorrectly admitted as acceptance')
    return dict(checkpoint=manifests[0]['checkpoint'],checkpoint_sha256=manifests[0]['checkpoint_sha256'],
        asset_sha256=manifests[0]['asset_sha256'],cases=9,seeds=3,samples_each=27000,resets_each=0,
        only_change={'rewards.tracking_goal_violation':[configs[0]['rewards']['tracking_goal_violation'],0.]},
        exact_policy_observations_max_abs_difference=observations_error,raw_actor_means_max_abs_difference=means_error,
        physical_telemetry_max_abs_difference=physical_error,all_other_rewards_max_abs_difference=reward_error,
        reward_total_delta_formula_max_abs_error=delta_error,
        diagnostic_rejected_by_acceptance=True,checkpoint_unchanged=True,
        weights_updated=False,optimizer_updated=False,training_improvement_proven=False,
        baseline_trace_sha256=hashlib.sha256((baseline/'trace.csv').read_bytes()).hexdigest(),
        candidate_trace_sha256=hashlib.sha256((candidate/'trace.csv').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.baseline,args.candidate)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
