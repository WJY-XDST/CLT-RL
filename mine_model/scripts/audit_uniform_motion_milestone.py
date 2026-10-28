"""One read-only saved-state/training-sample review at50/100/150/200.

This is a training diagnostic, not independent acceptance. No model forward,
optimizer construction, reward change or simulator is needed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import torch

from joint_pd_continuation import ROOT, RUNS, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates


SCRIPT=Path(__file__).resolve()


def audit(output,through):
    output=output.resolve()
    assert through in (50,100,150,200) and output.parent==ROOT/'mine_model/results'
    destination=output/f'agent_review_milestone_{through:04d}_readonly.json'
    assert not destination.exists()
    plan_path=output/'experiment_plan.json'
    plan=json.loads(plan_path.read_text())
    actual=plan['source_actual_ppo_updates']+through
    runs=list(RUNS.glob('*_'+plan['run_name']))
    assert len(runs)==1
    checkpoint=runs[0]/f'model_{actual-1}.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert saved_updates(checkpoint)==actual and tensor_finite(saved)
    states=saved['optimizer_state_dict']['state']
    assert len(states)==17 and all(float(s['step'])==actual*20 for s in states.values())
    assert saved['model_state_dict']['actor.0.weight'].shape==(256,27)
    assert saved['model_state_dict']['actor.6.weight'].shape==(6,64)
    fields=('native_commands','native_height','native_speed','native_yaw','native_gravity','dones')
    data={key:[] for key in fields}
    pins={str(p):sha(p) for p in (SCRIPT,plan_path,checkpoint)}
    ids=None
    for index in range(1,through+1):
        audit_path=output/f'native_learning_signal/audit_update_{index:04d}.json'
        record=json.loads(audit_path.read_text())
        assert record['update']==index and record['total_environments']==plan['native_num_envs']
        assert not any(record['all_environment_alignment_max_errors'].values())
        assert record['probe_RNG_unchanged'] and not record['added_loss_or_backward']
        if ids is None:ids=record['sampled_environment_ids']
        assert ids==record['sampled_environment_ids']
        path=output/f'native_learning_signal/rollout_{index:04d}.npz'
        with np.load(path,allow_pickle=False) as arrays:
            for key in fields:
                value=arrays[key]
                assert value.shape[:2]==(48,len(ids)) and np.isfinite(value).all()
                data[key].append(value.copy())
        pins.update({str(p):sha(p) for p in (audit_path,path)})
    x={key:np.concatenate(parts) for key,parts in data.items()}
    done=x['dones'][...,0].astype(bool)
    command=x['native_commands'].astype(np.float64)
    age=np.empty(done.shape);stable=np.empty(done.shape)
    dt=plan['training_config']['sim']['dt']*plan['training_config']['decimation']
    age[0],stable[0]=dt,0.
    for t in range(1,len(age)):
        age[t]=np.where(done[t-1],dt,age[t-1]+dt)
        changed=np.any(abs(command[t]-command[t-1])>1.e-7,axis=-1)
        stable[t]=np.where(changed|done[t-1],0.,stable[t-1]+dt)
    settled=(age>2.)&(stable>=1.)&~done
    latest=np.arange(len(age))[:,None]>=max(0,(through-20)*48)
    dedicated=np.broadcast_to(np.asarray(ids)[None,:]<plan['fixed_standing_environments'],done.shape)
    groups={}
    for name,mask in [('all_last20_settled',settled&latest),
                      ('dedicated_static_last20_settled',settled&latest&dedicated),
                      ('motion_last20_settled',settled&latest&~dedicated)]:
        n=int(mask.sum())
        tilt=np.rad2deg(np.arccos(np.clip(-x['native_gravity'][...,2],-1.,1.)))
        groups[name]=dict(samples=n,
            height_MAE_m=float(abs(x['native_height'][mask]-command[...,2][mask]).mean()) if n else None,
            speed_MAE_m_s=float(abs(x['native_speed'][mask]-command[...,0][mask]).mean()) if n else None,
            yaw_MAE_rad_s=float(abs(x['native_yaw'][mask]-command[...,1][mask]).mean()) if n else None,
            total_tilt_angle_P95_deg=float(np.percentile(tilt[mask],95)) if n else None)
    assert all(sha(path)==digest for path,digest in pins.items())
    result=dict(verified=True,observed_at=datetime.now(timezone.utc).isoformat(),
        through=through,checkpoint=str(checkpoint),checkpoint_sha256=pins[str(checkpoint)],
        actual_updates=actual,all17_Adam_step=actual*20,
        saved_std=saved['model_state_dict']['std'].tolist(),
        saved_actual_LR=[group['lr'] for group in saved['optimizer_state_dict']['param_groups']],
        actual_num_envs=plan['native_num_envs'],recorded_environment_ids=ids,
        all9_native_alignment_errors_zero=True,saved_model_and_optimizer_finite=True,
        groups=groups,source_pins=pins,optimizer_constructed=False,optimizer_executed=False,
        simulator_started=False,training_modified=False,policy_accepted=False,
        agent_decision_required=True,
        limitations=['32 evolving Gaussian training environments, not independent acceptance.',
            'Total tilt P95 differs from the original maximum roll/pitch P95 and is not used to replace it.',
            'Early failures and full-population event counts remain in original automatic capture/readers.'])
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--through',type=int,choices=(50,100,150,200),required=True)
    args=parser.parse_args()
    result=audit(args.output,args.through)
    print(json.dumps({key:result[key] for key in ('verified','through','actual_updates','groups')}))
