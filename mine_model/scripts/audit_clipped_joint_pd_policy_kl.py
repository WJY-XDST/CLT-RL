"""Read-only raw versus clipped conditional action-distribution KL.

The Gaussian's boundary atoms and interior density are both included. This
diagnostic does not alter PPO's own KL, optimizer, action map or acceptance.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import log_ndtr, ndtr
import torch

from audit_joint_pd_policy_updates import actor, gaussian_update
from joint_pd_continuation import ROOT, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates


def clipped_kl(m0, s0, m1, s1):
    a, b = (-1.-m0)/s0, (1.-m0)/s0
    pa, pb = np.exp(-.5*a*a)/np.sqrt(2.*np.pi), np.exp(-.5*b*b)/np.sqrt(2.*np.pi)
    mass = np.where(a > 0., ndtr(-a)-ndtr(-b), ndtr(b)-ndtr(a))
    central0 = s0*s0*(mass+a*pa-b*pb)
    central1 = central0+2.*(m0-m1)*s0*(pa-pb)+(m0-m1)**2*mass
    interior = mass*np.log(s1/s0)+central1/(2.*s1*s1)-central0/(2.*s0*s0)
    lp0, lp1 = log_ndtr(a), log_ndtr((-1.-m1)/s1)
    up0, up1 = log_ndtr(-b), log_ndtr((m1-1.)/s1)
    result = np.exp(lp0)*(lp0-lp1)+np.exp(up0)*(up0-up1)+interior
    if not np.isfinite(result).all() or result.min() < -1.e-10:
        raise ValueError('Invalid mixed continuous/discrete conditional KL')
    return np.maximum(result, 0.)


def numeric_kl(m0, s0, m1, s1):
    lp0, lp1 = log_ndtr((-1.-m0)/s0), log_ndtr((-1.-m1)/s1)
    up0, up1 = log_ndtr((m0-1.)/s0), log_ndtr((m1-1.)/s1)
    def integrand(x):
        density = np.exp(-.5*((x-m0)/s0)**2)/(np.sqrt(2.*np.pi)*s0)
        ratio = np.log(s1/s0)+.5*((x-m1)/s1)**2-.5*((x-m0)/s0)**2
        return density*ratio
    points = sorted(set(float(v) for v in (m0-8*s0,m0-4*s0,m0,m0+4*s0,m0+8*s0) if -1.<v<1.))
    integral, _ = quad(integrand, -1., 1., points=points, epsabs=1.e-12, epsrel=1.e-10)
    return np.exp(lp0)*(lp0-lp1)+np.exp(up0)*(up0-up1)+integral


def audit(output, through, selected):
    output = Path(output).resolve()
    target = output / f'clipped_action_policy_KL_prefix_{through:04d}_readonly.json'
    if target.exists():
        raise ValueError('Preserve the immutable existing diagnostic')
    plan_path, health_path = output/'experiment_plan.json', output/f'policy_update_health_prefix_{through:04d}_readonly.json'
    plan, health = json.loads(plan_path.read_text()), json.loads(health_path.read_text())
    if plan['native_num_envs'] != 4096 or plan['training_config']['leg_control_mode'] != 'implicit_joint_position':
        raise ValueError('Require the qualified4096/plain27 direct-joint execution contract')
    snapshots = {v['committed_update']:v for v in health['snapshots']}
    if len(snapshots) != through or not set(selected).issubset(snapshots):
        raise ValueError('Require complete contiguous saved-policy evidence')
    if torch.cuda.mem_get_info()[0] < 2*1024**3:
        raise ValueError('Defer read-only inference to protect native training memory')
    train_path = ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py'
    if 'torch.backends.cuda.matmul.allow_tf32 = True' not in train_path.read_text():
        raise ValueError('Requalify trainer precision')
    torch.backends.cuda.matmul.allow_tf32 = True
    cpu_rng, gpu_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    pins = {str(p):sha(p) for p in (Path(__file__),plan_path,health_path,train_path)}
    rows, max_numeric_error = [], 0.
    for i in selected:
        data_path = output/'native_learning_signal'/f'rollout_{i:04d}.npz'
        proof_path = data_path.with_name(f'audit_update_{i:04d}.json')
        proof = json.loads(proof_path.read_text())
        if any(proof['all_environment_alignment_max_errors'].values()):
            raise ValueError('Resolve native alignment')
        with np.load(data_path,allow_pickle=False) as f:
            inputs, m0, s0 = (f[k].copy() for k in ('observations','mu','sigma'))
        before = Path(plan['source_checkpoint']) if i == 1 else Path(snapshots[i-1]['checkpoint_after'])
        after = Path(snapshots[i]['checkpoint_after'])
        saved = []
        for path, actual in ((before,plan['source_actual_ppo_updates']+i-1),(after,plan['source_actual_ppo_updates']+i)):
            state = torch.load(path,map_location='cpu',weights_only=True)
            if not tensor_finite(state) or saved_updates(path)!=actual or len(state['optimizer_state_dict']['state'])!=17 or any(float(s['step'])!=20*actual for s in state['optimizer_state_dict']['state'].values()):
                raise ValueError('Wrong finite checkpoint or all17 Adam clock')
            saved.append(state['model_state_dict']);pins[str(path)]=sha(path)
        ids = torch.tensor(proof['sampled_environment_ids'],device='cuda')
        def infer(state):
            weights={k:v.cuda() for k,v in state.items()};full=torch.zeros((4096,27),device='cuda');result=[]
            with torch.inference_mode():
                for obs in inputs:
                    full[ids]=torch.as_tensor(obs,device='cuda');result.append(actor(full,weights)[ids].cpu().numpy())
            return np.stack(result)
        if not np.array_equal(infer(saved[0]),m0) or not np.array_equal(s0,np.broadcast_to(saved[0]['std'].numpy(),s0.shape)):
            raise ValueError('Native4096-row TF32 source distribution not reproduced exactly')
        m1=infer(saved[1]).astype(np.float64);s1=np.broadcast_to(saved[1]['std'].numpy(),s0.shape).astype(np.float64)
        m0,s0=m0.astype(np.float64),s0.astype(np.float64)
        raw, _ = gaussian_update(m0,s0,m1,s1,m0)
        clipped=clipped_kl(m0,s0,m1,s1)
        if np.any(clipped>raw+1.e-10) or abs(raw.sum(-1).mean()-snapshots[i]['exact_gaussian_KL_old_to_new']['mean'])>1.e-10:
            raise ValueError('Data-processing inequality or original raw KL reconstruction failed')
        flat=[x.ravel() for x in (m0,s0,m1,s1)]
        points=sorted(set([*np.linspace(0,m0.size-1,24,dtype=int),int(np.argmax(raw)),int(np.argmax(abs(m0)))]))
        for point in points:
            error=abs(numeric_kl(*(v[point] for v in flat))-clipped.ravel()[point]);max_numeric_error=max(max_numeric_error,float(error))
            if error>1.e-9:
                raise ValueError('Independent numerical quadrature differs')
        rawstep=np.sqrt(((m1-m0)**2).mean((0,1)));step=np.sqrt(((np.clip(m1,-1.,1.)-np.clip(m0,-1.,1.))**2).mean((0,1)))
        rows.append(dict(committed_update=i,raw_KL_mean=float(raw.sum(-1).mean()),clipped_KL_mean=float(clipped.sum(-1).mean()),clipped_over_raw_KL=float(clipped.sum()/raw.sum()),raw_KL_mean_by_head=raw.mean((0,1)).tolist(),clipped_KL_mean_by_head=clipped.mean((0,1)).tolist(),raw_mean_step_RMS_by_head=rawstep.tolist(),applied_mean_step_RMS_by_head=step.tolist(),applied_over_raw_mean_step_RMS_by_head=(step/rawstep).tolist(),source_native_mean_and_std_parity_exact=True))
        pins.update({str(p):sha(p) for p in (data_path,proof_path)})
    if not torch.equal(cpu_rng,torch.random.get_rng_state()) or not torch.equal(gpu_rng,torch.cuda.get_rng_state()):
        raise ValueError('Diagnostic drew RNG')
    if any(sha(p)!=h for p,h in pins.items()):
        raise ValueError('Immutable inputs changed')
    report=dict(observed_at=datetime.now().astimezone().isoformat(),committed_prefix=through,selected_updates=selected,updates=rows,independent_numeric_quadrature_max_error=max_numeric_error,source_pins=pins,simulation_started=False,optimizer_executed=False,training_modified=False,policy_KL_or_adaptive_LR_replaced=False,RNG_draws_added=0,policy_accepted=False,limitation='Conditional target-distribution comparison on32 immutable sampled environments at selected updates. Clipping includes both boundary atoms and the interior density. This is not plant-state KL, stability, convergence or independent acceptance. PPO keeps its original raw-Gaussian KL and learning-rate schedule.')
    with target.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    return target,report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--through',type=int,required=True)
    parser.add_argument('--updates',type=int,nargs='+',required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    path,report=audit(args.output,args.through,args.updates)
    print(json.dumps(dict(report=str(path),updates=report['updates'],numeric_error=report['independent_numeric_quadrature_max_error'])))
