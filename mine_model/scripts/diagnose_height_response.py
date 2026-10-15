"""Read-only fixed-state height response; never a closed-loop acceptance test."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import yaml

from wheel_legged_gym_isaaclab.joint_height_feedback import predict_support_heights


def height_cases(manifest):
    """Actual cases define commands; evaluator resampling bounds are not cases."""
    result={}
    for name in ('height_low','stand','height_high'):
        matches=[(index,case) for index,case in enumerate(manifest['cases']) if case['name']==name]
        if len(matches)!=1:
            raise ValueError('Require one actual case named '+name)
        index,case=matches[0]
        value=case['height']
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
            raise ValueError('Actual case height must be finite and positive')
        if case['speed']!=0. or case['yaw']!=0.:
            raise ValueError('Fixed-state height comparison requires stationary cases')
        result[name]=(index,float(value))
    if not result['height_low'][1]<result['stand'][1]<result['height_high'][1]:
        raise ValueError('Actual low, standing and high commands must be ordered')
    return result


def actor(value, state):
    for index in (0,2,4,6):
        value=torch.nn.functional.linear(value,state[f'actor.{index}.weight'],state[f'actor.{index}.bias'])
        if index!=6:
            value=torch.nn.functional.elu(value)
    return value


def load_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    agent=yaml.safe_load((path.parent/'params/agent.yaml').read_text())
    policy=agent['policy']
    if policy['activation']!='elu' or policy.get('actor_obs_normalization',False):
        raise ValueError('This diagnostic requires the original unnormalized ELU actor')
    state=saved['model_state_dict']
    if any(not torch.isfinite(value).all() for value in state.values() if torch.is_tensor(value)):
        raise ValueError('Nonfinite saved policy')
    model=agent['algorithm']['joint_height_feedback_model']
    return state,model,int(saved['iter'])


def diagnose(replay, checkpoints):
    torch.set_num_threads(1)
    replay=Path(replay)
    manifest=json.loads((replay/'manifest.json').read_text())
    cases=height_cases(manifest)
    if not manifest['policy_acceptance_test'] or manifest['weights_updated']:
        raise ValueError('Use a complete canonical frozen replay as the fixed input source')
    baseline=Path(manifest['checkpoint']).resolve()
    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=manifest['checkpoint_sha256']:
        raise ValueError('Fixed input source checkpoint changed')
    expected=(round(manifest['seconds_per_case']/manifest['policy_dt']),
              len(manifest['cases'])*len(manifest['seeds']))
    with np.load(replay/'policy_inputs.npz',allow_pickle=False) as data:
        observations,means=data['observations'],data['means']
        if observations.shape!=(*expected,27) or means.shape!=(*expected,6):
            raise ValueError('Frozen replay policy input record is incomplete')
        if not np.isfinite(observations).all() or not np.isfinite(means).all():
            raise ValueError('Nonfinite frozen replay policy input record')
        count=round(3./manifest['policy_dt'])
        if expected[0]<count:
            raise ValueError('At least three seconds of fixed inputs are required')
        tail=torch.as_tensor(observations[-count:].copy(),dtype=torch.float32)
        recorded=torch.as_tensor(means[-count:].copy(),dtype=torch.float32)
    source_state,_,_=load_checkpoint(baseline)
    with torch.no_grad():
        parity=float((actor(tail,source_state)-recorded).abs().max())
    if parity>1.e-4:
        raise ValueError('Fixed input source does not reproduce recorded inference means')
    low,high=cases['height_low'][1],cases['height_high'][1]
    outputs=[]
    paths=list(dict.fromkeys([baseline,*[Path(p).resolve() for p in checkpoints]]))
    with torch.no_grad():
        for path in paths:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            state,model,iteration=load_checkpoint(path)
            rows=[]
            for name,(index,_) in cases.items():
                inputs=tail[:,index*len(manifest['seeds']):(index+1)*len(manifest['seeds'])].reshape(-1,27)
                changed=inputs.repeat(2,1).clone()
                changed[:,8]=changed.new_tensor([low,high]).repeat_interleave(len(inputs))*model['height_scale']
                unchanged=[*range(8),*range(9,27)]
                if not torch.equal(changed[:,unchanged],inputs.repeat(2,1)[:,unchanged]):
                    raise AssertionError('Fixed inputs changed outside the height command')
                raw=actor(changed,state)
                height,_,valid=predict_support_heights(raw,changed,model)
                if not valid.all() or not torch.isfinite(height).all():
                    raise ValueError('Invalid forward geometry in fixed-state response')
                values=height.reshape(2,len(inputs),2)
                wheel=raw[:,[2,5]].clamp(-1.,1.).reshape(2,len(inputs),2)
                rows.append(dict(fixed_state_case=name,samples=len(inputs),
                    low_command_support_height_mean_m=float(values[0].mean()),
                    high_command_support_height_mean_m=float(values[1].mean()),
                    height_command_response_span_m=float((values[1]-values[0]).mean()),
                    low_command_support_proxy_mae_m=float((values[0]-low).abs().mean()),
                    high_command_support_proxy_mae_m=float((values[1]-high).abs().mean()),
                    wheel_raw_target_response_span=(wheel[1]-wheel[0]).mean(0).tolist()))
            outputs.append(dict(checkpoint=str(path),checkpoint_sha256=digest,iteration=iteration,
                training_profile=model.get('command_counterfactuals'),conditions=rows))
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
                raise ValueError('Checkpoint changed during read-only diagnosis')
    return dict(replay=str(replay.resolve()),fixed_input_source_checkpoint=str(baseline),
        source_actor_parity_max_abs=parity,height_command_endpoints_m=[low,high],
        height_command_span_m=high-low,command_source='Actual manifest case heights; never resolved_env resampling range',
        only_input_column_changed=8,checkpoints=outputs,weights_updated=False,optimizer_updated=False,
        policy_accepted=False,limitation='Forward kinematics of direct joint targets on frozen states. '
        'Not actual body height, a plant dynamics Jacobian, stability proof or closed-loop acceptance.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,action='append',default=[])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=diagnose(args.replay,args.checkpoint)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
