"""Isolate height-command fitting from policy initialization and Adam history.

This CPU-only necessary-skill experiment does not simulate the robot, train
PPO, replace a checkpoint, or constitute tracking/stability acceptance.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict
import yaml
from rsl_rl.modules import ActorCritic

from wheel_legged_gym_isaaclab.joint_height_feedback import height_proxy_loss,predict_support_heights
from wheel_legged_gym_isaaclab.joint_mean_guard import joint_mean_overshoot_loss
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M


def compare(checkpoint,replay,steps=600):
    torch.set_num_threads(1)
    checkpoint,replay=Path(checkpoint),Path(replay)
    before_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    agent=yaml.safe_load((checkpoint.parent/'params/agent.yaml').read_text())
    model=agent['algorithm']['joint_height_feedback_model']
    manifest=json.loads((replay/'manifest.json').read_text())
    with np.load(replay/'policy_inputs.npz') as data:
        assert np.isfinite(data['observations']).all()
        selected=[i for i,c in enumerate(manifest['cases']) if c['name'] in ('stand','height_low','height_high')]
        assert len(selected)==3 and manifest['seeds']==[83,84,85]
        train=np.concatenate([data['observations'][-150:,i*3:i*3+2].reshape(-1,27) for i in selected])
        test=np.concatenate([data['observations'][-150:,i*3+2:i*3+3].reshape(-1,27) for i in selected])
    training=torch.tensor(train,dtype=torch.float32)
    testing=torch.tensor(test,dtype=torch.float32)
    assert isinstance(model['joint_limits'],list)
    config=dict(agent['policy']);config.pop('class_name')
    template=TensorDict({'policy':training[:8]},batch_size=[8])
    def policy(seed):
        torch.manual_seed(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            result=ActorCritic(template,agent['obs_groups'],6,**config)
        if not isinstance(result.actor_obs_normalizer,torch.nn.Identity):
            raise ValueError('Compare only the original unnormalized direct-PD policy')
        return result
    generator=torch.Generator().manual_seed(832)
    indices=torch.randint(len(training),(steps,128),generator=generator)
    source_lr=saved['optimizer_state_dict']['param_groups'][0]['lr']
    assert len(saved['optimizer_state_dict']['param_groups'])==1
    source_group=saved['optimizer_state_dict']['param_groups'][0]
    settings={key:source_group[key] for key in ('betas','eps','weight_decay','amsgrad','maximize')}
    if source_group.get('capturable'):
        raise ValueError('CPU diagnosis requires the original noncapturable Adam configuration')
    commands=model['command_counterfactuals']['commands_m']
    def augmented(value,targets):
        result=value.repeat(len(targets),1).clone()
        result[:,8]=value.new_tensor(targets).repeat_interleave(len(value))*model['height_scale']
        return TensorDict({'policy':result},batch_size=[len(result)])
    evaluation=augmented(testing,list(SAFE_BASE_HEIGHT_RANGE_M))
    trajectories={}
    for name,use_saved_weights,use_saved_adam in [('retained_weights_and_adam',True,True),
            ('retained_weights_new_adam',True,False),('fresh_weights_and_adam',False,False)]:
        network=policy(404)
        if use_saved_weights:network.load_state_dict(saved['model_state_dict'],strict=True)
        optimizer=torch.optim.Adam(network.parameters(),lr=source_lr,**settings)
        if use_saved_adam:optimizer.load_state_dict(copy.deepcopy(saved['optimizer_state_dict']))
        assert optimizer.param_groups[0]['lr']==source_lr
        rows=[]
        for step in range(steps+1):
            if step in (0,20,100,300,steps):
                with torch.no_grad():
                    means=network.act_inference(evaluation)
                    heights,length,valid=predict_support_heights(means,evaluation['policy'],model)
                    assert valid.all() and torch.isfinite(heights).all()
                    command=evaluation['policy'][:,8]/model['height_scale']
                    error=(heights-command[:,None]).abs()
                    halves=heights.reshape(2,len(testing),2)
                    rows.append(dict(adam_updates=step,heldout_height_proxy_mae_m=float(error.mean()),
                        low_command_proxy_mae_m=float(error[:len(testing)].mean()),
                        high_command_proxy_mae_m=float(error[len(testing):].mean()),
                        height_command_response_span_m=float(halves[1].mean()-halves[0].mean()),
                        raw_leg_mean_clipped_fraction=float((means[:,[0,1,3,4]].abs()>=1.).float().mean())))
            if step==steps:break
            batch=augmented(training[indices[step]],commands)
            optimizer.zero_grad(set_to_none=True)
            means=network.act_inference(batch)
            loss=(agent['algorithm']['joint_height_feedback_coef']*height_proxy_loss(means,batch['policy'],model)
                  +agent['algorithm']['joint_mean_guard_coef']*joint_mean_overshoot_loss(means))
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
            optimizer.step()
        trajectories[name]=rows
        print(name,json.dumps(rows),flush=True)
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==before_hash
    return dict(checkpoint=str(checkpoint),checkpoint_sha256=before_hash,replay=str(replay),
        main_checkpoint_modified=False,main_optimizer_modified=False,simulator_used=False,policy_accepted=False,
        original_architecture_and_forward_geometry=True,learning_rate=source_lr,
        adam_hyperparameters=settings,adam_updates_per_branch=steps,
        train_seeds=[83,84],test_seeds=[85],train_states=len(training),test_states=len(testing),
        training_height_commands_m=commands,heldout_height_commands_m=list(SAFE_BASE_HEIGHT_RANGE_M),
        trajectories=trajectories,limitation='Auxiliary actor fitting on frozen states only. This isolates '
        'height-command learnability and initialization effects, not PPO sample efficiency, balance or closed-loop tracking.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=600)
    args=parser.parse_args()
    if not 1<=args.steps<=3000:raise ValueError('Use a bounded diagnostic budget of 1..3000 Adam updates')
    result=compare(args.checkpoint,args.replay,args.steps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
