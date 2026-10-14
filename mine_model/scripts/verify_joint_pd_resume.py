"""Verify actual full-policy/Adam progress and unchanged direct-PD contracts."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from joint_pd_continuation import optimizer_step, tensor_finite, validate_continuation_config
from watch_joint_pd_scratch import validate_contract
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


def verify(source, checkpoint):
    source,checkpoint=Path(source).resolve(),Path(checkpoint).resolve()
    before=torch.load(source,map_location='cpu',weights_only=True)
    after=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not tensor_finite(before) or not tensor_finite(after):
        raise ValueError('Nonfinite checkpoint or Adam state')
    if len(after['optimizer_state_dict']['state']) != 17:
        raise ValueError('Incomplete optimizer state')
    difference=optimizer_step(after)-optimizer_step(before)
    if difference <= 0 or difference % 20:
        raise ValueError('Saved actual Adam progress is not complete PPO updates')
    old,new=before['model_state_dict'],after['model_state_dict']
    rows=[not torch.equal(old['actor.6.weight'][i],new['actor.6.weight'][i]) for i in range(6)]
    shared=not torch.equal(old['actor.0.weight'],new['actor.0.weight'])
    critic=not torch.equal(old['critic.0.weight'],new['critic.0.weight'])
    if not all(rows) or not shared or not critic:
        raise ValueError('Some full-policy parameters did not update')
    validate_contract(checkpoint.parent,fresh_weights=False,source_checkpoint=source)
    previous=validate_checkpoint_robot(source)
    current=validate_checkpoint_robot(checkpoint)
    validate_continuation_config(previous,current,previous['commands'])
    return dict(source=str(source),checkpoint=str(checkpoint),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        saved_iteration=int(after['iter']),actual_ppo_updates=int(difference/20),
        optimizer_step_before=optimizer_step(before),optimizer_step_after=optimizer_step(after),
        optimizer_states=17,all_tensors_finite=True,actor_rows_updated=rows,
        shared_actor_updated=shared,critic_updated=critic,control_contract_unchanged=True,
        environment_config_commands_rewards_and_asset_unchanged=True,
        weights_reset=False,optimizer_reset=False,policy_accepted=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.source,args.checkpoint)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
