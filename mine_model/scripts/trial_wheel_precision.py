"""One controlled wheel-head PPO continuation; no automatic promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import torch
from summarize_mine_trial import assess, plot

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-replay',type=Path,required=True)
    p.add_argument('--overrides',type=Path,required=True)
    p.add_argument('--target',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((args.source_replay/'manifest.json').read_text())
    source=Path(manifest['checkpoint'])
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    if source_sha!=manifest['checkpoint_sha256']:raise ValueError('Source checkpoint changed')
    overrides=json.loads(args.overrides.read_text())
    (output/'overrides.json').write_text(json.dumps(overrides,indent=2)+'\n')
    run_name='mine_wheel_head_precision_'+output.name
    cmd=[str(ROOT/'run_python.sh'),str(ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py'),
         '--task','WheelLeggedVMC-Flat-v0','--headless','--num_envs','256','--seed','43',
         '--run_name',run_name,'--target_iteration',str(args.target),'--checkpoint_path',str(source),
         '--wheel_head_only_finetune','agent.algorithm.entropy_coef=0.001']
    cmd+=['env.'+k+'='+json.dumps(v) for k,v in overrides.items()]
    env=dict(os.environ,PYTHONPATH=str(ROOT/'wheel_legged_isaaclab'))
    for k in ('ROS_DISTRO','ROS_VERSION','ROS_PYTHON_VERSION','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH'):env.pop(k,None)
    provenance={'source_checkpoint':str(source),'source_sha256':source_sha,'target_iteration':args.target,
                'mode':'wheel_head_only','command':cmd,'status':'controlled trial, not promoted'}
    (output/'trial.json').write_text(json.dumps(provenance,indent=2)+'\n')
    with (output/'training.log').open('w') as stream:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    checkpoints=[f for d in (ROOT/'IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat').glob('*_'+run_name)
                 for f in d.glob('model_*.pt') if re.fullmatch(r'model_\d+',f.stem)]
    checkpoint=max(checkpoints,key=lambda x:(int(x.stem.split('_')[1]),x.stat().st_mtime_ns))
    if int(checkpoint.stem.split('_')[1])<args.target-1:raise ValueError('Trial ended early')
    before=torch.load(source,map_location='cpu',weights_only=True)['model_state_dict']
    after=torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict']
    frozen={}
    for key in before:
        if key.startswith('actor.'):
            frozen[key]=torch.equal(before[key][[0,1,3,4]],after[key][[0,1,3,4]]) if key.startswith('actor.6.') else torch.equal(before[key],after[key])
    frozen['leg_std']=torch.equal(before['std'][[0,1,3,4]],after['std'][[0,1,3,4]])
    if not all(frozen.values()):raise ValueError('Frozen actor parameters changed during live PPO training')
    provenance.update(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      frozen_parameters_verified=frozen)
    (output/'trial.json').write_text(json.dumps(provenance,indent=2)+'\n')
    cmd=[str(ROOT/'run_python.sh'),str(ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
         '--headless','--robot','mine','--checkpoint',str(checkpoint),'--seconds','12',
         '--overrides_json',str(output/'overrides.json'),'--output',str(output/'replay')]
    with (output/'evaluation.log').open('w') as stream:
        subprocess.run(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    report=assess(output/'replay');plot(output/'replay')
    print(json.dumps({'checkpoint':str(checkpoint),'survivors':report['survivors'],'passed':report['passed']}))


if __name__=='__main__':main()
