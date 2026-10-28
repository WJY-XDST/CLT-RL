"""Evaluate two actors on the same retained lowheight steering inputs.

No dynamics or policy sample is executed. Both AF7001 and the preserved6b7201
state distributions are retained. Checkpoint/Adam clocks, recorded own-policy
means and input commands are independently verified before comparing wheel
references and target FK. This is not candidate tracking acceptance.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch

from diagnose_height_response import actor
from joint_pd_continuation import ROOT,sha,tensor_finite
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
from wheel_legged_gym_isaaclab.leg_control import joint_position_targets,LEG_ACTION_COLUMNS
from wheel_legged_gym_isaaclab.five_bar_vmc import five_bar_state
from wheel_legged_gym_isaaclab.wheel_control import checkpoint_wheel_action_limit

SCRIPT=Path(__file__).resolve()
RESULTS=ROOT/'mine_model/results'
RETAINED=[
    RESULTS/'joint_pd_preserved6801_yawcost200_review7001_20261022/combined/candidate/low',
    RESULTS/'joint_pd_AF7001_leg005_fallback4096_review7201_v2_20261023_after_finalizer_repair_v2/combined/candidate/low',
]

def read(p):return json.loads(Path(p).read_text())

def checked(path,actual):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    assert tensor_finite(saved) and saved_updates(path)==actual
    states=saved['optimizer_state_dict']['state']
    assert len(states)==17 and all(float(v['step'])==actual*20 for v in states.values())
    return saved['model_state_dict'],validate_checkpoint_robot(path)

def audit(directory,through):
    directory=Path(directory).resolve()
    destination=directory/f'frozen_lowheight_turn_reference_prefix_{through:04d}_readonly.json'
    assert not destination.exists() and 1<=through<=200
    plan_path=directory/'experiment_plan.json';plan=read(plan_path)
    health_path=directory/f'policy_update_health_prefix_{through:04d}_readonly.json'
    health=read(health_path)
    assert health['committed_updates']==through and health['native_num_envs']==8192
    assert plan['source_actual_ppo_updates']==7001
    source=Path(plan['source_checkpoint']);candidate=Path(health['snapshots'][-1]['checkpoint_after'])
    source_state,cfg=checked(source,7001)
    new_state,new_cfg=checked(candidate,7001+through)
    assert cfg['leg_control_mode']==new_cfg['leg_control_mode']=='implicit_joint_position'
    assert cfg['observation_space']==new_cfg['observation_space']==27
    for key in ['leg_joint_position_offset','leg_joint_position_scale','leg_joint_target_limit_margin',
            'five_bar_geometry','wheel_radius','wheel_track_width','action_scale_vel','wheel_action_limit']:
        assert cfg[key]==new_cfg[key],key
    geometry_path=directory/f'joint_target_geometric_exploration_prefix_{through:04d}_readonly.json'
    limits=read(geometry_path)['USD_composed_joint_limits_rad']
    torch.set_num_threads(1)
    rng=torch.random.get_rng_state().clone()
    paths=[SCRIPT,Path(actor.__code__.co_filename),Path(joint_position_targets.__code__.co_filename),
        Path(five_bar_state.__code__.co_filename),plan_path,health_path,geometry_path,source,candidate]
    groups=[]
    for index,folder in enumerate(RETAINED):
        manifest_path=folder/'manifest.json';npz=folder/'policy_inputs.npz'
        manifest=read(manifest_path);own=Path(manifest['checkpoint'])
        assert sha(own)==manifest['checkpoint_sha256']
        assert not manifest['sampled_policy_actions'] and not manifest['weights_updated']
        assert manifest['seeds']==[683,684,685] and len(manifest['cases'])==36
        assert manifest['policy_dt']==.02 and manifest['seconds_per_case']==20.
        own_state,own_cfg=(source_state,cfg) if own==source else checked(own,7201)
        with np.load(npz,allow_pickle=False) as arrays:
            observations=arrays['observations'][-150:].copy()
            recorded=arrays['means'][-150:].copy()
            assert observations.shape==(150,108,27) and recorded.shape==(150,108,6)
            assert np.array_equal(arrays['seeds'],manifest['seeds'])
        assert np.isfinite(observations).all() and np.isfinite(recorded).all()
        inputs=torch.from_numpy(observations)
        with torch.inference_mode():
            own_means=actor(inputs,own_state)
            parity=float(np.max(np.abs(own_means.numpy()-recorded)))
            assert parity<=1e-4,parity
            models=[]
            for label,state,config in [('AF7001',source_state,cfg),('new_prefix',new_state,new_cfg)]:
                raw=actor(inputs,state)
                assert torch.isfinite(raw).all()
                wheel=raw[..., (2,5)].clamp(-checkpoint_wheel_action_limit(config),
                    checkpoint_wheel_action_limit(config))*config['action_scale_vel']
                yaw_proxy=config['wheel_radius']*(wheel[...,1]-wheel[...,0])/config['wheel_track_width']
                rolling=wheel.mean(-1)*config['wheel_radius']
                q,_=joint_position_targets(raw[...,LEG_ACTION_COLUMNS].reshape(-1,4),
                    config['leg_joint_position_offset'],config['leg_joint_position_scale'],
                    raw.new_tensor(limits).expand(150*108,-1,-1),config['leg_joint_target_limit_margin'])
                length,angle,_,valid=five_bar_state(q[:,::2],q[:,1::2],**config['five_bar_geometry'])
                assert torch.isfinite(length).all() and torch.isfinite(angle).all()
                models.append(dict(label=label,raw=raw.numpy(),wheel=wheel.numpy(),yaw=yaw_proxy.numpy(),
                    rolling=rolling.numpy(),length=length.reshape(150,108,2).numpy(),
                    angle=angle.reshape(150,108,2).numpy(),valid=valid.reshape(150,108,2).numpy()))
        rows=[]
        for case_id,case in enumerate(manifest['cases']):
            selection=slice(case_id*3,(case_id+1)*3)
            yaw_command=float(case['yaw']);assert yaw_command!=0.
            speed_command=float(case['speed'])
            command_parity=max(float(np.max(abs(observations[:,selection,6]/cfg['obs_scales']['lin_vel']-speed_command))),
                float(np.max(abs(observations[:,selection,7]/cfg['obs_scales']['ang_vel']-yaw_command))),
                float(np.max(abs(observations[:,selection,8]/cfg['obs_scales']['height_measurements']-case['height']))))
            assert command_parity<1e-7
            policies=[]
            for item in models:
                y=item['yaw'][:,selection].astype(np.float64)
                policies.append(dict(label=item['label'],reference_yaw_proxy_mean_rad_s=float(y.mean()),
                    reference_yaw_proxy_minus_command_mean_rad_s=float((y-yaw_command).mean()),
                    reference_yaw_proxy_MAE_rad_s=float(abs(y-yaw_command).mean()),
                    reference_rolling_proxy_mean_m_s=float(item['rolling'][:,selection].mean()),
                    wheel_reference_peak_abs_rad_s=float(abs(item['wheel'][:,selection]).max()),
                    FK_target_length_mean_m=item['length'][:,selection].astype(np.float64).mean((0,1)).tolist(),
                    FK_target_angle_mean_rad=item['angle'][:,selection].astype(np.float64).mean((0,1)).tolist(),
                    invalid_target_geometry_samples=int((~item['valid'][:,selection]).sum())))
            rows.append(dict(case=case,seeds=manifest['seeds'],frozen_input_samples=450,
                recorded_own_body_yaw_mean_rad_s=float((observations[:,selection,2]/cfg['obs_scales']['ang_vel']).mean()),
                command_observation_parity_max_error=command_parity,policies=policies,
                new_minus_AF_reference_yaw_proxy_mean_rad_s=float((models[1]['yaw'][:,selection]-models[0]['yaw'][:,selection]).mean())))
        groups.append(dict(recorded_state_source='AF7001' if index==0 else 'preserved_6b7201',
            checkpoint=str(own),checkpoint_sha256=sha(own),own_recorded_mean_reproduction_max_error=parity,
            all36_cases_and_three_seeds_preserved=True,cases=rows))
        paths.extend([manifest_path,npz,own,own.parent/'params/env.yaml'])
    assert torch.equal(rng,torch.random.get_rng_state())
    result=dict(verified=True,observed_at=datetime.now().astimezone().isoformat(),prefix=through,
        source_actual_updates=7001,candidate_actual_updates=7001+through,
        source_checkpoint=str(source),source_checkpoint_sha256=sha(source),
        candidate_checkpoint=str(candidate),candidate_checkpoint_sha256=sha(candidate),
        total_frozen_inputs=32400,groups=groups,source_pins={str(p):sha(p) for p in paths},
        RNG_draws_added=0,simulator_started=False,optimizer_executed=False,training_modified=False,
        can_extend_training=False,policy_accepted=False,
        limitations=['Both actors receive identical retained states and OLD action histories. No candidate closed-loop trajectory is produced.',
            'Reference yaw/rolling proxies use ideal differential kinematics, not measured current candidate body velocity.',
            'Target FK is a reference diagnostic, not actual body height or mechanical-safety certification.',
            'The complete original fixed735, static810, moving/steering, signed height and sustained reviews remain required.'])
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return destination,result

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory',type=Path);cli.add_argument('--through',type=int,required=True)
    args=cli.parse_args();path,report=audit(args.directory,args.through)
    print(json.dumps({'verified':True,'report':str(path),'prefix':args.through,'frozen_inputs':32400,'simulator_started':False}),flush=True)
