"""Independently reconstruct sampled native rewards and the direct-PD map.

Read-only training diagnosis. These sampled transitions are not independent
steady-state acceptance trials and cannot establish convergence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, sha
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


def audit(output):
    output=Path(output)
    plan=json.loads((output/'experiment_plan.json').read_text())
    run=next(RUNS.glob('*_'+plan['run_name']))
    checkpoints=sorted(run.glob('model_*.pt'))
    cfg=validate_checkpoint_robot(checkpoints[0])
    r=cfg['rewards'];dt=cfg['sim']['dt']*cfg['decimation']
    assert cfg['leg_control_mode']=='implicit_joint_position'
    assert cfg['commands']['training_motion_scale']==0.
    assert r['base_height_relative_deadband']==0. and r['base_height']>0.
    known={name:r.get(name,0.) for name in (
        'tracking_lin_vel_enhance','tracking_lin_vel_precise','spin_center_velocity','nominal_state',
        'standing_leg_angle','standing_wheel_tracking','lin_vel_z','ang_vel_xy','dof_vel','dof_acc',
        'torques','action_smooth','leg_length_action_saturation','leg_length_action_difference',
        'leg_length_target_underreach','wheel_action_saturation',
        'tracking_goal_violation','tracking_goal_component_violation')}
    assert all(value==0. for value in known.values()),known
    assert -1000. <= r.get('joint_target_leg_length_error_sq', 0.) <= 0.
    assert r['collision']<=0. and r['dof_pos_limits']<=0.
    assert r['tracking_task_quality_gate_floor']==1.
    rows=[]
    for file in sorted((output/'native_learning_signal').glob('rollout_*.npz')):
        with np.load(file,allow_pickle=False) as a:
            o=a['observations'];c=a['native_commands'];h=a['native_height'];g=a['native_gravity']
            assert not np.count_nonzero(c[...,:2])
            assert all(np.isfinite(a[key]).all() for key in a.files)
            # Native reward uses post-physics velocity. Successive nonreset
            # observations contain that lateral component; o[t] is pre-action.
            v=np.zeros_like(o[...,17:19]);v[...,0]=a['native_speed']
            v[:-1,:,1]=o[1:,:,18]/cfg['obs_scales']['lin_vel']
            yaw=a['native_yaw'];raw=a['native_raw_actions']
            applied=np.clip(raw,-1.,1.).copy()
            offset=np.array(cfg['leg_joint_position_offset'],dtype=raw.dtype)
            scale=np.array(cfg['leg_joint_position_scale'],dtype=raw.dtype)
            target=offset+np.clip(raw[...,[0,1,3,4]],-1.,1.)*scale
            map_error=float(np.max(np.abs(target-a['native_leg_targets'])))
            applied[...,[0,1,3,4]]=(a['native_leg_targets']-offset)/scale
            hsq=(h-c[...,2])**2
            signals=dict(tracking_lin_vel=np.exp(-(c[...,0]-v[...,0])**2/r['tracking_sigma']),
                tracking_ang_vel=np.exp(-(c[...,1]-yaw)**2/r['tracking_sigma_ang']),
                lin_vel_error_sq=(v[...,0]-c[...,0])**2+r['lin_vel_error_lateral_multiplier']*v[...,1]**2,
                standing_velocity=np.sum(v*v,axis=-1), yaw_rate_error_sq=(yaw-c[...,1])**2,
                base_height_error_sq=hsq,base_height=np.exp(-hsq/r['base_height_sigma']),
                joint_target_leg_length_error_sq=np.mean((a['native_leg_references']-
                    (c[...,2:3]+r['leg_length_target_height_offset']))**2,axis=-1),
                orientation=g[...,0]**2+r['orientation_roll_multiplier']*g[...,1]**2,
                action_rate=np.sum((o[...,-6:]-applied)**2,axis=-1),
                joint_position_action_saturation=np.sum(np.maximum(
                    np.abs(np.clip(raw[...,[0,1,3,4]],-10.,10.))-r['joint_position_action_soft_limit'],0.)**2,axis=-1))
            bounds=dict(lin_vel_error_sq='lin_vel_penalty_clip',yaw_rate_error_sq='yaw_rate_penalty_clip',
                base_height_error_sq='base_height_penalty_clip',orientation='orientation_penalty_clip',
                joint_target_leg_length_error_sq='base_height_penalty_clip',
                action_rate='action_rate_penalty_clip')
            reward=np.zeros_like(h)
            for key,signal in signals.items():
                reward+=np.clip(signal*r[key]*dt,-r[bounds.get(key,'clip_single_reward')]*dt,
                                r['clip_single_reward']*dt)
            unsafe=a['dones'][...,0].astype(bool)&~a['native_timeouts'].astype(bool)
            reward+=unsafe*r['termination']
            valid=~a['dones'][:-1,:,0].astype(bool)
            residual=(a['native_reward'][:-1]-reward[:-1])[valid]
            err=float(np.max(np.abs(residual)))
            rows.append(dict(file=file.name,command_observation_max_error=float(np.max(np.abs(
                o[...,8]-c[...,2]*cfg['obs_scales']['height_measurements']))),
                direct_joint_target_map_max_error_rad=map_error,
                total_native_reward_reconstruction_max_error=err,
                verified_nonreset_transitions=int(valid.sum()),
                unobserved_nonpositive_remainder_min=float(residual.min()),
                unobserved_nonpositive_remainder_max=float(residual.max()),
                sampled_body_height_MAE_m=float(np.mean(np.abs(h-c[...,2]))),
                actual_vs_target_leg_length_MAE_m=float(np.mean(np.abs(
                    a['native_leg_lengths']-a['native_leg_references']))),
                sampled_height_command_min_m=float(c[...,2].min()),
                sampled_height_command_max_m=float(c[...,2].max()),
                sampled_unsafe_transitions=int(unsafe.sum())))
    if not rows:raise ValueError('No completed native observations to audit')
    result=dict(completed_native_updates_sampled=len(rows),sampled_envs_per_update=32,steps_per_update=48,
        code_and_source_pins_unchanged=all(sha(Path(p))==digest for p,digest in plan['source_files'].items()),
        diagnostic_script_sha256=sha(Path(__file__)),
        command_observation_max_error=max(x['command_observation_max_error'] for x in rows),
        direct_joint_target_map_max_error_rad=max(x['direct_joint_target_map_max_error_rad'] for x in rows),
        total_native_reward_reconstruction_max_error=max(x['total_native_reward_reconstruction_max_error'] for x in rows),
        rows=rows,simulator_started=False,actions_rewards_or_optimizer_changed=False,policy_accepted=False,
        reconstruction_excludes_nonpositive_collision_and_joint_limit_terms=True,
        reward_time_alignment='Native post-physics forward/yaw/gravity/height; next nonreset observation lateral velocity; chunk-last steps excluded',
        interpretation='Sampled training-chain audit only. No independent closed-loop convergence or full-gate claim.')
    write_json(output/('height_chain_prefix_audit_'+str(len(rows)).zfill(4)+'.json'),result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);args=p.parse_args()
    value=audit(args.output)
    print(json.dumps({key:item for key,item in value.items() if key!='rows'}))
