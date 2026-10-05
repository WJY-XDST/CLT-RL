"""Measured, resumable optimization of the mine robot for sim2sim preparation."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from summarize_mine_trial import acceptance_criteria, assess, plot

ROOT=Path(__file__).resolve().parents[2]
LOG_ROOT=ROOT/'IsaacLab/logs/rsl_rl/mine_wheel_legged_vmc_flat'

def checkpoints_for(run_name):
    return [x for run in LOG_ROOT.glob('*_'+run_name) for x in run.glob('model_*.pt')
            if re.fullmatch(r'model_\d+(?:_interrupted)?',x.stem)]

def checkpoint_iteration(path):
    return int(path.stem.split('_')[1])

def round_contract(state, trial_target, round_iterations, round_id, output_name):
    """Keep an explicit warm-start branch separate from old run recovery."""
    suffix=state.get('run_name_suffix','')
    if not re.fullmatch(r'[a-zA-Z0-9_]*',suffix):
        raise ValueError('Invalid training run suffix')
    target=trial_target+round_id*round_iterations+int(state.get('iteration_offset',0))
    if target<=0:raise ValueError('Training target must be positive')
    return target,f'mine_auto_r{round_id:03d}_{output_name}{suffix}'

def training_checkpoint(latest,best,report,best_report,full_motion_exposure=True):
    stance=report['conditions'][0] if 'conditions' in report else None
    unsafe=report['survivors']<report['total_trials']
    if not full_motion_exposure and stance and stance['survivors']==len(stance['trials']):
        unsafe=False
    if unsafe and best_report['survivors']==best_report['total_trials']:
        return best
    return latest

def write_json(path,value):
    path=Path(path)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    temp.replace(path)

def quality(report):
    stand=report['conditions'][0]
    duration=sum(t['first_failure_s'] if t['first_failure_s'] is not None else 12. for c in report['conditions'] for t in c['trials'])/report['total_trials']
    errors=[t['metrics'][key]/limit for c in report['conditions'] for t in c['trials'] if t['metrics'] for key,limit in t['limits'].items()]
    passed_trials=sum(t['passed'] for c in report['conditions'] for t in c['trials'])
    # One excellent stationary case must not outrank a broadly improved
    # running candidate that narrowly misses several different tolerances.
    return (report['survivors'],duration,-max(errors) if errors else -1e6,
            -sum(errors)/len(errors) if errors else -1e6,passed_trials,int(stand['passed']))

def preserve_tracking_signal(overrides):
    """Keep penalties distinct across the actual command/drive error range.

    Increasing a coefficient behind a fixed -1 reward/s cap reduces the
    informative error range. These caps follow coefficients rather than
    accepting an optimizer change that simply saturates sooner.
    """
    changed=dict(overrides)
    for weight,clip,range_sq in (
        ('lin_vel_error_sq','lin_vel_penalty_clip',1.2**2),
        ('yaw_rate_error_sq','yaw_rate_penalty_clip',.6**2),
        ('standing_wheel_tracking','standing_wheel_penalty_clip',2.*2.**2),
    ):
        default={'lin_vel_error_sq':-30.,'yaw_rate_error_sq':-5.,'standing_wheel_tracking':0.}[weight]
        changed['rewards.'+clip]=max(1.,abs(changed.get('rewards.'+weight,default))*range_sq)
    if 'rewards.action_rate' in changed:
        # All six normalized references stay in [-1,1]: sum(delta^2)<=24.
        # A larger smoothness weight must not erase differences between
        # partial chatter and a full sign flip at the 50 Hz control boundary.
        changed['rewards.action_rate_penalty_clip']=max(1.,abs(changed['rewards.action_rate'])*24.)
    # Posture also saturated at about 8 degrees with weight -150/cap 3.
    # Keep differences through the observed ~16 degree pitch regression.
    changed['rewards.orientation_penalty_clip']=max(3.,abs(changed.get('rewards.orientation',-150.))*.35**2)
    # gamma=.99 at 50 Hz gives an effective two-second reward horizon.
    # Once continuous tracking costs are widened, the old -12 terminal cost
    # can make immediate collapse cheaper than correcting a running error.
    # Price unsafe termination above twice the discounted cost of ignoring
    # the largest acceptance commands; no survival thresholds are relaxed.
    terminal_floor=4.*(abs(changed.get('rewards.lin_vel_error_sq',-30.))*.6**2
                      +abs(changed.get('rewards.yaw_rate_error_sq',-5.))*.3**2+3.)
    changed['rewards.termination']=-max(abs(changed.get('rewards.termination',-12.)),terminal_floor)
    return changed

def plan_after(report,overrides,iteration):
    changed=dict(overrides)
    stand=report['conditions'][0]
    mapping={'speed_mae_m_s':('rewards.lin_vel_error_sq',-30.,150.),'yaw_mae_rad_s':('rewards.yaw_rate_error_sq',-5.,80.),
             'height_mae_m':('rewards.base_height_error_sq',-1000.,3000.),'body_p95_deg':('rewards.orientation',-150.,450.),
             'leg_angle_mae_rad':('kp_theta',10.,50.),'leg_length_mae_m':('kp_l0',900.,1800.),
             'wheel_speed_mae_rad_s':('wheel_damping',.5,2.)}
    if changed.get('leg_control_mode')=='implicit_joint_reference':
        mapping['leg_angle_mae_rad']=('leg_joint_stiffness',300.,1000.)
        mapping['leg_length_mae_m']=('leg_joint_stiffness',300.,1000.)
    def adjust(conditions,standing_phase=False):
        worst=(1.,None)
        for condition in conditions:
            for trial in condition['trials']:
                if not trial['metrics']:continue
                for metric,spec in mapping.items():
                    stationary=(abs(condition.get('speed',0.))<.01 and abs(condition.get('yaw',0.))<.01)
                    if (standing_phase or stationary) and metric=='wheel_speed_mae_rad_s':
                        spec=('rewards.standing_wheel_tracking',-.5,10.)
                    ratio=trial['metrics'][metric]/trial['limits'][metric]
                    if ratio>worst[0]:worst=(ratio,spec)
        if worst[1]:
            key,default,cap=worst[1]
            current=abs(changed.get(key,default))
            if current>=cap:
                chatter=max((t.get('diagnostics',{}).get('wheel_reference_delta_rms_rad_s',0.)
                    for c in conditions for t in c['trials'] if t.get('diagnostics')),default=0.)
                safe=all(c['survivors']==len(c['trials']) for c in conditions)
                smooth=abs(changed.get('rewards.action_rate',-.01))
                if (safe and chatter>1. and smooth<2.
                        and key in ('rewards.yaw_rate_error_sq','rewards.standing_wheel_tracking')):
                    changed['rewards.action_rate']=-min(2.,smooth*1.5)
                    return f'capped {key}; adjust action_rate for measured wheel reference delta RMS {chatter:.3f} rad/s'
                return f'continue learning at {key} cap; measured error ratio {worst[0]:.3f}'
            changed[key]=(1. if default>0 else -1.)*min(cap,current*1.25)
            if key=='wheel_damping':changed['robot.actuators.wheels.damping']=changed[key]
            if key=='rewards.base_height_error_sq':
                changed['rewards.base_height_penalty_clip']=max(30.,abs(changed[key])*.18**2)
            return f'adjust {key} for measured error ratio {worst[0]:.3f}'
        return 'continue learning without parameter change'
    if not stand['passed']:
        if (stand['survivors']==len(stand['trials'])
                and changed.get('commands.standing_only_steps',iteration*48)<iteration*48):
            return changed,'retain established motion course; '+adjust(report['conditions'])
        # Keep failed balance pilots out of motion/large reset perturbations.
        start=(iteration+1000)*48
        for key in ('standing_only_steps','reverse_ramp_start_steps','yaw_start_steps'):
            changed['commands.'+key]=start
        if stand['survivors']==len(stand['trials']):
            # A safe stance with actuator tracking error is not a collapse.
            # Tune its measured mismatch rather than repeatedly penalize death.
            return changed,'retain balance phase; '+adjust([stand],standing_phase=True)
        changed['rewards.termination']=-min(30.,abs(changed.get('rewards.termination',-10.))*1.2)
        height_errors=[t['metrics']['height_mae_m'] for t in stand['trials'] if t['metrics']]
        if not height_errors or max(height_errors)>.02:
            scale=min(3000.,abs(changed.get('rewards.base_height_error_sq',-1000.))*1.2)
            changed['rewards.base_height_error_sq']=-scale
            changed['rewards.base_height_penalty_clip']=max(30.,scale*.18**2)
        return changed,'extend balance phase; restore exploration; strengthen unsafe termination cost'
    changed['commands.standing_only_steps']=min(changed['commands.standing_only_steps'],iteration*48)
    changed['commands.reverse_ramp_start_steps']=changed['commands.standing_only_steps']
    changed['commands.yaw_start_steps']=changed['commands.standing_only_steps']
    # Change one reward term selected by the measured worst normalized error.
    return changed,'continue mixed motion; '+adjust(report['conditions'])

def main():
    p=argparse.ArgumentParser()
    init=p.add_mutually_exclusive_group(required=True)
    init.add_argument('--checkpoint',type=Path)
    init.add_argument('--fresh',action='store_true',help='Train from fresh network weights; recover only this output directory on restart')
    p.add_argument('--initial-overrides',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trial_target',type=int,default=1000)
    p.add_argument('--round_iterations',type=int,default=1000)
    p.add_argument('--num_envs',type=int,default=None,help='Parallel training environments; preserve the saved count on restart')
    p.add_argument('--max_rounds',type=int,default=0,help='0: keep optimizing until acceptance')
    p.add_argument('--shutdown-after-success',action='store_true',help='Finalize verified results and documents, then power off')
    p.add_argument('--wheel-head-only-finetune',action='store_true',help='Persist controlled wheel-head precision training mode')
    args=p.parse_args()
    if args.num_envs is not None and args.num_envs<=0:
        raise ValueError('num_envs must be positive')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(LOG_ROOT/'.auto_training.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=output/'status.json'
    env=dict(os.environ)
    env['PYTHONPATH']=str(ROOT/'wheel_legged_isaaclab')
    for key in ('ROS_DISTRO','ROS_VERSION','ROS_PYTHON_VERSION','AMENT_PREFIX_PATH','COLCON_PREFIX_PATH'):
        env.pop(key,None)
    state={'state':'starting','pid':os.getpid(),'checkpoint':str(args.checkpoint.resolve()) if args.checkpoint else None,'round':0,
           'initialization':'fresh' if args.fresh else 'resume',
           'objective':'tracking and genuine upright running policy for new-model sim2sim',
           'num_envs':args.num_envs or 256,'trial_target':args.trial_target,'round_iterations':args.round_iterations}
    if state_path.exists():
        old=json.loads(state_path.read_text())
        if old.get('state')=='accepted':
            command=[sys.executable,str(ROOT/'mine_model/scripts/complete_mine_training.py'),'--state',str(state_path)]
            if args.shutdown_after_success:command+=['--shutdown']
            subprocess.run(command,check=True)
            return
        state.update(old);state['pid']=os.getpid();state['state']='restarting'
    state['shutdown_after_success']=args.shutdown_after_success
    if args.num_envs is not None:state['num_envs']=args.num_envs
    state['round_iterations']=args.round_iterations
    state['trial_target']=args.trial_target
    state['wheel_head_only_finetune']=bool(args.wheel_head_only_finetune or state.get('wheel_head_only_finetune',False))
    criteria=acceptance_criteria()
    if state.get('acceptance_criteria')!=criteria:
        state['passed']=False
    state['acceptance_criteria']=criteria
    state['objective']=f"{100*criteria['relative_error']:g} percent tracking and genuine upright running policy for new-model sim2sim"
    if state['wheel_head_only_finetune'] and not state.get('checkpoint'):
        raise ValueError('Wheel-head precision mode requires a pretrained checkpoint')
    initial_overrides=json.loads(args.initial_overrides.read_text()) if args.initial_overrides else {}
    overrides=state.get('overrides',initial_overrides or {'commands.standing_only_steps':48000,'commands.motion_ramp_steps':48000,
                    'commands.reverse_ramp_start_steps':48000,'commands.reverse_ramp_steps':48000,
                    'commands.yaw_start_steps':48000,'commands.yaw_ramp_steps':48000,
                    'commands.ranges_ang_vel_yaw':[-.3,.3],'commands.heading_command':False,'commands.grouped_training':True})
    agent_overrides=state.get('agent_overrides',{})
    if any(not re.fullmatch(r'[a-zA-Z0-9_.]+',key) for key in agent_overrides):
        raise ValueError('Invalid PPO override key')
    checkpoint=Path(state['checkpoint']) if state['checkpoint'] else None
    best_report=None
    if state.get('best_assessment'):
        incumbent=Path(state['best_assessment'])
        best_report=json.loads(incumbent.read_text())
        if best_report.get('acceptance_criteria')!=criteria:
            report_name=f"assessment_{100*criteria['relative_error']:g}pct.json"
            best_report=assess(incumbent.parent,report_name=report_name)
            state['best_assessment']=str(incumbent.parent/report_name)
    best_checkpoint=Path(state['best_checkpoint']) if state.get('best_checkpoint') else checkpoint
    def save(**values):
        if 'state' in values and values['state']!='error':state.pop('error',None)
        state.update(values);state['updated_at']=datetime.now().astimezone().isoformat();write_json(state_path,state)
        print(json.dumps(values,ensure_ascii=False),flush=True)
    def job(command,log):
        log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as stream:
            stream.write('\nJob started '+datetime.now().astimezone().isoformat()+'\n')
            stream.flush()
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            save(child_pid=child.pid,child_command=command,child_log=str(log))
            code=child.wait()
        save(child_pid=None)
        if code:raise RuntimeError(f'Job failed ({code}): {log}')
    try:
        round_id=int(state['round'])
        while not args.max_rounds or round_id<args.max_rounds:
            phase='pilot' if round_id==0 else 'optimization'
            folder=output/f'round_{round_id:03d}_{phase}';folder.mkdir(parents=True,exist_ok=True)
            override_path=folder/'overrides.json';write_json(override_path,overrides)
            target,run_name=round_contract(state,args.trial_target,args.round_iterations,round_id,output.name)
            existing=checkpoints_for(run_name)
            recovered=max(existing,key=lambda x:(checkpoint_iteration(x),x.stat().st_mtime_ns)) if existing else None
            if recovered and (checkpoint is None or checkpoint_iteration(recovered)>checkpoint_iteration(checkpoint)):checkpoint=recovered
            command=[str(ROOT/'run_python.sh'),str(ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py'),
                     '--task','WheelLeggedVMC-Flat-v0','--headless','--num_envs',str(state['num_envs']),'--seed','43',
                     '--run_name',run_name,'--target_iteration',str(target)]
            if checkpoint is not None:command+=['--checkpoint_path',str(checkpoint)]
            if state['wheel_head_only_finetune']:command+=['--wheel_head_only_finetune']
            balance_failed=bool(best_report and best_report['conditions'][0]['survivors'] < len(best_report['conditions'][0]['trials']))
            rollback=(checkpoint is not None and state.get('candidate_checkpoint')
                      and checkpoint_iteration(checkpoint)<checkpoint_iteration(Path(state['candidate_checkpoint'])))
            wheel_transition=((recovered is None and state.get('motion_exploration_round',round_id)==round_id)
                              or 'motion_exploration_round' not in state
                              or (rollback and state.get('motion_exploration_round')!=round_id))
            if recovered is None and (round_id==0 or balance_failed) and not state['wheel_head_only_finetune']:
                command+=['--reset_optimizer','--reset_exploration_std','0.3']
            elif (wheel_transition and checkpoint is not None
                  and not state['wheel_head_only_finetune']
                  and overrides.get('leg_control_mode')=='implicit_joint_reference'
                  and overrides['commands.standing_only_steps']<=(checkpoint_iteration(checkpoint)+1)*48):
                # Motion needs wheel exploration beyond the stationary drive's
                # small response range; keep the already stable leg distribution.
                command+=['--wheel_exploration_std','0.08']
                save(motion_exploration_round=round_id)
            for key,value in overrides.items():command.append('env.'+key+'='+json.dumps(value))
            for key,value in agent_overrides.items():command.append('agent.'+key+'='+json.dumps(value))
            save(state='training_'+phase,round=round_id,target_iteration=target,overrides=overrides,agent_overrides=agent_overrides,checkpoint=str(checkpoint) if checkpoint else None)
            if recovered is None or checkpoint_iteration(recovered)<target-1:
                job(command,folder/'training.log')
            checkpoints=checkpoints_for(run_name)
            if not checkpoints:raise RuntimeError('No training checkpoints were produced')
            latest=max(checkpoints,key=lambda x:(checkpoint_iteration(x),x.stat().st_mtime_ns))
            run=latest.parent
            if checkpoint_iteration(latest)<target-1:raise RuntimeError('Training exited before target checkpoint')
            save(state='evaluating',training_run=str(run),candidate_checkpoint=str(latest))
            eval_dir=folder/'replay'
            base=[str(ROOT/'run_python.sh'),str(ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
                  '--headless','--robot','mine','--checkpoint',str(latest),'--seconds','12','--overrides_json',str(override_path)]
            job(base+['--output',str(eval_dir)],folder/'evaluation.log')
            report=assess(eval_dir);plot(eval_dir)
            if best_report is None or quality(report)>quality(best_report):
                best_report=report;best_checkpoint=latest
                save(best_checkpoint=str(latest),best_assessment=str(eval_dir/'assessment.json'))
            save(last_assessment=str(eval_dir/'assessment.json'),survivors=report['survivors'],resets=report['resets'],passed=report['passed'])
            if report['passed']:
                # Independent reset streams verify the SAME weights again.
                heldout=folder/'heldout_replay'
                job(base+['--seeds','83','84','85','--output',str(heldout)],folder/'heldout.log')
                validation=assess(heldout);plot(heldout)
                if validation['passed']:
                    export=output/'accepted_policy'
                    job(base+['--seconds','30','--seeds','83','84','85','--output',str(folder/'export_replay'),'--export_dir',str(export)],folder/'export.log')
                    extended=assess(folder/'export_replay')
                    if not extended['passed']:
                        # Optimize the errors actually seen in the extended
                        # run, rather than the already passing short replay.
                        report=extended
                        plot(folder/'export_replay')
                        save(state='extended_replay_failed',passed=False,
                             last_assessment=str(folder/'export_replay'/'assessment.json'))
                    else:
                        plot(folder/'export_replay')
                        write_json(export/'acceptance.json',{'checkpoint':str(latest),'training_assessment':str(eval_dir/'assessment.json'),'heldout_assessment':str(heldout/'assessment.json'),'criterion':report['criterion'],'acceptance_criteria':report['acceptance_criteria']})
                        save(state='accepted',checkpoint=str(latest),export_directory=str(export))
                        finalize=[sys.executable,str(ROOT/'mine_model/scripts/complete_mine_training.py'),'--state',str(state_path)]
                        if args.shutdown_after_success:finalize+=['--shutdown']
                        job(finalize,folder/'finalization.log')
                        return
                else:
                    report=validation
                    save(state='heldout_replay_failed',passed=False,
                         last_assessment=str(heldout/'assessment.json'))
            # Keep learning through failed balance pilots; only roll back a
            # regression once an incumbent already survives every condition.
            full_exposure=target*48>=max(
                overrides['commands.standing_only_steps']+overrides['commands.motion_ramp_steps'],
                overrides['commands.reverse_ramp_start_steps']+overrides['commands.reverse_ramp_steps'],
                overrides['commands.yaw_start_steps']+overrides['commands.yaw_ramp_steps'])
            checkpoint=training_checkpoint(latest,best_checkpoint,report,best_report,full_exposure)
            overrides,purpose=plan_after(report,overrides,target)
            overrides=preserve_tracking_signal(overrides)
            round_id+=1
            save(state='planning_next_round',round=round_id,checkpoint=str(checkpoint),overrides=overrides,purpose=purpose)
        save(state='round_limit_reached')
    except BaseException as error:
        # Finalization failures must retry the accepted artifact gates on restart,
        # rather than resume training and lose the accepted checkpoint status.
        save(state='accepted' if state.get('state')=='accepted' else 'error',error=str(error))
        raise
    finally:
        lock.close()

if __name__=='__main__':main()
