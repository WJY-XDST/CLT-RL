"""Retained-state PPO with leg-only extra entropy and complete gate reviews."""
import argparse
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

import continue_reviewed_joint_pd as core
from auto_train_mine import write_json
from summarize_mine_trial import assess, assess_stability
from wheel_legged_gym_isaaclab.joint_pd_exploration import entropy_multipliers

ROOT=core.ROOT
ENTRY=ROOT/'mine_model/scripts/train_joint_pd_split_exploration.py'
POLICY=ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_pd_exploration.py'
ORIGINAL_COMMAND=core.ORIGINAL_COMMAND
ORIGINAL_BLOCK_RUN=core.block.run
METRICS=('speed_mae_m_s','yaw_mae_rad_s','height_mae_m','body_p95_deg',
         'leg_length_mae_m','leg_angle_mae_rad','wheel_speed_mae_rad_s')


def review_summary(report):
    summary=core.review_summary(report)
    ratios={}
    for condition in report['conditions']:
        for trial in condition['trials']:
            if not trial['survived'] or not trial.get('metrics'):
                continue
            for metric in METRICS:
                key=condition['name']+'/'+metric
                value=trial['metrics'][metric]/trial['limits'][metric]
                if not math.isfinite(value) or value < 0.:
                    raise ValueError('Nonfinite or negative original tracking error ratio')
                ratios[key]=max(ratios.get(key,0.),value)
    return dict(**summary,measured_case_metric_ratios=ratios,
        missing_steady_trials=report['total_trials']-report['survivors'])


def next_entropy(history,current):
    result=core.next_entropy(history,current)
    if len(history)>1:
        before,after=history[-2:]
        old=before.get('measured_case_metric_ratios',{})
        new=after.get('measured_case_metric_ratios',{})
        regressions=[key for key,value in new.items()
                     if key in old and value>1. and value>max(1.2*old[key],old[key]+.25)]
        if regressions:
            return max(.001,current/2.)
    return result


def split_entropy_command(command,coefficient,evidence):
    entropy_multipliers(coefficient)
    command=core.entropy_command(command,coefficient)
    if any(v.startswith('agent.policy.class_name=') for v in command):
        raise ValueError('Do not replace another declared policy implementation')
    return [command[0],str(ENTRY),'--leg-entropy-coefficient',str(coefficient),
            '--evidence',str(evidence),'--',*command[2:],
            'agent.policy.class_name=JointPdExplorationActorCritic']


def prepare(previous,output,coefficient,updates):
    old=core.INTERVAL
    try:
        core.INTERVAL=updates
        plan,expected=core.prepare(previous,output,coefficient)
    finally:
        core.INTERVAL=old
    pins=plan['source_files']
    for path in (Path(__file__).resolve(),ENTRY,POLICY,
                 ROOT/'wheel_legged_isaaclab/tests/test_joint_pd_split_exploration.py',
                 ROOT/'wheel_legged_isaaclab/tests/test_split_exploration_continuation.py'):
        pins[str(path)]=core.sha(path)
    profile=Path(previous)/'native_entropy_profile.json'
    if profile.exists():pins[str(profile)]=core.sha(profile)
    plan.update(only_training_change='PPO leg entropy; wheel entropy stays .001. Sampling, inference, weights and all Adam states retained',
        effective_leg_entropy_coefficient=coefficient,effective_wheel_entropy_coefficient=.001,
        dimension_entropy_multipliers=entropy_multipliers(coefficient),
        first_pilot_or_normal_review_updates=updates)
    write_json(Path(output)/'experiment_plan.json',plan)
    return plan,expected


def environment():
    return dict(os.environ,PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
                PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')


def recover_handoff(previous,base):
    """Finish a deliberately saved handoff's exact latest checkpoint replay."""
    previous=Path(previous)
    status=json.loads((previous/'status.json').read_text())
    if status.get('training_active'):
        raise ValueError('The previous native owner must finish saving first')
    source=Path(status['checkpoint'])
    saved=torch.load(source,map_location='cpu',weights_only=True)
    actual=core.saved_updates(source)
    if (core.sha(source)!=status['checkpoint_sha256'] or not core.tensor_finite(saved)
            or len(saved['optimizer_state_dict']['state'])!=17 or core.optimizer_step(saved)!=20*actual):
        raise ValueError('Handoff did not retain a complete finite latest policy and Adam')
    plan=json.loads((previous/'experiment_plan.json').read_text())
    for name,digest in plan['source_files'].items():
        if core.sha(Path(name))!=digest:raise ValueError('Pinned source changed before handoff: '+name)
    replay=previous/'replay'
    if (replay/'completed.json').exists():
        report=json.loads((replay/'assessment.json').read_text())
        if report['checkpoint_sha256']==core.sha(source):return
        raise ValueError('A different replay already occupies this saved boundary')
    stop=previous/'STOP'
    if not stop.exists() or json.loads(stop.read_text()).get('reason')!='agent_selected_exploration_handoff':
        raise ValueError('Do not silently resume an owner stop or an undiagnosed native failure')
    # Keep the request as evidence; only the finished, reviewed handoff consumes it.
    stop.rename(previous/'consumed_exploration_handoff.json')
    core.block.runtime.OUTPUT=base
    core.block.runtime.run_child(core.block.replay_command(source,replay),previous/'recovery_replay.log',environment())
    report=assess(replay)
    if report['checkpoint_sha256']!=core.sha(source):raise ValueError('Handoff replay checkpoint changed')
    core.block.assess_directory(replay)
    proof=core.block.verify_update(Path(plan['source_checkpoint']),source,actual,
        core.validate_checkpoint_robot(source),Path(plan['source_checkpoint']))
    write_json(previous/'handoff_resume_verification.json',proof)
    write_json(previous/'pre_handoff_review_status.json',status)
    status.update(state='agent_review_required',training_active=False,actual_ppo_updates=actual,
                  live_ppo_updates=actual,updated_at=datetime.now().astimezone().isoformat(),
                  reason='agent_selected_exploration_handoff_replayed_with_latest_saved_state',
                  handoff_replay_completed=True,policy_accepted=False,goal_remains_active=True)
    write_json(previous/'status.json',status)


def verify_entropy(output,coefficient):
    evidence=json.loads((output/'native_entropy_profile.json').read_text())
    if (evidence['leg_entropy_coefficient']!=coefficient or evidence['wheel_entropy_coefficient']!=.001
            or evidence['modified_sampling_or_log_prob'] or evidence['modified_inference']
            or evidence['additional_parameters'] or evidence['additional_checkpoint_keys']
            or any(abs(a-b)>1.e-7 for a,b in zip(evidence['dimension_multipliers'],entropy_multipliers(coefficient)))
            or len(evidence['dimension_multipliers'])!=6):
        raise ValueError('Native entropy implementation differs from the qualified objective')
    state=json.loads((output/'status.json').read_text())
    agent=yaml.safe_load((Path(state['checkpoint']).parent/'params/agent.yaml').read_text())
    if (agent['algorithm']['class_name']!='PPO' or agent['algorithm']['entropy_coef']!=coefficient
            or agent['policy']['class_name']!='JointPdExplorationActorCritic'):
        raise ValueError('Native PPO did not use the declared exploration profile')


def run(previous,base,unit,recover=False):
    base=Path(base).resolve();base.mkdir(exist_ok=True)
    if base.parent!=ROOT/'mine_model/results':raise ValueError('Use an owned coordinator directory')
    with (base/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cursor=base/'coordinator.json'
        if cursor.exists():raise ValueError('Inspect an existing coordinator before restarting its owner')
        previous=Path(previous).resolve()
        if recover:
            write_json(cursor,dict(state='reviewing_saved_handoff',previous_output=str(previous),policy_accepted=False))
            pointer=dict(unit=unit,directory=str(previous),coordinator=str(cursor),
                controller_kind='continuous_split_exploration',phase='reviewing_saved_handoff',
                weights_reset=False,optimizer_reset=False,policy_accepted=False,goal_remains_active=True)
            for name in ('joint_pd_active_joint_tracking.json','joint_pd_active_standing_seed.json'):
                write_json(ROOT/'mine_model/results'/name,pointer)
            with (core.RUNS/'.cumulative_learning.lock').open('a') as native_owner:
                fcntl.flock(native_owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                recover_handoff(previous,base)
        status=json.loads((previous/'status.json').read_text())
        parent=json.loads((previous/'experiment_plan.json').read_text()).get('previous_output')
        history=[]
        if parent:
            p=Path(parent)
            report_path=p/'replay/assessment.json'
            if report_path.exists():
                report=json.loads(report_path.read_text())
                history.append(dict(actual_ppo_updates=json.loads((p/'status.json').read_text())['actual_ppo_updates'],**review_summary(report)))
        coefficient=.005;cycle=0
        while not (base/'STOP').exists():
            state=json.loads((previous/'status.json').read_text())
            report=json.loads((previous/'replay/assessment.json').read_text())
            summary=review_summary(report)
            history.append(dict(actual_ppo_updates=state['actual_ppo_updates'],**summary))
            # The first pilot keeps .005 for legs while reverting wheels to .001.
            # Subsequent changes respond to every measured original case/metric.
            if cycle:coefficient=next_entropy(history,coefficient)
            write_json(previous/'split_exploration_review.json',dict(**summary,next_leg_entropy=coefficient,
                wheel_entropy=.001,weights_or_Adam_reset=False,original_acceptance_unchanged=True,policy_accepted=False))
            if report['passed']:
                passed=True
                for label,seconds,seeds,window in [('heldout',30,(183,184,185),20),('sustained',60,(283,284,285),40)]:
                    directory=previous/label
                    command=core.block.replay_command(Path(state['checkpoint']),directory)
                    i=command.index('--seconds');command[i+1]=str(seconds)
                    i=command.index('--seeds');command[i+1:i+4]=list(map(str,seeds))
                    core.block.runtime.OUTPUT=base
                    core.block.runtime.run_child(command,previous/(label+'.log'),environment())
                    passed=bool(assess_stability(directory,window)['passed']) and passed
                if passed:
                    write_json(cursor,dict(state='all_original_and_sustained_gates_verified_requires_final_audit',
                        checkpoint=state['checkpoint'],history=history,policy_accepted=False))
                    return
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output=ROOT/'mine_model/results'/('joint_pd_optimization_'+stamp+'_split_exploration')
            updates=25 if cycle==0 else 100
            args=SimpleNamespace(previous=previous,output=output,reward_key='tracking_task_quality_gate_floor',
                value=.1,updates=updates,prepare_only=False)
            def command(*a,**kw):
                native,final,saved=ORIGINAL_COMMAND(*a,**kw)
                return split_entropy_command(native,coefficient,output/'native_entropy_profile.json'),final,saved
            core.block.continuation_command=command
            write_json(cursor,dict(state='training_and_review_cycle',cycle=cycle,current_output=str(output),
                previous_output=str(previous),history=history,leg_entropy_coefficient=coefficient,
                wheel_entropy_coefficient=.001,policy_accepted=False))
            pointer=dict(unit=unit,directory=str(output),source_checkpoint=state['checkpoint'],
                weights_reset=False,optimizer_reset=False,optimizer_states=17,
                optimizer_step_before=20*state['actual_ppo_updates'],controller_kind='continuous_split_exploration',
                coordinator=str(cursor),policy_accepted=False,goal_remains_active=True)
            for name in ('joint_pd_active_joint_tracking.json','joint_pd_active_standing_seed.json'):
                write_json(ROOT/'mine_model/results'/name,pointer)
            ORIGINAL_BLOCK_RUN(args,prepare_function=lambda *a:prepare(a[0],a[1],coefficient,updates))
            verify_entropy(output,coefficient)
            previous=output;cycle+=1
        write_json(cursor,dict(state='explicit_stop_respected',previous_output=str(previous),policy_accepted=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--coordinator',type=Path,required=True)
    parser.add_argument('--unit',required=True)
    parser.add_argument('--recover-handoff',action='store_true')
    args=parser.parse_args()
    run(args.previous,args.coordinator,args.unit,args.recover_handoff)
