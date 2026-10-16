"""Continuous retained-state PPO, with completed reviews between native blocks.

This owns the transitions that one-shot reward experiments deliberately left
to an agent. No checkpoint rollback, inference remapping, hardware or shutdown
path is present. A failed acceptance test remains a failed acceptance test.
"""
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

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, optimizer_step, sha, tensor_finite, qualified_wheel_mass_asset
from train_joint_pd_posture_repair import GATE_SHA
from train_joint_pd_standing_seed import saved_updates
from summarize_mine_trial import assess_stability
from watch_joint_pd_scratch import evaluate
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, validate_checkpoint_robot
import train_joint_pd_reward_block as block

INTERVAL=100
BASE_ENTROPY=.005
MAX_ENTROPY=.02
ORIGINAL_COMMAND=block.continuation_command


def entropy_command(command, coefficient):
    if (isinstance(coefficient,bool) or not isinstance(coefficient,(float,int))
            or not math.isfinite(coefficient) or not .001<=coefficient<=MAX_ENTROPY):
        raise ValueError('Require a finite bounded PPO entropy coefficient')
    if any(v.startswith('agent.algorithm.entropy_coef=') for v in command):
        raise ValueError('Do not silently replace an existing entropy override')
    if any(v in command for v in ('--reset_optimizer','--reset_exploration_std')):
        raise ValueError('Retain learned weights, exploration and complete Adam')
    return [*command,'agent.algorithm.entropy_coef='+str(coefficient)]


def review_summary(report):
    if len(report['conditions'])!=9 or report['total_trials']!=27:
        raise ValueError('Review all original conditions and seeds')
    height=[]
    for case in report['conditions']:
        if case['name'] in ('height_low','height_high'):
            if len(case['trials'])!=3 or any(not t['survived'] or not t.get('metrics') for t in case['trials']):
                return dict(survivors=report['survivors'],height_error_fraction=None,
                    full_gate_passed=False,resets=report['resets'])
            height.extend(t['metrics']['height_mae_m']/case['height'] for t in case['trials'])
    if len(height)!=6:
        raise ValueError('Both original height endpoints are required')
    return dict(survivors=report['survivors'],height_error_fraction=max(height),
        full_gate_passed=report['passed'],resets=report['resets'])


def next_entropy(history,current):
    """Respond to measured regression or repeated height stagnation, not logs."""
    if len(history)<2:
        return current
    before,after=history[-2:]
    if after['survivors']<before['survivors']:
        return max(.001,current/2.)
    a,b=before['height_error_fraction'],after['height_error_fraction']
    if a is None or b is None:
        return max(.001,current/2.)
    if len(history)>=3:
        values=[v['height_error_fraction'] for v in history[-3:]]
        if all(v is not None and v>.1 for v in values) and min(values)>=.98*values[0]:
            return min(MAX_ENTROPY,current*1.5)
    return current


def prepare(previous,output,coefficient):
    previous,output=Path(previous).resolve(),Path(output).resolve()
    old=json.loads((previous/'experiment_plan.json').read_text())
    status=json.loads((previous/'status.json').read_text())
    source=Path(status['checkpoint']).resolve()
    if status.get('training_active') or status['state'] not in ('agent_review_required','agent_diagnosis_required'):
        raise ValueError('Start from a saved and reviewed boundary')
    if sha(source)!=status['checkpoint_sha256'] or not source.is_relative_to(RUNS.resolve()):
        raise ValueError('The latest reviewed source checkpoint changed')
    saved=torch.load(source,map_location='cpu',weights_only=True)
    actual=saved_updates(source)
    if not tensor_finite(saved) or len(saved['optimizer_state_dict']['state'])!=17 or optimizer_step(saved)!=20*actual:
        raise ValueError('Require all finite weights and the complete continuous Adam clock')
    expected=validate_checkpoint_robot(source)
    if (expected['leg_control_mode']!='implicit_joint_position'
            or expected['joint_reset_profile']!='joint_target_nominal_v1'
            or tuple(expected['commands']['ranges_height'])!=SAFE_BASE_HEIGHT_RANGE_M
            or not expected['commands']['full_height_range_from_start']):
        raise ValueError('Preserve direct joint PD, nominal closed-chain reset and full safe height range')
    agent=yaml.safe_load((source.parent/'params/agent.yaml').read_text())
    if agent['algorithm']['class_name']!='PPO':
        raise ValueError('Preserve ordinary full-policy PPO')
    source_report=json.loads((previous/'replay/assessment.json').read_text())
    if source_report['checkpoint_sha256']!=sha(source):
        raise ValueError('Review must cover the exact latest checkpoint')
    review_summary(source_report)
    pins=dict(old['source_files'])
    for path,digest in pins.items():
        if sha(Path(path))!=digest:
            raise ValueError('Pinned source/evidence changed: '+path)
    asset,calibration=qualified_wheel_mass_asset(ASSETS/MANIFEST['wheel_mass_calibration'])
    if expected['robot']['spawn']['usd_path']!=asset or sha(ROOT/'mine_model/config/tracking_acceptance.json')!=GATE_SHA:
        raise ValueError('Measured-mass asset and original acceptance must remain unchanged')
    pins.update(calibration['asset_chain_sha256'])
    for path in [source,*sorted((source.parent/'params').glob('*.yaml')),
                 previous/'status.json',previous/'experiment_plan.json',previous/'replay/assessment.json',
                 Path(__file__).resolve()]:
        pins[str(path)]=sha(path)
    output.mkdir(exist_ok=False)
    plan=dict(previous_output=str(previous),source_checkpoint=str(source),source_sha256=sha(source),
        source_replay=str(previous/'replay'),protected_reference=old['protected_reference'],
        source_actual_ppo_updates=actual,source_Adam_step=optimizer_step(saved),source_Adam_states=17,
        target_actual_ppo_updates=actual+INTERVAL,bounded_training_updates=INTERVAL,
        run_name='reviewed_joint_pd_'+output.name+'_u'+str(actual),
        source_files=pins,entropy_coefficient=coefficient,
        previous_entropy_coefficient=agent['algorithm']['entropy_coef'],
        reward_key='tracking_task_quality_gate_floor',
        reward_value=expected['rewards']['tracking_task_quality_gate_floor'],only_reward_change={},
        only_training_change='PPO entropy coefficient; loaded weights/std/Adam/adaptive LR are retained',
        weights_std_Adam_and_adaptive_LR_reset=False,ordinary_full_policy_PPO=True,
        full_height_range=list(SAFE_BASE_HEIGHT_RANGE_M),original_acceptance_sha256=GATE_SHA,
        goal_remains_active=True,policy_accepted=False,hardware_enabled=False,shutdown=False)
    write_json(output/'experiment_plan.json',plan)
    return plan,expected


def run(previous,base,unit):
    base=Path(base).resolve();base.mkdir(exist_ok=True)
    if base.parent!=ROOT/'mine_model/results':
        raise ValueError('Use an owned coordinator directory')
    cursor_path=base/'coordinator.json'
    with (base/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if cursor_path.exists():
            raise ValueError('Inspect the saved coordinator boundary before restarting this owner')
        history=[];coefficient=BASE_ENTROPY;previous=Path(previous).resolve();cycle=0
        while True:
            if (base/'STOP').exists():
                write_json(cursor_path,dict(state='explicit_stop_respected',previous_output=str(previous)))
                return
            source_status=json.loads((previous/'status.json').read_text())
            report=json.loads((previous/'replay/assessment.json').read_text())
            summary=review_summary(report)
            history.append(dict(actual_ppo_updates=source_status['actual_ppo_updates'],**summary))
            coefficient=next_entropy(history,coefficient)
            write_json(previous/'continuous_optimization_review.json',dict(
                **summary,next_entropy_coefficient=coefficient,original_acceptance_unchanged=True,
                decision='Continue from this latest complete saved state; no weight or Adam reset',
                policy_accepted=False))
            if report['passed']:
                passed=True
                for label,seconds,seeds,stable in [('heldout',30,(183,184,185),20),('sustained',60,(283,284,285),40)]:
                    directory=previous/label
                    command=block.replay_command(Path(source_status['checkpoint']),directory)
                    index=command.index('--seconds');command[index+1]=str(seconds)
                    index=command.index('--seeds');command[index+1:index+4]=list(map(str,seeds))
                    block.runtime.OUTPUT=base
                    environment=dict(os.environ,PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
                        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
                    block.runtime.run_child(command,previous/(label+'.log'),environment)
                    passed=bool(assess_stability(directory,stable)['passed']) and passed
                if passed:
                    write_json(cursor_path,dict(state='all_original_and_sustained_gates_verified_requires_final_audit',
                        checkpoint=source_status['checkpoint'],history=history,policy_accepted=False))
                    return
            stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            output=ROOT/'mine_model/results'/('joint_pd_optimization_'+stamp+'_reviewed_exploration')
            args=SimpleNamespace(previous=previous,output=output,reward_key='tracking_task_quality_gate_floor',
                value=.1,updates=INTERVAL,prepare_only=False)
            def command_with_entropy(*arguments,**keywords):
                command,final,saved=ORIGINAL_COMMAND(*arguments,**keywords)
                return entropy_command(command,coefficient),final,saved
            block.continuation_command=command_with_entropy
            pointer=dict(unit=unit,directory=str(output),source_checkpoint=source_status['checkpoint'],
                optimizer_states=17,optimizer_step_before=source_status['actual_ppo_updates']*20,
                weights_reset=False,optimizer_reset=False,goal_remains_active=True,policy_accepted=False,
                controller_kind='continuous_reviewed_joint_pd',coordinator=str(cursor_path))
            write_json(cursor_path,dict(state='training_and_review_cycle',cycle=cycle,entropy_coefficient=coefficient,
                current_output=str(output),previous_output=str(previous),history=history))
            for name in ('joint_pd_active_joint_tracking.json','joint_pd_active_standing_seed.json'):
                write_json(ROOT/'mine_model/results'/name,pointer)
            block.run(args,prepare_function=lambda *a:prepare(a[0],a[1],coefficient))
            after=json.loads((output/'status.json').read_text())
            algorithm=yaml.safe_load((Path(after['checkpoint']).parent/'params/agent.yaml').read_text())['algorithm']
            if algorithm['entropy_coef']!=coefficient or algorithm['class_name']!='PPO':
                raise ValueError('Native runner did not retain the declared ordinary PPO exploration revision')
            previous=output;cycle+=1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--coordinator',type=Path,required=True)
    parser.add_argument('--unit',required=True)
    args=parser.parse_args()
    run(args.previous,args.coordinator,args.unit)
