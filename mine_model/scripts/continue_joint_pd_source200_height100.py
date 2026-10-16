"""Continue the model_200 height experiment, reviewing every 100 PPO updates.

Finish the already running short trial at its saved review boundary, then
adopt its reviewed incumbent. Rejected weights/Adam never become the source.
"""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
import shutil
import time

import torch
import yaml

from auto_train_mine import write_json
import continue_joint_pd_source200_height as experiment
from joint_pd_continuation import validate_continuation_config
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot

guard=experiment.guard
ROOT=guard.ROOT
UPDATES=100


def selected_incumbent(cursor):
    if cursor['state']!='explicit_stop_respected':
        raise ValueError('The previous controller must finish its saved independent review')
    previous=Path(cursor['incumbent']).resolve()
    history=cursor['history']
    if not history:raise ValueError('Keep the already-started short trial and its completed review')
    last=history[-1]
    expected=Path(last['output'] if last['promote'] else last['source']).resolve()
    if previous!=expected:raise ValueError('Never continue a rejected branch')
    if previous.parent!=ROOT/'mine_model/results':raise ValueError('Unexpected incumbent directory')
    return previous


def adopt(previous,candidate,base,previous_cursor):
    previous,candidate=Path(previous),Path(candidate)
    status=json.loads((previous/'status.json').read_text());source=Path(status['checkpoint'])
    parent=json.loads((previous/'experiment_plan.json').read_text())
    report=json.loads((previous/'replay/assessment.json').read_text())
    if (status['training_active'] or status['state']!='agent_review_required'
            or report['checkpoint_sha256']!=guard.core.sha(source)
            or status['checkpoint_sha256']!=guard.core.sha(source)):
        raise ValueError('Adopt a complete saved and independently reviewed incumbent only')
    saved=torch.load(source,map_location='cpu',weights_only=True)
    actual=guard.core.saved_updates(source)
    if (not guard.core.tensor_finite(saved) or len(saved['optimizer_state_dict']['state'])!=17
            or guard.core.optimizer_step(saved)!=20*actual):
        raise ValueError('Retain all 17 Adam states and the actual completed-update clock')
    if source.name=='model_200.pt' and guard.core.sha(source)!=experiment.SOURCE_SHA:
        raise ValueError('The rollback source must be the owner-selected original model_200')
    cfg=validate_checkpoint_robot(source)
    if cfg['rewards']['base_height_error_sq'] not in (0.,-50.,-100.,-250.,-500.):
        raise ValueError('Keep the declared bounded single-coefficient height experiment')
    asset,_=guard.core.qualified_wheel_mass_asset(experiment.ASSETS/experiment.MANIFEST['wheel_mass_calibration'])
    original=experiment.starting_config(validate_checkpoint_robot(experiment.SOURCE),asset)
    original['rewards']['base_height_error_sq']=cfg['rewards']['base_height_error_sq']
    validate_continuation_config(original,cfg,original['commands'])
    algorithm=yaml.safe_load((source.parent/'params/agent.yaml').read_text())['algorithm']
    if (algorithm['class_name']!='PPO' or algorithm.get('joint_mean_guard_coef',0.)
            or algorithm.get('joint_height_feedback_coef',0.)):
        raise ValueError('Keep ordinary PPO without auxiliary losses')
    pins=dict(parent['source_files'])
    if any(guard.core.sha(Path(path))!=digest for path,digest in pins.items()):
        raise ValueError('Frozen physics, rewards, code or evidence changed during the handoff')
    candidate.mkdir(exist_ok=False)
    shutil.copytree(previous/'replay',candidate/'replay')
    for path in [source,*sorted((source.parent/'params').glob('*.yaml')),
                 previous/'status.json',previous/'experiment_plan.json',previous_cursor,
                 Path(__file__).resolve(),ROOT/'wheel_legged_isaaclab/tests/test_source200_height100.py',
                 *(candidate/'replay'/n for n in ('assessment.json','manifest.json','completed.json','trace.csv'))]:
        pins[str(path)]=guard.core.sha(path)
    write_json(candidate/'experiment_plan.json',dict(parent,source_checkpoint=str(source),
        source_sha256=guard.core.sha(source),source_actual_ppo_updates=actual,source_files=pins,
        previous_output=str(previous),review_interval_actual_PPO_updates=UPDATES,
        original_model200_lineage_verified=True,weights_reinitialized=False,optimizer_reset=False,
        policy_accepted=False))
    write_json(candidate/'status.json',dict(status,state='agent_review_required',training_active=False,
        updated_at=datetime.now().astimezone().isoformat()))
    write_json(base/'handoff_verification.json',dict(previous_output=str(previous),
        checkpoint=str(source),checkpoint_sha256=guard.core.sha(source),actual_ppo_updates=actual,
        optimizer_states=17,optimizer_step=guard.core.optimizer_step(saved),
        next_review_after_actual_PPO_updates=UPDATES,optimizer_reset=False,
        only_reward_variable='base_height_error_sq',policy_accepted=False))
    return candidate


def run(previous_cursor,candidate,base,unit):
    previous_cursor,candidate,base=map(lambda p:Path(p).resolve(),(previous_cursor,candidate,base))
    if base.parent!=ROOT/'mine_model/results':raise ValueError('Use an owned review coordinator')
    base.mkdir(exist_ok=False)
    write_json(base/'cadence.json',dict(actual_PPO_updates_per_review=UPDATES,
        independent_all_cases_replay=True,standing_stability_before_height_before_speed=True,
        only_reward_variable='base_height_error_sq',hardware_enabled=False,shutdown=False))
    write_json(base/'handoff_status.json',dict(state='waiting_for_running_short_trial_saved_review',
        previous_coordinator=str(previous_cursor),next_review_interval=UPDATES,policy_accepted=False))
    with (previous_cursor.parent/'.lock').open('a') as lock:
        while True:
            if (base/'STOP').exists():return
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:time.sleep(5)
        previous=selected_incumbent(json.loads(previous_cursor.read_text()))
    old_updates=guard.UPDATES
    try:
        guard.UPDATES=UPDATES
        guard.run(candidate,previous,base,unit,
            bootstrap_function=lambda *unused:adopt(previous,candidate,base,previous_cursor),
            proposal_function=experiment.height_proposal)
    finally:guard.UPDATES=old_updates


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator',type=Path,required=True)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--coordinator',type=Path,required=True)
    parser.add_argument('--unit',required=True)
    args=parser.parse_args();run(args.previous_coordinator,args.candidate,args.coordinator,args.unit)
