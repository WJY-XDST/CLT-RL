"""Owner-selected model_200: ordinary PPO, one height cost, stability rollback.

The starting policy is selected by the owner, not advertised as an accepted
model. Its missing endpoint drift/height skills stay visible. Measured physics
and the corrected reset are held constant throughout the reward comparison.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil

import torch
import yaml

from auto_train_mine import write_json
import continue_joint_pd_rollback as guard
from joint_pd_priority import priority_summary
from joint_pd_continuation import HEIGHT_DEFAULTS
from summarize_mine_trial import assess
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.joint_reset import nominal_reset_config
from wheel_legged_gym_isaaclab.robot_model import ASSETS,MANIFEST,validate_checkpoint_robot

ROOT=guard.ROOT
SOURCE=guard.core.RUNS/'2026-10-15_standing_seed_recovery_source_snapshot/model_200.pt'
HISTORICAL=guard.core.RUNS/'2026-10-11_18-14-19_joint_pd_scratch_1k_20261011_181337/model_200.pt'
SOURCE_SHA='556ad6b47c108e3cb8522a700c8b0fec143666e3568d4d6de8b4afc6a85349d4'


def starting_config(source,asset):
    result=nominal_reset_config(source)
    result['robot']['spawn']['usd_path']=str(asset)
    result['commands'].update(HEIGHT_DEFAULTS,ranges_height=list(SAFE_BASE_HEIGHT_RANGE_M),
        full_height_range_from_start=True,standing_only_steps=0,yaw_start_steps=0,reverse_ramp_start_steps=0)
    if (not result['commands']['fixed_standing_groups'] or result['commands']['standing_env_fraction']!=.25
            or not result['commands']['grouped_training']):
        raise ValueError('Keep 25% standing environments alongside all motion groups')
    if (source['rewards']['base_height_error_sq']!=0. or source['rewards']['orientation']!=-20.
            or source['rewards'].get('tracking_goal_component_violation',0.)
            or source['rewards'].get('joint_target_leg_length_error_sq',0.)):
        raise ValueError('Use the simple original model_200 reward baseline')
    return result


def height_proposal(cfg,report,used):
    # A real quadratic error supplies feedback outside the narrow height kernel.
    # Do not repeat the earlier simultaneous -500 height / -400 orientation /
    # composite-cost bundle. Only one existing reward coefficient may change.
    current=cfg['rewards']['base_height_error_sq']
    options=[v for v in (-100.,-50.,-250.,-500.) if v!=current]
    if current<0.:options.insert(0,current)
    for seed in range(43,100000):
        for value in options:
            delta={'base_height_error_sq':value}
            signature=(seed,tuple(delta.items()))
            if signature not in used:return seed,delta,signature
    raise RuntimeError('No unused height-cost experiment remains')


def snapshot(candidate):
    candidate.mkdir(exist_ok=False)
    if guard.core.sha(SOURCE)!=SOURCE_SHA or guard.core.sha(HISTORICAL)!=SOURCE_SHA:
        raise ValueError('Owner-selected original model_200 bytes changed')
    saved=torch.load(SOURCE,map_location='cpu',weights_only=True)
    if (not guard.core.tensor_finite(saved) or len(saved['optimizer_state_dict']['state'])!=17
            or guard.core.optimizer_step(saved)!=4020 or guard.core.saved_updates(SOURCE)!=201):
        raise ValueError('Keep exact original weights/std/17 Adam states and their completed-update clock')
    asset,calibration=guard.core.qualified_wheel_mass_asset(ASSETS/MANIFEST['wheel_mass_calibration'])
    cfg=starting_config(validate_checkpoint_robot(SOURCE),asset)
    agent=yaml.safe_load((SOURCE.parent/'params/agent.yaml').read_text())
    if agent['algorithm']['class_name']!='PPO':raise ValueError('Use original ordinary PPO')
    directory=guard.core.RUNS/('2026-10-16_rollback_source200_'+candidate.name.removeprefix('joint_pd_optimization_'))
    directory.mkdir(exist_ok=False);params=directory/'params';params.mkdir()
    checkpoint=directory/'model_200.pt';shutil.copy2(SOURCE,checkpoint)
    for name in ('control_contract.yaml','precision_mode.yaml','curriculum_clock.yaml'):
        path=SOURCE.parent/'params'/name
        if path.exists():shutil.copy2(path,params/name)
    shutil.copy2(SOURCE.parent/'params/env.yaml',params/'source_env_legacy.yaml')
    for name,value in [('env.yaml',cfg),('agent.yaml',agent)]:
        (params/name).write_text(yaml.safe_dump(value,sort_keys=False))
    migration=dict(created_at=datetime.now().astimezone().isoformat(),historical_checkpoint=str(HISTORICAL),
        source_snapshot=str(SOURCE),resume_snapshot=str(checkpoint),checkpoint_sha256=SOURCE_SHA,
        checkpoint_bytes_identical=guard.core.sha(checkpoint)==SOURCE_SHA,
        all_policy_and_17_Adam_states_unchanged=True,restored_Adam_step=4020,
        restored_optimizer_completed_updates=201,historical_loop_label=200,
        retained_reward_dictionary_exactly=True,original_algorithm='PPO',resume_algorithm='PPO',
        auxiliary_gradient_losses_enabled=False,physical_action_mapping_unchanged=True,
        measured_physics_and_nominal_reset_retained=True,calibration=calibration,
        commands_changed_explicitly=dict(full_safe_height=True,all_task_groups_active=True,
            standing_env_fraction=.25,actual_optimizer_clock_retained=True),
        owner_request='回退到200；站立稳定前提下，用高度惩罚控制收敛；避免5550引入的多变量',
        policy_accepted=False,hardware_enabled=False,shutdown=False)
    write_json(directory/'resume_migration.json',migration);write_json(candidate/'resume_migration.json',migration)
    write_json(candidate/'status.json',dict(state='frozen_owner_selected_source200',checkpoint=str(checkpoint),
        checkpoint_sha256=SOURCE_SHA,actual_ppo_updates=201,live_ppo_updates=201,optimizer_step=4020,
        optimizer_states=17,training_active=False,policy_accepted=False,goal_remains_active=True))
    return checkpoint


def owner_bootstrap(candidate,abandoned,base):
    candidate,abandoned=Path(candidate),Path(abandoned)
    status=json.loads((candidate/'status.json').read_text());source=Path(status['checkpoint'])
    if (source.name!='model_200.pt' or guard.core.sha(source)!=SOURCE_SHA
            or guard.core.saved_updates(source)!=201 or guard.core.sha(HISTORICAL)!=SOURCE_SHA):
        raise ValueError('Never substitute model_5550 or a later model for owner-selected model_200')
    if guard.core.sha(ROOT/'mine_model/config/tracking_acceptance.json')!=guard.GATE_SHA:
        raise ValueError('Original acceptance changed')
    replay=candidate/'replay'
    if not (replay/'completed.json').exists():
        guard.core.block.runtime.OUTPUT=base
        guard.core.block.runtime.run_child(guard.core.block.replay_command(source,replay),candidate/'replay.log',guard.old.split.environment())
    report=assess(replay)
    nominal=next(c for c in report['conditions'] if c['name']=='stand')
    if (report['checkpoint_sha256']!=SOURCE_SHA or report['survivors']!=27 or report['resets']
            or any(not t['survived'] or any(t['metrics'][k]>t['limits'][k] for k in
                ('body_p95_deg','speed_mae_m_s','yaw_mae_rad_s')) for t in nominal['trials'])):
        raise ValueError('Owner-selected model_200 did not retain nominal standing and all-case survival with measured physics')
    # Starting an explicitly requested baseline is not a relaxation of any gate.
    # Endpoint drift/yaw failures remain in the report and in every comparison.
    previous=json.loads((abandoned/'status.json').read_text())
    retained=Path(previous['checkpoint'])
    if previous['training_active'] or guard.core.sha(retained)!=previous['checkpoint_sha256']:
        raise ValueError('Preserve the complete stopped branch before changing owners')
    asset,calibration=guard.core.qualified_wheel_mass_asset(ASSETS/MANIFEST['wheel_mass_calibration'])
    watched=[SOURCE,HISTORICAL,source,retained,abandoned/'status.json',
        candidate/'resume_migration.json',source.parent/'resume_migration.json',
        *sorted((source.parent/'params').glob('*.yaml')),
        ROOT/'run_python.sh',ROOT/'mine_model/config/tracking_acceptance.json',
        ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py',ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
        *sorted((ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab').rglob('*.py')),
        *sorted((ROOT/'mine_model/scripts').glob('*.py')),
        ROOT/'wheel_legged_isaaclab/tests/test_source200_height_guard.py',
        *(replay/n for n in ('manifest.json','completed.json','assessment.json','trace.csv'))]
    pins={str(path):guard.core.sha(path) for path in watched};pins.update(calibration['asset_chain_sha256'])
    write_json(candidate/'experiment_plan.json',dict(source_checkpoint=str(source),source_sha256=SOURCE_SHA,
        source_actual_ppo_updates=201,source_Adam_states=17,source_files=pins,
        protected_reference=str(replay/'assessment.json'),owner_selected_checkpoint=True,
        restored_original_reward_profile=True,single_reward_trial_only='base_height_error_sq',
        original_acceptance_sha256=guard.GATE_SHA,retained_5550_branch=str(abandoned),
        endpoint_drift_failures_not_hidden=not priority_summary(report)['stationary_pose_passed'],
        policy_accepted=False,hardware_enabled=False,shutdown=False))
    status.update(state='agent_review_required',updated_at=datetime.now().astimezone().isoformat())
    write_json(candidate/'status.json',status)
    write_json(base/'source200_baseline_verification.json',dict(checkpoint=str(source),checkpoint_sha256=SOURCE_SHA,
        original_weights_and_17_Adam_states_restored=True,**priority_summary(report),
        starting_baseline_only_not_acceptance=True,policy_accepted=False))
    return candidate


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--abandoned',type=Path,required=True)
    parser.add_argument('--coordinator',type=Path,required=True)
    parser.add_argument('--unit',required=True)
    parser.add_argument('--prepare-snapshot-only',action='store_true')
    args=parser.parse_args()
    if args.prepare_snapshot_only:snapshot(args.candidate)
    else:guard.run(args.candidate,args.abandoned,args.coordinator,args.unit,
        bootstrap_function=owner_bootstrap,proposal_function=height_proposal)
