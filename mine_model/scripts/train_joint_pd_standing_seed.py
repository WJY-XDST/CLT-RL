"""Resume an independently standing policy, never randomize a replacement.

The legacy source is copied byte-for-byte into an explicitly clock-adopted
snapshot. Adam, every policy parameter and its adaptive LR are retained. Native
training/replay share the existing simulation lock; the original gate is fixed.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path

import torch
import yaml

from auto_train_mine import write_json
from joint_pd_continuation import (
    ROOT, RUNS, continuation_command, optimizer_step, qualified_wheel_mass_asset,
    sha, tensor_finite,
)
from joint_pd_optimization import ratios, TASK_GOAL_METRICS, REFERENCE_TRACKING_METRICS
import resume_after_solver_check_20261011 as runtime
from summarize_mine_trial import assess, assess_stability
from assess_height_leg_command_tracking import assess_directory as assess_leg_commands
from train_joint_pd_standing_first import (
    commands_for_stage, simple_rewards, standing_readiness, verify_update,
)
from watch_joint_pd_scratch import numerical_anomaly
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST, validate_checkpoint_robot
from wheel_legged_gym_isaaclab.training_state import clocked_update_count, load_curriculum_clock
from wheel_legged_gym_isaaclab.joint_reset import (
    CAD_ZERO_RESET, NOMINAL_JOINT_RESET, nominal_reset_config, reset_profile,
)


METRICS = (*TASK_GOAL_METRICS, *REFERENCE_TRACKING_METRICS)
STATIONARY = ('stand', 'height_low', 'height_high')
INITIAL_REQUESTED_REINFORCEMENT = dict(
    base_height_error_sq=-50., tracking_goal_component_violation=-.1)


def reinforced_rewards(original):
    """Historical standing scales plus only the two declared tracking costs."""
    result = simple_rewards(original)
    result.update(INITIAL_REQUESTED_REINFORCEMENT)
    return result


def normal_joint_commands(original):
    """Restore the normal absolute-clock motion curriculum, without a new warmup."""
    return commands_for_stage(original, 0)


def retained_joint_rewards(original, saved_rewards):
    """Keep serialized current defaults as well as every learned-run reward scale."""
    expected = reinforced_rewards(original)
    if any(key not in saved_rewards or saved_rewards[key] != value for key,value in expected.items()):
        raise ValueError('Joint-training correction must retain the current rewards')
    return copy.deepcopy(saved_rewards)


def height_posture_regressions(report, protected):
    """Independent retention screen; these margins NEVER change acceptance.

    Noise-sized slack is 3 mm/5% height MAE and .25 deg/10% body P95.
    A speed improvement cannot compensate for breaching either bound. Body
    P95 must also remain strictly inside the unchanged original limit.
    """
    failures = []
    if ([c['name'] for c in report['conditions']] != [c['name'] for c in protected['conditions']]):
        raise ValueError('Height/posture comparison conditions differ')
    for current, previous in zip(report['conditions'], protected['conditions']):
        if current['height'] != previous['height']:
            raise ValueError('Height/posture comparison commands differ')
        if [t['seed'] for t in current['trials']] != [t['seed'] for t in previous['trials']]:
            raise ValueError('Height/posture comparison reset seeds differ')
        for trial, old in zip(current['trials'], previous['trials']):
            if not trial['survived'] or not trial.get('metrics') or not old.get('metrics'):
                failures.append(dict(condition=current['name'], seed=trial['seed'], reason='first_episode_failure'))
                continue
            for key, absolute_slack in (('height_mae_m', .003), ('body_p95_deg', .25)):
                fraction = .05 if key == 'height_mae_m' else .1
                value, before = trial['metrics'][key], old['metrics'][key]
                allowed = before+max(absolute_slack, fraction*before)
                original_limit_failed = key == 'body_p95_deg' and not value < trial['limits'][key]-1.e-12
                if (not torch.isfinite(torch.tensor(value)).item() or value > allowed+1.e-12
                        or original_limit_failed):
                    failures.append(dict(condition=current['name'], seed=trial['seed'], metric=key,
                        previous=before, measured=value, retention_bound=allowed,
                        original_acceptance_limit=trial['limits'][key]))
    return failures


def update_protected_height_posture(protected, report):
    """Retain the best measured per-trial height/posture, not a combined score."""
    result = copy.deepcopy(protected)
    for previous, current in zip(result['conditions'], report['conditions']):
        for old, trial in zip(previous['trials'], current['trials']):
            for key in ('height_mae_m', 'body_p95_deg'):
                old['metrics'][key] = min(old['metrics'][key], trial['metrics'][key])
    result.update(reference_only_not_an_accepted_policy=True, policy_accepted=False)
    return result


def stationary_task_ready(report):
    """All current heights must meet original body-task limits before motion."""
    standing = standing_readiness(report)
    trials = [t for c in report['conditions'] if c['name'] in STATIONARY for t in c['trials']]
    return bool(standing['ready'] and all(t['metrics']['upright_support']
        and t['metrics']['command_reached'] and all(
            t['metrics'][k] < t['limits'][k]-1.e-12 for k in TASK_GOAL_METRICS)
        for t in trials))


def nominal_standing_intact(report):
    """Preserve the independently established nominal standing, not mere life."""
    cases = [c for c in report['conditions'] if c['name'] == 'stand']
    if len(cases) != 1 or len(cases[0]['trials']) != 3:
        return False
    return all(t['survived'] and t.get('metrics')
        and t['metrics']['upright_support'] and t['metrics']['command_reached']
        and all(t['metrics'][k] < t['limits'][k]-1.e-12 for k in TASK_GOAL_METRICS)
        for t in cases[0]['trials'])


def strict_full_gate(report):
    trials = [t for c in report['conditions'] for t in c['trials']]
    return bool(len(report['conditions']) == 9 and len(trials) == 27
        and all(t['survived'] and t.get('metrics') and t['metrics']['upright_support']
            and t['metrics']['command_reached'] and all(
                t['metrics'][k] < t['limits'][k]-1.e-12 for k in METRICS)
            for t in trials))


def monitoring_ratios(report, motion_complete=False):
    """Full-height stationary scope until motion is trained; never acceptance."""
    scoped = dict(report)
    if not motion_complete:
        scoped['conditions'] = [c for c in report['conditions'] if c['name'] in STATIONARY]
    return ratios(scoped, METRICS)


def stagnant_axes(history):
    """Do not extend a surviving-but-wrong policy indefinitely."""
    recent = history[-3:]
    if len(recent) != 3 or len({x['monitoring_scope'] for x in recent}) != 1:
        return []
    if recent[-1]['additional_ppo_updates'] < 200:
        return []
    return [k for k in METRICS if all(x['monitoring_ratios'][k] > 1. for x in recent)
            and recent[-1]['monitoring_ratios'][k] >= .98*recent[0]['monitoring_ratios'][k]]


def saved_updates(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    clock = load_curriculum_clock(path)
    if clock is None or not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17:
        raise ValueError('Require an intact explicit-clock full-policy/Adam checkpoint')
    return clocked_update_count(saved, clock)


def next_target(actual, initial):
    extra = actual-initial
    if extra < 0:
        raise ValueError('Cannot roll back the adopted source Adam clock')
    return initial+5 if extra < 5 else initial+(extra//100+1)*100


def run(source, historical, output, joint_tracking_revision=None):
    source, historical, output = (Path(p).resolve() for p in (source, historical, output))
    if (not source.is_relative_to(RUNS.resolve()) or not historical.is_relative_to(RUNS.resolve())
            or output.parent != ROOT/'mine_model/results'
            or not output.name.endswith(('_standing_seed_recovery', '_joint_tracking'))):
        raise ValueError('Use owned standing-source and recovery output paths')
    if sha(source) != sha(historical):
        raise ValueError('The clock-adopted snapshot must preserve exact checkpoint bytes')
    legacy_initial = saved_updates(source)
    old = torch.load(source, map_location='cpu', weights_only=True)
    if legacy_initial != 201 or optimizer_step(old) != 4020:
        raise ValueError('Require the identified independently standing model_200, not an arbitrary branch')
    if not bool((old['model_state_dict']['std'] >= .01).all()) or not bool(
            (old['model_state_dict']['std'][[2,5]] >= .03).all()):
        raise ValueError('Exploration floor would change the identified saved source at load')
    output.mkdir(exist_ok=True)
    joint = joint_tracking_revision is not None
    revision = None
    reset_change = None
    resume = source
    if joint:
        revision_path = Path(joint_tracking_revision).resolve()
        if revision_path != output/'owner_joint_tracking_revision.json' or not output.name.endswith('_joint_tracking'):
            raise ValueError('Require the explicit owned joint-training correction record')
        revision = json.loads(revision_path.read_text())
        reset_change = revision.get('reset_pose_change')
        resume = Path(revision['resume_checkpoint']).resolve()
        if (not revision['request_supersedes_static_only_curriculum'] or not revision['rewards_unchanged']
                or revision['normal_motion_curriculum_start_step'] != 0 or revision['motion_ramp_steps'] != 48000
                or not resume.is_relative_to(RUNS.resolve())
                or sha(resume) != revision['resume_checkpoint_sha256']
                or saved_updates(resume) != revision['resume_actual_ppo_updates']
                or Path(revision['source_snapshot']).resolve() != source
                or Path(revision['historical_source']).resolve() != historical
                or sha(ROOT/'mine_model/config/tracking_acceptance.json') != revision['original_acceptance_sha256']):
            raise ValueError('Joint-training correction must preserve the latest saved policy/Adam and normal clock')
        if reset_change:
            qualification_path = Path(reset_change['qualification'])
            qualification_report = json.loads(qualification_path.read_text())
            reset = qualification_report['initial_reset']
            if (reset_change['source_profile'] != CAD_ZERO_RESET
                    or reset_change['target_profile'] != NOMINAL_JOINT_RESET
                    or reset_profile(validate_checkpoint_robot(resume)) != CAD_ZERO_RESET
                    or sha(qualification_path) != reset_change['qualification_sha256']
                    or not qualification_report['passed'] or qualification_report['resets']
                    or reset['profile'] != NOMINAL_JOINT_RESET
                    or reset['active_joint_position_error_max_rad'] != 0.
                    or reset['maximum_joint_anchor_gap_m'] >= 2.e-5):
                raise ValueError('Reset migration lacks intact native complete-chain qualification')
    elif not output.name.endswith('_standing_seed_recovery'):
        raise ValueError('Joint training requires the explicit owner correction record')
    initial = saved_updates(resume)
    with (output/'.controller.lock').open('a') as owned, (RUNS/'.cumulative_learning.lock').open('a') as global_lock:
        for handle in (owned, global_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = validate_checkpoint_robot(source)
        asset, calibration = qualified_wheel_mass_asset(ASSETS/MANIFEST['wheel_mass_calibration'])
        qualification = Path(revision['frozen_source_qualification_directory']) if joint else output/'source_200_transfer'
        manifest = json.loads((qualification/'manifest.json').read_text())
        baseline = json.loads((qualification/'assessment.json').read_text())
        if (manifest['checkpoint_sha256'] != sha(historical) or manifest['weights_updated']
                or manifest['seeds'] != [83,84,85] or manifest['seconds_per_case'] != 20
                or manifest['asset_sha256'] != sha(asset)
                or tuple(c['height'] for c in manifest['cases'][-2:]) != SAFE_BASE_HEIGHT_RANGE_M
                or baseline['survivors'] != 27 or baseline['resets']
                or not nominal_standing_intact(baseline)):
            raise ValueError('Current calibrated/full-height frozen standing qualification failed')
        protected = None
        if joint:
            initial_manifest = json.loads((output/'initial_replay/manifest.json').read_text())
            initial_replay = json.loads((output/'initial_replay/assessment.json').read_text())
            if (initial_manifest['checkpoint_sha256'] != sha(resume) or initial_manifest['weights_updated']
                    or initial_replay['checkpoint_sha256'] != sha(resume)
                    or initial_manifest['asset_sha256'] != sha(asset) or initial_manifest['seeds'] != [83,84,85]
                    or initial_manifest['seconds_per_case'] != 20 or initial_replay['total_trials'] != 27
                    or initial_replay['survivors'] != 27 or initial_replay['resets']
                    or not nominal_standing_intact(initial_replay)):
                raise ValueError('Require a complete saved-boundary frozen joint-training baseline')
            assess_leg_commands(output/'initial_replay')
            if reset_change and initial_manifest.get('joint_reset_profile') != NOMINAL_JOINT_RESET:
                raise ValueError('Frozen transfer replay did not use the qualified consistent reset')
            joint_rewards = retained_joint_rewards(original['rewards'], validate_checkpoint_robot(resume)['rewards'])
            protected = copy.deepcopy(initial_replay)
            if (output/'protected_height_posture.json').exists():
                protected = json.loads((output/'protected_height_posture.json').read_text())
        watched = [*sorted((ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab').rglob('*.py')),
            ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
            ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py', Path(__file__).resolve(),
            ROOT/'mine_model/scripts/train_joint_pd_standing_first.py',
            ROOT/'mine_model/scripts/train_joint_pd_fresh500.py',
            ROOT/'mine_model/scripts/auto_train_mine.py',
            ROOT/'mine_model/scripts/joint_pd_optimization.py',
            ROOT/'mine_model/scripts/guard_mcu_vmc_training.py',
            ROOT/'mine_model/scripts/joint_pd_continuation.py',
            ROOT/'mine_model/scripts/watch_joint_pd_scratch.py',
            ROOT/'mine_model/scripts/summarize_mine_trial.py',
            ROOT/'mine_model/scripts/assess_height_leg_command_tracking.py',
            ROOT/'mine_model/scripts/resume_after_solver_check_20261011.py', ROOT/'run_python.sh',
            ROOT/'mine_model/config/tracking_acceptance.json', historical, source,
            *sorted((source.parent/'params').glob('*.yaml')),
            qualification/'manifest.json', qualification/'assessment.json',
            output/'qualification_overrides.json', Path(calibration['evidence'])]
        if joint:
            watched.remove(output/'qualification_overrides.json')
            watched.extend([revision_path, resume, Path(revision['previous_output'])/'experiment_plan.json',
                *sorted((resume.parent/'params').glob('*.yaml')),
                output/'initial_replay/manifest.json', output/'initial_replay/assessment.json',
                output/'initial_replay/completed.json', output/'initial_replay/trace.csv'])
            if reset_change:
                watched.extend([Path(reset_change['qualification']), output/'reset_pose_overrides.json'])
        handoff = json.loads((ROOT/'mine_model/results/joint_pd_optimization_20261015_165400_standing_first/owner_standing_seed_switch_handoff.json').read_text())
        retained = Path(handoff['retained_fresh_checkpoint'])
        if sha(retained) != handoff['retained_fresh_checkpoint_sha256']:
            raise ValueError('Latest fresh branch must remain intact')
        watched.extend([retained, ROOT/'IsaacLab/logs/rsl_rl/mine_wheel_legged_joint_pd_flat/2026-10-15_14-26-18_joint_pd_cumulative_joint_pd_optimization_20261015_140845_908048_mass_recovery_cumulative_c0001/model_5751_interrupted.pt'])
        fingerprints = {str(p):sha(p) for p in watched}
        fingerprints.update(calibration['asset_chain_sha256'])
        plan = dict(historical_source=str(historical), source_checkpoint=str(source),
            source_sha256=sha(source), source_actual_ppo_updates=legacy_initial, source_Adam_step=4020,
            continuation_start_checkpoint=str(resume), continuation_start_actual_ppo_updates=initial,
            joint_tracking_owner_revision=revision,
            source_snapshot_is_byte_identical=True, weights_std_and_Adam_reset=False,
            legacy_clock_adoption='offset 0, 48 control steps/update, 20 Adam steps/update; source bytes unchanged',
            calibration=calibration, full_height_range_from_first_reset=list(SAFE_BASE_HEIGHT_RANGE_M),
            historical_rewards=simple_rewards(original['rewards']),
            initial_rewards=joint_rewards if joint else reinforced_rewards(original['rewards']),
            initial_reward_changes={} if joint else INITIAL_REQUESTED_REINFORCEMENT,
            historical_reinforcement_retained=INITIAL_REQUESTED_REINFORCEMENT,
            reinforcement='actual height squared cost and mild sum of only original-tolerance excesses; legs, wheels, posture, height and command tracking remain separate measured metrics',
            motion_release=('normal mixed-motion curriculum from the retained absolute clock; no static-only freeze'
                if joint else 'two independent full-height stationary task passes at unchanged original limits'),
            height_posture_retention=('independent per-case/seed best-measured bounds; speed gains cannot offset '
                'height or posture regressions; original acceptance unchanged' if joint else None),
            additional_rewards='none automatically; diagnose measured failures before any later one-term revision',
            measured_stagnation='three comparable independent milestones with any still-failed axis lacking 2 percent improvement; retain latest policy/Adam for agent diagnosis',
            first_resume_smoke_updates=5, independent_replay_every_additional_updates=100,
            full_policy_training=True, num_envs=8192, headless=True,
            old_fresh_1500_course_superseded_by_owner_standing_seed_request=True,
            historical_fresh_and_original_main_retained=True, source_files=fingerprints,
            original_acceptance_sha256=sha(ROOT/'mine_model/config/tracking_acceptance.json'),
            policy_accepted=False, hardware_enabled=False, shutdown=False)
        plan = json.loads(json.dumps(plan))
        path = output/'experiment_plan.json'
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError('Inspect the existing recovery plan before changing pinned settings')
        if not path.exists():
            write_json(path, plan)
        state = dict(state='prepared', actual_ppo_updates=initial, additional_ppo_updates=0,
            live_ppo_updates=initial, history=[], motion_start=0 if joint else None,
            stage='joint_height_posture_speed_tracking' if joint else 'height_leg_precision',
            policy_accepted=False, goal_remains_active=True)
        if (output/'status.json').exists():
            state.update(json.loads((output/'status.json').read_text()))

        def save(phase, **values):
            if values.get('latest_update'):
                values['live_ppo_updates'] = values['latest_update']['iteration']+1
                values['live_additional_ppo_updates'] = values['live_ppo_updates']-initial
            state.update(state=phase, updated_at=datetime.now().astimezone().isoformat(),
                         training_active=phase in ('training_initialization','training'))
            state.update(values)
            write_json(output/'status.json',state)
            print(json.dumps(dict(state=phase,**values),ensure_ascii=False),flush=True)

        def unchanged():
            if any(not Path(p).is_file() or sha(p) != digest for p,digest in fingerprints.items()):
                raise ValueError('Pinned source, calibrated physics, gate or preserved checkpoints changed')
            if (output/'STOP').exists():
                raise RuntimeError('cooperative_stop_marker_present')

        environment = dict(os.environ, PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
            PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        runtime.OUTPUT, runtime.anomaly = output, numerical_anomaly
        prefix = 'joint_pd_standing_seed_'+output.name
        actual, checkpoint = initial, resume
        try:
            candidates = [(saved_updates(p),p) for p in RUNS.glob('*_'+prefix+'_*/model_*.pt')]
            if candidates:
                actual, checkpoint = max(candidates, key=lambda v:v[0])
            while True:
                unchanged()
                measured = next((h for h in state['history'] if h['actual_ppo_updates'] == actual),None)
                extra = actual-initial
                if actual == initial or (extra != 5 and extra % 100) or measured is not None:
                    target = next_target(actual, initial)
                    block = output/f'updates_{target-initial:04d}'
                    block.mkdir(exist_ok=True)
                    current = validate_checkpoint_robot(checkpoint)
                    cfg = copy.deepcopy(current)
                    cfg['robot']['spawn']['usd_path'] = asset
                    cfg['commands'] = normal_joint_commands(original['commands']) if joint else commands_for_stage(original['commands'],state['motion_start'])
                    cfg['rewards'] = copy.deepcopy(joint_rewards) if joint else reinforced_rewards(original['rewards'])
                    if reset_change:
                        cfg = nominal_reset_config(cfg)
                    label = prefix+'_u'+str(actual)+'_'+datetime.now().strftime('%H%M%S_%f')
                    command,final,_ = continuation_command(checkpoint,current,label,target-actual,
                        cfg['commands'],output/'STOP',minimum_wheel_std=.03,robot_asset=asset,
                        reward_overrides={'rewards.'+k:v for k,v in cfg['rewards'].items()},
                        joint_mean_guard_coef=0.,joint_height_feedback_coef=0.,
                        joint_reset_profile=NOMINAL_JOINT_RESET if reset_change else None,
                        allow_reset_pose_change=bool(reset_change and reset_profile(current) != NOMINAL_JOINT_RESET))
                    if final != target-1 or any(flag in command for flag in (
                            '--reset_optimizer','--reset_exploration_std','--save_initial_checkpoint')):
                        raise ValueError('Continuation must retain the explicit full Adam/policy clock')
                    write_json(block/'launch.json',dict(command=command,source_checkpoint=str(checkpoint),
                        source_sha256=sha(checkpoint),actual_updates_before=actual,
                        target_actual_ppo_updates=target,target_additional_ppo_updates=target-initial,
                        weights_std_and_Adam_reset=False,calibrated_asset=asset))
                    save('training_initialization',actual_ppo_updates=actual,additional_ppo_updates=extra,
                        checkpoint=str(checkpoint),target_actual_ppo_updates=target,
                        target_additional_ppo_updates=target-initial,child_pid=None,reason=None)
                    runtime.run_child(command,block/'training.log',environment,training=True,
                        reporter=save,cooperative_stop=True)
                    unchanged()
                    runs=list(RUNS.glob('*_'+label))
                    if len(runs)!=1:
                        raise ValueError('Owned native continuation missing or ambiguous')
                    result=runs[0]/f'model_{final}.pt'
                    proof=verify_update(checkpoint,result,target,cfg,historical)
                    start=yaml.safe_load((runs[0]/'params/curriculum_start.yaml').read_text())
                    if (tuple(start['height_bounds_m']) != SAFE_BASE_HEIGHT_RANGE_M or any(
                            abs(start[k]-value)>1.e-6 for k,value in zip((
                            'sampled_command_height_min_m','sampled_command_height_max_m'),SAFE_BASE_HEIGHT_RANGE_M))):
                        raise ValueError('Native first reset failed the requested full height range')
                    if state['motion_start'] is None and any(start[k] != 0. for k in (
                            'motion_progress','sampled_command_speed_min_m_s','sampled_command_speed_max_m_s',
                            'sampled_command_yaw_min_rad_s','sampled_command_yaw_max_rad_s')):
                        raise ValueError('Motion enabled before actual stationary task verification')
                    if joint and (start['motion_progress'] <= 0. or not (
                            start['sampled_command_yaw_min_rad_s'] < 0. < start['sampled_command_yaw_max_rad_s'])):
                        raise ValueError('Owner-requested normal mixed-motion curriculum is still frozen')
                    proof.update(additional_ppo_updates=target-initial,first_reset=start,
                        retained_original_source_sha256=sha(historical),all_requested_source_states_retained=True)
                    write_json(block/'resume_verification.json',proof)
                    actual,checkpoint=target,result
                extra=actual-initial
                block=output/f'updates_{extra:04d}'
                save('independent_replay',actual_ppo_updates=actual,additional_ppo_updates=extra,
                    checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),child_pid=None)
                replay=block/'replay'
                if not (replay/'completed.json').exists():
                    command=[str(ROOT/'run_python.sh'),str(ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
                        '--checkpoint',str(checkpoint),'--output',str(replay),'--seconds','20',
                        '--seeds','83','84','85','--record_policy_inputs','--diagnose_contacts','--headless','--device','cuda:0']
                    runtime.run_child(command,block/'replay.log',environment,reporter=save)
                report=assess(replay)
                if report['checkpoint_sha256'] != sha(checkpoint):
                    raise ValueError('Replay does not belong to the latest saved policy')
                ready=standing_readiness(report)
                complete=joint or (state['motion_start'] is not None and actual*48-state['motion_start'] >= 48000)
                retention_failures=height_posture_regressions(report,protected) if joint else []
                leg_tracking=assess_leg_commands(replay) if joint else None
                if leg_tracking is not None:
                    write_json(replay/'height_leg_command_assessment.json',leg_tracking)
                entry=dict(actual_ppo_updates=actual,additional_ppo_updates=extra,checkpoint=str(checkpoint),
                    standing=ready,stationary_task_ready=stationary_task_ready(report),
                    nominal_standing_intact=nominal_standing_intact(report),
                    monitoring_scope='all_original_conditions' if complete else 'all_three_full_height_stationary_conditions',
                    monitoring_ratios=monitoring_ratios(report,complete),
                    full_original_metric_ratios=ratios(report,METRICS),
                    survivors=report['survivors'],resets=report['resets'],full_original_gate_passed=report['passed'],
                    strict_full_original_goal_passed=strict_full_gate(report))
                if joint:
                    entry.update(height_posture_retention_failures=retention_failures,
                        actual_leg_static_command_tracking_passed=leg_tracking['actual_leg_static_command_tracking_passed'],
                        height_posture_passed=leg_tracking['height_posture_passed'])
                state['history']=[h for h in state['history'] if h['actual_ppo_updates']!=actual]+[entry]
                write_json(block/'comparison.json',entry)
                save('milestone_measured',actual_ppo_updates=actual,additional_ppo_updates=extra,latest_measurement=entry)
                if report['survivors'] != report['total_trials'] or not entry['nominal_standing_intact']:
                    raise RuntimeError('standing_or_survival_regressed_preserve_latest_Adam_and_diagnose')
                if retention_failures:
                    raise RuntimeError('speed_gain_cannot_trade_away_height_or_posture_preserve_latest_and_diagnose')
                if joint:
                    protected=update_protected_height_posture(protected,report)
                    write_json(output/'protected_height_posture.json',protected)
                previous=state['history'][:-1][-2:]
                if not joint and len(previous)==2 and all(h['standing']['ready'] for h in previous) and not ready['ready']:
                    raise RuntimeError('previously_confirmed_full_height_standing_lost_preserve_latest_and_diagnose')
                recent=state['history'][-2:]
                if len(recent)==2 and all(h['standing']['ready'] for h in recent) and not (output/'standing_confirmed.json').exists():
                    write_json(output/'standing_confirmed.json',dict(event='standing_confirmed_notify_user',
                        actual_ppo_updates=actual,additional_ppo_updates=extra,checkpoint=str(checkpoint),
                        checkpoint_sha256=sha(checkpoint),consecutive_additional_updates=[h['additional_ppo_updates'] for h in recent],
                        original_limits_unchanged=True,full_original_gate_passed=report['passed'],policy_accepted=False))
                if strict_full_gate(report) and (not joint or leg_tracking['actual_leg_static_command_tracking_passed']):
                    passed=True
                    for name,seconds,seeds,steady in (('heldout',30,(183,184,185),20),('sustained',60,(283,284,285),40)):
                        unchanged()
                        directory=block/name
                        if not (directory/'completed.json').exists():
                            command=[str(ROOT/'run_python.sh'),str(ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
                                '--checkpoint',str(checkpoint),'--output',str(directory),'--seconds',str(seconds),
                                '--seeds',*map(str,seeds),'--record_policy_inputs','--diagnose_contacts','--headless','--device','cuda:0']
                            runtime.run_child(command,block/(name+'.log'),environment,reporter=save)
                        passed=bool(assess_stability(directory,steady)['passed']) and passed
                    if passed:
                        save('full_gate_and_sustained_verified_requires_final_package_audit',
                            training_active=False,independent_sustained_verified=True,policy_accepted=False)
                        return
                stalled=stagnant_axes(state['history'])
                if stalled:
                    raise RuntimeError('measured_tracking_plateau_requires_one_term_diagnosis:'+','.join(stalled))
                if not joint and state['motion_start'] is None and len(recent)==2 and all(h['stationary_task_ready'] for h in recent):
                    state.update(motion_start=actual*48,stage='motion_ramp')
                    save('stationary_task_verified_motion_released_without_new_rewards')
        except BaseException as error:
            owned_checkpoints=[(saved_updates(p),p) for p in RUNS.glob('*_'+prefix+'_*/model_*.pt')]
            if owned_checkpoints:
                actual,checkpoint=max(owned_checkpoints,key=lambda v:v[0])
            save('agent_diagnosis_required',reason=repr(error),actual_ppo_updates=actual,
                additional_ppo_updates=actual-initial,checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),training_active=False,
                all_weights_std_Adam_and_historical_branches_retained=True)
            raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--historical',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--joint-tracking-revision',type=Path,
        help='Explicit saved-boundary owner correction: normal mixed tasks with independent height/posture retention')
    args=parser.parse_args()
    run(args.source,args.historical,args.output,args.joint_tracking_revision)
