"""Cumulative direct-PD learning, checkpointed replay and explicit diagnosis.

No round/time budget terminates a healthy learning chain. A measured fault or
stagnation saves the current state for code diagnosis; the user's goal remains
active. This controller has no hardware or shutdown path.
"""
import copy
from datetime import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import torch
import yaml

import resume_after_solver_check_20261011 as runtime
from auto_train_mine import write_json
from diagnose_policy_feedback import diagnose_joint_pd
from joint_pd_optimization import (ratios, quality, stop_outcome, prepared_reward_trial,
                                   TASK_GOAL_METRICS, REFERENCE_TRACKING_METRICS)
from summarize_mine_trial import acceptance_criteria, assess, assess_stability
from watch_joint_pd_scratch import RUNS, evaluate, validate_contract
from wheel_legged_gym_isaaclab.config_overrides import strip_retired_config_fields
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M, training_height_bounds
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot
from wheel_legged_gym_isaaclab.asset_integrity import verify_asset_chain

ROOT = runtime.ROOT
HEIGHT_DEFAULTS = dict(height_expansion_from=None, height_expansion_start_steps=0,
                       height_expansion_ramp_steps=48000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(tensor_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensor_finite(v) for v in value)
    return True


def optimizer_step(saved):
    states = saved['optimizer_state_dict']['state']
    steps = {float(s['step']) for s in states.values() if 'step' in s}
    if not states or len(steps) != 1:
        raise ValueError('Missing or inconsistent Adam update steps')
    return steps.pop()


def qualified_wheel_mass_asset(evidence_path):
    """Admit only the versioned owner-confirmed profile and its verified USD chain."""
    from wheel_legged_gym_isaaclab.robot_model import ASSETS, MANIFEST
    evidence_path = Path(evidence_path).resolve()
    expected = (ASSETS / MANIFEST['wheel_mass_calibration']).resolve()
    if evidence_path != expected:
        raise ValueError('Use the current owner-confirmed wheel-mass calibration evidence')
    profile = json.loads(evidence_path.read_text())
    asset = (evidence_path.parent / profile['overlay']).resolve()
    smoke = profile.get('validation', {}).get('physx_loading_smoke', {})
    if (profile.get('authority') != 'owner_confirmation_in_chat'
            or profile.get('wheel_masses_kg') != {'LW_link': .665, 'RW_link': .665}
            or asset != (ASSETS / MANIFEST['training_usd']).resolve()
            or not smoke.get('finite_observations') or not smoke.get('finite_rewards')
            or any(not math.isclose(smoke.get('wheel_masses_kg', {}).get(name, 0.), .665, abs_tol=1.e-6)
                   for name in ('LW_link', 'RW_link'))):
        raise ValueError('Wheel mass calibration lacks matched owner/PhysX evidence')
    hashes, _ = verify_asset_chain(asset)
    return str(asset), dict(evidence=str(evidence_path), evidence_sha256=sha(evidence_path),
                           calibration_id=profile['calibration_id'], asset=str(asset),
                           asset_chain_sha256=hashes, wheel_masses_kg=profile['wheel_masses_kg'],
                           inertia_measured=profile['inertia']['measured'])


def continuation_command(source, cfg, label, updates, commands, stop_file, reward_overrides=None,
                         minimum_wheel_std=None, robot_asset=None):
    if minimum_wheel_std is not None and (
            not math.isfinite(minimum_wheel_std) or not .01 <= minimum_wheel_std <= .05):
        raise ValueError('A diagnosed wheel exploration floor must be finite and within [0.01, 0.05].')
    saved = torch.load(source, map_location='cpu', weights_only=True)
    if (not tensor_finite(saved) or not saved.get('optimizer_state_dict')
            or not saved.get('model_state_dict')):
        raise ValueError('Continuation requires intact finite weights and Adam state')
    start = int(saved['iter'])
    target = start + updates
    argv = [str(ROOT/'run_python.sh'), str(ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py'),
            '--task','WheelLeggedJointPD-Flat-v0','--require_leg_control_mode','implicit_joint_position',
            '--checkpoint_path',str(source),'--target_iteration',str(target),
            '--num_envs','8192','--seed','43','--run_name',label,'--headless','--device','cuda:0',
            '--min_exploration_std','.01','--stop_file',str(stop_file),'agent.save_interval=25']
    if minimum_wheel_std is not None:
        argv.extend(['--min_wheel_exploration_std', str(minimum_wheel_std)])
    # An ordinary continuation retains its saved CAD/mass contract. Moving to
    # the owner-calibrated asset is a separate, explicitly qualified experiment.
    previous_asset = cfg.get('robot', {}).get('spawn', {}).get('usd_path')
    asset = robot_asset if robot_asset is not None else previous_asset
    if asset is not None:
        argv.append('env.robot.spawn.usd_path='+json.dumps(asset))
    if robot_asset is not None and robot_asset != previous_asset:
        argv.append('--allow_physics_contract_change')
    rewards = dict(cfg['rewards'])
    rewards.update({key.split('.',1)[1]:value for key,value in (reward_overrides or {}).items()})
    for group, values in (('commands',commands), ('rewards',rewards)):
        argv.extend('env.'+group+'.'+key+'='+json.dumps(value) for key,value in values.items())
    return argv, target-1, saved


def validate_continuation_config(previous, current, commands, reward_overrides=None, robot_asset=None):
    previous = strip_retired_config_fields(previous)
    current = strip_retired_config_fields(current)
    expected = copy.deepcopy(previous)
    if robot_asset is not None:
        expected['robot']['spawn']['usd_path'] = robot_asset
    expected['commands'] = dict(HEIGHT_DEFAULTS, **commands)
    for name,default in (('base_height_relative_deadband',0.),('wheel_action_saturation',0.),
                         ('wheel_action_saturation_penalty_clip',40.),('tracking_goal_violation',0.),
                         ('tracking_goal_penalty_clip',40.),('tracking_goal_component_violation',0.),
                         ('tracking_goal_component_penalty_clip',200.)):
        if name in current['rewards']:
            expected['rewards'].setdefault(name,default)
    expected['rewards'].update({key.split('.',1)[1]:value for key,value in (reward_overrides or {}).items()})
    for obj in (expected,current):
        obj.pop('log_dir',None)
    if expected != current:
        changed = [key for key in expected if expected[key] != current.get(key)]
        raise ValueError('Unexpected changes outside the height command expansion: '+str(changed))


def stagnant(history):
    recent = history[-3:]
    return (len(recent)==3 and min(recent)>1. and min(recent)>=.95*recent[0])


def tracking_monitoring_scope(report, cfg, iteration, rollout_steps):
    """Monitor every gate metric and include endpoints once training reaches them."""
    if type(iteration) is not int or iteration < 0 or type(rollout_steps) is not int or rollout_steps <= 0:
        raise ValueError('Monitoring requires the saved iteration and positive rollout length')
    commands = SimpleNamespace(**cfg['commands'])
    step = iteration * rollout_steps  # Same curriculum restoration clock as train.py.
    progress = min(max((step - commands.standing_only_steps) / max(commands.motion_ramp_steps, 1), 0.), 1.)
    bounds = training_height_bounds(commands, cfg['rewards']['base_height_target'], step, progress)
    expanded = all(math.isclose(a, b, rel_tol=0., abs_tol=1.e-10)
                   for a, b in zip(bounds, commands.ranges_height))
    scoped = dict(report)
    scoped['conditions'] = [c for c in report['conditions']
                            if expanded or c['name'] not in ('height_low', 'height_high')]
    metric_ratios = ratios(scoped, (*TASK_GOAL_METRICS, *REFERENCE_TRACKING_METRICS))
    return dict(scope='all_conditions' if expanded else 'central_until_height_expanded',
                conditions=[c['name'] for c in scoped['conditions']],
                excluded_conditions=[c['name'] for c in report['conditions'] if c not in scoped['conditions']],
                height_course_complete=expanded, training_height_bounds=list(bounds),
                restored_curriculum_control_steps=step, metric_ratios=metric_ratios,
                worst_ratio=max(metric_ratios.values()), acceptance_unchanged=True)


def stagnant_monitoring_metrics(history):
    """A passing/improving axis must not conceal another persistently failed one."""
    recent = history[-3:]
    if len(recent) != 3 or any((v['scope'], v['conditions']) !=
                              (recent[0]['scope'], recent[0]['conditions']) for v in recent):
        return []
    return [key for key in recent[0]['metric_ratios']
            if stagnant([v['metric_ratios'][key] for v in recent])]


def run(args):
    if (args.fresh or not args.checkpoint
            or args.max_rounds or args.shutdown_after_success
            or not 1<=args.round_iterations<=200 or args.num_envs not in (None,8192)):
        raise ValueError('Cumulative mode preserves weights/Adam, has no round limit or shutdown, and uses 8192 environments')
    prepared = prepared_reward_trial(args.initial_overrides)
    source, output = args.checkpoint.resolve(), args.output.resolve()
    if (not source.is_file() or not source.is_relative_to(RUNS.resolve())
            or output.parent != ROOT/'mine_model/results'
            or not output.name.startswith('joint_pd_optimization_')):
        raise ValueError('Unscoped continuation source/output')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'status.json').exists():
        raise ValueError('Inspect the existing continuation status before reusing an output directory')
    if subprocess.run(['pgrep','-f','^[^ ]*python.*scripts/rsl_rl/train[.]py'],
                      stdout=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('Another training process is active')
    with (output/'.controller.lock').open('a') as lock, (RUNS/'.cumulative_learning.lock').open('a') as global_lock:
        for handle in (lock,global_lock):
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cfg = validate_checkpoint_robot(source)
        if cfg['leg_control_mode']!='implicit_joint_position':
            raise ValueError('Continuation requires the direct motor-angle PD policy')
        commands = dict(HEIGHT_DEFAULTS, **cfg['commands'])
        calibrated_asset, calibration = None, None
        if getattr(args, 'wheel_mass_calibration', None) is not None:
            calibrated_asset, calibration = qualified_wheel_mass_asset(args.wheel_mass_calibration)
            calibration['previous_asset'] = cfg['robot']['spawn']['usd_path']
        if args.height_envelope_evidence:
            evidence_path = args.height_envelope_evidence.resolve()
            evidence = json.loads(evidence_path.read_text())
            guide = json.loads((evidence_path.parent/'guide/assessment.json').read_text())
            if (evidence['source_sha256']!=sha(source) or not guide['passed'] or guide['resets']
                    or not evidence['asset_hashes_match']
                    or tuple(evidence['height_command_range_m'])!=SAFE_BASE_HEIGHT_RANGE_M
                    or evidence['acceptance_sha256']!=sha(ROOT/'mine_model/config/tracking_acceptance.json')):
                raise ValueError('Height expansion has missing or mismatched mechanical/guide evidence')
            old_range = list(commands['ranges_height'])
            initial_saved = torch.load(source,map_location='cpu',weights_only=True)
            commands.update(ranges_height=list(SAFE_BASE_HEIGHT_RANGE_M), height_expansion_from=old_range,
                            height_expansion_start_steps=int(initial_saved['iter'])*48,
                            height_expansion_ramp_steps=48000)
        watched = [ROOT/'mine_model/config/tracking_acceptance.json',
                   *sorted((ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab').rglob('*.py')),
                   ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
                   ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
                   *[ROOT/'mine_model/scripts'/name for name in
                     ('joint_pd_continuation.py','joint_pd_optimization.py','auto_train_mine.py',
                      'resume_after_solver_check_20261011.py','guard_mcu_vmc_training.py',
                      'watch_joint_pd_scratch.py','diagnose_policy_feedback.py','summarize_mine_trial.py',
                      'complete_mine_training.py')]]
        fingerprints = {str(p):sha(p) for p in watched}
        if calibration:
            fingerprints.update(calibration['asset_chain_sha256'])
            fingerprints[calibration['evidence']] = calibration['evidence_sha256']
        state = dict(initial_source=str(source), initial_source_sha256=sha(source),
                     checkpoint=str(source), full_policy_training=True, optimizer_reset=False,
                     num_envs=8192, updates_per_block=args.round_iterations, round_limit=None,
                     goal_remains_active=True, all_errors_under_10_percent=False,
                     acceptance_criteria=acceptance_criteria(), source_files=fingerprints,
                     commands=commands, history=[], shutdown_after_success=False, prepared_reward_trial=prepared,
                     wheel_exploration_std_floor=getattr(args, 'min_wheel_exploration_std', None),
                     wheel_mass_calibration=calibration)
        if state['acceptance_criteria']['relative_error']>.1:
            raise ValueError('Original ten-percent acceptance is required')
        def save(phase, **values):
            state.update(state=phase, training_active=phase=='training',
                         updated_at=datetime.now().astimezone().isoformat(), **values)
            write_json(output/'status.json',state)
            print(json.dumps(dict(state=phase,**values),ensure_ascii=False),flush=True)
        environment = dict(os.environ,PYTHONPATH=str(ROOT/'wheel_legged_isaaclab'),
                           PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1')
        runtime.OUTPUT = output
        def job(command, log, training=False):
            if any(sha(path)!=digest for path,digest in fingerprints.items()):
                raise RuntimeError('Source changed while a block was active; inspect at a checkpoint boundary')
            runtime.run_child(command,log,environment,training=training,reporter=save,
                              cooperative_stop=training)
        def replay(checkpoint, directory, seconds=20, seeds=(83,84,85)):
            command=[str(ROOT/'run_python.sh'),str(ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
                     '--checkpoint',str(checkpoint),'--output',str(directory),'--seconds',str(seconds),
                     '--seeds',*[str(s) for s in seeds],'--record_policy_inputs','--diagnose_contacts',
                     '--headless','--device','cuda:0']
            job(command,directory.with_suffix('.log'))
            report=assess(directory)
            diagnose_joint_pd(directory,directory/'joint_pd_command_diagnosis.json')
            return report
        try:
            cycle=0
            monitoring_history=[]
            rollout_steps=int(yaml.safe_load((source.parent/'params/agent.yaml').read_text())['num_steps_per_env'])
            best=None
            while True:
                if (output/'STOP').exists():
                    phase, details=stop_outcome(output)
                    save(phase,child_pid=None,**details)
                    return
                label=f'joint_pd_cumulative_{output.name}_c{cycle:04d}'
                block=output/f'block_{cycle:04d}'
                block.mkdir()
                command,final,before=continuation_command(source,cfg,label,args.round_iterations,
                                                         commands,output/'STOP',reward_overrides=prepared,
                                                         minimum_wheel_std=getattr(args, 'min_wheel_exploration_std', None),
                                                         robot_asset=calibrated_asset)
                write_json(block/'launch.json',dict(source_checkpoint=str(source),source_sha256=sha(source),
                           command=command,updates=args.round_iterations,optimizer_step_before=optimizer_step(before),
                           reset_optimizer=False,reset_exploration=False,commands=commands,
                           wheel_mass_calibration=calibration))
                save('training_initialization',block=cycle,source_checkpoint=str(source),
                     expected_final_iteration=final,log=str(block/'training.log'),child_pid=None)
                job(command,block/'training.log',training=True)
                runs=list(RUNS.glob('*_'+label))
                if len(runs)!=1:
                    raise RuntimeError('Missing or ambiguous owned cumulative training output')
                checkpoint=runs[0]/f'model_{final}.pt'
                after=torch.load(checkpoint,map_location='cpu',weights_only=True)
                if not tensor_finite(after) or optimizer_step(after)-optimizer_step(before)!=20*args.round_iterations:
                    raise RuntimeError('Actual Adam progress or saved tensors failed verification')
                validate_contract(runs[0],'joint_pd_balance_v1',fresh_weights=False,source_checkpoint=source)
                current=validate_checkpoint_robot(checkpoint)
                validate_continuation_config(cfg,current,commands,reward_overrides=prepared,
                                             robot_asset=calibrated_asset)
                changed=[bool((after['model_state_dict']['actor.6.weight'][i]
                              !=before['model_state_dict']['actor.6.weight'][i]).any()) for i in range(6)]
                if not all(changed):
                    raise RuntimeError('One or more actor output rows did not update')
                write_json(block/'resume_verification.json',dict(optimizer_step_before=optimizer_step(before),
                           optimizer_step_after=optimizer_step(after),actual_updates=args.round_iterations,
                           actor_rows_updated=changed,tensors_finite=True,control_contract_preserved=True,
                           checkpoint_sha256=sha(checkpoint)))
                save('checkpoint_evaluation',checkpoint=str(checkpoint),child_pid=None)
                measured=replay(checkpoint,block/'replay')
                monitoring=tracking_monitoring_scope(measured,current,int(after['iter']),rollout_steps)
                entry=dict(checkpoint=str(checkpoint),assessment=str(block/'replay/assessment.json'),
                           ratios=ratios(measured),survivors=measured['survivors'],resets=measured['resets'],
                           passed=measured['passed'],monitoring=monitoring)
                state['history'].append(entry)
                source,cfg=checkpoint,current  # Always accumulate actual updates, not a new branch from 2400.
                save('block_evaluated',checkpoint=str(source),last_result=entry,child_pid=None)
                if best is None or quality(measured)>quality(best):
                    best=measured
                    save('best_measured_checkpoint_saved',best_checkpoint=str(checkpoint),
                         best_assessment=entry['assessment'],all_errors_under_10_percent=False)
                if measured['passed']:
                    heldout=block/'heldout_30s'; extended=block/'extended_60s'
                    replay(checkpoint,heldout,30,(183,184,185))
                    independent=assess_stability(heldout,20.)
                    if independent['passed']:
                        replay(checkpoint,extended,60,(283,284,285))
                        sustained=assess_stability(extended,40.)
                        if sustained['passed']:
                            save('simulation_accepted_pending_sim2sim_and_randomization',best_checkpoint=str(checkpoint),
                                 best_assessment=entry['assessment'],all_errors_under_10_percent=True,
                                 heldout_assessment=str(heldout/'assessment_stability.json'),
                                 extended_assessment=str(extended/'assessment_stability.json'),child_pid=None)
                            from complete_mine_training import finish_joint_pd
                            finish_joint_pd(output/'status.json',shutdown=False)
                            return
                monitoring_history.append(monitoring)
                stalled=stagnant_monitoring_metrics(monitoring_history)
                if stalled:
                    save('agent_diagnosis_required',reason='three_independent_tracking_milestones_stagnant',
                         stagnant_metrics=stalled,monitoring=monitoring,user_pause_requested=False,
                         goal_remains_active=True,checkpoint=str(source),child_pid=None)
                    return
                cycle+=1
        except BaseException as error:
            runs=sorted(RUNS.glob('*_'+label)) if 'label' in locals() else []
            saved=sorted(runs[0].glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[1])) if len(runs)==1 else []
            preserved=str(saved[-1] if saved else source)
            if (output/'STOP').exists():
                phase,details=stop_outcome(output)
                save(phase,reason_detail=repr(error),checkpoint=preserved,child_pid=None,**details)
            else:
                save('agent_diagnosis_required',reason=repr(error),checkpoint=preserved,
                     user_pause_requested=False,goal_remains_active=True,child_pid=None)
            raise
