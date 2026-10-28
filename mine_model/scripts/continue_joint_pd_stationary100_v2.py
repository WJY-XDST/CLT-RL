"""Reviewed stationary-first PPO; use native command archive fields and retain completed handoff reviews.

There is no total-update stopping point. The original canonical acceptance
is always reported separately from training-stage decisions and probes.
"""
import argparse
import copy
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from auto_train_mine import write_json
import continue_joint_pd_repair100 as native
from joint_pd_priority import STATIC_CASES
import joint_pd_stationary_stage as stage
from summarize_mine_trial import assess, assess_stability
from joint_pd_continuation import validate_continuation_config
from joint_pd_rollback import quality as full_quality

ROOT, core, observed = native.ROOT, native.core, native.observed
SCALES = (0., .125, .25, .5, .75, 1.)


def environment():
    return dict(os.environ, PYTHONPATH=str(ROOT/'wheel_legged_isaaclab')+':'+str(ROOT/'mine_model/scripts'),
        PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')


def verify_prior_pins(previous, migration):
    old = json.loads((Path(previous)/'experiment_plan.json').read_text())
    revisions = {row['path']: row for row in migration['revisions']}
    for name, digest in old['source_files'].items():
        current = core.sha(Path(name))
        if current != digest:
            row = revisions.get(name)
            if (not row or row['before_sha256'] != digest or row['after_sha256'] != current
                    or core.sha(Path(row['archived_before'])) != digest):
                raise ValueError('Unreviewed source change: '+name)


def prepare(previous, output, delta, scale, migration_path):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    migration = json.loads(Path(migration_path).read_text())
    verify_prior_pins(previous, migration)
    status = json.loads((previous/'status.json').read_text())
    if status['training_active'] or status['state'] not in ('agent_review_required', 'agent_diagnosis_required'):
        raise ValueError('Keep a completed independently reviewed checkpoint')
    source = Path(status['checkpoint'])
    saved = torch.load(source, map_location='cpu', weights_only=True)
    actual = core.saved_updates(source)
    if (not core.tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
            or core.optimizer_step(saved) != actual*20 or core.sha(source) != status['checkpoint_sha256']):
        raise ValueError('Retain the exact complete finite model, std and 17 Adam states')
    report = observed.drift.completed_report(previous)
    if not stage.summary(report)['stationary_pose_passed']:
        raise ValueError('Do not use a source with lost stationary posture/drift')
    expected = copy.deepcopy(core.validate_checkpoint_robot(source))
    from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
    if (expected['leg_control_mode'] != 'implicit_joint_position'
            or expected['joint_reset_profile'] != 'joint_target_nominal_v1'
            or tuple(expected['commands']['ranges_height']) != SAFE_BASE_HEIGHT_RANGE_M
            or not expected['commands']['full_height_range_from_start'] or scale not in SCALES):
        raise ValueError('Keep direct joint PD, full safe heights and a reviewed motion scale')
    if len(delta) > 1 or any(key not in ('orientation', 'standing_velocity', 'yaw_rate_error_sq',
                                        'base_height_error_sq', 'lin_vel_error_sq') for key in delta):
        raise ValueError('At most one diagnosed existing reward scale per 100-update block')
    allowed=dict(orientation=(-40.,-80.,-160.,-320.,-400.), standing_velocity=(-25.,-50.,-100.,-200.),
        yaw_rate_error_sq=(-2.,-4.,-8.,-16.), base_height_error_sq=(-250.,-500.,-1000.),
        lin_vel_error_sq=(-4.,-8.,-16.,-32.))
    if any(type(value) not in (int,float) or value not in allowed[key]
           or not value<expected['rewards'][key]<=0. for key,value in delta.items()):
        raise ValueError('Use a bounded strengthening of one diagnosed existing cost')
    commands_before = copy.deepcopy(expected['commands'])
    expected['commands']['training_motion_scale'] = scale
    changes = {key: [expected['rewards'][key], value] for key, value in delta.items()}
    expected['rewards'].update(delta)
    expected['rewards'].setdefault('tracking_task_quality_gate_floor', 1.)
    validate_continuation_config(core.validate_checkpoint_robot(source), expected, expected['commands'],
        reward_overrides={'rewards.'+key:value for key,value in delta.items()})
    agent = yaml.safe_load((source.parent/'params/agent.yaml').read_text())
    if agent['algorithm']['class_name'] != 'PPO':
        raise ValueError('Keep ordinary coupled PPO')
    coefficient = agent['algorithm']['entropy_coef']
    asset, calibration = core.qualified_wheel_mass_asset(native.entropy.observed.drift.handoff.experiment.ASSETS/
        native.entropy.observed.drift.handoff.experiment.MANIFEST['wheel_mass_calibration'])
    if expected['robot']['spawn']['usd_path'] != asset:
        raise ValueError('Keep measured-mass physics')
    if core.sha(ROOT/'mine_model/config/tracking_acceptance.json') != native.guard.GATE_SHA:
        raise ValueError('Never change original acceptance')
    watched = [*sorted((ROOT/'wheel_legged_isaaclab/wheel_legged_gym_isaaclab').rglob('*.py')),
        ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py',
        ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py', ROOT/'run_python.sh',
        ROOT/'mine_model/config/tracking_acceptance.json', Path(__file__).resolve(),
        ROOT/'mine_model/scripts/joint_pd_stationary_stage.py',
        ROOT/'wheel_legged_isaaclab/tests/test_stationary_motion_curriculum.py',
        Path(migration_path), source, *sorted((source.parent/'params').glob('*.yaml')),
        previous/'status.json', previous/'experiment_plan.json',
        *(previous/'replay'/name for name in ('assessment.json', 'manifest.json', 'completed.json', 'trace.csv')),
        observed.ENTRY, observed.OBSERVER, Path(observed.__file__), Path(core.__file__),
        ROOT/'mine_model/scripts/joint_pd_continuation.py',
        ROOT/'mine_model/scripts/train_joint_pd_reward_block.py',
        ROOT/'mine_model/scripts/train_joint_pd_standing_first.py']
    pins = {str(path): core.sha(path) for path in watched}
    pins.update(calibration['asset_chain_sha256'])
    key, value = next(iter(delta.items())) if delta else ('base_height_error_sq', expected['rewards']['base_height_error_sq'])
    output.mkdir(exist_ok=False)
    plan = dict(previous_output=str(previous), source_checkpoint=str(source), source_sha256=core.sha(source),
        source_replay=str(previous/'replay'), protected_reference=str(previous/'replay/assessment.json'),
        source_actual_ppo_updates=actual, source_Adam_step=core.optimizer_step(saved), source_Adam_states=17,
        target_actual_ppo_updates=actual+100, bounded_training_updates=100,
        run_name='stationary_reviewed_'+output.name+'_u'+str(actual), source_files=pins,
        reward_key=key, reward_value=value, only_reward_change=changes,
        entropy_coefficient=coefficient, previous_entropy_coefficient=coefficient,
        commands_before=commands_before, training_motion_scale=scale,
        only_command_change={'training_motion_scale': [commands_before.get('training_motion_scale', 1.), scale]},
        only_training_change='Review-controlled motion amplitude; one existing reward only when diagnosed after a completed block',
        native_num_envs=8192, full_height_range=list(SAFE_BASE_HEIGHT_RANGE_M),
        ordinary_full_policy_PPO=True, auxiliary_losses_enabled=False, optimizer_reset=False,
        learned_std_reset=False, weights_std_Adam_and_adaptive_LR_reset=False,
        original_acceptance_sha256=native.guard.GATE_SHA, original_acceptance_unchanged=True,
        no_total_training_update_limit=True, policy_accepted=False, hardware_enabled=False, shutdown=False)
    write_json(output/'experiment_plan.json', plan)
    return plan, expected


def train(previous, output, delta, scale, migration_path, base):
    coefficient = None
    old_command = core.block.continuation_command
    def prepared(*unused):
        nonlocal coefficient
        plan, expected = prepare(previous, output, delta, scale, migration_path)
        coefficient = plan['entropy_coefficient']
        observed.record_resource_wait(output, plan)
        observed.drift.wait_memory(base)
        return plan, expected
    def command(*args, **kwargs):
        return native.entropy.observed_command(*args, coefficient=coefficient, seed=43, **kwargs)
    try:
        core.block.continuation_command = command
        native.guard.NATIVE(SimpleNamespace(previous=previous, output=output, reward_key='unused',
            value=0., updates=100, prepare_only=False), prepare_function=prepared)
        observed.finalize_observations(output)
        status = json.loads((output/'status.json').read_text())
        source = Path(status['checkpoint'])
        start = yaml.safe_load((source.parent/'params/curriculum_start.yaml').read_text())
        if scale == 0. and any(start[key] != 0. for key in (
                'sampled_command_speed_min_m_s', 'sampled_command_speed_max_m_s',
                'sampled_command_yaw_min_rad_s', 'sampled_command_yaw_max_rad_s')):
            raise ValueError('Stationary first reset contains nonzero motion commands')
        for path in (output/'native_learning_signal').glob('rollout_*.npz'):
            verify_rollout_motion(path, scale)
    finally:
        core.block.continuation_command = old_command


def verify_rollout_motion(path, scale):
    with np.load(path, allow_pickle=False) as arrays:
        commands = arrays['native_commands']
        if commands.shape != (48,32,3) or not np.isfinite(commands).all():
            raise ValueError('Invalid native command observation archive')
        if scale == 0. and np.count_nonzero(commands[..., :2]):
            raise ValueError('Stationary PPO data contains a nonzero speed or turn command')


def sustained_stationary(output, base, scale=0.):
    checkpoint = Path(json.loads((output/'status.json').read_text())['checkpoint'])
    results = []
    for label, seconds, seeds, window in [('stationary_heldout',30,(183,184,185),20),
                                           ('stationary_sustained',60,(283,284,285),40)]:
        directory = output/label
        command = core.block.replay_command(checkpoint, directory)
        command[command.index('--seconds')+1] = str(seconds)
        index = command.index('--seeds'); command[index+1:index+4] = list(map(str, seeds))
        core.block.runtime.OUTPUT = base
        core.block.runtime.run_child(command, output/(label+'.log'), environment())
        report = assess_stability(directory, window)
        if report['checkpoint_sha256'] != core.sha(checkpoint):
            raise ValueError('Sustained standing report used a different checkpoint')
        results.append(dict(label=label, **stage.summary(report)))
    passed = all(row['stationary_all_original_metrics_passed'] for row in results)
    if scale > 0.:
        for label, seconds, seeds, window in [('motion_heldout',30,(183,184,185),20),
                                              ('motion_sustained',60,(283,284,285),40)]:
            directory=output/label
            command=core.block.replay_command(checkpoint,directory)+['--command-scale',str(scale)]
            command[command.index('--seconds')+1]=str(seconds)
            index=command.index('--seeds');command[index+1:index+4]=list(map(str,seeds))
            core.block.runtime.OUTPUT=base
            core.block.runtime.run_child(command,output/(label+'.log'),environment())
            report=assess_stability(directory,window)
            if report['checkpoint_sha256'] != core.sha(checkpoint):
                raise ValueError('Motion-stage sustained checkpoint changed')
            passed=bool(report['passed']) and passed
            results.append(dict(label=label,command_scale=scale,passed=bool(report['passed']),
                                diagnostic_only=scale<1.))
    return dict(passed=passed, results=results, original_full_gate_acceptance=scale==1. and passed)


def motion_replay(output, directory, scale, base):
    checkpoint=Path(json.loads((output/'status.json').read_text())['checkpoint'])
    command=core.block.replay_command(checkpoint,directory)+['--command-scale',str(scale)]
    core.block.runtime.OUTPUT=base
    core.block.runtime.run_child(command,directory.with_suffix('.log'),environment())
    report=assess(directory)
    if report['checkpoint_sha256'] != core.sha(checkpoint):
        raise ValueError('Motion-stage probe checkpoint changed')
    write_json(directory/'stage_scope.json',dict(command_scale=scale,
        full_height_stationary_gate_separately_required=True, stage_probe_only=scale<1.,
        final_canonical_gate_unchanged=True, policy_accepted=False))
    return report


def motion_reward(cfg, evidence, used):
    rows=[(case,trial) for case in evidence['conditions'] for trial in case['trials']]
    failed=lambda key:any(not t['survived'] or t['metrics'][key]>=t['limits'][key]-1.e-12 for _,t in rows)
    if failed('body_p95_deg'):
        key,levels='orientation',(40.,80.,160.,320.,400.)
    elif failed('speed_mae_m_s'):
        key,levels='lin_vel_error_sq',(4.,8.,16.,32.)
    elif failed('yaw_mae_rad_s'):
        key,levels='yaw_rate_error_sq',(2.,4.,8.,16.)
    elif () not in used:
        return {}
    else:
        raise RuntimeError('Motion-stage reference failure needs diagnosis before an identical trial')
    for magnitude in levels:
        delta={key:-magnitude}
        if -magnitude<cfg['rewards'][key] and tuple(delta.items()) not in used:
            return delta
    raise RuntimeError('Inspect measured motion-stage plateau before further reward changes')


def run(previous_cursor, base, unit, migration_path):
    previous_cursor, base, migration_path = [Path(p).resolve() for p in (previous_cursor, base, migration_path)]
    base.mkdir(exist_ok=True)
    cursor = base/'coordinator.json'
    with (base/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cursor.exists():
            raise ValueError('Inspect existing progress before restarting an owner')
        prior = json.loads(previous_cursor.read_text())
        if prior['state'] != 'explicit_stop_respected':
            raise ValueError('Previous native owner must complete its checkpoint and full replay first')
        options = {Path(prior['incumbent']), Path(prior['history'][-1]['output'])}
        eligible = [p for p in options if stage.summary(observed.drift.completed_report(p))['stationary_pose_passed']]
        if not eligible:
            raise ValueError('No independently stationary-safe retained source')
        incumbent = min(eligible, key=lambda p: stage.quality(observed.drift.completed_report(p)))
        history = list(prior.get('stationary_history', []))
        attempts, scale, streak = {}, 0., 0
        work = 100*len(history)
        for row in history:
            plan=json.loads((Path(row['output'])/'experiment_plan.json').read_text())
            signature=tuple((key,pair[1]) for key,pair in plan['only_reward_change'].items())
            attempts.setdefault((plan['source_sha256'],plan['training_motion_scale']),set()).add(signature)
        source_motion=None
        def save(state, **values):
            write_json(cursor, dict(state=state, unit=unit, updated_at=datetime.now().astimezone().isoformat(),
                incumbent=str(incumbent), training_motion_scale=scale, history=history,
                actual_updates_executed_this_controller=work, no_total_training_update_limit=True,
                original_acceptance_unchanged=True, goal_remains_active=True, policy_accepted=False, **values))
        save('preparing_stationary_training')
        while not (base/'STOP').exists():
            report = observed.drift.completed_report(incumbent)
            source = Path(json.loads((incumbent/'status.json').read_text())['checkpoint'])
            cfg = core.validate_checkpoint_robot(source)
            used = attempts.setdefault((core.sha(source), scale), set())
            evidence = observed.drift.completed_report(Path(history[-1]['output'])) if history else report
            if scale>0. and source_motion is None:
                source_motion=motion_replay(incumbent,base/('stage_'+str(scale)+'_baseline'),scale,base)
            if scale>0. and stage.summary(evidence)['stationary_all_original_metrics_passed']:
                last=history[-1] if history and history[-1].get('training_motion_scale')==scale else None
                delta={} if last is None or last['promote'] else motion_reward(cfg,last['motion_stage_report'],used)
            else:
                delta = stage.next_reward(cfg, evidence, history, used)
            signature = tuple(delta.items())
            if signature in used:
                raise RuntimeError('Refuse an identical failed source/reward/seed trial; inspect evidence')
            used.add(signature)
            output = ROOT/'mine_model/results'/('joint_pd_optimization_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_stationary100')
            save('training_and_review', current_output=str(output), source_checkpoint=str(source), reward_trial=delta)
            pointer = dict(unit=unit, directory=str(output), coordinator=str(cursor), source_checkpoint=str(source),
                controller_kind='stationary_first_reviewed_100', training_motion_scale=scale,
                optimizer_reset=False, weights_reinitialized=False, no_total_training_update_limit=True,
                policy_accepted=False, goal_remains_active=True)
            for name in ('joint_pd_active_joint_tracking.json', 'joint_pd_active_standing_seed.json'):
                write_json(ROOT/'mine_model/results'/name, pointer)
            try:
                train(incumbent, output, delta, scale, migration_path, base)
            except BaseException as error:
                save('agent_diagnosis_required', current_output=str(output), reason=repr(error), all_checkpoints_retained=True)
                raise
            work += 100
            after = observed.drift.completed_report(output)
            decision = stage.review(after, report)
            motion=None
            if scale>0.:
                motion=motion_replay(output,output/'motion_stage_replay',scale,base)
                safe=(decision['candidate_stationary']['stationary_all_original_metrics_passed']
                      and not decision['stationary_stability_regressions'])
                decision['promote']=bool(safe and full_quality(motion)<full_quality(source_motion))
                decision.update(motion_stage_report=motion,
                    motion_stage_source_quality=full_quality(source_motion),
                    motion_stage_candidate_quality=full_quality(motion))
            decision['training_motion_scale']=scale
            write_json(output/'stationary_review.json', decision)
            history.append(dict(output=str(output), source=str(incumbent), **decision))
            if decision['promote']:
                incumbent = output
                if motion is not None:source_motion=motion
            else:
                write_json(output/'checkpoint_rejected_but_preserved.json', dict(
                    restore_checkpoint=str(source), restore_complete_optimizer_from_same_checkpoint=True,
                    rejected_checkpoint=json.loads((output/'status.json').read_text())['checkpoint'], **decision))
            ready = decision['candidate_stationary']['stationary_all_original_metrics_passed']
            if scale>0.:ready=bool(ready and motion['passed'])
            streak = streak+1 if ready and not decision['stationary_stability_regressions'] else 0
            if streak >= 2:
                verified = sustained_stationary(incumbent, base, scale)
                write_json(output/'stationary_release_verification.json', verified)
                if verified['passed']:
                    write_json(base/('verified_motion_scale_'+str(scale)+'.json'),dict(
                        checkpoint=json.loads((incumbent/'status.json').read_text())['checkpoint'],
                        verified=verified, training_motion_scale=scale, policy_accepted=False))
                    if scale<1.:
                        scale=SCALES[SCALES.index(scale)+1]
                        source_motion=None
                    else:
                        write_json(base/'full_scope_verification_requires_final_audit.json',dict(
                            checkpoint=json.loads((incumbent/'status.json').read_text())['checkpoint'],
                            canonical_report=str(incumbent/'replay/assessment.json'),
                            verification=verified,policy_accepted=False,continue_reviewed_training=True))
                    streak=0
            save('block_review_complete', last_output=str(output), last_stationary_review=decision)
        save('explicit_stop_respected')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-coordinator', type=Path, required=True)
    parser.add_argument('--coordinator', type=Path, required=True)
    parser.add_argument('--environment-revision', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    args = parser.parse_args()
    run(args.previous_coordinator, args.coordinator, args.unit, args.environment_revision)
