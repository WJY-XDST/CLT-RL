"""Two unchanged native PPO updates with observation-only pipeline auditing."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import runpy
import sys

import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, continuation_command, optimizer_step, sha, tensor_finite
from joint_pd_training_observer import TrainingObserver
from train_joint_pd_posture_repair import GATE_SHA
from train_joint_pd_standing_first import verify_update
from train_joint_pd_standing_seed import saved_updates
from wheel_legged_gym_isaaclab.joint_height_feedback import model_from_config
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot


def run_original_training(arguments):
    script = ROOT/'wheel_legged_isaaclab/scripts/rsl_rl/train.py'
    previous_path, previous_argv = list(sys.path), sys.argv
    sys.path.insert(0,str(script.parent))
    sys.argv = [str(script),*arguments]
    try:
        return runpy.run_path(str(script),run_name='__main__')
    finally:
        sys.path[:] = previous_path
        sys.argv = previous_argv


def prepare(previous, output):
    previous, output = Path(previous).resolve(), Path(output).resolve()
    if (previous.parent != ROOT/'mine_model/results' or output.parent != previous.parent
            or previous == output or not output.name.startswith('joint_pd_optimization_')):
        raise ValueError('Use a separate owned diagnostic output')
    state = json.loads((previous/'status.json').read_text())
    old_plan = json.loads((previous/'experiment_plan.json').read_text())
    source = Path(state['checkpoint']).resolve()
    review = previous/'learning_review_after_height_cost_revision.json'
    evidence = json.loads(review.read_text())
    if (state['state'] != 'agent_review_required' or state['training_active']
            or not source.is_relative_to(RUNS.resolve()) or sha(source) != state['checkpoint_sha256']
            or evidence['checkpoint_sha256'] != sha(source)):
        raise ValueError('Audit only the latest reviewed saved policy, not an older branch')
    saved = torch.load(source,map_location='cpu',weights_only=True)
    initial = saved_updates(source)
    if (not tensor_finite(saved) or optimizer_step(saved) != initial*20
            or len(saved['optimizer_state_dict']['state']) != 17):
        raise ValueError('Require the complete finite policy and Adam state')
    cfg = validate_checkpoint_robot(source)
    output.mkdir(exist_ok=True)
    status = output/'status.json'
    if status.exists() and json.loads(status.read_text())['state'] != 'prepared':
        raise ValueError('Inspect the already-started audit; never repeat older source updates')
    parity_path = output/'instrumentation_parity.json'
    parity = json.loads(parity_path.read_text())
    observer_path = ROOT/'mine_model/scripts/joint_pd_training_observer.py'
    if (parity['observer_sha256'] != sha(observer_path)
            or parity['source_checkpoint_sha256'] != sha(source)
            or parity['source_actual_ppo_updates'] != initial
            or parity['source_Adam_step'] != optimizer_step(saved)
            or not all(parity[k] for k in ('weights_bitwise_equal','Adam_bitwise_equal',
                'gradients_bitwise_equal','losses_equal','RNG_bitwise_equal','timeout_and_reset_tests_passed'))):
        raise ValueError('Read-only instrumentation lacks current controlled parity evidence')
    pins = dict(old_plan['source_files'])
    paths = [source,*sorted((source.parent/'params').glob('*.yaml')),
        previous/'status.json',previous/'experiment_plan.json',review,
        observer_path,Path(__file__).resolve(),parity_path]
    pins.update({str(p):sha(p) for p in paths})
    if (any(sha(Path(p)) != v for p,v in pins.items())
            or sha(ROOT/'mine_model/config/tracking_acceptance.json') != GATE_SHA):
        raise ValueError('Pinned sources, physics, original gate or retained checkpoints changed')
    path = output/'experiment_plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['source_files'] != pins or plan['source_checkpoint'] != str(source):
            raise ValueError('Do not change a pinned audit')
    else:
        label = 'joint_pd_native_alignment_'+output.name+'_u'+str(initial)+'_'+datetime.now().strftime('%H%M%S_%f')
        command,final,_ = continuation_command(source,cfg,label,2,cfg['commands'],output/'STOP',
            minimum_wheel_std=.03,joint_mean_guard_coef=0.,joint_height_feedback_coef=0.)
        command[command.index('agent.save_interval=5')] = 'agent.save_interval=1'
        command = [command[0],str(Path(__file__).resolve()),'--gradient-audit-output',str(output),*command[2:]]
        plan = dict(created_at=datetime.now().astimezone().isoformat(),previous_output=str(previous),
            source_checkpoint=str(source),source_sha256=sha(source),source_actual_ppo_updates=initial,
            source_Adam_step=optimizer_step(saved),source_Adam_states=17,target_actual_ppo_updates=initial+2,
            actual_updates=2,final_iteration=final,run_name=label,command=command,source_files=pins,
            weights_std_Adam_and_adaptive_LR_reset=False,rewards_changed=False,control_changed=False,
            normal_commands_and_full_height_unchanged=True,only_logging_change='save every update',
            original_acceptance_sha256=GATE_SHA,policy_accepted=False,hardware_enabled=False,shutdown=False)
        write_json(path,plan)
    write_json(output/'status.json',dict(state='prepared',actual_ppo_updates=initial,
        checkpoint=str(source),checkpoint_sha256=sha(source),training_active=False,
        goal_remains_active=True,policy_accepted=False))
    return plan


def worker(output, arguments):
    from rsl_rl.runners import OnPolicyRunner
    output = Path(output).resolve()
    if output.parent != ROOT/'mine_model/results' or not output.name.startswith('joint_pd_optimization_'):
        raise ValueError('Native audit output is not an owned optimization directory')
    plan = json.loads((output/'experiment_plan.json').read_text())
    if list(arguments) != plan['command'][4:]:
        raise ValueError('Native arguments differ from the qualified unchanged continuation')
    if json.loads((output/'status.json').read_text())['state'] != 'prepared':
        raise ValueError('Do not rerun an already-started audit from an older source')
    original_init = OnPolicyRunner.__init__
    observers = []

    def unchanged():
        if any(sha(Path(p)) != digest for p,digest in plan['source_files'].items()):
            raise ValueError('Pinned audit source, gate, physics or preserved checkpoints changed')

    def init(runner,env,*args,**kwargs):
        original_init(runner,env,*args,**kwargs)
        raw = env.unwrapped
        cfg = raw.cfg.to_dict()
        limits = raw._robot.data.joint_pos_limits[0,raw._leg_joint_ids].cpu().tolist()
        observers.append(TrainingObserver(runner.alg,model_from_config(cfg,limits),
            env=raw,cfg=cfg,output=output))

    def save(phase, **values):
        state = dict(state=phase,updated_at=datetime.now().astimezone().isoformat(),
            actual_ppo_updates=plan['source_actual_ppo_updates'],checkpoint=plan['source_checkpoint'],
            checkpoint_sha256=plan['source_sha256'],training_active=phase=='native_training',
            goal_remains_active=True,policy_accepted=False)
        state.update(values)
        write_json(output/'status.json',state)

    with (output/'.controller.lock').open('a') as owned, (RUNS/'.cumulative_learning.lock').open('a') as global_lock:
        for handle in (owned,global_lock):
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            unchanged()
            save('native_training')
            OnPolicyRunner.__init__ = init
            run_original_training(arguments)
            unchanged()
            if len(observers) != 1 or len(observers[0].updates) != 2:
                raise RuntimeError('The complete two-update native audit was not recorded')
            runs = list(RUNS.glob('*_'+plan['run_name']))
            if len(runs) != 1:
                raise RuntimeError('Missing or ambiguous owned audit checkpoint')
            checkpoint = runs[0]/f"model_{plan['final_iteration']}.pt"
            proof = verify_update(Path(plan['source_checkpoint']),checkpoint,
                plan['target_actual_ppo_updates'],validate_checkpoint_robot(plan['source_checkpoint']),
                Path(plan['source_checkpoint']))
            write_json(output/'resume_verification.json',proof)
            errors = {key:max(r['all_environment_alignment_max_errors'][key] for r in observers[0].updates)
                      for key in observers[0].updates[0]['all_environment_alignment_max_errors']}
            summary = dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                actual_ppo_updates=plan['target_actual_ppo_updates'],all_environment_alignment_max_errors=errors,
                transition_alignment_passed=bool(all(v<=5.e-6 for v in errors.values())),
                optimizer_minibatches=sum(len(r['optimizer_steps']) for r in observers[0].updates),
                all_recorded_probes_RNG_unchanged=True,rewards_and_control_unchanged=True,
                policy_accepted=False,independent_tracking_acceptance_not_proven_by_audit=True)
            write_json(output/'audit_summary.json',summary)
            save('agent_review_required',actual_ppo_updates=plan['target_actual_ppo_updates'],
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),training_active=False)
        except BaseException as error:
            candidates = [(saved_updates(p),p) for p in RUNS.glob('*_'+plan['run_name']+'/model_*.pt')]
            actual,checkpoint = max(candidates,key=lambda v:v[0]) if candidates else (
                plan['source_actual_ppo_updates'],Path(plan['source_checkpoint']))
            save('agent_diagnosis_required',reason=repr(error),training_active=False,
                actual_ppo_updates=actual,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
            raise
        finally:
            OnPolicyRunner.__init__ = original_init
            for observer in observers:
                observer.close()


if __name__ == '__main__':
    if '--gradient-audit-output' in sys.argv:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--gradient-audit-output',type=Path,required=True)
        known,rest = parser.parse_known_args()
        worker(known.gradient_audit_output,rest)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--previous',type=Path,required=True)
        parser.add_argument('--output',type=Path,required=True)
        args = parser.parse_args()
        prepare(args.previous,args.output)
