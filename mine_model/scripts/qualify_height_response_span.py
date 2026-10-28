"""Qualify paired height-command learning in an isolated checkout only.

Never deploy the diagnostic weights, change production files, stop training,
relax acceptance, reset the primary optimizer, or operate real hardware.
Native jobs use run_python.sh's shared simulation lock to avoid host/GPU OOM.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch
import yaml

from auto_train_mine import write_json
from joint_pd_continuation import (
    continuation_command, optimizer_step, tensor_finite, validate_continuation_config,
)
from joint_pd_optimization import ratios, TASK_GOAL_METRICS, REFERENCE_TRACKING_METRICS
from summarize_mine_trial import assess
from watch_joint_pd_scratch import RUNS, validate_contract
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot

ROOT=Path(__file__).resolve().parents[2]
FILES=('wheel_legged_isaaclab/wheel_legged_gym_isaaclab/joint_height_feedback.py',
       'wheel_legged_isaaclab/tests/test_joint_height_response_span.py',
       'wheel_legged_isaaclab/tests/test_joint_height_gradient_scale.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_diagnostic_config(source,current,commands,env_count=512):
    """Only the declared parallel batch counts may differ from production."""
    if type(env_count) is not int or env_count<=0:
        raise ValueError('Diagnostic environment count must be a positive integer')
    if source['terrain']['num_envs']!=source['scene']['num_envs']:
        raise ValueError('Source scene and terrain batch counts differ')
    expected=json.loads(json.dumps(source))
    expected['scene']['num_envs']=env_count
    expected['terrain']['num_envs']=env_count
    validate_continuation_config(expected,current,commands)


def qualify(evidence,replay,resume_native=False,require_wheel_feedback=False):
    evidence,replay=Path(evidence).resolve(),Path(replay).resolve()
    if evidence.parent!=ROOT/'mine_model/results' or not evidence.name.startswith('joint_height_response_span_'):
        raise ValueError('Use a dedicated owned qualification directory')
    stage=evidence/'stage'
    files=FILES+(('wheel_legged_isaaclab/tests/test_joint_height_commands.py',
                  'wheel_legged_isaaclab/tests/test_steady_wheel_feedback.py') if require_wheel_feedback else ())
    if not stage.is_dir() or any((stage/name).resolve()==(ROOT/name).resolve() for name in files):
        raise ValueError('Candidate source must be separate from the running checkout')
    manifest=json.loads((replay/'manifest.json').read_text())
    source=Path(manifest['checkpoint']).resolve()
    control=json.loads((replay/'assessment.json').read_text())
    if (source.parent.parent!=RUNS or manifest['checkpoint_sha256']!=sha(source)
            or not manifest['policy_acceptance_test'] or manifest['weights_updated']
            or manifest['seconds_per_case']!=20 or manifest['seeds']!=[83,84,85]
            or control['total_trials']!=27 or control['survivors']!=27 or control['resets']):
        raise ValueError('Require the intact canonical surviving 27-trial baseline')
    before_hash=sha(source)
    acceptance=ROOT/'mine_model/config/tracking_acceptance.json'
    acceptance_hash=sha(acceptance)
    production_hashes={name:sha(ROOT/name) for name in files if (ROOT/name).is_file()}
    candidate_hashes={name:sha(stage/name) for name in files}
    if (evidence/'qualification_status.json').exists():
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        write_json(evidence/f'qualification_attempt_before_{stamp}.json',
                   json.loads((evidence/'qualification_status.json').read_text()))
    record=dict(source_checkpoint=str(source),source_sha256=before_hash,
                control_replay=str(replay),acceptance_sha256=acceptance_hash,
                production_source_sha256=production_hashes,stage_source_sha256=candidate_hashes,
                weights_reset=False,optimizer_reset=False,main_training_changed=False,
                policy_accepted=False,hardware_enabled=False,shutdown_requested=False)
    def save(phase,**values):
        record.update(state=phase,updated_at=datetime.now().astimezone().isoformat(),**values)
        write_json(evidence/'qualification_status.json',record)
        print(json.dumps(dict(state=phase,**values),ensure_ascii=False),flush=True)
    environment=dict(os.environ,PYTHONPATH=str(stage/'wheel_legged_isaaclab')+':'+str(stage/'mine_model/scripts'),
                     PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',PYTHONDONTWRITEBYTECODE='1',
                     OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    def run(command,log,timeout=3600):
        with Path(log).open('w') as output:
            result=subprocess.run(command,cwd=stage,env=environment,stdout=output,
                                  stderr=subprocess.STDOUT,timeout=timeout)
        if result.returncode:
            raise RuntimeError(f'Qualification command failed ({result.returncode}); inspect {log}')
    try:
        save('staged_regression_tests')
        run([str(stage/'run_python.sh'),'-m','pytest','wheel_legged_isaaclab/tests','-q'],
            evidence/'staged_regression_tests.log',120)
        summary=(evidence/'staged_regression_tests.log').read_text().splitlines()[-1]
        if 'passed' not in summary or 'failed' in summary:
            raise ValueError('Missing successful complete regression summary')
        cfg=validate_checkpoint_robot(source)
        label=evidence.name+'_native_qualification'
        command,final,before=continuation_command(source,cfg,label,3,cfg['commands'],evidence/'NATIVE_STOP',
                        minimum_wheel_std=.03,joint_mean_guard_coef=10.,joint_height_feedback_coef=.225)
        command[0]=str(stage/'run_python.sh')
        command[1]=str(stage/'wheel_legged_isaaclab/scripts/rsl_rl/train.py')
        command[command.index('--num_envs')+1]='512'
        command[command.index('agent.save_interval=5')]='agent.save_interval=1'
        plan=dict(command=command,source_checkpoint=str(source),source_sha256=before_hash,
                  expected_final_iteration=final,actual_updates=3,diagnostic_only=True,
                  main_training_unchanged=True,stage_source_sha256=candidate_hashes)
        if require_wheel_feedback:
            plan['required_standing_wheel_feedback_weight']=.2
        if resume_native:
            if json.loads((evidence/'native_plan.json').read_text())!=plan:
                raise ValueError('Cannot reuse native evidence after its exact plan/source/candidate changed')
            save('rechecking_saved_native_qualification',regression_summary=summary)
        else:
            if list(RUNS.glob('*_'+label)):
                raise ValueError('Native diagnostic already exists; inspect and explicitly resume its evidence')
            write_json(evidence/'native_plan.json',plan)
            save('native_qualification_waiting_shared_simulation_lock',regression_summary=summary)
            run(command,evidence/'native_training.log')
        runs=list(RUNS.glob('*_'+label))
        if len(runs)!=1:
            raise ValueError('Missing or ambiguous diagnostic run')
        checkpoint=runs[0]/f'model_{final}.pt'
        after=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if (not tensor_finite(after) or len(after['optimizer_state_dict']['state'])!=17
                or optimizer_step(after)-optimizer_step(before)!=60):
            raise ValueError('Incomplete/nonfinite diagnostic policy or retained Adam progress')
        changed=[bool((after['model_state_dict']['actor.6.weight'][i]
                        !=before['model_state_dict']['actor.6.weight'][i]).any()) for i in range(6)]
        if not all(changed):
            raise ValueError('Every actor output must remain trainable')
        validate_contract(runs[0],'joint_pd_balance_v1',fresh_weights=False,source_checkpoint=source)
        current=validate_checkpoint_robot(checkpoint)
        validate_diagnostic_config(cfg,current,cfg['commands'])
        agent=yaml.safe_load((runs[0]/'params/agent.yaml').read_text())
        profile=agent['algorithm']['joint_height_feedback_model']['command_counterfactuals']
        if profile.get('response_span_fraction')!=.8 or agent['algorithm']['joint_height_feedback_coef']!=.225:
            raise ValueError('Native job did not use the staged response-span training term')
        wheel_feedback=agent['algorithm']['joint_height_feedback_model'].get('standing_wheel_feedback')
        if require_wheel_feedback and (not wheel_feedback or wheel_feedback['weight']!=.2):
            raise ValueError('Native job did not use the qualified steady-wheel training feedback')
        verification=dict(source_checkpoint=str(source),source_sha256=before_hash,
            checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),native_actual_updates=3,
            optimizer_step_before=optimizer_step(before),optimizer_step_after=optimizer_step(after),
            optimizer_states=17,actor_rows_updated=changed,all_tensors_finite=True,
            control_contract_unchanged=True,rewards_and_acceptance_unchanged=True,
            diagnostic_weights_not_for_primary_continuation=True,
            training_profile=profile,standing_wheel_feedback=wheel_feedback,stage_source_sha256=candidate_hashes)
        write_json(evidence/'native_verification.json',verification)
        output=evidence/'candidate_replay'
        command=[str(stage/'run_python.sh'),str(stage/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py'),
                 '--checkpoint',str(checkpoint),'--output',str(output),'--seconds','20',
                 '--seeds','83','84','85','--record_policy_inputs','--diagnose_contacts',
                 '--headless','--device','cuda:0']
        save('candidate_canonical_replay_waiting_shared_simulation_lock',diagnostic_checkpoint=str(checkpoint))
        run(command,evidence/'candidate_replay.log')
        candidate=assess(output)
        keys=(*TASK_GOAL_METRICS,*REFERENCE_TRACKING_METRICS)
        controls,candidates=ratios(control,keys),ratios(candidate,keys)
        regressions={name:dict(before=value,after=candidates[name]) for name,value in controls.items()
                     if candidates[name]>value+max(.05,.05*value)}
        eligible=(candidate['total_trials']==27 and candidate['survivors']==27 and not candidate['resets']
                  and not regressions)
        if (sha(source)!=before_hash or sha(acceptance)!=acceptance_hash
                or any(sha(ROOT/name)!=digest for name,digest in production_hashes.items())
                or any(sha(stage/name)!=digest for name,digest in candidate_hashes.items())):
            raise ValueError('Source, candidate or original acceptance changed during qualification')
        write_json(evidence/'paired_native_replay_comparison.json',dict(
            reports=dict(control=dict(checkpoint=str(source),survivors=control['survivors'],resets=control['resets'],
                                      ratios=controls),
                         candidate=dict(checkpoint=str(checkpoint),survivors=candidate['survivors'],
                                        resets=candidate['resets'],ratios=candidates)),
            regressions=regressions,eligible_for_training_experiment=eligible,policy_accepted=False,
            limitation='Three native updates and a frozen replay qualify a bounded learning experiment, '
                       'not long-term convergence or independent sustained acceptance.'))
        save('qualification_finished',eligible_for_training_experiment=eligible,
             regressions=regressions,survivors=candidate['survivors'],resets=candidate['resets'],
             diagnostic_checkpoint=str(checkpoint),full_gate_passed=candidate['passed'])
    except BaseException as error:
        save('qualification_requires_inspection',reason=repr(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--replay',type=Path,required=True)
    parser.add_argument('--resume-native',action='store_true',help='Recheck the exact existing native plan/checkpoint; never rerun its PPO updates')
    parser.add_argument('--require-wheel-feedback',action='store_true',help='Verify the bounded training-only stationary wheel term')
    args=parser.parse_args()
    qualify(args.evidence,args.replay,args.resume_native,args.require_wheel_feedback)
