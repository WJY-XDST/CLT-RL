"""Restore a verified historical policy/rewards/Adam, reject regressing branches."""
import argparse
from datetime import datetime
import fcntl
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from auto_train_mine import write_json
import continue_joint_pd_priority as old
from joint_pd_rollback import admit,proposal,quality,restoration_ready,normalize_expanded_height
from summarize_mine_trial import assess,assess_stability
from train_joint_pd_posture_repair import GATE_SHA

core=old.core
ROOT=core.ROOT
UPDATES=25
NATIVE=old.NATIVE_BLOCK


def prepare(previous,output,delta,seed):
    interval=core.INTERVAL
    original_load=core.validate_checkpoint_robot
    def normalized_load(path):
        from wheel_legged_gym_isaaclab.training_state import load_curriculum_clock
        clock=load_curriculum_clock(path)
        step=clock['offset_steps']+core.saved_updates(path)*clock['steps_per_update']
        return normalize_expanded_height(original_load(path),step)
    try:
        core.INTERVAL=UPDATES
        core.validate_checkpoint_robot=normalized_load
        plan,expected=core.prepare(previous,output,.001)
    finally:
        core.INTERVAL=interval
        core.validate_checkpoint_robot=original_load
    if len(delta)>1:raise ValueError('Only one diagnosed reward scale per trial')
    changes={k:[expected['rewards'].get(k,0.),v] for k,v in delta.items()}
    expected['rewards'].update(delta)
    key,value=next(iter(delta.items())) if delta else ('tracking_task_quality_gate_floor',expected['rewards'].get('tracking_task_quality_gate_floor',1.))
    expected['rewards'].setdefault(key,value)
    plan.update(reward_key=key,reward_value=value,only_reward_change=changes,
        only_training_change='One measured reward scale; reject a regressing branch and load the incumbent full checkpoint',
        checkpoint_rollback_authorized=True,full_safe_command_dictionary_unchanged=True,
        legacy_full_height_metadata_normalized_without_changing_current_bounds=True,
        no_training_phases=True,training_seed=seed,weights_reinitialized=False,optimizer_reset=False)
    write_json(Path(output)/'experiment_plan.json',plan)
    return plan,expected


def bootstrap(candidate,abandoned,base):
    candidate,abandoned=Path(candidate),Path(abandoned)
    status=json.loads((candidate/'status.json').read_text());source=Path(status['checkpoint'])
    migration=json.loads((candidate/'resume_migration.json').read_text())
    if core.sha(source)!=migration['checkpoint_sha256'] or not migration['checkpoint_bytes_identical']:
        raise ValueError('Historical checkpoint bytes or coupled optimizer changed')
    cfg=core.validate_checkpoint_robot(source)
    if cfg['joint_reset_profile']!='joint_target_nominal_v1':raise ValueError('Keep the corrected closed-chain reset')
    if core.sha(ROOT/'mine_model/config/tracking_acceptance.json')!=GATE_SHA:raise ValueError('Acceptance changed')
    report=assess(candidate/'replay')
    if report['checkpoint_sha256']!=core.sha(source) or not restoration_ready(report):
        raise ValueError('Fresh full-command replay did not preserve stationary stability and all 27 episodes')
    # Source bytes are unchanged. Historical inference is reverified with the
    # corrected nominal reset, then held-out seeds; this is no full-gate claim.
    heldout=candidate/'rollback_heldout'
    if not (heldout/'completed.json').exists():
        command=core.block.replay_command(source,heldout)
        i=command.index('--seconds');command[i+1]='30'
        i=command.index('--seeds');command[i+1:i+4]=['183','184','185']
        core.block.runtime.OUTPUT=base
        core.block.runtime.run_child(command,candidate/'rollback_heldout.log',old.split.environment())
    held=assess_stability(heldout,20)
    if held['checkpoint_sha256']!=core.sha(source) or not restoration_ready(held):
        raise ValueError('Independent sustained stationary stability did not verify the rollback candidate')
    core.block.assess_directory(candidate/'replay')
    archive=json.loads((abandoned/'experiment_plan.json').read_text())
    pins=dict(archive['source_files']);changed=[]
    for name,digest in list(pins.items()):
        current=core.sha(Path(name))
        if current!=digest:
            if Path(name)!=ROOT/'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py':
                raise ValueError('Unexpected implementation change before rollback: '+name)
            backup=candidate/'evaluate_before_disabled_reward_telemetry_fix.py.txt'
            if core.sha(backup)!=digest:raise ValueError('Missing pre-fix diagnostic provenance')
            changed.append(dict(path=name,before_sha256=digest,after_sha256=current,
                reason='Disabled optional reward telemetry compatibility only; production reward equations unchanged'))
            pins[name]=current;pins[str(backup)]=digest
    for path in [source,*sorted((source.parent/'params').glob('*.yaml')),source.parent/'resume_migration.json',
                 candidate/'resume_migration.json',abandoned/'status.json',abandoned/'experiment_plan.json',
                 Path(__file__).resolve(),ROOT/'mine_model/scripts/joint_pd_rollback.py',
                 ROOT/'wheel_legged_isaaclab/tests/test_joint_pd_rollback.py',
                 ROOT/'wheel_legged_isaaclab/tests/test_disabled_reward_replay_telemetry.py',
                 *(candidate/'replay'/n for n in ('assessment.json','manifest.json','completed.json','trace.csv')),
                 *(heldout/n for n in ('assessment.json','manifest.json','completed.json','trace.csv'))]:
        pins[str(path)]=core.sha(path)
    plan=dict(source_checkpoint=str(source),source_sha256=core.sha(source),source_files=pins,
        source_actual_ppo_updates=core.saved_updates(source),source_Adam_states=17,
        protected_reference=str(candidate/'replay/assessment.json'),historical_loop_label=migration['historical_loop_label'],
        restored_original_reward_profile=True,restored_corresponding_optimizer=True,
        physical_reset_fix_preserved=True,code_revision=changed,abandoned_branch=str(abandoned),
        all_abandoned_checkpoints_preserved=True,checkpoint_rollback_authorized=True,
        original_acceptance_sha256=GATE_SHA,policy_accepted=False,hardware_enabled=False,shutdown=False)
    write_json(candidate/'experiment_plan.json',plan)
    status.update(state='agent_review_required',training_active=False,
        actual_ppo_updates=core.saved_updates(source),live_ppo_updates=core.saved_updates(source),
        updated_at=datetime.now().astimezone().isoformat(),policy_accepted=False,
        rollback_source_qualified_for_learning_only=True,full_original_gate_passed=report['passed'])
    write_json(candidate/'status.json',status)
    write_json(base/'rollback_verification.json',dict(source_checkpoint=str(source),checkpoint_sha256=core.sha(source),
        original_weights_and_17_Adam_states_restored=True,canonical_survivors=report['survivors'],
        heldout_survivors=held['survivors'],heldout_stationary_stability_passed=True,
        full_original_gate_passed=report['passed'],policy_accepted=False))
    return candidate


def train_trial(incumbent,output,delta,seed):
    args=SimpleNamespace(previous=incumbent,output=output,reward_key='unused',value=0.,updates=UPDATES,prepare_only=False)
    def command(*a,**kw):
        kw['minimum_wheel_std']=None  # Preserve the historical learned wheel std.
        argv,final,saved=old.split.ORIGINAL_COMMAND(*a,**kw)
        i=argv.index('--seed');argv[i+1]=str(seed)
        argv.extend(['agent.algorithm.entropy_coef=0.001','agent.policy.class_name=ActorCritic'])
        return argv,final,saved
    core.block.continuation_command=command
    NATIVE(args,prepare_function=lambda *unused:prepare(incumbent,output,delta,seed))


def run(candidate,abandoned,base,unit):
    candidate,abandoned,base=Path(candidate).resolve(),Path(abandoned).resolve(),Path(base).resolve()
    if base.parent!=ROOT/'mine_model/results':raise ValueError('Use an owned rollback coordinator')
    base.mkdir(exist_ok=True)
    with (base/'.lock').open('a') as own:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cursor=base/'coordinator.json'
        if cursor.exists():raise ValueError('Inspect an existing rollback coordinator before restarting')
        def record(state,**values):
            write_json(cursor,dict(state=state,unit=unit,updated_at=datetime.now().astimezone().isoformat(),
                checkpoint_rollback_allowed=True,policy_accepted=False,goal_remains_active=True,**values))
        record('verifying_historical_rollback',candidate=str(candidate),abandoned=str(abandoned))
        with (core.RUNS/'.cumulative_learning.lock').open('a') as global_lock:
            fcntl.flock(global_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            incumbent=bootstrap(candidate,abandoned,base)
        anchor=json.loads((incumbent/'replay/assessment.json').read_text())
        attempts={};history=[];work=0
        while not (base/'STOP').exists():
            state=json.loads((incumbent/'status.json').read_text());source=Path(state['checkpoint'])
            report=json.loads((incumbent/'replay/assessment.json').read_text())
            if report['passed']:
                passed=True
                for label,seconds,seeds,window in [('heldout',30,(183,184,185),20),('sustained',60,(283,284,285),40)]:
                    directory=incumbent/('final_'+label);command=core.block.replay_command(source,directory)
                    i=command.index('--seconds');command[i+1]=str(seconds)
                    i=command.index('--seeds');command[i+1:i+4]=list(map(str,seeds))
                    core.block.runtime.OUTPUT=base
                    core.block.runtime.run_child(command,incumbent/('final_'+label+'.log'),old.split.environment())
                    passed=bool(assess_stability(directory,window)['passed']) and passed
                if passed:
                    record('all_original_and_sustained_gates_verified_requires_final_audit',checkpoint=str(source),history=history);return
            identity=core.sha(source);used=attempts.setdefault(identity,set())
            seed,delta,signature=proposal(core.validate_checkpoint_robot(source),report,used);used.add(signature)
            output=ROOT/'mine_model/results'/('joint_pd_optimization_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_rollback_guard')
            record('training_with_incumbent_rollback_guard',incumbent=str(incumbent),current_output=str(output),
                source_checkpoint=str(source),branch_actual_ppo_updates=core.saved_updates(source),
                optimizer_updates_executed_this_controller=work,history=history,reward_trial=delta,seed=seed)
            pointer=dict(unit=unit,directory=str(output),coordinator=str(cursor),source_checkpoint=str(source),
                checkpoint_rollback_allowed=True,optimizer_reset=False,weights_reinitialized=False,
                policy_accepted=False,goal_remains_active=True,controller_kind='verified_incumbent_with_rollback')
            for name in ('joint_pd_active_joint_tracking.json','joint_pd_active_standing_seed.json'):
                write_json(ROOT/'mine_model/results'/name,pointer)
            train_trial(incumbent,output,delta,seed);work+=UPDATES
            after=json.loads((output/'replay/assessment.json').read_text());decision=admit(after,report,anchor)
            write_json(output/'rollback_review.json',decision)
            history.append(dict(output=str(output),source=str(incumbent),**decision))
            if decision['promote']:incumbent=output
            else:
                write_json(output/'checkpoint_rejected_but_preserved.json',dict(
                    restore_checkpoint=str(source),restore_Adam_from_same_checkpoint=True,
                    rejected_checkpoint=json.loads((output/'status.json').read_text())['checkpoint'],
                    all_failed_branch_artifacts_preserved=True,**decision))
            record('branch_review_complete',incumbent=str(incumbent),history=history,
                optimizer_updates_executed_this_controller=work)
        record('explicit_stop_respected',incumbent=str(incumbent),history=history)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,required=True)
    parser.add_argument('--abandoned',type=Path,required=True)
    parser.add_argument('--coordinator',type=Path,required=True)
    parser.add_argument('--unit',required=True)
    args=parser.parse_args()
    run(args.candidate,args.abandoned,args.coordinator,args.unit)
