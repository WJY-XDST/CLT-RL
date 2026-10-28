import json
from pathlib import Path

import pytest
import numpy as np

import continue_joint_pd_stationary100_v2 as owner
import handoff_stationary100_reader_fix as handoff
from test_stationary_motion_curriculum import completed_report


def fixture(tmp_path, monkeypatch, *, improved=True):
    prior=tmp_path/'prior';prior.mkdir()
    old=tmp_path/'old';old.mkdir()
    output=tmp_path/'output';output.mkdir()
    signal=output/'native_learning_signal';signal.mkdir()
    for n in range(1,101):(signal/f'rollout_{n:04d}.npz').touch()
    write=lambda p,d:p.write_text(json.dumps(d))
    cursor=prior/'coordinator.json'
    schema=tmp_path/'actual_observer_schema.npz'
    np.savez(schema,native_commands=np.zeros((48,32,3)))
    with np.load(schema) as archive:
        try:
            archive['commands']
        except KeyError as error:
            actual_numpy_reason=repr(error)
    write(cursor,dict(state='agent_diagnosis_required',reason=actual_numpy_reason,
                      current_output=str(output),incumbent=str(old)))
    write(output/'status.json',dict(state='agent_review_required',training_active=False,
        actual_ppo_updates=426,checkpoint=str(output/'model_425.pt')))
    write(output/'experiment_plan.json',dict(target_actual_ppo_updates=426,
        source_checkpoint=str(old/'model_325.pt')))
    write(output/'resume_verification.json',dict(actual_ppo_updates=426,optimizer_step=8520,
        optimizer_states=17,all_tensors_finite=True,checkpoint_sha256='retained'))
    reports={old:completed_report(.06),output:completed_report(.04 if improved else .08)}
    monkeypatch.setattr(owner.observed.drift,'completed_report',lambda p:reports[Path(p)])
    monkeypatch.setattr(owner.core,'sha',lambda p:'retained')
    calls=[]
    monkeypatch.setattr(owner.observed,'finalize_observations',lambda p:calls.append(('finalize',p)))
    monkeypatch.setattr(owner,'verify_rollout_motion',lambda p,s:calls.append(('archive',p,s)))
    return cursor,output,old,calls


@pytest.mark.parametrize('improved',[True,False])
def test_recovery_preserves_all_100_updates_and_reviews_before_next_source(tmp_path,monkeypatch,improved):
    cursor,output,old,calls=fixture(tmp_path,monkeypatch,improved=improved)
    bridge=handoff.recover(cursor,tmp_path/'recovery','next-unit')
    state=json.loads(bridge.read_text())
    assert state['incumbent']==str(output if improved else old)
    assert state['no_native_updates_repeated'] and len(state['stationary_history'])==1
    assert len([c for c in calls if c[0]=='archive'])==100
    assert len([c for c in calls if c[0]=='finalize'])==1
    decision=json.loads((output/'stationary_review.json').read_text())
    assert decision['promote'] is improved and not decision['policy_accepted']
    if not improved:
        cfg=dict(rewards=dict(orientation=-20.,standing_velocity=-50.,yaw_rate_error_sq=-1.,base_height_error_sq=-100.))
        delta=owner.stage.next_reward(cfg,completed_report(.08),state['stationary_history'],{()})
        assert delta=={'base_height_error_sq':-250.}


def test_incomplete_optimizer_progress_is_not_mistaken_for_reader_only_error(tmp_path,monkeypatch):
    cursor,output,old,calls=fixture(tmp_path,monkeypatch)
    proof=json.loads((output/'resume_verification.json').read_text());proof['optimizer_step']=8500
    (output/'resume_verification.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='incomplete'):
        handoff.recover(cursor,tmp_path/'recovery','next-unit')
    assert calls==[] and not (output/'stationary_review.json').exists()


def test_an_unrelated_native_failure_is_never_silently_recovered(tmp_path,monkeypatch):
    cursor,output,old,calls=fixture(tmp_path,monkeypatch)
    state=json.loads(cursor.read_text());state['reason']="RuntimeError('native failure')";cursor.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='Only recover'):
        handoff.recover(cursor,tmp_path/'recovery','next-unit')
    assert calls==[]
