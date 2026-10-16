import copy
import json
from pathlib import Path

import pytest

import continue_joint_pd_rollback as controller
from joint_pd_rollback import admit,proposal,restoration_ready,normalize_expanded_height
from test_joint_pd_priority import report,trial


def test_a_falling_branch_cannot_replace_the_restored_incumbent_for_better_height():
    before=report();after=copy.deepcopy(before)
    row=trial(after,'reverse_0p6');row.update(survived=False,resets=1,metrics=None,first_failure_s=1.)
    after.update(survivors=26,resets=1)
    for c in after['conditions']:
        for t in c['trials']:
            if t['metrics']:t['metrics']['height_mae_m']=0.
    decision=admit(after,before,before)
    assert not decision['promote']
    assert decision['decision']=='restore_incumbent_weights_rewards_and_Adam'


def test_inside_original_pose_limits_height_can_improve_without_posture_polishing():
    before=report();after=copy.deepcopy(before)
    for c in before['conditions']:
        for t in c['trials']:t['metrics']['height_mae_m']=.06
    for c in after['conditions']:
        for t in c['trials']:t['metrics'].update(body_p95_deg=2.,height_mae_m=.04)
    assert admit(after,before,before)['promote']
    assert not after['passed']


def test_rollback_source_needs_all_episodes_and_stationary_pose_but_is_not_full_acceptance():
    value=report()
    assert restoration_ready(value) and not value['passed']
    trial(value,'height_high')['metrics']['speed_mae_m_s']=.04
    assert not restoration_ready(value)


def test_rejected_source_profile_seed_is_never_blindly_repeated():
    value=report();trial(value,'forward_0p3')['metrics']['body_p95_deg']=8.
    cfg=dict(rewards=dict(orientation=-60.,base_height_error_sq=0.))
    used=set()
    s0,d0,k0=proposal(cfg,value,used);used.add(k0)
    s1,d1,k1=proposal(cfg,value,used)
    assert (s0,d0)==(43,{}) and (s1,d1)==(43,{'orientation':-120.})
    assert k0!=k1


def test_legacy_height_metadata_only_normalizes_after_full_expansion():
    cfg=dict(commands=dict(ranges_height=[.225,.375],height_expansion_from=[.28,.32],
        height_expansion_start_steps=117024,height_expansion_ramp_steps=48000))
    original=copy.deepcopy(cfg)
    normalized=normalize_expanded_height(cfg,4601*48)
    assert cfg==original
    assert normalized['commands'].pop('full_height_range_from_start') is True
    assert normalized==cfg
    with pytest.raises(ValueError):normalize_expanded_height(cfg,117024+47999)


def test_rejected_branch_reloads_the_same_incumbent_checkpoint_and_matching_optimizer(tmp_path,monkeypatch):
    results=tmp_path/'mine_model/results';results.mkdir(parents=True)
    incumbent=results/'source';(incumbent/'replay').mkdir(parents=True)
    state=dict(checkpoint=str(incumbent/'model.pt'),checkpoint_sha256='stable',actual_ppo_updates=4601)
    (incumbent/'status.json').write_text(json.dumps(state))
    value=report();trial(value,'forward_0p3')['metrics']['body_p95_deg']=8.
    (incumbent/'replay/assessment.json').write_text(json.dumps(value))
    base=results/'coordinator';calls=[]
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    monkeypatch.setattr(controller.core,'RUNS',tmp_path)
    monkeypatch.setattr(controller,'bootstrap',lambda *unused:incumbent)
    monkeypatch.setattr(controller.core,'sha',lambda p:'stable')
    monkeypatch.setattr(controller.core,'saved_updates',lambda p:4601)
    monkeypatch.setattr(controller.core,'validate_checkpoint_robot',lambda p:dict(rewards=dict(orientation=-60.,base_height_error_sq=0.)))
    def train(previous,output,delta,seed):
        calls.append((previous,delta,seed,json.loads((previous/'status.json').read_text())['checkpoint']))
        (output/'replay').mkdir(parents=True)
        result=copy.deepcopy(value);trial(result,'height_high')['metrics']['body_p95_deg']=4.
        (output/'replay/assessment.json').write_text(json.dumps(result))
        (output/'status.json').write_text(json.dumps(dict(checkpoint=str(output/'failed.pt'))))
        if len(calls)==2:(base/'STOP').touch()
    monkeypatch.setattr(controller,'train_trial',train)
    controller.run(incumbent,results/'abandoned',base,'rollback.service')
    assert [row[0] for row in calls]==[incumbent,incumbent]
    assert [row[3] for row in calls]==[state['checkpoint'],state['checkpoint']]
    assert [row[1] for row in calls]==[{}, {'orientation':-120.}]
    saved=json.loads((base/'coordinator.json').read_text())
    assert saved['incumbent']==str(incumbent)
    assert all(not row['promote'] for row in saved['history'])
    assert len(list(results.glob('joint_pd_optimization_*/checkpoint_rejected_but_preserved.json')))==2
