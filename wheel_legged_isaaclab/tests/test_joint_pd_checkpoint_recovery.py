"""Recovered progress must come from intact Adam files, not iteration labels."""
import json

import pytest
import torch

import launch_joint_pd_continuation as recovery


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery, 'RUNS', tmp_path/'runs')
    monkeypatch.setattr(recovery, 'validate_checkpoint_robot', lambda path:dict(leg_control_mode='implicit_joint_position'))
    directory=tmp_path/'mine_model/results/joint_pd_optimization_recovery_test'
    directory.mkdir(parents=True)
    run=tmp_path/'runs/date_joint_pd_cumulative_joint_pd_optimization_recovery_test_c0000'
    run.mkdir(parents=True)
    (directory/'status.json').write_text('{}')
    return directory,run


def checkpoint(run, filename, iteration, steps, *, nonfinite=False):
    path=run/filename
    torch.save(dict(iter=iteration, model_state_dict={'actor.6.weight':torch.tensor(float('nan') if nonfinite else 1.)},
        optimizer_state_dict={'state':{i:{'step':torch.tensor(float(steps)),
                                         'exp_avg':torch.zeros(1), 'exp_avg_sq':torch.ones(1)} for i in range(17)}}),path)
    return path


def test_stale_status_does_not_hide_newer_owned_checkpoint(owned):
    directory,run=owned
    old=checkpoint(run,'model_10.pt',10,200)
    latest=checkpoint(run,'model_20.pt',20,400)
    (directory/'status.json').write_text(json.dumps(dict(checkpoint=str(old))))
    selected,rejected=recovery.newest_owned_checkpoint(directory)
    assert selected==latest
    assert not rejected


def test_actual_adam_progress_wins_over_inclusive_resume_labels(owned):
    directory,run=owned
    checkpoint(run,'model_100.pt',100,1000)
    latest=checkpoint(run,'model_99_interrupted.pt',99,1020)
    assert recovery.newest_owned_checkpoint(directory)[0]==latest


def test_nonfinite_latest_checkpoint_is_preserved_but_not_resumed(owned):
    directory,run=owned
    intact=checkpoint(run,'model_20.pt',20,400)
    corrupt=checkpoint(run,'model_21.pt',21,420,nonfinite=True)
    selected,rejected=recovery.newest_owned_checkpoint(directory)
    assert selected==intact
    assert rejected[0]['checkpoint']==str(corrupt)
    assert corrupt.exists()


def test_partial_checkpoint_does_not_prevent_intact_file_recovery(owned):
    directory,run=owned
    intact=checkpoint(run,'model_20.pt',20,400)
    (run/'model_21.pt').write_bytes(b'partial checkpoint file')
    selected,rejected=recovery.newest_owned_checkpoint(directory)
    assert selected==intact
    assert len(rejected)==1


def test_missing_adam_states_cannot_count_as_preserved_optimizer(owned):
    directory,run=owned
    path=checkpoint(run,'model_20.pt',20,400)
    saved=torch.load(path,weights_only=True)
    saved['optimizer_state_dict']['state'].pop(0)
    torch.save(saved,path)
    with pytest.raises(ValueError,match='No intact policy'):
        recovery.newest_owned_checkpoint(directory)


def test_foreign_checkpoint_in_status_is_not_resumed(owned,tmp_path):
    directory,run=owned
    intact=checkpoint(run,'model_20.pt',20,400)
    outside=tmp_path/'foreign'
    outside.mkdir()
    foreign=checkpoint(outside,'model_100.pt',100,2000)
    (directory/'status.json').write_text(json.dumps(dict(checkpoint=str(foreign))))
    selected,rejected=recovery.newest_owned_checkpoint(directory)
    assert selected==intact
    assert len(rejected)==1
