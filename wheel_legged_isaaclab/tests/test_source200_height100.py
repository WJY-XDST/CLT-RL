import json
from pathlib import Path

import pytest
import continue_joint_pd_source200_height100 as controller


@pytest.mark.parametrize('promote',[False,True])
def test_handoff_loads_reviewed_incumbent_and_never_rejected_branch(tmp_path,monkeypatch,promote):
    results=tmp_path/'mine_model/results';results.mkdir(parents=True)
    source,output=results/'model200',results/'trial'
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    row=dict(source=str(source),output=str(output),promote=promote)
    selected=output if promote else source
    cursor=dict(state='explicit_stop_respected',incumbent=str(selected),history=[row])
    assert controller.selected_incumbent(cursor)==selected
    cursor['incumbent']=str(source if promote else output)
    with pytest.raises(ValueError):controller.selected_incumbent(cursor)


def test_unfinished_25_update_review_cannot_silently_become_100_update_source(tmp_path):
    with pytest.raises(ValueError):
        controller.selected_incumbent(dict(state='training_with_incumbent_rollback_guard'))


def test_all_following_native_blocks_use_100_actual_updates_and_restore_module_setting(tmp_path,monkeypatch):
    results=tmp_path/'mine_model/results';results.mkdir(parents=True)
    previous=results/'short';previous.mkdir()
    source,trial=results/'model200',results/'trial'
    cursor=previous/'coordinator.json'
    cursor.write_text(json.dumps(dict(state='explicit_stop_respected',incumbent=str(source),
        history=[dict(source=str(source),output=str(trial),promote=False)])))
    monkeypatch.setattr(controller,'ROOT',tmp_path)
    calls=[]
    original=controller.guard.UPDATES
    def run(candidate,abandoned,base,unit,**kw):
        calls.append(controller.guard.UPDATES)
        assert abandoned==source and kw['proposal_function'] is controller.experiment.height_proposal
    monkeypatch.setattr(controller.guard,'run',run)
    controller.run(cursor,results/'bridge',results/'next','100-update.service')
    assert calls==[100] and controller.guard.UPDATES==original
