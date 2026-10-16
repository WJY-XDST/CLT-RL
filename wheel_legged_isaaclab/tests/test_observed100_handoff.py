import json
from pathlib import Path

import pytest

import continue_joint_pd_observed100 as controller


def test_lineage_reaches_exact_original_model200_and_rejects_cycles(tmp_path, monkeypatch):
    source = tmp_path/'new/model_325.pt'
    original = tmp_path/'old/model_200.pt'
    (source.parent/'params').mkdir(parents=True)
    (source.parent/'params/precision_mode.yaml').write_text('source_checkpoint: '+str(original))
    monkeypatch.setattr(controller.guard.core, 'sha', lambda p: controller.drift.handoff.experiment.SOURCE_SHA if p == original else 'new')
    assert controller.lineage(source) == [source, original]
    (source.parent/'params/precision_mode.yaml').write_text('source_checkpoint: '+str(source))
    with pytest.raises(ValueError, match='Cyclic'):
        controller.lineage(source)


def test_handoff_retains_selected_source_and_wraps_original_native_arguments(tmp_path, monkeypatch):
    results = tmp_path/'mine_model/results'
    results.mkdir(parents=True)
    old = results/'old';old.mkdir()
    previous = results/'source';previous.mkdir()
    cursor = old/'coordinator.json'
    cursor.write_text(json.dumps(dict(state='explicit_stop_respected', incumbent=str(previous),
        history=[dict(output=str(results/'rejected'), source=str(previous), promote=False)])))
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller.drift.handoff, 'ROOT', tmp_path)
    monkeypatch.setattr(controller, 'verify_source', lambda *args: dict(optimizer_states=17))
    monkeypatch.setattr(controller.guard, 'train_trial', lambda *args: calls.append(controller.guard.old.split.ORIGINAL_COMMAND()))
    original_prepare, original_train, original_updates = controller.guard.prepare, controller.guard.train_trial, controller.guard.UPDATES
    original_command = controller.guard.old.split.ORIGINAL_COMMAND
    def command():
        return ['/run_python.sh', '/train.py', '--task', 'WheelLeggedJointPD-Flat-v0', '--headless'], 325, 'retained-Adam'
    monkeypatch.setattr(controller.guard.old.split, 'ORIGINAL_COMMAND', command)
    calls = []
    def run(candidate, abandoned, base, unit, **kwargs):
        assert candidate == previous and controller.guard.UPDATES == 100
        assert kwargs['bootstrap_function']() == previous
        controller.guard.train_trial(previous, results/'trial', {'orientation': -40.}, 43)
        assert controller.guard.old.split.ORIGINAL_COMMAND is command
    monkeypatch.setattr(controller.guard, 'run', run)
    controller.run(cursor, results/'next', 'observed.service', tmp_path/'parity.json')
    argv, final, saved = calls[0]
    assert argv[1] == str(controller.ENTRY) and argv[4] == '--'
    assert argv[5:] == ['--task', 'WheelLeggedJointPD-Flat-v0', '--headless']
    assert final == 325 and saved == 'retained-Adam'
    assert controller.guard.prepare is original_prepare and controller.guard.train_trial is original_train
    assert controller.guard.UPDATES == original_updates
