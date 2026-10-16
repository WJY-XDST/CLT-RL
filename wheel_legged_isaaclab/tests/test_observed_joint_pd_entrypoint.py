import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import train_joint_pd_observed as worker


def test_observer_source_requires_retained_direct_pd_and_exact_actual_update_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'saved_updates', lambda p: 226)
    monkeypatch.setattr(worker, 'validate_checkpoint_robot', lambda p: dict(leg_control_mode='implicit_joint_position'))
    arguments = ['--checkpoint_path', str(tmp_path/'model_225.pt'), '--target_iteration', '326']
    assert worker.continuation_source(arguments)[1:] == (226, 326)
    with pytest.raises(ValueError, match='100 actual'):
        worker.continuation_source([*arguments[:-1], '327'])
    with pytest.raises(ValueError, match='complete optimizer'):
        worker.continuation_source([*arguments, '--reset_optimizer'])


@pytest.mark.parametrize('failure', [False, True])
def test_observer_uses_original_training_arguments_and_restores_runner_after_exit(tmp_path, monkeypatch, failure):
    output = tmp_path/'mine_model/results/joint_pd_optimization_test/native_learning_signal'
    output.parent.mkdir(parents=True)
    source = tmp_path/'run/model_225.pt'
    (source.parent/'params').mkdir(parents=True)
    (source.parent/'params/agent.yaml').write_text('algorithm:\n  class_name: PPO\npolicy:\n  activation: elu\n')
    monkeypatch.setattr(worker, 'ROOT', tmp_path)
    monkeypatch.setattr(worker, 'continuation_source', lambda args: (source, 226, 326))
    monkeypatch.setattr(worker, 'sha', lambda p: 'unchanged')
    calls = []
    class Runner:
        def __init__(self, env, **kwargs):
            self.alg = 'original-PPO'
            calls.append(kwargs)
    original_init = Runner.__init__
    monkeypatch.setattr(worker, 'OnPolicyRunner', Runner)
    monkeypatch.setattr(worker, 'geometry', lambda cfg, limits: dict(limits=limits))
    class Observer:
        def __init__(self, algorithm, model, **kwargs):
            assert algorithm == 'original-PPO'
            self.updates = [dict(all_environment_alignment_max_errors=dict(action_storage_max_error=0.),
                                 probe_RNG_unchanged=True, optimizer_steps=[{}]*20) for _ in range(100)]
        def close(self):
            calls.append('observer-closed')
    monkeypatch.setattr(worker, 'TrainingObserver', Observer)
    import torch
    raw = SimpleNamespace(cfg=SimpleNamespace(to_dict=lambda: {}), _leg_joint_ids=[0,1,2,3],
        _robot=SimpleNamespace(data=SimpleNamespace(joint_pos_limits=torch.tensor([[[-1.,1.]]*4]))))
    arguments = ['--task', 'WheelLeggedJointPD-Flat-v0', '--headless']
    def training(args):
        assert args is arguments
        Runner(SimpleNamespace(unwrapped=raw), log_dir='unchanged-log', device='cuda:0')
        if failure:
            raise RuntimeError('native failure')
    monkeypatch.setattr(worker, 'run_original_training', training)
    if failure:
        with pytest.raises(RuntimeError, match='native failure'):
            worker.run(output, arguments)
        assert json.loads((output/'diagnosis_required.json').read_text())['all_native_checkpoints_preserved']
    else:
        worker.run(output, arguments)
        assert json.loads((output/'completed.json').read_text())['optimizer_minibatches'] == 2000
    assert Runner.__init__ is original_init and calls[-1] == 'observer-closed'
    assert calls[0] == dict(log_dir='unchanged-log', device='cuda:0')
