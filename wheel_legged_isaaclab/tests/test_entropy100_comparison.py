import json
from pathlib import Path

import pytest

import continue_joint_pd_entropy100 as controller


def test_failed_reward_trials_trigger_single_bounded_entropy_comparison():
    value, signature = controller.next_coefficient(.001, set())
    assert value == .003 and signature == (43, ('entropy_coef', .003))
    assert controller.next_coefficient(.001, set(), initial_promoted=True)[0] == .001


def test_stability_regression_reduces_trial_entropy_without_repeating_source_seed():
    review = dict(promote=False, comparison=dict(stability_retained=False))
    assert controller.next_coefficient(.001, set(), reviews=[review])[0] == .002
    with pytest.raises(RuntimeError, match='evidence-backed'):
        controller.next_coefficient(.001, {(43, ('entropy_coef', .002))}, reviews=[review])


def test_measured_improvement_continues_same_entropy_until_height_stagnates():
    row = dict(promote=True, candidate_quality=[0,0,0,1.,0,1.,.32,5.])
    assert controller.next_coefficient(.003, set(), reviews=[row])[0] == .003
    assert controller.next_coefficient(.003, set(), reviews=[row]*3)[0] == pytest.approx(.0045)
    with pytest.raises(RuntimeError, match='stagnant'):
        controller.next_coefficient(.005, set(), reviews=[row]*3)


def test_native_command_preserves_args_and_std_and_changes_only_existing_entropy(tmp_path, monkeypatch):
    calls = []
    def original(*args, **kwargs):
        calls.append(kwargs)
        return ['/run_python.sh','/train.py','--seed','43','--checkpoint_path','retained.pt',
                '--target_iteration','326','--num_envs','8192','--headless'],325,'full-Adam'
    monkeypatch.setattr(controller.core, 'ORIGINAL_COMMAND', original)
    argv, final, saved = controller.observed_command('source',{},'trial',100,{},tmp_path/'STOP',
        coefficient=.003, minimum_wheel_std=.03)
    assert calls[0]['minimum_wheel_std'] is None
    assert argv[1] == str(controller.observed.ENTRY) and argv[4] == '--'
    assert [v for v in argv if v.startswith('agent.algorithm.entropy_coef=')] == ['agent.algorithm.entropy_coef=0.003']
    assert argv[5:-2] == ['--seed','43','--checkpoint_path','retained.pt','--target_iteration','326',
                         '--num_envs','8192','--headless']
    assert not any(v.startswith('env.') for v in argv)
    assert final == 325 and saved == 'full-Adam'


def test_entropy_command_rejects_exploration_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(controller.core, 'ORIGINAL_COMMAND', lambda *a, **kw: (
        ['/run_python.sh','/train.py','--seed','43','--reset_exploration_std','.1'],325,None))
    with pytest.raises(ValueError, match='Retain learned'):
        controller.observed_command('source',{},'trial',100,{},tmp_path/'STOP',coefficient=.003)


@pytest.mark.parametrize('native_failure', [False, True])
def test_real_handoff_routes_exact_entropy_to_native_block_and_restores_hooks(tmp_path, monkeypatch, native_failure):
    results = tmp_path/'mine_model/results'; results.mkdir(parents=True)
    previous = results/'source'; previous.mkdir()
    source = tmp_path/'source_run/model_225.pt'; (source.parent/'params').mkdir(parents=True)
    (source.parent/'params/agent.yaml').write_text('algorithm:\n  class_name: PPO\n  entropy_coef: 0.001\n')
    (previous/'status.json').write_text(json.dumps(dict(checkpoint=str(source))))
    old = results/'old'; old.mkdir(); cursor = old/'coordinator.json'
    cursor.write_text(json.dumps(dict(state='explicit_stop_respected', incumbent=str(previous),
        history=[dict(output='rejected', source=str(previous), promote=False)])))
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller.observed.drift.handoff, 'ROOT', tmp_path)
    monkeypatch.setattr(controller, 'evidence_paths', lambda p: [])
    monkeypatch.setattr(controller.observed, 'verify_source', lambda *args: dict(optimizer_states=17))
    monkeypatch.setattr(controller.observed.drift, 'completed_report', lambda *args: {})
    monkeypatch.setattr(controller, 'priority_summary', lambda *args: dict(stationary_pose_passed=True))
    monkeypatch.setattr(controller.core, 'sha', lambda p: 'frozen')
    monkeypatch.setattr(controller.observed.drift, 'wait_memory', lambda *args: None)
    monkeypatch.setattr(controller.observed, 'finalize_observations', lambda *args: None)
    def prepared(previous_output, output, coefficient):
        output.mkdir()
        return dict(source_actual_ppo_updates=226, source_checkpoint=str(source), source_sha256='frozen',
                    target_actual_ppo_updates=326, source_files={}, entropy_coefficient=coefficient), {}
    monkeypatch.setattr(controller.core, 'prepare', prepared)
    monkeypatch.setattr(controller.core, 'ORIGINAL_COMMAND', lambda *a, **kw: (
        ['/python.sh','/train.py','--seed','43','--target_iteration','326','--headless'],325,'retained-Adam'))
    calls = []
    def native(args, prepare_function):
        plan, expected = prepare_function()
        assert plan['only_reward_change'] == {} and plan['entropy_coefficient'] == .003
        argv, final, saved = controller.core.block.continuation_command(
            source, {}, 'same', 100, {}, args.output/'STOP')
        assert 'agent.algorithm.entropy_coef=0.003' in argv and saved == 'retained-Adam'
        assert json.loads((args.output/'status.json').read_text())['state'] == 'prepared'
        calls.append((argv, final))
        if native_failure: raise RuntimeError('native failure')
        checkpoint = tmp_path/'trained/model_325.pt'; (checkpoint.parent/'params').mkdir(parents=True)
        (checkpoint.parent/'params/agent.yaml').write_text('algorithm:\n  class_name: PPO\n  entropy_coef: 0.003\n')
        (args.output/'status.json').write_text(json.dumps(dict(checkpoint=str(checkpoint))))
    monkeypatch.setattr(controller.guard, 'NATIVE', native)
    original_train, original_updates = controller.guard.train_trial, controller.guard.UPDATES
    original_command = controller.core.block.continuation_command
    def guard_run(candidate, abandoned, base, unit, **kwargs):
        assert candidate == previous and controller.guard.UPDATES == 100
        assert kwargs['bootstrap_function']() == previous
        (base/'coordinator.json').write_text(json.dumps(dict(incumbent=str(previous), history=[])))
        seed, delta, signature = kwargs['proposal_function']({}, {}, set())
        assert delta == {} and signature == (43, ('entropy_coef', .003))
        controller.guard.train_trial(previous, results/'trial', delta, seed)
    monkeypatch.setattr(controller.guard, 'run', guard_run)
    if native_failure:
        with pytest.raises(RuntimeError, match='native failure'):
            controller.run(cursor, results/'next', 'entropy.service', tmp_path/'parity', tmp_path/'evidence')
    else:
        controller.run(cursor, results/'next', 'entropy.service', tmp_path/'parity', tmp_path/'evidence')
    assert calls and controller.guard.train_trial is original_train and controller.guard.UPDATES == original_updates
    assert controller.core.block.continuation_command is original_command
