import copy
import json
from pathlib import Path

import pytest

import continue_joint_pd_priority as controller
from joint_pd_priority import REWARD_KEY
from test_joint_pd_priority import report, trial


def config():
    return dict(commands=dict(grouped_training=True, fixed_standing_groups=True, standing_env_fraction=.25,
        ranges_height=[.22534645531,.37534645531], full_height_range_from_start=True,
        ranges_lin_vel_x=[.2,.8], ranges_reverse_lin_vel_x=[-.8,-.2], ranges_ang_vel_yaw=[-.3,.3],
        standing_only_steps=0, yaw_start_steps=0, reverse_ramp_start_steps=0, motion_ramp_steps=48000),
        rewards=dict(orientation=-400., base_height_error_sq=-500.,
            joint_target_leg_length_error_sq=-500., tracking_lin_vel=2., tracking_ang_vel=1.))


def fake_preparation(monkeypatch, tmp_path, configs):
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller.core, 'validate_checkpoint_robot', lambda source: copy.deepcopy(configs[str(source)]))
    monkeypatch.setattr(controller.core, 'sha', lambda unused: 'digest')
    def prepare(previous, output, coefficient):
        status = json.loads((previous/'status.json').read_text())
        source = Path(status['checkpoint'])
        output.mkdir()
        return dict(source_checkpoint=str(source), source_files={},
                    source_actual_ppo_updates=status['actual_ppo_updates']), copy.deepcopy(configs[str(source)])
    monkeypatch.setattr(controller.core, 'prepare', prepare)


def boundary(path, actual, value):
    (path/'replay').mkdir(parents=True)
    (path/'status.json').write_text(json.dumps(dict(checkpoint=str(path/'model.pt'), actual_ppo_updates=actual)))
    (path/'replay/assessment.json').write_text(json.dumps(value))


def test_preparation_changes_one_reward_and_no_commands_or_curriculum(tmp_path, monkeypatch):
    previous = tmp_path/'source'; previous.mkdir()
    source = previous/'model.pt'
    (previous/'status.json').write_text(json.dumps(dict(checkpoint=str(source), actual_ppo_updates=1400)))
    configs = {str(source): config()}
    fake_preparation(monkeypatch, tmp_path, configs)
    plan, expected = controller.prepare(previous, tmp_path/'output', -250., 25)
    assert expected['commands'] == configs[str(source)]['commands']
    assert plan['no_training_phases'] and plan['command_dictionary_unchanged']
    assert plan['dedicated_standing_fraction'] == .25
    assert plan['only_reward_change'] == {REWARD_KEY: [-500., -250.]}
    assert expected['rewards']['orientation'] == -400.
    assert expected['rewards']['base_height_error_sq'] == -500.


def test_continuation_reviews_stability_before_height_and_keeps_all_tasks(tmp_path, monkeypatch):
    results = tmp_path/'mine_model/results'; results.mkdir(parents=True)
    previous = results/'source'; base = results/'coordinator'
    boundary(previous, 1400, report())
    configs = {str(previous/'model.pt'): config()}
    fake_preparation(monkeypatch, tmp_path, configs)
    calls = []
    monkeypatch.setattr(controller.split, 'verify_entropy', lambda *unused: None)
    def native_block(args, prepare_function):
        plan, expected = prepare_function()
        native, final, saved = controller.core.block.continuation_command('source', 'cfg')
        assert '--reset_optimizer' not in native and '--reset_exploration_std' not in native
        assert expected['commands'] == config()['commands']
        assert expected['rewards']['tracking_lin_vel'] > 0 and expected['rewards']['tracking_ang_vel'] > 0
        calls.append((args, plan, expected))
        before = json.loads((args.previous/'replay/assessment.json').read_text())
        trial(before)['metrics']['body_p95_deg'] = 3. if len(calls) == 1 else 2.8
        trial(before)['metrics']['height_mae_m'] = .001  # Cannot offset posture regression.
        actual = json.loads((args.previous/'status.json').read_text())['actual_ppo_updates']+args.updates
        (args.output/'replay').mkdir()
        (args.output/'status.json').write_text(json.dumps(dict(checkpoint=str(args.output/'model.pt'), actual_ppo_updates=actual)))
        (args.output/'replay/assessment.json').write_text(json.dumps(before))
        configs[str(args.output/'model.pt')] = expected
        if len(calls) == 3: (base/'STOP').touch()
    monkeypatch.setattr(controller, 'NATIVE_BLOCK', native_block)
    monkeypatch.setattr(controller.split, 'ORIGINAL_COMMAND', lambda *a,**kw: (
        ['python','train.py','--checkpoint_path','latest.pt'], 42, {}))
    monkeypatch.setattr(controller.core.block, 'continuation_command', controller.core.block.continuation_command)
    controller.run(previous, base, 'owned.service')
    assert [a.updates for a,_,_ in calls] == [25,50,50]
    assert [a.value for a,_,_ in calls] == [-250.,-100.,-100.]
    assert calls[1][0].previous == calls[0][0].output
    assert calls[2][0].previous == calls[1][0].output
    cursor = json.loads((base/'coordinator.json').read_text())
    assert cursor['state'] == 'explicit_stop_respected'
    assert cursor['comparisons'][0]['verdict'] == 'stability_regression'
    assert not cursor['policy_accepted']


def test_owner_stop_is_not_consumed_as_agent_handoff(tmp_path):
    previous = tmp_path/'old'; previous.mkdir()
    (previous/'STOP').write_text(json.dumps(dict(reason='owner_stop')))
    (previous/'status.json').write_text(json.dumps(dict(training_active=False)))
    with pytest.raises(ValueError, match='owner pause'):
        controller.recover_handoff(previous, tmp_path/'new')
    assert (previous/'STOP').exists()


def test_full_gate_pass_requires_both_independent_sustained_checks(tmp_path, monkeypatch):
    results = tmp_path/'mine_model/results'; results.mkdir(parents=True)
    previous = results/'source'; base = results/'coordinator'
    value = report(); value['passed'] = True
    boundary(previous, 1400, value)
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    native = []
    def replay(source, output):
        return ['python','evaluate','--checkpoint',str(source),'--output',str(output),
                '--seconds','20','--seeds','83','84','85']
    monkeypatch.setattr(controller.core.block, 'replay_command', replay)
    monkeypatch.setattr(controller.core.block.runtime, 'OUTPUT', controller.core.block.runtime.OUTPUT)
    monkeypatch.setattr(controller.core.block.runtime, 'run_child', lambda command,*a,**kw: native.append(command))
    measured = []
    monkeypatch.setattr(controller, 'assess_stability', lambda path,window: measured.append((path.name,window)) or {'passed':True})
    controller.run(previous, base, 'owned.service')
    assert measured == [('heldout',20),('sustained',40)]
    assert [cmd[cmd.index('--seconds')+1] for cmd in native] == ['30','60']
    assert native[0][native[0].index('--seeds')+1:] == ['183','184','185']
    assert native[1][native[1].index('--seeds')+1:] == ['283','284','285']
    cursor = json.loads((base/'coordinator.json').read_text())
    assert cursor['state'] == 'all_original_and_sustained_gates_verified_requires_final_audit'
    assert not cursor['policy_accepted']
