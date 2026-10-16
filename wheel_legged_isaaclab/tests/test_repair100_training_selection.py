import copy
import json
from pathlib import Path

import pytest

import continue_joint_pd_repair100 as controller
from joint_pd_priority import STATIC_CASES
from test_joint_pd_priority import report, trial


def sample(height=.07, moving_body=1.):
    value = report()
    for case in value['conditions']:
        for row in case['trials']:
            row['metrics']['height_mae_m'] = height
            if case['name'] not in STATIC_CASES:
                row['metrics']['body_p95_deg'] = moving_body
    return value


def test_retained_repair_learning_is_distinct_from_original_stability_and_acceptance():
    protected, before, after = sample(), sample(.06, 5.), sample(.065, 4.)
    decision = controller.review_for_learning(after, before, protected, protected)
    assert decision['promote'] and decision['retain_experimental_learning_only']
    assert not decision['protected_reference_promoted']
    assert not decision['full_original_gate_passed'] and not decision['policy_accepted']
    assert protected == sample() and after['passed'] is False


@pytest.mark.parametrize('metric', ['speed_mae_m_s', 'yaw_mae_rad_s', 'body_p95_deg'])
def test_better_height_or_moving_posture_cannot_keep_lost_stationary_stability(metric):
    before, after = sample(.06, 5.), sample(.02, 4.)
    trial(after, 'height_high')['metrics'][metric] = trial(after, 'height_high')['limits'][metric]*2
    decision = controller.review_for_learning(after, before, sample(), sample())
    assert not decision['promote'] and not decision['retain_experimental_learning_only']


def test_any_lost_survival_or_reset_rejects_repair_learning():
    before, after = sample(.06, 5.), sample(.02, 4.)
    trial(after, 'reverse_0p6').update(survived=False, resets=1, metrics=None, first_failure_s=4.)
    after.update(survivors=26, resets=1)
    assert not controller.review_for_learning(after, before, sample(), sample())['promote']


def test_height_improvement_does_not_cancel_new_moving_posture_regression():
    assert not controller.review_for_learning(sample(.02, 6.), sample(.06, 5.), sample(), sample())['promote']


def test_original_stability_promotion_still_requires_original_moving_limits():
    protected = sample()
    decision = controller.review_for_learning(sample(.05, 2.), sample(.06, 5.), protected, protected)
    assert decision['promote'] and decision['protected_reference_promoted']
    assert not decision['retain_experimental_learning_only'] and not decision['policy_accepted']


def test_original_threshold_changes_remain_an_error():
    after = sample(.05, 4.)
    trial(after)['limits']['body_p95_deg'] = 3.
    with pytest.raises(ValueError, match='tolerance'):
        controller.review_for_learning(after, sample(.06, 5.), sample(), sample())


def test_seed_selection_excludes_lost_standing_and_height_regression():
    bad = sample(.001, 1.)
    trial(bad)['metrics']['yaw_mae_rad_s'] = .1
    rows = {'good': sample(.06, 5.), 'lost_standing': bad, 'worse_height': sample(.08, 3.)}
    assert controller.select_seed(rows, sample()) == 'good'
    with pytest.raises(ValueError, match='No reviewed'):
        controller.select_seed({'lost_standing': bad}, sample())


def test_posture_repair_precedes_dense_height_without_repeating_profile_or_changing_tasks():
    cfg = dict(rewards=dict(orientation=-20., base_height_error_sq=-100.))
    _, delta, signature = controller.choose(cfg, sample(.06, 5.), set(), [])
    assert delta == {'orientation': -80.}
    assert controller.choose(cfg, sample(.06, 5.), {signature}, [])[1] == {'orientation': -160.}
    assert controller.choose(cfg, sample(.06, 2.), set(), [])[1] == {'base_height_error_sq': -250.}
    prior = dict(promote=True, candidate_quality=[0,0,0,1.,18,2.,.2,5.])
    assert controller.choose(cfg, sample(.06, 5.), set(), [prior])[1] == {}


def test_stagnation_requires_an_existing_cost_revision_instead_of_blind_extension():
    cfg = dict(rewards=dict(orientation=-80., base_height_error_sq=-100.))
    row = dict(promote=True, candidate_quality=[0,0,0,1.,18,2.,.2,5.])
    assert controller.choose(cfg, sample(.06, 5.), set(), [row]*3)[1] == {'orientation': -160.}


def test_prepare_changes_exactly_one_cost_and_keeps_inherited_entropy(tmp_path, monkeypatch):
    cfg = dict(rewards=dict(orientation=-20., base_height_error_sq=-100.), commands={'standing_env_fraction': .25})
    monkeypatch.setattr(controller.core, 'prepare', lambda previous, output, coefficient: (
        dict(source_files={}, entropy_coefficient=coefficient, source_actual_ppo_updates=326,
             target_actual_ppo_updates=426, source_Adam_step=6520, source_Adam_states=17), copy.deepcopy(cfg)))
    monkeypatch.setattr(controller.core, 'sha', lambda p: 'pinned')
    interval = controller.core.INTERVAL
    plan, expected = controller.prepare(tmp_path, tmp_path, {'orientation': -80.}, .003)
    assert expected['rewards'] == dict(orientation=-80., base_height_error_sq=-100.)
    assert expected['commands'] == cfg['commands']
    assert plan['entropy_coefficient'] == .003 and plan['source_Adam_step'] == 6520
    assert plan['only_reward_change'] == {'orientation': [-20., -80.]}
    assert plan['target_actual_ppo_updates']-plan['source_actual_ppo_updates'] == 100
    assert controller.core.INTERVAL == interval
    with pytest.raises(ValueError, match='at most one'):
        controller.prepare(tmp_path, tmp_path, {'orientation': -80., 'base_height_error_sq': -250.}, .003)


@pytest.mark.parametrize('native_failure', [False, True])
def test_native_training_inherits_entropy_and_full_checkpoint_and_restores_command(tmp_path, monkeypatch, native_failure):
    previous = tmp_path/'previous'; previous.mkdir()
    source = tmp_path/'run/model_325.pt'; (source.parent/'params').mkdir(parents=True)
    (source.parent/'params/agent.yaml').write_text('algorithm:\n  class_name: PPO\n  entropy_coef: 0.003\n')
    (previous/'status.json').write_text(json.dumps(dict(checkpoint=str(source))))
    output = tmp_path/'output'; output.mkdir()
    monkeypatch.setattr(controller, 'prepare', lambda *a, **kw: (dict(source_actual_ppo_updates=326,
        source_checkpoint=str(source), source_sha256='retained', target_actual_ppo_updates=426), {}))
    monkeypatch.setattr(controller.observed.drift, 'wait_memory', lambda *a: None)
    monkeypatch.setattr(controller.observed, 'finalize_observations', lambda *a: None)
    def original(*a, **kw):
        assert kw['minimum_wheel_std'] is None
        assert kw['reward_overrides'] == {'rewards.orientation': -80.}
        return ['/python.sh','/train.py','--seed','43','--checkpoint_path',str(source),
                '--target_iteration','426','--num_envs','8192','--headless'],425,'all17Adam'
    monkeypatch.setattr(controller.core, 'ORIGINAL_COMMAND', original)
    original_command = controller.core.block.continuation_command
    def native(args, prepare_function):
        prepare_function()
        assert json.loads((output/'status.json').read_text())['state'] == 'prepared'
        argv, final, saved = controller.core.block.continuation_command(
            source, {}, 'repair', 100, {}, output/'STOP', reward_overrides={'rewards.orientation': -80.})
        assert 'agent.algorithm.entropy_coef=0.003' in argv and saved == 'all17Adam' and final == 425
        assert not any(flag in argv for flag in ('--reset_optimizer','--reset_exploration_std','--save_initial_checkpoint'))
        if native_failure:
            raise RuntimeError('native failed')
        (output/'status.json').write_text(json.dumps(dict(checkpoint=str(source))))
    monkeypatch.setattr(controller.guard, 'NATIVE', native)
    if native_failure:
        with pytest.raises(RuntimeError, match='native failed'):
            controller.train(previous, output, {'orientation': -80.}, tmp_path, [])
    else:
        controller.train(previous, output, {'orientation': -80.}, tmp_path, [])
    assert controller.core.block.continuation_command is original_command


def test_real_review_loop_retains_candidate_without_replacing_protected_stable_checkpoint(tmp_path, monkeypatch):
    results = tmp_path/'mine_model/results'; results.mkdir(parents=True)
    protected, seed = results/'protected', results/'seed'
    for path, value, cp in [(protected, sample(), 'model_225.pt'), (seed, sample(.06, 5.), 'model_325.pt')]:
        (path/'replay').mkdir(parents=True)
        (path/'status.json').write_text(json.dumps(dict(checkpoint=str(path/cp), checkpoint_sha256='retained')))
        (path/'replay/assessment.json').write_text(json.dumps(value))
    old = results/'old'; old.mkdir(); cursor = old/'coordinator.json'
    cursor.write_text(json.dumps(dict(state='explicit_stop_respected', incumbent=str(protected),
        history=[dict(output=str(seed), source=str(protected), promote=False)])))
    base = results/'new'
    monkeypatch.setattr(controller, 'ROOT', tmp_path)
    monkeypatch.setattr(controller.guard, 'ROOT', tmp_path)
    monkeypatch.setattr(controller.observed.drift.handoff, 'ROOT', tmp_path)
    monkeypatch.setattr(controller.core, 'RUNS', tmp_path)
    monkeypatch.setattr(controller.core, 'sha', lambda p: 'retained')
    monkeypatch.setattr(controller.core, 'saved_updates', lambda p: 326)
    monkeypatch.setattr(controller.core, 'validate_checkpoint_robot', lambda p: dict(
        rewards=dict(orientation=-20., base_height_error_sq=-100.)))
    monkeypatch.setattr(controller.observed, 'verify_source', lambda *a: dict(optimizer_states=17))
    monkeypatch.setattr(controller.observed.drift, 'completed_report', lambda p: json.loads((Path(p)/'replay/assessment.json').read_text()))
    calls = []
    def trained(previous, output, delta, base_directory, pins):
        assert previous == seed and delta == {'orientation': -80.}
        calls.append(previous)
        (output/'replay').mkdir(parents=True)
        (output/'replay/assessment.json').write_text(json.dumps(sample(.055, 4.)))
        (output/'status.json').write_text(json.dumps(dict(checkpoint=str(output/'model_425.pt'))))
        (base/'STOP').touch()
    monkeypatch.setattr(controller, 'train', trained)
    before = (controller.guard.train_trial, controller.guard.admit, controller.guard.write_json, controller.guard.UPDATES)
    controller.run(cursor, seed, base, 'repair.service', tmp_path/'parity')
    assert len(calls) == 1
    saved = json.loads((base/'coordinator.json').read_text())
    assert saved['incumbent'] != str(protected)
    assert saved['protected_stability_reference'] == str(protected)
    row = saved['history'][0]
    assert row['promote'] and row['retain_experimental_learning_only']
    assert not row['protected_reference_promoted'] and not row['policy_accepted']
    assert before == (controller.guard.train_trial, controller.guard.admit, controller.guard.write_json, controller.guard.UPDATES)
