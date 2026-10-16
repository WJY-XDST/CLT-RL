import copy
import json

import pytest

import continue_joint_pd_priority_repair100 as controller
from test_repair100_training_selection import sample
from test_joint_pd_priority import trial


def cfg():
    return dict(rewards=dict(orientation=-20., base_height_error_sq=-100.,
                            standing_velocity=-50., yaw_rate_error_sq=-1.))


def test_failed_candidate_drift_is_not_hidden_by_restored_safe_source():
    source = sample(.06, 5.); failed = sample(.055, 8.)
    trial(failed, 'height_low')['metrics']['speed_mae_m_s'] = .059
    assert controller.choose(cfg(), source, set(), [], failed)[1] == {'standing_velocity': -100.}


def test_new_stationary_yaw_failure_precedes_moving_posture_and_height():
    source = sample(.06, 5.); failed = sample(.055, 8.)
    trial(failed)['metrics']['yaw_mae_rad_s'] = .06
    assert controller.choose(cfg(), source, set(), [], failed)[1] == {'yaw_rate_error_sq': -2.}


def test_survival_and_stationary_body_failure_keep_highest_priority():
    source = sample(.06, 5.); failed = sample(.055, 8.)
    trial(failed)['metrics'].update(body_p95_deg=4., speed_mae_m_s=.1)
    assert controller.choose(cfg(), source, set(), [], failed)[1] == {'orientation': -80.}
    trial(failed, 'reverse_0p6').update(survived=False, resets=1, metrics=None, first_failure_s=3.)
    failed.update(survivors=26, resets=1)
    assert controller.choose(cfg(), source, set(), [], failed)[1] == {'orientation': -80.}


def test_drift_trials_are_not_repeated_with_identical_source_and_seed():
    source = sample(.06, 5.); failed = copy.deepcopy(source)
    trial(failed, 'height_low')['metrics']['speed_mae_m_s'] = .059
    used = {(43, (('standing_velocity', -100.),))}
    assert controller.choose(cfg(), source, used, [], failed)[1] == {'standing_velocity': -200.}
    used.add((43, (('standing_velocity', -200.),)))
    with pytest.raises(RuntimeError, match='exhausted'):
        controller.choose(cfg(), source, used, [], failed)


def test_improving_safe_learning_keeps_existing_continuation_rule():
    source = sample(.06, 5.)
    row = dict(promote=True, candidate_quality=[0,0,0,1.,18,2.,.2,5.])
    assert controller.choose(cfg(), source, set(), [row], source)[1] == {}


def test_changed_original_thresholds_are_rejected():
    source = sample(.06, 5.); failed = copy.deepcopy(source)
    trial(failed)['limits']['yaw_mae_rad_s'] = .1
    with pytest.raises(ValueError, match='tolerance'):
        controller.choose(cfg(), source, set(), [], failed)


def test_drift_prepare_preserves_entropy_weights_optimizer_and_other_rewards(tmp_path, monkeypatch):
    def baseline(previous, output, delta, coefficient, **kwargs):
        assert delta == {} and coefficient == .001
        return dict(source_files={}, entropy_coefficient=coefficient,
                    source_Adam_step=6520, source_actual_ppo_updates=326,
                    target_actual_ppo_updates=426), dict(rewards=cfg()['rewards'], commands={'standing_env_fraction': .25})
    monkeypatch.setattr(controller, 'ORIGINAL_PREPARE', baseline)
    plan, expected = controller.prepare(tmp_path, tmp_path, {'standing_velocity': -100.}, .001)
    assert plan['only_reward_change'] == {'standing_velocity': [-50., -100.]}
    assert expected['rewards']['orientation'] == -20. and expected['rewards']['base_height_error_sq'] == -100.
    assert expected['commands']['standing_env_fraction'] == .25
    assert plan['source_Adam_step'] == 6520 and plan['entropy_coefficient'] == .001
    with pytest.raises(ValueError, match='at most one'):
        controller.prepare(tmp_path, tmp_path, {'standing_velocity': -100., 'orientation': -80.}, .001)


@pytest.mark.parametrize('failure', [False, True])
def test_evidence_wrapper_routes_latest_failed_report_and_restores_hooks(tmp_path, monkeypatch, failure):
    old = tmp_path/'old'; old.mkdir(); failed = tmp_path/'failed'; failed.mkdir()
    source_report = sample(.06, 5.); source_report['checkpoint_sha256'] = 'same-source'
    failed_report = sample(.055, 8.); failed_report['checkpoint_sha256'] = 'failed-result'
    trial(failed_report, 'height_low')['metrics']['speed_mae_m_s'] = .059
    (failed/'experiment_plan.json').write_text(json.dumps(dict(source_sha256='same-source',
        training_seed=43, only_reward_change={'orientation': [-20., -80.]})))
    cursor = old/'coordinator.json'; cursor.write_text(json.dumps(dict(history=[dict(output=str(failed), promote=False)])))
    monkeypatch.setattr(controller.legacy.observed.drift, 'completed_report', lambda path: failed_report)
    def run(*args):
        assert controller.legacy.choose(cfg(), source_report, set(), [])[1] == {'standing_velocity': -100.}
        if failure:
            raise RuntimeError('native preflight failed')
    monkeypatch.setattr(controller.legacy, 'run', run)
    choose, prepare = controller.legacy.choose, controller.legacy.prepare
    if failure:
        with pytest.raises(RuntimeError, match='native preflight failed'):
            controller.run(cursor, tmp_path/'seed', tmp_path/'new', 'unit', tmp_path/'parity')
    else:
        controller.run(cursor, tmp_path/'seed', tmp_path/'new', 'unit', tmp_path/'parity')
    assert controller.legacy.choose is choose and controller.legacy.prepare is prepare


def test_handoff_keeps_a_measured_improvement_from_inherited_reviews(tmp_path, monkeypatch):
    previous = tmp_path/'previous'; previous.mkdir()
    report = sample(.055, 4.); report['checkpoint_sha256'] = 'improved'
    (previous/'experiment_plan.json').write_text(json.dumps(dict(source_sha256='older-source',
        training_seed=43, only_reward_change={'orientation': [-20., -160.]})))
    cursor = tmp_path/'cursor.json'; cursor.write_text(json.dumps(dict(history=[dict(
        output=str(previous), promote=True, candidate_quality=[0,0,0,1.,18,1.6,.18,5.])])) )
    monkeypatch.setattr(controller.legacy.observed.drift, 'completed_report', lambda path: report)
    def run(*args):
        assert controller.legacy.choose(cfg(), report, set(), [])[1] == {}
    monkeypatch.setattr(controller.legacy, 'run', run)
    controller.run(cursor, tmp_path/'seed', tmp_path/'new', 'unit', tmp_path/'parity')
