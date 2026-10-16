import copy

import pytest
import continue_joint_pd_stationary_drift100 as controller
from joint_pd_priority import CASE_NAMES


def report(speed=0., yaw=0., body=0., height=.02):
    rows = []
    for name in CASE_NAMES:
        rows.append(dict(name=name, speed=0., yaw=0., height=.3, trials=[
            dict(seed=seed, survived=True, resets=0,
                 metrics=dict(speed_mae_m_s=speed, yaw_mae_rad_s=yaw,
                              body_p95_deg=body, height_mae_m=height),
                 limits=dict(speed_mae_m_s=.025, yaw_mae_rad_s=.025,
                             body_p95_deg=2.5, height_mae_m=.03))
            for seed in (83, 84, 85)]))
    return dict(conditions=rows, total_trials=27, survivors=27, resets=0, passed=False)


def cfg():
    return dict(rewards=dict(standing_velocity=0., base_height_error_sq=-100.,
                            orientation=-20., yaw_rate_error_sq=-1.))


def test_two_measured_drift_failures_change_only_existing_standing_cost():
    config = cfg(); original = copy.deepcopy(config)
    seed, delta, signature, diagnosis = controller.choose(config, report(), set(),
        initial_reports=[report(speed=.063), report(speed=.085)])
    assert config == original and seed == 43 and delta == {'standing_velocity': -25.}
    assert diagnosis['actual_PPO_updates'] == 100 and diagnosis['hypothesis_only']
    with pytest.raises(ValueError, match='both observed drift'):
        controller.choose(config, report(), set(), initial_reports=[report(speed=.063), report()])


def test_rejected_source_profile_is_not_repeated_or_assigned_a_different_seed():
    used = {(43, (('standing_velocity', -25.),))}
    assert controller.choose(cfg(), report(speed=.063), used)[1] == {'standing_velocity': -50.}
    used.update((43, (('standing_velocity', -v),)) for v in (50., 100., 200.))
    with pytest.raises(RuntimeError, match='diagnostics'):
        controller.choose(cfg(), report(speed=.063), used)


def test_stability_failure_prevents_another_height_cost_sweep():
    assert controller.choose(cfg(), report(body=5., height=.1), set())[1] == {'orientation': -40.}
    assert controller.choose(cfg(), report(yaw=.08, height=.1), set())[1] == {'yaw_rate_error_sq': -2.}
    assert controller.choose(cfg(), report(height=.1), set())[1] == {'base_height_error_sq': -150.}


def test_accepted_improvement_continues_same_reward_for_100_updates():
    assert controller.choose(cfg(), report(height=.1), set(), continue_improvement=True)[1] == {}


def test_native_memory_gate_preserves_batch_and_requires_headroom():
    gib = 1024**3
    assert controller.memory_ready(dict(MemAvailable=16*gib, SwapFree=11*gib))
    assert not controller.memory_ready(dict(MemAvailable=16*gib, SwapFree=3*gib))
    assert not controller.memory_ready(dict(MemAvailable=5*gib, SwapFree=25*gib))
