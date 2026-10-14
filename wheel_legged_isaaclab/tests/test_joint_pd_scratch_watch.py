"""Pure monitoring decisions: random-policy collapse is not a numerical fault."""
import importlib.util
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[2] / 'mine_model/scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('joint_pd_watch', SCRIPTS / 'watch_joint_pd_scratch.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def entry(iteration, lifetime=1., survivors=0):
    return dict(iteration=iteration, mean_first_lifetime_s=lifetime, survivors=survivors, trials=3)


def test_random_exploration_collapse_does_not_trigger_numerical_guard():
    update = dict(iteration=1, metrics={'Mean reward': -50., 'Mean episode length': 8.,
                                     'Episode_Termination/standing_unsafe': 1.})
    assert watch.numerical_anomaly([update] * 3, '') is None


def test_nonfinite_metric_pauses():
    assert watch.numerical_anomaly([dict(metrics={'Mean reward': float('nan')})], '') == 'nonfinite_training_metric'


def test_numerical_termination_pauses():
    assert watch.numerical_anomaly([dict(metrics={'Episode_Termination/numerical': .001})], '') == 'numerical_termination'


def test_fatal_error_not_warning():
    assert watch.numerical_anomaly([], '[Error] [omni.physx.plugin] bad joints')
    assert watch.numerical_anomaly([], '[Warning] [omni.physx.plugin] TGS warning') is None
    assert watch.numerical_anomaly([], '[Error] [omni.platforminfo.plugin] CPU data unavailable') is None


def test_two_early_failed_milestones_are_not_enough_to_pause():
    assert watch.support_anomaly([entry(200), entry(400)]) is None


def test_persistent_immediate_failure_pauses_after_three_milestones():
    assert watch.support_anomaly([entry(200), entry(400), entry(600)]) == 'persistent_early_support_failure_without_improvement'


def test_improving_lifetime_keeps_learning():
    assert watch.support_anomaly([entry(200, .5), entry(400, .8), entry(600, 1.)]) is None


def test_longer_partial_balance_not_confused_with_immediate_collapse():
    assert watch.support_anomaly([entry(200, 10.), entry(400, 10.), entry(600, 10.)]) is None


def test_lost_previously_learned_standing_pauses():
    assert watch.support_anomaly([entry(200, 20., 3), entry(400)]) == 'previously_learned_standing_lost'


def test_motion_failures_do_not_gate_standing_curriculum():
    assessment = dict(iteration=200, survivors=3, conditions=[dict(name='stand', survivors=3, passed=False,
                      trials=[dict(survived=True, metrics={'speed_mae_m_s': .1})] * 3)])
    summary = watch.standing_summary(assessment)
    assert summary['mean_first_lifetime_s'] == 20.
    assert summary['survivors'] == 3
    assert watch.support_anomaly([summary, summary, summary]) is None


def test_scope_cannot_control_old_vmc_service():
    import pytest
    output = watch.ROOT / 'mine_model/results/joint_pd_scratch_test'
    launch = dict(unit='clt-vmc-feedback-curriculum-20261011.service', run_name='joint_pd_scratch_1k_20261011_163122',
                  fresh_weights=True, leg_control_mode='implicit_joint_position', training_log=str(output / 'training.log'))
    with pytest.raises(ValueError, match='unrelated'):
        watch.validate_scope(launch, output)


def test_scope_allows_only_owned_fresh_training():
    output = watch.ROOT / 'mine_model/results/joint_pd_scratch_test'
    launch = dict(unit='clt-joint-pd-scratch-20261011_163122.service', run_name='joint_pd_scratch_1k_20261011_163122',
                  fresh_weights=True, leg_control_mode='implicit_joint_position', training_log=str(output / 'training.log'))
    watch.validate_scope(launch, output)
    launch['fresh_weights'] = False
    import pytest
    with pytest.raises(ValueError, match='fresh'):
        watch.validate_scope(launch, output)


def test_resume_scope_requires_identified_checkpoint_and_hash(tmp_path, monkeypatch):
    import hashlib
    import pytest
    runs = tmp_path / 'runs'
    runs.mkdir()
    checkpoint = runs / 'model_999.pt'
    checkpoint.write_bytes(b'identified weights')
    monkeypatch.setattr(watch, 'RUNS', runs)
    output = watch.ROOT / 'mine_model/results/joint_pd_motion_20261011_230000'
    launch = dict(unit='clt-joint-pd-motion-20261011_230000.service',
                  run_name='joint_pd_motion_1k_20261011_230000', fresh_weights=False,
                  leg_control_mode='implicit_joint_position', training_log=str(output / 'training.log'),
                  source_checkpoint=str(checkpoint), source_iteration=999,
                  source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  iterations=2000, steps_per_env=48, standing_only_steps=48000, motion_ramp_steps=48000,
                  checkpoint_milestones=[1000, 1200, 1400, 1600, 1800, 1999])
    watch.validate_scope(launch, output)
    checkpoint.write_bytes(b'replaced weights')
    with pytest.raises(ValueError, match='changed'):
        watch.validate_scope(launch, output)


def test_final_checkpoint_and_curriculum_clock_are_not_hardcoded():
    import pytest
    launch = dict(fresh_weights=False, iterations=2000, steps_per_env=48,
                  standing_only_steps=48000, motion_ramp_steps=48000,
                  checkpoint_milestones=[1200, 1400, 1600, 1800, 1999])
    assert watch.checkpoint_milestones(launch)[-1] == 1999
    assert watch.curriculum_progress(launch, 999) == 0.
    assert watch.curriculum_progress(launch, 1200) == .2
    assert watch.curriculum_progress(launch, 1500) == .5
    assert watch.curriculum_progress(launch, 2500) == 1.
    launch['monitoring'] = dict(final_acceptance_requires_full_commands=True)
    assert watch.curriculum_progress(launch, 1999) == 1.
    launch['checkpoint_milestones'] = [1200, 1400, 1800]
    with pytest.raises(ValueError, match='final'):
        watch.checkpoint_milestones(launch)


def motion_entry(iteration, progress, ratio):
    conditions = {}
    for name, speed in (('forward_0p3', .3), ('forward_0p6', .6),
                        ('reverse_0p3', -.3), ('reverse_0p6', -.6)):
        conditions[name] = dict(speed=speed, survivors=3, trials=3,
                               metrics=[{'speed_mae_m_s': abs(speed) * ratio}] * 3)
    return dict(entry(iteration, 20., 3), curriculum_progress=progress, conditions=conditions)


def test_untrained_motion_at_early_curriculum_is_not_a_fault():
    history = [motion_entry(1000, 0., 1.), motion_entry(1200, .2, 1.), motion_entry(1400, .4, 1.)]
    assert watch.learning_anomaly(history) is None


def test_three_nonresponsive_motion_milestones_trigger_diagnosis():
    history = [motion_entry(1400, .4, 1.), motion_entry(1600, .6, .99), motion_entry(1800, .8, .98)]
    assert watch.learning_anomaly(history) == 'speed_command_response_stagnant_three_milestones'


def test_improving_motion_response_does_not_trigger_stagnation():
    history = [motion_entry(1400, .4, 1.), motion_entry(1600, .6, .93), motion_entry(1800, .8, .87)]
    assert watch.learning_anomaly(history) is None


def test_failed_replay_is_not_interpreted_as_steady_motion_error():
    history = [motion_entry(1400, .4, 1.), motion_entry(1600, .6, 1.), motion_entry(1800, .8, 1.)]
    history[-1]['conditions']['forward_0p3'].update(survivors=0, metrics=[])
    assert watch.learning_anomaly(history) is None


def test_yaw_command_blindness_also_triggers_diagnosis():
    history = [motion_entry(i, p, .1) for i, p in ((1400, .4), (1600, .6), (1800, .8))]
    for h in history:
        h['conditions'].update({name: dict(yaw=yaw, survivors=3, trials=3,
                                          metrics=[{'yaw_mae_rad_s': abs(yaw)}] * 3)
                                for name, yaw in (('spin_left', .3), ('spin_right', -.3))})
    assert watch.learning_anomaly(history) == 'yaw_command_response_stagnant_three_milestones'


def test_runtime_contract_accepts_resume_but_rejects_physical_changes(tmp_path):
    import pytest
    import yaml
    run, source = tmp_path / 'run', tmp_path / 'source'
    (run / 'params').mkdir(parents=True)
    (source / 'params').mkdir(parents=True)
    contract = dict(leg_control_mode='implicit_joint_position', fresh_weights=False,
                    leg_joint_gravity_feedforward=True,
                    joint_pd_total_effort_limit='PhysxDrivePerformanceEnvelopeAPI:angular',
                    leg_effort_limit=35.574, wheel_effort_limit=4.80944,
                    leg_joint_drive_stiffness=300., policy_dt=.02)
    (run / 'params/control_contract.yaml').write_text(yaml.safe_dump(contract))
    (source / 'params/control_contract.yaml').write_text(yaml.safe_dump(dict(contract, fresh_weights=True)))
    (run / 'params/precision_mode.yaml').write_text(yaml.safe_dump(
        dict(mode='full_policy', shared_actor_features_frozen=False)))
    watch.validate_contract(run, fresh_weights=False, source_checkpoint=source / 'model_999.pt')
    with pytest.raises(ValueError, match='runtime contract'):
        watch.validate_contract(run, fresh_weights=True)
    contract['leg_joint_drive_stiffness'] = 200.
    (run / 'params/control_contract.yaml').write_text(yaml.safe_dump(contract))
    with pytest.raises(ValueError, match='physical contract changed'):
        watch.validate_contract(run, fresh_weights=False, source_checkpoint=source / 'model_999.pt')


def test_runtime_contract_rejects_wheel_head_only_training(tmp_path):
    import pytest
    import yaml
    (tmp_path / 'params').mkdir()
    contract = dict(leg_control_mode='implicit_joint_position', fresh_weights=True,
                    leg_joint_gravity_feedforward=True,
                    joint_pd_total_effort_limit='PhysxDrivePerformanceEnvelopeAPI:angular',
                    leg_effort_limit=35.574, wheel_effort_limit=4.80944)
    (tmp_path / 'params/control_contract.yaml').write_text(yaml.safe_dump(contract))
    (tmp_path / 'params/precision_mode.yaml').write_text(yaml.safe_dump(
        dict(mode='wheel_head_only', shared_actor_features_frozen=True)))
    with pytest.raises(ValueError, match='all actor features'):
        watch.validate_contract(tmp_path)
