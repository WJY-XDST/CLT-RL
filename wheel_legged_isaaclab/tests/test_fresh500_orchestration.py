"""CPU-only orchestration checks: no simulation, training process or gate waiver."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
import yaml

import train_joint_pd_fresh500 as experiment


def payload(updates=0, *, baseline=False):
    value = 10. if baseline else .01 + updates * .001
    state = {i: {'step': torch.tensor(float(96180 if baseline else updates * 20)),
                 'exp_avg': torch.full((1,), value)} for i in range(17)}
    return dict(iter=5751 if baseline else max(0, updates - 1),
        model_state_dict={'actor.0.weight': torch.full((2, 2), value),
            'actor.6.weight': torch.full((6, 64), value),
            'critic.0.weight': torch.full((2, 2), value), 'std': torch.full((6,), .2)},
        optimizer_state_dict={'state': state if updates or baseline else {},
                              'param_groups': [{'lr': .0003}]})


@pytest.fixture
def harness(tmp_path, monkeypatch):
    root, runs = tmp_path / 'project', tmp_path / 'runs'
    results = root / 'mine_model/results'
    results.mkdir(parents=True)
    runs.mkdir()
    output = results / 'joint_pd_optimization_unit_fresh500'
    baseline = runs / 'baseline/model_5751_interrupted.pt'
    baseline.parent.mkdir()
    torch.save(payload(baseline=True), baseline)
    asset = str(root / 'owner_calibrated.usda')
    cfg = dict(robot={'spawn': {'usd_path': asset}},
        commands={'ranges_height': list(experiment.SAFE_BASE_HEIGHT_RANGE_M)},
        rewards={'orientation': -60., 'base_height_target': .3})
    events, reports, verification_deltas = [], {}, []
    active = {'native': False}

    def digest(path):
        path = Path(path)
        return hashlib.sha256(path.read_bytes() if path.exists() else str(path).encode()).hexdigest()

    def native_run(label, updates, *, initial=False, start=0):
        run = runs / ('2026-10-15_00-00-00_' + label)
        params = run / 'params'
        params.mkdir(parents=True)
        clock = dict(offset_steps=276096, steps_per_update=48, adam_steps_per_update=20)
        (params / 'curriculum_clock.yaml').write_text(yaml.safe_dump(clock))
        (params / 'curriculum_start.yaml').write_text(yaml.safe_dump(dict(
            motion_progress=1., height_bounds_m=list(experiment.SAFE_BASE_HEIGHT_RANGE_M),
            initial_common_step_counter=276096 + start * 48,
            sampled_command_height_min_m=experiment.SAFE_BASE_HEIGHT_RANGE_M[0],
            sampled_command_height_max_m=experiment.SAFE_BASE_HEIGHT_RANGE_M[1])))
        checkpoint = run / f'model_{updates - 1}.pt'
        torch.save(payload(updates), checkpoint)
        if initial:
            torch.save(payload(), run / 'model_initial.pt')
        return checkpoint

    def run_child(command, log, environment, *, training=False, **kwargs):
        assert training and not active['native'], 'No second native job may overlap'
        assert '--reset_optimizer' not in command and '--reset_exploration_std' not in command
        active['native'] = True
        try:
            target = int(command[command.index('--target_iteration') + 1])
            label = command[command.index('--run_name') + 1]
            source = Path(command[command.index('--checkpoint_path') + 1]) if '--checkpoint_path' in command else None
            assert source != baseline, 'Fresh experiment must not load old policy weights'
            count = 0 if source is None else experiment.clocked_update_count(
                torch.load(source, map_location='cpu', weights_only=True),
                experiment.load_curriculum_clock(source), allow_initial=source.name == 'model_initial.pt')
            assert 0 <= count < target <= 500
            events.append(('train', count, target))
            native_run(label, target, initial=source is None, start=count)
        finally:
            active['native'] = False

    def evaluate(checkpoint, directory, stop_path):
        assert not active['native'], 'Release training resources before replay'
        directory.mkdir(parents=True, exist_ok=True)
        events.append(('replay', str(checkpoint), str(directory)))
        metrics = {key: .05 if key == 'height_mae_m' else 2. for key in experiment.METRICS}
        limits = {key: .03 if key == 'height_mae_m' else 1. for key in experiment.METRICS}
        conditions = [dict(name='case_' + str(index), speed=0., yaw=0., height=.3,
            trials=[dict(seed=seed, metrics=metrics, limits=limits) for seed in (83, 84, 85)])
            for index in range(9)]
        report = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            survivors=27, total_trials=27, resets=0, passed=False, conditions=conditions)
        reports[str(directory)] = report
        (directory / 'completed.json').write_text('{"completed": true}')
        return report

    def verify(source, checkpoint):
        previous = torch.load(source, map_location='cpu', weights_only=True)
        current = torch.load(checkpoint, map_location='cpu', weights_only=True)
        delta = int((experiment.optimizer_step(current) - experiment.optimizer_step(previous)) / 20)
        assert delta > 0
        verification_deltas.append(delta)
        return dict(actual_ppo_updates=delta, optimizer_reset=False, weights_reset=False)

    monkeypatch.setattr(experiment, 'ROOT', root)
    monkeypatch.setattr(experiment, 'RUNS', runs)
    monkeypatch.setattr(experiment, 'sha', digest)
    monkeypatch.setattr(experiment, 'validate_checkpoint_robot', lambda path: cfg)
    monkeypatch.setattr(experiment, 'qualified_wheel_mass_asset', lambda path: (asset, {'asset_chain_sha256': {}}))
    monkeypatch.setattr(experiment, 'guard_coefficient', lambda path: 10.)
    monkeypatch.setattr(experiment, 'height_coefficient', lambda path: .225)
    monkeypatch.setattr(experiment, 'validate_contract', lambda *args, **kwargs: None)
    monkeypatch.setattr(experiment, 'validate_continuation_config', lambda previous, current, commands: None)
    monkeypatch.setattr(experiment, 'verify', verify)
    monkeypatch.setattr(experiment, 'evaluate', evaluate)
    monkeypatch.setattr(experiment, 'assess', lambda directory: reports[str(directory)])
    monkeypatch.setattr(experiment, 'drift_summary', lambda directory: {'all_stationary_trials_survived': True})
    monkeypatch.setattr(experiment.runtime, 'run_child', run_child)
    # run() assigns these globals; restore them to avoid contaminating other CPU tests.
    monkeypatch.setattr(experiment.runtime, 'OUTPUT', experiment.runtime.OUTPUT)
    monkeypatch.setattr(experiment.runtime, 'anomaly', experiment.runtime.anomaly)
    return dict(root=root, runs=runs, output=output, baseline=baseline, events=events,
                digest=digest, native_run=native_run, verification_deltas=verification_deltas)


def test_exactly_500_new_updates_first_replay_200_and_never_train_old_branch(harness):
    h = harness
    old_digest = h['digest'](h['baseline'])
    experiment.run(h['baseline'], h['output'])
    jobs = [event for event in h['events'] if event[0] == 'train']
    replays = [event for event in h['events'] if event[0] == 'replay']
    assert jobs == [('train', 0, 200)] + [('train', start, start + 50) for start in range(200, 500, 50)]
    assert len(replays) == 8  # One read-only baseline, seven milestones at/after 200.
    assert h['verification_deltas'] == [50] * 6
    assert [event[0] for event in h['events'][:4]] == ['train', 'replay', 'replay', 'train']
    assert h['digest'](h['baseline']) == old_digest
    status = json.loads((h['output'] / 'status.json').read_text())
    assert status['state'] == 'comparison_500_updates_complete'
    assert status['actual_ppo_updates'] == status['live_ppo_updates'] == 500
    assert len(status['history']) == 7 and not status['training_active']
    assert status['policy_accepted'] is False and status['goal_remains_active'] is True


def test_partial_recovery_counts_saved_adam_not_500_additional_updates(harness):
    h = harness
    label = 'joint_pd_fresh500_' + h['output'].name + '_u0_preserved'
    h['native_run'](label, 4)
    experiment.run(h['baseline'], h['output'])
    jobs = [event for event in h['events'] if event[0] == 'train']
    assert jobs[0] == ('train', 4, 200)
    assert sum(target - start for _, start, target in jobs) == 496
    assert h['verification_deltas'] == [196] + [50] * 6


@pytest.mark.parametrize('saved_updates', [50, 100, 150])
def test_recovery_of_early_checkpoint_never_triggers_early_replay(harness, saved_updates):
    h = harness
    label = 'joint_pd_fresh500_' + h['output'].name + '_u0_preserved'
    h['native_run'](label, saved_updates)
    experiment.run(h['baseline'], h['output'])
    assert h['events'][0] == ('train', saved_updates, 200)
    replays = [event for event in h['events'] if event[0] == 'replay']
    assert not any('updates_0050' in event[2] or 'updates_0100' in event[2]
                   or 'updates_0150' in event[2] for event in replays)


def test_recover_saved_500th_update_runs_missing_replay_and_does_not_extend(harness):
    h = harness
    label = 'joint_pd_fresh500_' + h['output'].name + '_u450_preserved'
    h['native_run'](label, 500)
    experiment.run(h['baseline'], h['output'])
    assert not any(event[0] == 'train' for event in h['events'])
    assert len(h['events']) == 2 and all(event[0] == 'replay' for event in h['events'])
    status = json.loads((h['output'] / 'status.json').read_text())
    assert status['actual_ppo_updates'] == 500 and len(status['history']) == 1
    assert status['history'][0]['actual_ppo_updates'] == 500
    assert not status['policy_accepted'] and not status['all_errors_under_10_percent']
