"""Reproduce one completed735 linear bonus and audit global width candidates.

All candidates are constant across every command. This only evaluates reward
values and state partial derivatives on fixed recordings, not policy gradients,
new dynamics, convergence, acceptance, or a selected training configuration.
"""
import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from audit_frozen_yaw_reward_sensitivity import quality_from_row


ROOT = Path('/home/aaa/studyRL/src/CLT-RL')


def read(p): return json.loads(Path(p).read_text())


def sha(p):
    digest = hashlib.sha256()
    with Path(p).open('rb') as stream:
        while block := stream.read(1024 * 1024): digest.update(block)
    return digest.hexdigest()


def accumulator(variances):
    return dict(samples=0, error_sum=0., candidates={str(v): dict(
        bonus_sum=0., partial_abs_sum=0., clipped_count=0) for v in variances})


def accumulate(state, error, weight, clip, variances):
    state['samples'] += 1
    state['error_sum'] += abs(error)
    for variance in variances:
        raw = weight * math.exp(-error * error / variance)
        item = state['candidates'][str(variance)]
        item['bonus_sum'] += min(clip, raw)
        item['clipped_count'] += raw >= clip
        item['partial_abs_sum'] += 0. if raw >= clip else 2. * abs(error) * raw / variance


def finish(state):
    n = state['samples']
    assert n > 0
    state['speed_MAE_m_s'] = state.pop('error_sum') / n
    for item in state['candidates'].values():
        item['mean_bonus_per_s'] = item.pop('bonus_sum') / n
        item['mean_absolute_partial_d_bonus_d_measured_speed'] = item.pop('partial_abs_sum') / n
        item['positive_clip_fraction'] = item.pop('clipped_count') / n


def run(matrix, output, variances, native_output):
    assert not output.exists() and len(set(variances)) == len(variances)
    assert all(math.isfinite(v) and v > 0 for v in variances)
    full_path, scope_path = matrix / 'complete_frozen_command_matrix.json', matrix / 'qualification.json'
    full, scope = read(full_path), read(scope_path)
    assert full['complete'] and full['all_failures_retained']
    assert len(full['all_trials']) == scope['trial_count'] == 735
    assert scope['unique_numeric_commands'] == 219
    scores = {(v['case']['name'], v['seed']): v for v in full['all_trials']}
    assert len(scores) == 735
    env = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/tasks/direct/wheel_legged_vmc_flat/wheel_legged_vmc_flat_env.py'
    tree = ast.parse(env.read_text())
    nodes = [v for v in ast.walk(tree) if isinstance(v, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == 'r_track_lin' for t in v.targets)]
    expected = ast.parse('torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)', mode='eval').body
    assert len(nodes) == 1 and ast.dump(nodes[0].value) == ast.dump(expected)
    pins = {str(p): sha(p) for p in [Path(__file__), env, full_path, scope_path,
                                    Path(quality_from_row.__code__.co_filename),
                                    env.parents[3] / 'reward_signals.py',
                                    ROOT / 'mine_model/config/tracking_acceptance.json']}
    rows, max_reward_error, contract = [], 0., None
    for job in scope['batches']:
        folder = Path(job['replay'])
        manifest, cfg, completed = [read(folder / n) for n in ['manifest.json', 'resolved_env.json', 'completed.json']]
        assert completed['completed'] and completed['resets'] == 0
        assert manifest['checkpoint_sha256'] == full['checkpoint_sha256']
        assert not manifest['sampled_policy_actions'] and not manifest['weights_updated']
        reward, dt, seconds = cfg['rewards'], manifest['policy_dt'], manifest['seconds_per_case']
        evidence = manifest['tracking_goal_feedback_criteria']
        assert sha(evidence['path']) == evidence['sha256']
        criteria = evidence['criteria']
        assert criteria['relative_error'] == .1
        current = dict(weight=reward['tracking_lin_vel'], variance=reward['tracking_sigma'],
                       clip=reward['clip_single_reward'], quality_gate_floor=reward['tracking_task_quality_gate_floor'],
                       existing_quality_guard_criteria=criteria, dt=dt, seconds=seconds)
        assert .05 <= current['quality_gate_floor'] <= 1. and dt == .02 and seconds == 20.
        assert current['variance'] in variances
        if contract is None: contract = current
        assert current == contract
        entries = [dict(case=case, seed=seed, strict10_passed=scores[(case['name'], seed)]['original_AND_literal10_passed'],
                        nonzero_relative_percent=scores[(case['name'], seed)]['nonzero_relative_percent'],
                        windows={name: accumulator(variances) for name in ['full20s', 'first2s', 'tail3s']})
                   for case in manifest['cases'] for seed in manifest['seeds']]
        nrows, width = 0, len(entries)
        with (folder / 'trace.csv').open(newline='') as stream:
            reader = csv.reader(stream)
            index = {name: n for n, name in enumerate(next(reader))}
            for row in reader:
                number = nrows % width
                entry = entries[number]
                nrows += 1
                assert int(float(row[index['case_id']])) == number // len(manifest['seeds'])
                assert int(float(row[index['seed']])) == entry['seed']
                t = float(row[index['time_s']])
                error = float(row[index['speed_m_s']]) - float(row[index['speed_cmd_m_s']])
                actual = float(row[index['reward_track_lin_vel']])
                pose = {key: float(row[index[key]]) for key in
                        ['height_m', 'height_cmd_m', 'pitch_rad', 'roll_rad']}
                quality = (1. if contract['quality_gate_floor'] == 1. else
                           quality_from_row(pose, criteria, contract['quality_gate_floor']))
                effective_weight = contract['weight'] * quality
                assert all(math.isfinite(v) for v in [t, error, actual])
                predicted = min(contract['clip'], effective_weight * math.exp(-error * error / contract['variance'])) * dt
                max_reward_error = max(max_reward_error, abs(predicted - actual))
                windows = ['full20s'] + (['first2s'] if t <= 2. + 1.e-9 else []) + (['tail3s'] if t >= seconds - 3. else [])
                for name in windows:
                    accumulate(entry['windows'][name], error, effective_weight, contract['clip'], variances)
        assert nrows == width * round(seconds / dt)
        for entry in entries:
            for state in entry['windows'].values(): finish(state)
            expected_MAE = scores[(entry['case']['name'], entry['seed'])]['metrics']['speed_mae_m_s']
            assert abs(entry['windows']['tail3s']['speed_MAE_m_s'] - expected_MAE) < 1.e-12
        rows += entries
        for name in ['manifest.json', 'resolved_env.json', 'completed.json', 'trace.csv']:
            p = folder / name
            pins[str(p)] = sha(p)
    assert len(rows) == 735 and max_reward_error < 2.e-6

    def summarize(chosen):
        assert chosen
        return {name: dict(trials=len(chosen), speed_MAE_m_s=statistics.mean(r['windows'][name]['speed_MAE_m_s'] for r in chosen),
                           candidates={str(v): {k: statistics.mean(r['windows'][name]['candidates'][str(v)][k] for r in chosen)
                                                for k in chosen[0]['windows'][name]['candidates'][str(v)]}
                                       for v in variances}) for name in ['full20s', 'first2s', 'tail3s']}

    native = None
    if native_output is not None:
        import numpy as np
        proof_path = native_output / 'final_checkpoint_verification_readonly.json'
        proof = read(proof_path)
        assert proof['verified'] and proof['all50_native_updates_verified']
        assert proof['checkpoint_sha256'] == full['checkpoint_sha256']
        assert read(native_output / 'original11_health50_complete.json')['qualified']
        files = sorted((native_output / 'native_learning_signal').glob('rollout_*.npz'))
        assert len(files) == 50
        state = accumulator(variances)
        for p in files:
            with np.load(p, allow_pickle=False) as z:
                error = (z['native_speed'] - z['native_commands'][..., 0]).astype(np.float64)
                assert error.shape == (48, 32) and np.isfinite(error).all()
                gravity = z['native_gravity'].astype(np.float64)
                height = z['native_height'].astype(np.float64)
                command_height = z['native_commands'][..., 2].astype(np.float64)
                height_ratio = np.abs(height-command_height) / (
                    criteria['relative_error'] * np.maximum(np.abs(command_height), .001))
                pitch = np.abs(np.arcsin(np.clip(gravity[..., 0], -1., 1.)))
                roll = np.abs(np.arctan2(-gravity[..., 1], -gravity[..., 2]))
                body_ratio = np.maximum(pitch, roll) / math.radians(criteria['body_p95_deg'])
                floor = contract['quality_gate_floor']
                quality = floor + (1.-floor) / (
                    (1.+np.maximum(height_ratio-.5, 0.)**2) * (1.+np.maximum(body_ratio-.5, 0.)**2))
                state['samples'] += error.size
                state['error_sum'] += float(np.abs(error).sum())
                for variance in variances:
                    raw = contract['weight'] * quality * np.exp(-error * error / variance)
                    item = state['candidates'][str(variance)]
                    item['bonus_sum'] += float(np.minimum(contract['clip'], raw).sum())
                    item['clipped_count'] += int((raw >= contract['clip']).sum())
                    item['partial_abs_sum'] += float(np.where(raw >= contract['clip'], 0., 2. * np.abs(error) * raw / variance).sum())
            pins[str(p)] = sha(p)
        finish(state)
        native = dict(cached_environments=32, native_population=8192, completed_updates=50,
                      recorded_samples=76800, all_cached_samples_retained=state)
        pins[str(proof_path)] = sha(proof_path)
    failed = [r for r in rows if r['nonzero_relative_percent']['speed'] is not None and r['nonzero_relative_percent']['speed'] >= 10.]
    result = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
                  checkpoint_sha256=full['checkpoint_sha256'], original_trials=735, numeric_commands=219,
                  actual_linear_bonus_contract=contract,
                  exact_original_bonus_reproduction_max_abs_error_per_step=max_reward_error,
                  uniform_candidate_variances=variances,
                  equivalent_gaussian_width_sqrt_variance_m_s=[math.sqrt(v) for v in variances],
                  summary_all735=summarize(rows), summary_nonzero_speed=summarize([r for r in rows if r['case']['speed'] != 0]),
                  summary_zero_speed=summarize([r for r in rows if r['case']['speed'] == 0]),
                  summary_existing_speed_failures=summarize(failed), cached32_completed_training=native,
                  all_trials=rows, source_pins=pins,
                  limitations=[
                      'The actual current clip is read from saved configuration. A historical clip1 claim must not be applied to these recordings; the maximum positive weight can be lower than the actual clip.',
                      'Each candidate is one common constant for every height, velocity, yaw, sign and zero command. Failure groups describe recorded data only and do not select a training rule.',
                      'These are scalar reward values and partial derivatives with respect to measured speed on fixed states. They omit the dynamics and policy score/Jacobian and are not PPO gradients or predictions of learning.',
                      'Frozen735 uses its original20s deterministic replay and reset. Cached32 native data contains actual training transients but is not the complete8192 population.',
                      'Narrower positive kernels can reduce feedback for large errors and change return distributions. Negative costs and every stability/acceptance requirement remain unchanged in this arithmetic comparison.'
                  ], parameter_selected=False, training_modified=False, simulator_started=False,
                  policy_evaluated=False, optimizer_executed=False, acceptance_changed=False, policy_accepted=False)
    assert all(sha(p) == digest for p, digest in pins.items())
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ['verified', 'actual_linear_bonus_contract',
                     'exact_original_bonus_reproduction_max_abs_error_per_step', 'summary_all735',
                     'summary_existing_speed_failures', 'cached32_completed_training', 'parameter_selected']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variances', type=float, nargs='+', required=True)
    parser.add_argument('--native-output', type=Path)
    a = parser.parse_args()
    run(a.matrix.resolve(), a.output.resolve(), a.variances,
        a.native_output.resolve() if a.native_output else None)
