"""Pair complete original 735-trial reports without selecting a policy.

Keep all original metrics and failures, and report literal nonzero 10%
separately from the original absolute-floor-aware gates. Compare all initial
27-dimensional native inputs before summarizing a completed candidate.
"""
import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
GATE = ROOT / 'mine_model/config/tracking_acceptance.json'
GATE_SHA = '08fbf1aa50cc450bd7170ec310dba9acba3f8bd7ceded330ccffe9e6015cb91d'
CHANNELS = (('speed_mae_m_s', 'speed'), ('yaw_mae_rad_s', 'yaw'), ('height_mae_m', 'height'))
FIXED_LIMITS = ('speed_mae_m_s', 'yaw_mae_rad_s', 'height_mae_m', 'leg_angle_mae_rad', 'body_p95_deg')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def identity(row):
    return row['scope'], row['case']['name'], row['seed']


def literal(row):
    if row['metrics'] is None:
        assert row['nonzero_command_relative_error_percent'] is None and not row['passed']
        return False
    for metric, command in CHANNELS:
        value = row['metrics'][metric]
        assert math.isfinite(value) and value >= 0.
        target = row['case'][command]
        percent = 100. * value / abs(target) if target else None
        recorded = row['nonzero_command_relative_error_percent'][metric]
        assert (percent is None) == (recorded is None)
        assert percent is None or abs(percent - recorded) <= 1.e-12
        if percent is not None and not percent < 10.:
            return False
    return True


def audit(request_path):
    request_path = Path(request_path).resolve()
    request = read(request_path)
    destination = Path(request['output']).resolve()
    assert not destination.exists() and destination.is_relative_to(ROOT / 'mine_model/results')
    assert sha(GATE) == GATE_SHA
    pins = {str(p): sha(p) for p in (SCRIPT, request_path, GATE)}
    data, scopes, finals = {}, {}, {}
    for label in ('source', 'candidate'):
        spec = request[label]
        final_path = Path(spec['final_verification'])
        final = read(final_path)
        assert final['verified'] and final['checkpoint_sha256'] == sha(final['checkpoint'])
        assert final['all17_Adam_step'] == final['actual_ppo_updates'] * 20
        assert final['exact_actual_update_delta'] == 200 and final['all200_original_RNG_unchanged']
        alignment_keys = [k for k in final if k.startswith('all200_all') and k.endswith('_alignment_max_errors')]
        assert len(alignment_keys) == 1 and not any(final[alignment_keys[0]].values())
        finals[label] = final
        pins.update({str(final_path): sha(final_path), final['checkpoint']: final['checkpoint_sha256']})
        rows = []
        assert len(spec['raw_reports']) == 3
        for path, count in zip(map(Path, spec['raw_reports']), (657, 54, 24)):
            report = read(path)
            assert report['verified'] and report['total_trials'] == len(report['trials']) == count
            assert report['actual_ppo_updates'] == final['actual_ppo_updates']
            assert report['checkpoint_sha256'] == final['checkpoint_sha256']
            assert report['all7_metrics_and_limits_raw_reproduction_max_error'] == 0.
            assert report['failed_trial_None_metrics_not_zeroed']
            assert report['source_pins'][str(GATE)] == GATE_SHA
            pins[str(path)] = sha(path)
            pins.update(report['source_pins'])
            rows.extend(report['trials'])
        data[label] = {identity(row): row for row in rows}
        assert len(data[label]) == 735
        entries = read(spec['original657_input_specification'])
        assert len(entries) == 10
        entries += [dict(scope='supplemental_interior18', directory=spec['known54_directory']),
                    dict(scope='retained_left_interior8', directory=spec['left24_directory'])]
        scopes[label] = {entry['scope']: Path(entry['directory']) for entry in entries}
        assert len(scopes[label]) == 12
        pins[spec['original657_input_specification']] = sha(spec['original657_input_specification'])
    assert finals['candidate']['actual_ppo_updates'] - finals['source']['actual_ppo_updates'] == 200
    assert data['source'].keys() == data['candidate'].keys()
    assert scopes['source'].keys() == scopes['candidate'].keys()
    initial_pairs = []
    for scope, source_directory in scopes['source'].items():
        directories = (source_directory, scopes['candidate'][scope])
        manifests = [read(directory / 'manifest.json') for directory in directories]
        for label, manifest in zip(('source', 'candidate'), manifests):
            assert manifest['checkpoint_sha256'] == finals[label]['checkpoint_sha256']
            assert manifest['seconds_per_case'] == 20. and not manifest['sampled_policy_actions']
            assert not manifest['weights_updated'] and not manifest['policy_input_remapping']
        assert manifests[0]['cases'] == manifests[1]['cases']
        assert manifests[0]['seeds'] == manifests[1]['seeds']
        initial = []
        for directory in directories:
            with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as archive:
                initial.append(archive['observations'][0].copy())
            for name in ('manifest.json', 'policy_inputs.npz', 'completed.json', 'resolved_env.json'):
                pins[str(directory / name)] = sha(directory / name)
            assert read(directory / 'completed.json')['completed']
        assert initial[0].shape[1] == 27 and np.array_equal(*initial)
        initial_pairs.append(dict(scope=scope, trials=len(initial[0]), initial27_bitwise_equal=True))
    assert sum(row['trials'] for row in initial_pairs) == 735
    fixed_limit_error = 0.
    for key, before in data['source'].items():
        after = data['candidate'][key]
        assert before['case'] == after['case']
        for metric in FIXED_LIMITS:
            fixed_limit_error = max(fixed_limit_error, abs(before['limits'][metric] - after['limits'][metric]))
        literal(before)
        literal(after)
    assert fixed_limit_error == 0.

    def summarize(rows):
        defined = [row for row in rows if row['metrics'] is not None]
        yaw = [row for row in defined if row['case']['yaw']]
        relative = lambda row, metric, command: 100. * row['metrics'][metric] / abs(row['case'][command])
        return dict(fixed_trials=len(rows), survivors=sum(row['survived'] for row in rows),
            undefined_failed_metrics=len(rows) - len(defined), original_passed=sum(row['passed'] for row in rows),
            original_AND_literal10_passed=sum(row['passed'] and literal(row) for row in rows),
            nonzero_yaw_trials=sum(bool(row['case']['yaw']) for row in rows),
            nonzero_yaw_strict10_passed=sum(relative(row, 'yaw_mae_rad_s', 'yaw') < 10. for row in yaw),
            nonzero_yaw_mean_MAE_rad_s=float(np.mean([row['metrics']['yaw_mae_rad_s'] for row in yaw])),
            worst_nonzero_yaw_percent=max(relative(row, 'yaw_mae_rad_s', 'yaw') for row in yaw),
            worst_nonzero_speed_percent=max(relative(row, 'speed_mae_m_s', 'speed') for row in defined if row['case']['speed']),
            worst_height_percent=max(relative(row, 'height_mae_m', 'height') for row in defined),
            worst_body_P95_deg=max(row['metrics']['body_p95_deg'] for row in defined))

    strict = lambda row: bool(row['passed'] and literal(row))
    rescues = [list(key) for key in data['source'] if not strict(data['source'][key]) and strict(data['candidate'][key])]
    regressions = [list(key) for key in data['source'] if strict(data['source'][key]) and not strict(data['candidate'][key])]
    assert all(sha(path) == digest for path, digest in pins.items()), 'Immutable raw evidence changed'
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        source_actual_updates=finals['source']['actual_ppo_updates'],
        candidate_actual_updates=finals['candidate']['actual_ppo_updates'],
        source_checkpoint_sha256=finals['source']['checkpoint_sha256'],
        candidate_checkpoint_sha256=finals['candidate']['checkpoint_sha256'],
        summaries={label: summarize(list(rows.values())) for label, rows in data.items()},
        original735_cases_seeds_commands_and_all_initial27_observations_bitwise_paired=True,
        initial_observation_pairs=initial_pairs, all7_current_metrics_raw_reproduction_error=0.,
        five_fixed_limit_pairing_max_error=fixed_limit_error,
        literal_rescues=rescues, literal_regressions=regressions, source_pins=pins,
        all_original_and_literal_failed_trials_preserved=True,
        interpretation='This comparison cannot select a source, accept a policy or extend PPO. Static810, phase810, signed-height and both sustained120 reviews still apply.',
        training_modified=False, optimizer_executed=False, simulator_started=False, acceptance_changed=False,
        can_select_source=False, can_extend_training=False, policy_accepted=False)
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.request)
    print(json.dumps(dict(verified=result['verified'], summaries=result['summaries'],
        literal_rescues=len(result['literal_rescues']), literal_regressions=len(result['literal_regressions']))))
