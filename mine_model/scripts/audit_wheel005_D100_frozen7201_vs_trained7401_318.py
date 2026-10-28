"""Read completed, matched D1 replays before/after200 ordinary PPO updates.

Only the frozen diagnostic's report filename is rebound in the original yaw
trace reader. No native simulation, optimizer, controller or gate is changed.
This318-case diagnosis is not a substitute for full735 and sustained reviews.
"""
from datetime import datetime
from pathlib import Path
import json

import numpy as np

import audit_paired_yaw_kernel_trace_budget as original
import audit_paired_retained735_tracking as retained
import plain27_sampler_metadata_adapter as adapter


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]
BASE = ROOT / 'mine_model/results'
TRAINING = BASE / 'joint_pd_optimization_20261024_034949_008168_04717aae_wheel005_source7201_D100_200'
REVIEW = BASE / 'joint_pd_wheel005_source7201_D100_review7401_20261024'
FROZEN = BASE / 'joint_pd_frozen_wheel005_7201_D100_diagnostic_20261024'
OUTPUT = TRAINING / 'sameD100_frozen7201_vs_trained7401_completed318_signed_yaw_readonly_20261024.json'
SOURCE_SHA = '3964b3a21906dad8160f08d8ca4e1d2310a6bcc2d179277729a63e8a92756785'
TARGET_SHA = 'c201737323d06f4e6465981717731078c829184e4f1fccb22ae0c6141cc82b12'
SCOPES = (
    ('small51', 'motion_scale0.25', 'damping100_small51', 'full_height_0.25/candidate', 51),
    ('fullspeed51', 'motion_scale1.0', 'damping100_fullspeed51', 'full_height_1.0/candidate', 51),
    ('low_turn108', 'combined_low', 'damping100_low_turn108', 'combined/candidate/low', 108),
    ('high_turn108', 'combined_high', 'damping100_high_turn108', 'combined/candidate/high', 108),
)


def frozen_row(row):
    metrics = row['metrics']
    return dict(case=row['case'], seed=row['seed'], survived=row['survived'],
        metrics=metrics, limits=row['limits'], passed=row['stage_constraints_passed'],
        nonzero_command_relative_error_percent={metric: 100. * metrics[metric] / abs(row['case'][command])
            if row['case'][command] else None for metric, command in retained.CHANNELS}
            if metrics is not None else None)


def summary(rows):
    result = dict(trials=len(rows), survivors=sum(r['survived'] for r in rows),
        undefined_failed_metrics=sum(r['metrics'] is None for r in rows),
        original_passed=sum(r['passed'] for r in rows),
        original_AND_literal10_passed=sum(r['passed'] and retained.literal(r) for r in rows))
    nonzero = [r for r in rows if r['case']['yaw'] and r['metrics'] is not None]
    result.update(nonzero_yaw_trials=len(nonzero), yaw_strict10_passed=sum(
        r['nonzero_command_relative_error_percent']['yaw_mae_rad_s'] < 10. for r in nonzero),
        worst_nonzero_yaw_percent=max(r['nonzero_command_relative_error_percent']['yaw_mae_rad_s']
            for r in nonzero), mean_nonzero_yaw_MAE_rad_s=float(np.mean([
                r['metrics']['yaw_mae_rad_s'] for r in nonzero])))
    return result


def run():
    assert not OUTPUT.exists()
    read, sha = original.read, original.sha
    prior_path = FROZEN / 'complete318_paired_readonly_20261024.json'
    raw_path = TRAINING / 'early_completed_original657_candidate7401_all7_raw_20261024.json'
    final_path = TRAINING / 'final_checkpoint_verification_readonly_20261019.json'
    prior, raw, final = map(read, (prior_path, raw_path, final_path))
    assert prior['verified'] and prior['completed_probe_trials'] == 318
    assert raw['verified'] and raw['total_trials'] == 657
    assert raw['all7_metrics_and_limits_raw_reproduction_max_error'] == 0.
    assert final['verified'] and final['actual_ppo_updates'] == 7401
    assert final['all17_Adam_step'] == 148020 and final['exact_actual_update_delta'] == 200
    assert final['source_checkpoint_sha256'] == SOURCE_SHA
    assert final['checkpoint_sha256'] == raw['checkpoint_sha256'] == TARGET_SHA
    assert sha(original.GATE) == original.GATE_SHA
    frozen_api = adapter.private_module(original.SCRIPT, '_sameD100_frozen_original_yaw_reader_318')
    evidence = adapter.retarget(frozen_api, ('process',), exact_strings={
        'motion025_full_height_diagnosis.json': 'metrics_diagnostic_only.json'})
    assert evidence[0]['declared_metadata_changes'] == {'declared_label_or_command_option_metadata': 2}
    combined_api = adapter.private_module(original.SCRIPT, '_sameD100_combined_original_yaw_reader_318')
    combined_evidence = adapter.retarget(combined_api, ('process',), exact_strings={
        'motion025_full_height_diagnosis.json': 'combined_turn_tracking_diagnosis.json'})
    assert combined_evidence[0]['declared_metadata_changes'] == {'declared_label_or_command_option_metadata': 2}
    pins = {str(p): sha(p) for p in (SCRIPT, original.SCRIPT, retained.SCRIPT,
        adapter.SCRIPT, prior_path, raw_path, final_path, original.GATE)}
    paired, summaries, aggregate = [], {}, {'frozen': [], 'trained': []}
    for scope, raw_scope, old_name, new_name, count in SCOPES:
        before_dir, after_dir = FROZEN / old_name, REVIEW / new_name
        before, bm, bc, bi, bp = frozen_api.process(before_dir)
        after_api = combined_api if raw_scope.startswith('combined_') else original
        after, am, ac, ai, ap = after_api.process(after_dir)
        assert before['total_trials'] == after['total_trials'] == count
        assert before['checkpoint_sha256'] == SOURCE_SHA and after['checkpoint_sha256'] == TARGET_SHA
        assert bm['cases'] == am['cases'] and bm['seeds'] == am['seeds'] and np.array_equal(bi, ai)
        cfg_differences = original.differences(bc, ac)
        assert not [r for r in cfg_differences if r[0] != '/log_dir'], cfg_differences
        assert bc['wheel_damping'] == ac['wheel_damping'] == 1.
        frozen_diagnosis = read(before_dir / 'metrics_diagnostic_only.json')
        old_rows = [frozen_row(r) for r in frozen_diagnosis['trials']]
        new_rows = [r for r in raw['trials'] if r['scope'] == raw_scope]
        assert len(old_rows) == len(new_rows) == count
        prior_summary = prior['results'][scope]['frozen100']
        sums = {'frozen': summary(old_rows), 'trained': summary(new_rows)}
        assert all(sums['frozen'][k] == prior_summary[k] for k in sums['frozen'] if k != 'mean_nonzero_yaw_MAE_rad_s')
        assert abs(sums['frozen']['mean_nonzero_yaw_MAE_rad_s'] - prior_summary['mean_nonzero_yaw_MAE_rad_s']) < 1.e-14
        assert prior['results'][scope]['original_numeric_report_trials_reproduced_exactly']
        summaries[scope] = dict(**sums, exact_saved_config_differences=cfg_differences,
            original_yaw_raw_reader_parity=dict(frozen=before['parity'], trained=after['parity']),
            initial_cases_seeds_and27_inputs_bitwise_paired=True)
        before_index = {(r['case']['name'], r['seed']): r for r in before['trials']}
        after_index = {(r['case']['name'], r['seed']): r for r in after['trials']}
        old_index = {(r['case']['name'], r['seed']): r for r in old_rows}
        new_index = {(r['case']['name'], r['seed']): r for r in new_rows}
        assert before_index.keys() == after_index.keys() == old_index.keys() == new_index.keys()
        for key, br in before_index.items():
            ar, old, new = after_index[key], old_index[key], new_index[key]
            assert br['case'] == ar['case'] == old['case'] == new['case']
            assert br['survived'] == old['survived'] and ar['survived'] == new['survived']
            for checked, recorded in ((br, old), (ar, new)):
                assert checked['original_stage_constraints_passed'] == recorded['passed']
                if recorded['metrics'] is None:
                    assert checked['metrics'] is None
                else:
                    assert abs(checked['metrics']['yaw_MAE_rad_s'] - recorded['metrics']['yaw_mae_rad_s']) < 1.e-8
            paired.append(dict(scope=scope, case=br['case'], seed=br['seed'],
                frozen=br, trained=ar, original_frozen=old, original_trained=new,
                frozen_original_AND_literal10=bool(old['passed'] and retained.literal(old)),
                trained_original_AND_literal10=bool(new['passed'] and retained.literal(new))))
        aggregate['frozen'].extend(old_rows)
        aggregate['trained'].extend(new_rows)
        pins.update(bp)
        pins.update(ap)
    assert len(paired) == 318
    assert all(sha(Path(path)) == digest for path, digest in pins.items())
    result = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        scope='Completed318 frozen D1 and trained D1 only; no nominal combined or retained interiors, dynamic or sustained replacement',
        source_actual_updates=7201, candidate_actual_updates=7401,
        frozen_checkpoint_sha256=SOURCE_SHA, trained_checkpoint_sha256=TARGET_SHA,
        same_D100_physics_commands_rewards_floors_and_initial27_inputs_verified=True,
        sole_new_learned_factor='200 ordinary full-policy PPO updates',
        summaries={k: summary(v) for k, v in aggregate.items()}, per_scope=summaries,
        strict_rescues=[dict(scope=r['scope'], case=r['case'], seed=r['seed']) for r in paired
            if not r['frozen_original_AND_literal10'] and r['trained_original_AND_literal10']],
        strict_regressions=[dict(scope=r['scope'], case=r['case'], seed=r['seed']) for r in paired
            if r['frozen_original_AND_literal10'] and not r['trained_original_AND_literal10']],
        original_gate_regressions=[dict(scope=r['scope'], case=r['case'], seed=r['seed']) for r in paired
            if r['original_frozen']['passed'] and not r['original_trained']['passed']],
        paired_trials=paired, source_pins=pins,
        original_process_only_report_filename_adapters=dict(frozen=evidence, combined=combined_evidence),
        prior_readonly_report_lookup_failure_preserved=str(BASE /
            'joint_pd_wheel005_source7201_D100_qualification_20261024/preserved_readonly318_combined_report_filename_failure_20261024'),
        limitations=['Signed residuals close algebraically; their MAEs do not add and these are not independent causal plant factors.',
            'This reuses the original seven-metric frozen diagnosis verified in prior318 and current raw657; only yaw/rewards are independently re-reproduced here.',
            'All735, original120, small120 and static/dynamic checks remain required.'],
        optimizer_executed=False, simulator_started=False, production_training_modified=False,
        next_parameter_or_source_selected=False, acceptance_changed=False, policy_accepted=False)
    with OUTPUT.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(verified=True, report=str(OUTPUT), summaries=result['summaries'],
        per_scope={k: {t: z for t, z in v.items() if t in ('frozen', 'trained')} for k, v in summaries.items()},
        strict_rescues=len(result['strict_rescues']), strict_regressions=len(result['strict_regressions']),
        original_gate_regressions=len(result['original_gate_regressions']), policy_accepted=False)))


if __name__ == '__main__':
    run()
