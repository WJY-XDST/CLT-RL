"""Retain every original full/static/travel/120s review for orientation060 actual6201.

Only source clocks, owned paths and checkpoint-profile metadata are rebound.
No native command, physical metric, strict limit, window or failed None value
is changed. This worker cannot accept a policy or extend or roll back PPO.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import review_wheel125_training_block as original
import verify_completed_plain27_joint_pd_trial as final_check
import joint_pd_source6001_orientation060_runtime as runtime
import qualify_source6001_orientation060_transfer as qualifier

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = qualifier.OUTPUT / 'post200_original_review_qualification.json'
OUTPUT = RESULTS / 'joint_pd_orientation060_review6201_20261020'
STATIC = RESULTS / 'joint_pd_static_retention_orientation060_20261020_source6201'
_api = adapter.private_module(original.SCRIPT, '_source6201_orientation060_original_all_scope_review')
_api.SCRIPT, _api.QUALIFICATION, _api.OUTPUT, _api.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_api.runtime, _api.final_check = runtime, final_check
ADAPTER_EVIDENCE = adapter.retarget(_api, ('run',),
    integers={'*': {6001: 6201, 120020: 124020}},
    strings={'source6001_wheel125': 'source6201_orientation060', 'source6001': 'source6201'})


def read(path):
    return json.loads(Path(path).read_text())


def qualify():
    launch_path = qualifier.OUTPUT / 'training_launch.json'
    launch = read(launch_path)
    training = Path(launch['output'])
    plan = read(training / 'experiment_plan.json')
    frozen_path = qualifier.OUTPUT / 'native_reward_transfer_qualification.json'
    frozen = read(frozen_path)
    old = read(original.QUALIFICATION)
    source_full = read(runtime.TRAINING / 'retained_six_motion_envelopes_review_20261020.json')
    if (QUALIFICATION.exists() or OUTPUT.exists() or STATIC.exists()
            or not frozen['qualified'] or not frozen['physical_outputs_observations_actions_and_other_rewards_bitwise_equal']
            or launch['source_checkpoint_sha256'] != runtime.ANCHOR_SHA
            or launch['source_actual_updates'] != 6001 or launch['target_actual_updates'] != 6201
            or launch['planned_actual_updates'] != 200 or launch['target_all17_Adam_step'] != 124020
            or plan['next_configuration_changes'] != runtime.DELTA
            or plan['profile'] != runtime.profile() or plan['action_map_or_controller_changed']
            or plan['early_learning_inspection_actual_updates'] != [50, 100]):
        raise ValueError('Require the owned exact200 reward-only source6001→6201 trial')
    pins = dict(plan['source_files'])
    pins.update(frozen['source_data_sha256'])
    for scale in original.SCALES:
        directory = Path(source_full['suites'][str(scale)]['output']) / 'candidate'
        manifest = read(directory / 'manifest.json')
        if (manifest['cases'] != old['full_height_cases'][str(scale)]
                or manifest['seeds'] != old['full_seeds'] or len(manifest['cases']) != 17):
            raise ValueError('Keep every original full-height case and reset seed')
        pins[str(directory / 'manifest.json')] = sha(directory / 'manifest.json')
    current_review = RESULTS / 'joint_pd_wheel125_review6001_20261020'
    for level, cases in old['combined_batches'].items():
        path = current_review / 'combined/candidate' / level / 'manifest.json'
        manifest = read(path)
        if manifest['cases'] != cases or manifest['seeds'] != old['combined_seeds']:
            raise ValueError('Keep every signed moving-turning case and paired seed')
        pins[str(path)] = sha(path)
    sustained_cases = original.sustained.cases_for(plan['training_config'])
    current_sustained = read(current_review / 'sustained120/candidate/manifest.json')
    if (sustained_cases != current_sustained['cases']
            or current_sustained['seeds'] != old['sustained_seeds']
            or old['original_static_height_plan'] != original.static_steps.step_plan()):
        raise ValueError('Preserve the complete full-speed120s and full-height step coverage')
    for p in (SCRIPT, original.SCRIPT, adapter.SCRIPT, runtime.SCRIPT, final_check.SCRIPT,
            original.QUALIFICATION, launch_path, frozen_path,
            runtime.TRAINING / 'retained_six_motion_envelopes_review_20261020.json',
            original.numeric.SCRIPT, original.static_raw.SCRIPT, original.errors.SCRIPT,
            original.sustained.SCRIPT, Path(original.static_steps.__file__),
            Path(original.static_template.__file__), Path(original.static_template.sustained.__file__),
            Path(original.numeric.acceptance.assessor.__file__),
            original.numeric.acceptance.CURRENT, original.numeric.acceptance.POLICY):
        pins[str(p)] = sha(p)
    decision = runtime.repair_ready()
    interior_manifest_path = Path(decision['required_retained_interior54_cases'])
    interior = read(interior_manifest_path)
    interior_raw_path = Path(decision['source_posture_weakness_review'])
    interior_raw = read(interior_raw_path)
    if (len(interior['cases']) != 18 or interior['seeds'] != [301483, 301484, 301485]
            or not interior_raw['verified'] or interior_raw['total_trials'] != 54
            or interior_raw['all7_metrics_and_limits_raw_reproduction_max_error'] != 0.):
        raise ValueError('Retain every known interior posture case and original strict gates')
    pins.update({str(p): sha(p) for p in (interior_manifest_path, interior_raw_path)})
    original.intact(pins)
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        training_launch=str(launch_path), training=str(training), after_unit=launch['unit'],
        output=str(OUTPUT), static_output=str(STATIC), target_actual_updates=6201,
        target_all17_Adam_step=124020, full_height_cases=old['full_height_cases'],
        full_seeds=old['full_seeds'], combined_batches=old['combined_batches'],
        combined_seeds=old['combined_seeds'], moving_plan=old['moving_plan'],
        moving_seeds=old['moving_seeds'], steering_plan=old['steering_plan'],
        steering_seeds=old['steering_seeds'],
        original_static_height_plan=old['original_static_height_plan'],
        sustained_cases=sustained_cases, sustained_seeds=old['sustained_seeds'],
        canonical27_plus_six51_plus_combined324_total=657,
        current_source_frozen27_reward_only_transfer_qualified=True,
        original_metrics_thresholds_absolute_floors_windows_and_failed_None_preserved=True,
        metadata_adapter=ADAPTER_EVIDENCE, source_pins=pins,
        acceptance_changed=False, policy_accepted=False,
        known_interior_cases=interior['cases'], known_interior_seeds=interior['seeds'],
        known_interior_source_directory=str(interior_manifest_path.parent),
        known_interior_source_raw=str(interior_raw_path),
        all_known_interior54_required_in_addition_to_original657=True,
        can_extend_training=False, can_select_or_rollback=False)
    original.save(QUALIFICATION, result)
    return result


def run():
    result = _api.run()
    proof = read(QUALIFICATION)
    original.intact(result['source_pins'])
    source = Path(result['checkpoint'])
    directory = _api.native(source, OUTPUT / 'known_interior/candidate', 20.,
        proof['known_interior_seeds'], proof['known_interior_cases'])
    reader, _, _ = original.combined.reader(proof['known_interior_cases'], proof['known_interior_seeds'])
    report = reader(directory, source)
    report_path = directory / 'original_metric_posture_diagnosis.json'
    original.save(report_path, report)
    import numpy as np
    with np.load(directory / 'policy_inputs.npz') as a, np.load(
            Path(proof['known_interior_source_directory']) / 'policy_inputs.npz') as b:
        if not np.array_equal(a['observations'][0], b['observations'][0]):
            raise ValueError('Known interior paired initial policy inputs changed')
    specification = OUTPUT / 'known_interior/input_specification.json'
    original.save(specification, [dict(scope='supplemental_interior18',kind='fixed_command_probe',
        directory=str(directory),report=str(report_path))])
    verified_path = OUTPUT / 'known_interior/all54_raw_verified.json'
    verified = original.errors.audit(Path(proof['training']) / 'final_checkpoint_verification_readonly_20261019.json',
        specification, verified_path)
    result.update(known_interior54_raw_verified=str(verified_path),
        known_interior54_survivors=verified['total_survivors'],
        known_interior54_original_passed=verified['total_passed'],
        known_interior54_initial_observations_bitwise_paired=True,
        all_original_and_known_interior_review_data_complete=True,
        policy_accepted=False)
    original.intact(result['source_pins'])
    original.save(OUTPUT / 'complete_with_known_interior_review.json',result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true')
    args = parser.parse_args()
    result = qualify() if args.qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'actual_updates',
        'all_required_review_data_complete', 'policy_accepted') if k in result}))
