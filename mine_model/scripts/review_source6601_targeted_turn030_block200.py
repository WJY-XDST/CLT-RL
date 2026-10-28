"""All original/retained frozen reviews for the exact6601 to6801 sampler trial."""
import ast
from datetime import datetime
import inspect
import json
from pathlib import Path
import re

from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import review_wheel125_training_block as core
import review_source6001_orientation060_block200 as known_template
import review_source6201_lin300_block200 as retained_template
import verify_completed_targeted_turn030_trial as final_check
import joint_pd_source6601_targeted_turn030_runtime as runtime
import qualify_source6601_targeted_turn030_transfer as qualifier

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = qualifier.OUTPUT / 'post200_original_review_qualification.json'
OUTPUT = RESULTS / 'joint_pd_targeted_turn030_review6801_20261021'
STATIC = RESULTS / 'joint_pd_static_retention_targeted_turn030_20261021_source6801'
SOURCE_REVIEW = RESULTS / 'joint_pd_yaw200_review6601_20261021'
SOURCE_QUALIFICATION = RESULTS / 'joint_pd_source6401_yaw200_qualification_20261021/post200_original_review_qualification.json'


def metadata_strings(function, clocks):
    values = {n.value for n in ast.walk(ast.parse(inspect.getsource(function)))
        if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    result = {}
    for value in values:
        changed = re.sub(r'(source|candidate)(6001|6201|6401)',
            lambda m: m.group(1) + str(clocks.get(int(m.group(2)), int(m.group(2)))), value)
        changed = changed.replace('_20261020', '_20261021')
        if value == '/rewards/tracking_lin_vel':
            changed = '/commands/' + runtime.sampler.CONFIG_KEY
        if value == 'joint_pd_verified_static_source6001_wheel125_20261020.json':
            changed = 'joint_pd_verified_static_source6801_targeted_turn030_20261021.json'
        if changed != value:
            result[value] = changed
    return result


_base = adapter.private_module(core.SCRIPT, '_targeted_turn030_original_all_scope_review')
_base.SCRIPT, _base.QUALIFICATION, _base.OUTPUT, _base.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_base.runtime, _base.final_check = runtime, final_check
BASE_ADAPTER = adapter.retarget(_base, ('run',),
    integers={'*': {6001: 6801, 120020: 136020}},
    exact_strings=metadata_strings(core.run, {6001: 6801}))
_known = adapter.private_module(known_template.SCRIPT, '_targeted_turn030_retained_known54_review')
_known.SCRIPT, _known.QUALIFICATION, _known.OUTPUT, _known.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_known.runtime, _known.qualifier, _known._api = runtime, qualifier, _base
_root = adapter.private_module(retained_template.SCRIPT, '_targeted_turn030_retained_left24_review')
_root.SCRIPT, _root.QUALIFICATION, _root.OUTPUT, _root.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_root.runtime, _root.qualifier, _root.final_check = runtime, qualifier, final_check
_root._base, _root._api = _base, _known
ROOT_ADAPTER = adapter.retarget(_root, ('run',),
    integers={'*': {6201: 6601, 6401: 6801, 128020: 136020}},
    exact_strings=metadata_strings(retained_template.run, {6201: 6601, 6401: 6801}))


def read(path):
    return json.loads(Path(path).read_text())


def qualify():
    if QUALIFICATION.exists() or OUTPUT.exists() or STATIC.exists():
        raise ValueError('Preserve original review proofs and data')
    decision = runtime.repair_ready()
    launch_path = qualifier.OUTPUT / 'training_launch.json'
    launch = read(launch_path)
    training = Path(launch['output'])
    plan = read(training / 'experiment_plan.json')
    frozen_path = qualifier.OUTPUT / 'native_sampler_transfer_qualification.json'
    frozen = read(frozen_path)
    old = read(SOURCE_QUALIFICATION)
    if (not frozen['qualified'] or not frozen['physical_outputs_observations_actions_and_all_rewards_bitwise_equal']
            or launch['source_checkpoint_sha256'] != runtime.ANCHOR_SHA
            or launch['source_actual_updates'] != 6601 or launch['target_actual_updates'] != 6801
            or launch['planned_actual_updates'] != 200 or launch['target_all17_Adam_step'] != 136020
            or plan['profile'] != runtime.profile() or plan['next_configuration_changes'] != runtime.DELTA
            or plan['early_learning_inspection_actual_updates'] != [50, 100]):
        raise ValueError('Require the qualified exact200 one-command-field trial')
    source, _, _ = runtime.anchor()
    fixture = SOURCE_REVIEW / 'full_height_1.0/candidate'
    cases = old['full_height_cases']['1.0']
    reader, captured, evidence = core.combined.reader(cases, old['full_seeds'])
    rebuilt = reader(fixture, source)
    original = read(fixture / 'motion025_full_height_diagnosis.json')
    if (len(captured) != 1 or rebuilt['total_trials'] != 51 or rebuilt['survivors'] != 51
            or rebuilt['resets'] or rebuilt['trials'] != original['trials']):
        raise ValueError('Original real51 numerical metrics or failed flags changed')
    proof = dict(old)
    proof.update(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        training_launch=str(launch_path), training=str(training), after_unit=launch['unit'],
        output=str(OUTPUT), static_output=str(STATIC), target_actual_updates=6801,
        target_all17_Adam_step=136020,
        known_interior_source_directory=str(SOURCE_REVIEW / 'known_interior/candidate'),
        known_interior_source_raw=str(SOURCE_REVIEW / 'known_interior/all54_raw_verified.json'),
        retained_left_source_directory=str(SOURCE_REVIEW / 'retained_left_interior/candidate'),
        retained_left_source_raw=str(SOURCE_REVIEW / 'retained_left_interior/all24_raw_verified.json'),
        complete_review_expected=str(OUTPUT / 'complete_with_retained_interior_reviews.json'),
        source6601_all_original_and_retained_reviews_complete=True,
        original_real51_metrics_limits_and_failed_flags_reproduced=True,
        original_numeric_reader_metadata_adapter=evidence,
        source_target_metadata_adapters=BASE_ADAPTER + ROOT_ADAPTER,
        exact_added_command_field_diff_metadata_normalized=True,
        acceptance_changed=False, policy_accepted=False)
    pins = {**old['source_pins'], **plan['source_files'], **decision['source_pins'], **frozen['source_data_sha256']}
    for p in (SCRIPT, core.SCRIPT, known_template.SCRIPT, retained_template.SCRIPT, adapter.SCRIPT,
            final_check.SCRIPT, final_check.original.SCRIPT, runtime.SCRIPT, qualifier.SCRIPT,
            launch_path, frozen_path, SOURCE_QUALIFICATION, fixture / 'manifest.json', fixture / 'trace.csv',
            SOURCE_REVIEW / 'known_interior/candidate/manifest.json',
            SOURCE_REVIEW / 'retained_left_interior/candidate/manifest.json'):
        pins[str(p)] = sha(p)
    proof['source_pins'] = pins
    core.intact(pins)
    core.save(QUALIFICATION, proof)
    return proof


def run():
    return _root.run()


if __name__ == '__main__':
    import argparse
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualify-only', action='store_true')
    args = cli.parse_args()
    result = qualify() if args.qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'actual_updates',
        'all_original_and_retained_reviews_complete', 'policy_accepted') if k in result}))
