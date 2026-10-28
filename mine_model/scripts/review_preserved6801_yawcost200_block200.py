"""Original735/static810/120s review of the prepared one-field yaw-cost trial.

Only source clocks, owned paths and source reference metadata are adapted.
The original numerical readers, gates, windows and failed None values remain
unchanged. This worker cannot select or extend another PPO block.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path

from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import review_wheel125_training_block as core
import review_source6001_orientation060_block200 as middle_template
import review_source6201_yawcost20_block200 as previous
import verify_completed_plain27_joint_pd_trial as final_check
import joint_pd_preserved6801_yawcost200_runtime as runtime
import qualify_preserved6801_yawcost200_transfer as qualifier

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = qualifier.OUTPUT / 'post200_original_review_qualification.json'
OUTPUT = RESULTS / 'joint_pd_preserved6801_yawcost200_review7001_20261022'
STATIC = RESULTS / 'joint_pd_static_retention_preserved6801_yawcost200_20261022_source7001'
SOURCE_REVIEW = RESULTS / 'joint_pd_targeted_turn030_review6801_20261021'
_base = adapter.private_module(core.SCRIPT, '_yawcost200_original_all_scope_review')
_base.SCRIPT, _base.QUALIFICATION, _base.OUTPUT, _base.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_base.runtime, _base.final_check = runtime, final_check
BASE_ADAPTER = adapter.retarget(_base, ('run',),
    integers={'*': {6001: 7001, 120020: 140020}},
    strings={'source6001_wheel125': 'source7001_preserved6801_yawcost200', 'source6001': 'source7001'})
_middle = adapter.private_module(middle_template.SCRIPT, '_yawcost200_original_known54_review')
_middle.SCRIPT, _middle.QUALIFICATION, _middle.OUTPUT, _middle.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_middle.runtime, _middle.qualifier, _middle._api = runtime, qualifier, _base
_top = adapter.private_module(previous.SCRIPT, '_yawcost200_original_retained735_review')
_top.SCRIPT, _top.QUALIFICATION, _top.OUTPUT, _top.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
_top.runtime, _top.qualifier, _top._api, _top._base = runtime, qualifier, _middle, _base
TOP_ADAPTER = adapter.retarget(_top, ('run',),
    integers={'*': {6201: 6801, 6401: 7001, 128020: 140020}},
    strings={'candidate6401': 'candidate7001', 'source6401': 'source7001', 'source6201': 'source6801'})
run = _top.run


def read(path):
    return json.loads(Path(path).read_text())


def qualify():
    if QUALIFICATION.exists() or OUTPUT.exists() or STATIC.exists():
        raise ValueError('Preserve every previous qualification and native result')
    d = runtime.repair_ready()
    launch_path = qualifier.OUTPUT / 'training_launch.json'
    launch = read(launch_path)
    training = Path(launch['output'])
    plan = read(training / 'experiment_plan.json')
    frozen_path = qualifier.OUTPUT / 'native_reward_transfer_qualification.json'
    frozen = read(frozen_path)
    old = read(previous.QUALIFICATION)
    if (launch['source_checkpoint_sha256'] != runtime.ANCHOR_SHA
            or launch['source_actual_updates'] != 6801 or launch['target_actual_updates'] != 7001
            or launch['target_all17_Adam_step'] != 140020 or launch['planned_actual_updates'] != 200
            or plan['next_configuration_changes'] != runtime.DELTA or plan['profile'] != runtime.profile()
            or plan['only_reward_change'] != runtime.DELTA['rewards']
            or any(plan[k] for k in ('only_exploration_change', 'only_command_change', 'only_controller_change'))
            or plan['early_learning_inspection_actual_updates'] != [50, 100]
            or not frozen['physical_outputs_observations_actions_and_other_rewards_bitwise_equal']
            or frozen['only_reward_change'] != runtime.DELTA['rewards']):
        raise ValueError('Require exactly the qualified preserved6801 cost-only200 transfer')
    proof = copy.deepcopy(old)
    pins = {**old['source_pins'], **plan['source_files'], **frozen['source_data_sha256']}
    for scale in core.SCALES:
        path = SOURCE_REVIEW / ('full_height_' + str(scale)) / 'candidate/manifest.json'
        manifest = read(path)
        if manifest['cases'] != old['full_height_cases'][str(scale)] or manifest['seeds'] != old['full_seeds']:
            raise ValueError('Original full-height cases or paired seeds changed')
        pins[str(path)] = sha(path)
    for level, cases in old['combined_batches'].items():
        path = SOURCE_REVIEW / 'combined/candidate' / level / 'manifest.json'
        manifest = read(path)
        if manifest['cases'] != cases or manifest['seeds'] != old['combined_seeds']:
            raise ValueError('Original forward/reverse moving-turning coverage changed')
        pins[str(path)] = sha(path)
    for prefix, manifest_key, raw_key, count in (
            ('known_interior', 'required_retained_interior54_cases', 'source_posture_weakness_review', 18),
            ('retained_left', 'required_retained_left_interior24_cases', 'source_retained_left24_raw', 8)):
        manifest_path, raw_path = Path(d[manifest_key]), Path(d[raw_key])
        manifest, raw = read(manifest_path), read(raw_path)
        if (len(manifest['cases']) != count or not raw['verified']
                or raw['total_trials'] != count * 3
                or raw['all7_metrics_and_limits_raw_reproduction_max_error'] != 0.):
            raise ValueError('Retain both interior command sets and every original metric')
        proof.update({prefix + '_cases': manifest['cases'], prefix + '_seeds': manifest['seeds'],
            prefix + '_source_directory': str(manifest_path.parent), prefix + '_source_raw': str(raw_path)})
        pins.update({str(p): sha(p) for p in (manifest_path, raw_path)})
    if (core.sustained.cases_for(plan['training_config']) != old['sustained_cases']
            or core.static_steps.step_plan() != old['original_static_height_plan']):
        raise ValueError('Original sustained120s or full-height step coverage changed')
    for p in (SCRIPT, previous.SCRIPT, previous.QUALIFICATION, middle_template.SCRIPT, core.SCRIPT,
            adapter.SCRIPT, runtime.SCRIPT, qualifier.SCRIPT, final_check.SCRIPT, launch_path, frozen_path):
        pins[str(p)] = sha(p)
    proof.update(observed_at=datetime.now().astimezone().isoformat(), training_launch=str(launch_path),
        training=str(training), after_unit=launch['unit'], output=str(OUTPUT), static_output=str(STATIC),
        target_actual_updates=7001, target_all17_Adam_step=140020,
        source_pins=pins, current_source_frozen27_reward_only_transfer_qualified=True,
        complete_review_expected=str(OUTPUT / 'complete_with_retained_interior_reviews.json'),
        metadata_adapter=BASE_ADAPTER + TOP_ADAPTER, acceptance_changed=False, policy_accepted=False)
    proof.pop('current_source_frozen27_exploration_only_transfer_qualified', None)
    core.intact(pins)
    core.save(QUALIFICATION, proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-only', action='store_true')
    result = qualify() if parser.parse_args().qualify_only else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'actual_updates',
        'all_original_and_retained_reviews_complete', 'policy_accepted') if k in result}))
