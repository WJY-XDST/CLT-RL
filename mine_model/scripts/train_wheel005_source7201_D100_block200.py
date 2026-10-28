"""One admitted ordinary200 block from complete7201 state, driveD100 only.

The original runner,4000 Adam minibatches, native observer and finalizer are
retained. Source metadata point at current7201 evidence, not an older branch.
Full post200 review qualification is mandatory before production starts.
"""
import argparse
import copy
import json
from pathlib import Path

import train_AF7001_wheelD075_wheel005_block200 as original
import joint_pd_wheel005_source7201_D100_runtime as runtime
import verify_wheel005_source7201_D100_completed200 as final_check
import plain27_sampler_metadata_adapter as adapter
from auto_train_mine import write_json


SCRIPT = Path(__file__).resolve()
QUALIFICATION = runtime.QUALIFICATION / 'post200_complete_qualification.json'
api = adapter.private_module(original.SCRIPT, '_source7201_D100_original200_controller')
api.SCRIPT, api.runtime, api.final_check, api.QUALIFICATION = SCRIPT, runtime, final_check, QUALIFICATION
METADATA = adapter.retarget(api, ('prepare',),
    integers={'*': {7001: 7201, 7201: 7401, 140020: 144020, 144020: 148020}},
    exact_strings={
        'plain_wheelD075_wheel005_n': 'plain_wheelD100_wheel005_n',
        '_u7001': '_u7201',
        'review_AF7001_wheelD075_wheel005_block200.py': 'review_wheel005_source7201_D100_block200.py',
        'recover_AF7001_wheelD075_completed200_boundary.py': 'verify_wheel005_source7201_D100_completed200.py'})
METADATA += adapter.retarget(api, ('run',), exact_strings={
    'ordinary_joint_pd_declared_wheelD075_future_wheel005':
        'ordinary_joint_pd_source7201_D100_unchanged_wheel005'})
_prepare = api.prepare


def prepare(output, previous):
    plan, expected = _prepare(output, previous)
    source = runtime.SOURCE_REPORT.parent
    review = runtime.RESULTS / 'joint_pd_AF7001_D075_wheel005_review7201_20261024'
    static = runtime.RESULTS / 'joint_pd_static_retention_AF7001_D075_wheel005_source7201_20261024'
    signed = runtime.RESULTS / 'joint_pd_AF7001_D075_wheel005_signed_height_20261024'
    small = runtime.RESULTS / 'joint_pd_AF7001_D075_wheel005_small_yaw120_20261024'
    paired_path = source / 'paired_sourceAF7001_candidate_D075_wheel005_7201_all735_readonly_20261024.json'
    paired = json.loads(paired_path.read_text())
    summary = paired['summaries']['candidate']
    assert paired['verified'] and paired['candidate_checkpoint_sha256'] == runtime.ANCHOR_SHA
    assert summary['original_passed'] == summary['survivors'] == 735
    stable_keys = {'source_checkpoint', 'source_sha256', 'source_actual_ppo_updates',
        'source_Adam_step', 'source_Adam_states', 'source_optimizer_LR', 'source_std',
        'source_replay', 'source_files', 'source_model200_reloaded',
        'source_model_and_optimizer_reloaded_from_original200', 'source_state_selection'}
    current = dict(
        source_final_verification=str(runtime.SOURCE_REPORT),
        source_full_review=str(review / 'complete_all_scopes_AF7001_wheelD075.json'),
        source_all735_paired_report=str(paired_path), source_all735_summary=summary,
        source_all735_original_passed=735, source_all735_literal_both_passed=678,
        source_all735_survivors=735,
        source_all_errors=str(source / 'all_errors_all657_source7201_readonly_20261022.json'),
        source_static_retention_report=str(static / 'static_retention.json'),
        source_original_sustained120=str(source / 'independent120_raw_verified_paired_source7001_readonly_20261022.json'),
        source_fullspeed_height_transition_review=str(signed / 'complete_paired_fullspeed_height_transitions.json'),
        source_small_yaw_sustained120_complete=str(small / 'complete_paired_small_yaw_sustained120.json'),
        source_is_completely_reviewed_unaccepted_checkpoint=True)
    legacy = {key: copy.deepcopy(value) for key, value in plan.items()
              if key.startswith('source_') and key not in stable_keys and key not in current}
    for key in legacy:
        plan.pop(key)
    if 'inherited_legacy_source_metadata' not in plan:
        plan['inherited_legacy_source_metadata'] = legacy
    plan.update(current)
    plan.update(source_state_selection='complete3964_nominal7201_no_reset',
        previous_completed_profile_is_lineage_not_current_recipe=True,
        current_only_new_factor=runtime.DELTA['controller'],
        only_exploration_change={}, future_six_std_floors=[.01, .01, .005, .01, .01, .005],
        complete_original_controller_metadata_adapters=METADATA,
        all_post200_scopes_preflight_required_before_training=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    write_json(Path(output) / 'experiment_plan.json', plan)
    return plan, expected


api.prepare = prepare


def run(output, previous, unit):
    return api.run(output, previous, unit)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--previous', type=Path, required=True)
    cli.add_argument('--output', type=Path, required=True)
    cli.add_argument('--unit')
    cli.add_argument('--prepare-only', action='store_true')
    args = cli.parse_args()
    if args.prepare_only:
        prepare(args.output, args.previous)
    else:
        run(args.output, args.previous, args.unit)
