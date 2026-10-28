"""Complete diagnostics/replay after PPO finished but source metadata was stale.

Retain the training plan and completed optimizer updates verbatim. Record a
fresh canonical replay of the actual immediate source, use a separate corrected
diagnostic view, and resume only the original numerical reviews. No PPO launch
is available in this recovery path.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import joint_pd_calibrated_common_noise_runtime as runtime
import review_frozen_joint_pd_command_matrix as matrix
import review_saved_joint_pd_segment as segment
import train_saved_joint_pd_review50 as workflow


SCRIPT = Path(__file__).resolve()
common_train_wrappers = runtime.profile_for_count.__globals__['train']


def run(recovery_path):
    recovery = matrix.read(recovery_path)
    original_request = Path(recovery['original_request'])
    request = matrix.read(original_request)
    runtime.configure(request['recipe'])
    matrix.verify_pins(request['source_pins'])
    owner = Path(request['output'])
    native = matrix.read(owner / 'native_learning_signal/completed_external.json')
    final_path = owner / 'final_checkpoint_verification_readonly.json'
    final = matrix.read(final_path)
    assert final['verified'] and final['all50_native_updates_verified']
    assert native['actual_observed_updates'] == 50 and native['PPO_updates_repeated'] == 0
    assert all(value == 0. for value in native['all_environment_alignment_max_errors'].values())
    assert final['actual_ppo_updates'] == request['target_actual_updates'] == 7526
    checkpoint = Path(final['checkpoint'])
    assert final['checkpoint_sha256'] == native['checkpoint_sha256'] == runtime.sha(checkpoint)
    assert final['all17_Adam_step'] == 150520
    saved, _ = runtime.require_source(checkpoint)
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert {int(s['step']) for s in saved['optimizer_state_dict']['state'].values()} == {150520}
    immutable_paths = [checkpoint, owner / 'experiment_plan.json', owner / 'training.log',
        owner / 'native_learning_signal/completed_external.json', final_path]
    immutable = {str(path): runtime.sha(path) for path in immutable_paths}
    assert not (owner / 'independent_review189').exists()
    canonical = Path(recovery['canonical_source_replay'])
    assert not canonical.exists(), 'Inspect an existing source replay instead of repeating it'
    source = Path(request['source_checkpoint'])
    assert runtime.saved_updates(source) == request['source_actual_updates'] == 7476
    source_cases = matrix.read(Path(request['source_mapping_replay']) / 'manifest.json')['cases']
    assert len(source_cases) == 9
    case_file = recovery_path.parent / 'immediate_source7476_canonical_commands.json'
    assert not case_file.exists()
    matrix.write(case_file, source_cases)
    argv = runtime.replay_command(source, canonical)
    argv[argv.index('--seconds') + 1] = '20'
    at = argv.index('--seeds') + 1
    argv[at:at + 3] = ['83', '84', '85']
    argv += ['--command-probe-json', str(case_file)]
    assert argv[2] == 'replay'
    matrix.write(recovery_path.parent / 'canonical_source_replay_command.json', dict(
        command=argv, source_checkpoint=str(source), source_checkpoint_sha256=runtime.sha(source),
        optimizer_executed=False, no_PPO_repeated=True, policy_accepted=False))
    subprocess.run(argv, cwd=matrix.ROOT, check=True)
    manifest = matrix.read(canonical / 'manifest.json')
    completed = matrix.read(canonical / 'completed.json')
    assert Path(manifest['checkpoint']) == source and manifest['checkpoint_sha256'] == runtime.sha(source)
    assert manifest['cases'] == source_cases and manifest['seeds'] == [83, 84, 85]
    assert completed['completed'] and completed['samples'] == 27000 and completed['resets'] == 0
    assert not manifest['weights_updated'] and not manifest['sampled_policy_actions']

    old_view = owner.with_name(owner.name + '_diagnostic_view')
    view = owner.with_name(owner.name + '_diagnostic_source7476_review_v2')
    assert not view.exists()
    view.mkdir()
    old_plan = matrix.read(old_view / 'experiment_plan.json')
    plan = copy.deepcopy(old_plan)
    plan['source_replay'] = str(canonical)
    assert {key for key in plan if plan[key] != old_plan[key]} == {'source_replay'}
    matrix.write(view / 'experiment_plan.json', plan)
    for name in ('native_learning_signal', 'training.log', 'resume_verification.json',
                 'final_checkpoint_verification_readonly.json', 'review_interval_runtime.json'):
        (view / name).symlink_to(owner / name, target_is_directory=(owner / name).is_dir())
    reused = []
    for path in old_view.glob('*_prefix_0050_readonly.json'):
        (view / path.name).symlink_to(path)
        reused.append(dict(path=str(path), sha256=runtime.sha(path)))
    assert len(reused) == 9, 'Only the nine completed readers may be reused'
    matrix.write(view / 'source_replay_metadata_correction.json', dict(
        verified=True, original_training_plan_unchanged=str(owner / 'experiment_plan.json'),
        original_training_plan_sha256=runtime.sha(owner / 'experiment_plan.json'),
        original_diagnostic_plan=str(old_view / 'experiment_plan.json'),
        corrected_plan=str(view / 'experiment_plan.json'), only_changed_field='source_replay',
        previous_source_replay=old_plan['source_replay'], actual_immediate_source_replay=str(canonical),
        actual_immediate_source_sha256=runtime.sha(source),
        nine_completed_numerical_reports_reused=reused, training_modified=False,
        optimizer_executed=False, policy_accepted=False))

    old_run, old_view_function = workflow.run, segment.diagnostic_view

    def corrected_view(output):
        assert Path(output) == owner
        return view

    def resume_reviews(path):
        assert Path(path) == original_request
        workflow.health50(owner)
        workflow.review(owner, request, checkpoint)
        matrix.verify_pins(request['source_pins'])
        matrix.verify_pins(immutable)
        matrix.write(owner / 'status.json', dict(
            state='complete50_independently_reviewed_agent_decision_required',
            actual_ppo_updates=7526, checkpoint=str(checkpoint), training_active=False,
            goal_remains_active=True, policy_accepted=False,
            reviews_completed_by_recovery_service=True, PPO_updates_repeated=0,
            corrected_diagnostic_view=str(view)))

    workflow.run, segment.diagnostic_view = resume_reviews, corrected_view
    try:
        # Install only the previously qualified read-only metadata and routing
        # wrappers. Their controller dispatch is replaced above by review only.
        common_train_wrappers(original_request)
    finally:
        workflow.run, segment.diagnostic_view = old_run, old_view_function
    matrix.verify_pins(immutable)
    matrix.write(recovery_path.parent / 'review_recovery_completed.json', dict(
        verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint), checkpoint_sha256=runtime.sha(checkpoint),
        actual_updates=7526, all17_Adam_step=150520, PPO_updates_repeated=0,
        original_training_plan_and_checkpoint_and_native_logs_unchanged=True,
        original11_numeric_readers_and_matched189_completed=True,
        corrected_diagnostic_view=str(view), immutable_source_pins=immutable,
        policy_accepted=False, full735_and_independent_sustained_still_required=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    run(args.request.resolve())
