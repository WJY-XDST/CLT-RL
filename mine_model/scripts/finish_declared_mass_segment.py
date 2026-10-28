"""Finish read-only validation/replay of an already completed mass-transfer50.

Never invoke training, optimizer.step or modify the source/checkpoints. Admit
only the recipe's exact source-to-current gravity-force delta in the legacy
physical-contract comparison; retain every other original numerical check.
"""
import argparse
import json
from pathlib import Path

import yaml

import joint_pd_saved_mass_runtime as runtime
from joint_pd_continuation import ROOT, RUNS, sha


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def finish(request_path):
    import train_joint_pd_standing_first as boundary
    import continue_joint_pd_observed100 as original_external
    import plain27_sampler_metadata_adapter as adapter
    import joint_pd_review_interval200 as interval
    import review_saved_joint_pd_segment as segment
    import train_saved_joint_pd_review50 as workflow
    from native_admitted_population_finalizer import install
    request = read(request_path)
    runtime.configure(request['recipe'])
    assert all(sha(path) == digest for path, digest in request['source_pins'].items())
    output, source = Path(request['output']), Path(request['source_checkpoint'])
    plan = read(output / 'experiment_plan.json')
    assert plan['bounded_training_updates'] == 50 and plan['native_num_envs'] == 8192
    assert plan['only_physics_change'] == runtime.recipe()['physics_changes']
    target = runtime.saved_updates(source) + 50
    assert target == plan['target_actual_ppo_updates']
    audits = sorted((output / 'native_learning_signal').glob('audit_update_*.json'))
    assert len(audits) == 50
    for index, path in enumerate(audits, 1):
        report = read(path)
        assert report['update'] == index and len(report['optimizer_steps']) == 20
        assert all(value == 0. for value in report['all_environment_alignment_max_errors'].values())
    runs = list(RUNS.glob('*_' + plan['run_name']))
    assert len(runs) == 1
    checkpoint = runs[0] / f'model_{target - 1}.pt'
    saved, current = runtime.require_source(checkpoint)
    assert runtime.saved_updates(checkpoint) == target
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert all(float(value['step']) == target * 20 for value in saved['optimizer_state_dict']['state'].values())
    _, before = runtime.require_source(source)
    expected = runtime.expected_configuration(before, 8192)
    original_contract = boundary.validate_contract
    admitted = []

    def declared_contract(run, reward_profile=None, *, fresh_weights=True, source_checkpoint=None):
        # Run the original per-run mode, cap and precision checks first.
        original_contract(run, reward_profile, fresh_weights=fresh_weights, source_checkpoint=None)
        if source_checkpoint:
            previous = yaml.safe_load((Path(source_checkpoint).parent / 'params/control_contract.yaml').read_text())
            actual = yaml.safe_load((Path(run) / 'params/control_contract.yaml').read_text())
            differences = {key: [value, actual.get(key)] for key, value in previous.items()
                           if key != 'fresh_weights' and actual.get(key) != value}
            allowed = {'feedforward_force': runtime.recipe()['physics_changes']['feedforward_force']}
            assert differences == allowed, differences
            admitted.append(differences)

    boundary.validate_contract = declared_contract
    try:
        proof = boundary.verify_update(source, checkpoint, target, expected, source)
    finally:
        boundary.validate_contract = original_contract
    assert admitted == [{'feedforward_force': runtime.recipe()['physics_changes']['feedforward_force']}]
    proof.update(calibrated_asset_and_full_height_unchanged=False,
                 physical_control_contract_unchanged=False,
                 declared_mass_and_gravity_transition_verified=True,
                 all_other_source_physical_contract_fields_equal=True,
                 declared_physics_changes=runtime.recipe()['physics_changes'],
                 postprocessor_does_not_train=True)
    write(output / 'resume_verification.json', proof)
    write(output / 'declared_mass_contract_postcheck.json', dict(
        verified=True, legacy_rejected_field='feedforward_force',
        admitted_exact_delta=admitted[0], new_USD_and_complete_configuration_equal=expected ==
        dict(current, log_dir=expected['log_dir']), original_per_run_caps_and_precision_checks_retained=True,
        all50_native_transition_alignment_checks_zero=True,
        source_and_all_checkpoints_unmodified=True, optimizer_executed=False, policy_accepted=False))

    completed = output / 'native_learning_signal/completed_external.json'
    if not completed.exists():
        interval.UPDATES = 50
        external = adapter.private_module(original_external.__file__, '_declared_mass50_external_finalizer')
        interval.install_finalizer(external)
        install(external, 8192)
        external.finalize_observations(output)
    write(output / 'final_checkpoint_verification_readonly.json', dict(
        verified=True, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        actual_ppo_updates=target, source_checkpoint=str(source), source_checkpoint_sha256=sha(source),
        all17_Adam_step=target * 20, all50_native_updates_verified=True,
        saved_std=saved['model_state_dict']['std'].tolist(),
        saved_actual_LR=[group['lr'] for group in saved['optimizer_state_dict']['param_groups']],
        source_initial_NN_std_Adam_and_LR_preserved=True, policy_accepted=False))
    if not (output / 'original11_health50_complete.json').exists():
        previous_processes = segment.processes

        def processes(delegate):
            base = previous_processes(delegate)

            class DeclaredMassReaders(type(base)):
                def routed(self, argv):
                    argv = super().routed(argv)
                    if (isinstance(argv, (list, tuple)) and len(argv) > 2
                            and Path(argv[1]).name == segment.SCRIPT.name and argv[2] == 'compare'):
                        return [argv[0], str(runtime.SCRIPT), 'compare', '--recipe',
                                str(runtime.RECIPE_PATH), *argv[3:]]
                    return argv
            return DeclaredMassReaders(delegate)

        segment.processes = processes
        try:
            segment.health(output)
        finally:
            segment.processes = previous_processes
    if not (output / 'complete50_matched189_agent_decision_required.json').exists():
        workflow.review(output, request, checkpoint)
    assert all(sha(path) == digest for path, digest in request['source_pins'].items())
    write(output / 'status.json', dict(state='complete50_independently_reviewed_agent_decision_required',
          actual_ppo_updates=target, checkpoint=str(checkpoint), training_active=False,
          original_training_completed_before_readonly_postprocessing=True,
          goal_remains_active=True, policy_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    finish(args.request.resolve())
