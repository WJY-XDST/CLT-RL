"""A full-state fifty-update PPO continuation followed by matched frozen review.

Use the explicitly selected8192 population, original PPO and original numerical
readers. The interval change never changes rollout length or Adam minibatches.
"""
import argparse
import ast
import copy
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys

import joint_pd_saved_bundle_review50_runtime as runtime
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, sha


SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def original50_tick(api):
    """Adapt only the three review-counter predicates in the existing reader."""
    tree = ast.parse(inspect.getsource(api.tick))
    original = ast.dump(tree)
    changes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and ast.unparse(node.iter) == '(50, 100, 200)':
            changes.append((node, 'iter', node.iter))
            node.iter = ast.Tuple(elts=[ast.Constant(50)], ctx=ast.Load())
        elif isinstance(node, ast.Compare) and ast.unparse(node.left) == "plan['bounded_training_updates']":
            assert len(node.comparators) == 1 and ast.literal_eval(node.comparators[0]) == 200
            changes.append((node, 'comparators', node.comparators))
            node.comparators = [ast.Constant(50)]
        elif isinstance(node, ast.Compare) and ast.unparse(node.left) == 'completed':
            assert len(node.comparators) == 1 and ast.literal_eval(node.comparators[0]) == [50, 100, 200]
            changes.append((node, 'comparators', node.comparators))
            node.comparators = [ast.List(elts=[ast.Constant(50)], ctx=ast.Load())]
    assert len(changes) == 3
    changed = copy.deepcopy(tree)
    for node, attribute, value in changes:
        setattr(node, attribute, value)
    assert ast.dump(tree) == original
    exec(compile(ast.fix_missing_locations(changed), str(SCRIPT), 'exec'), api.__dict__)
    return dict(original_numeric_reader_restored_AST_identical=True,
                only_review_boundary_predicates_changed=3, updates=50)


def health50(output):
    import plain27_sampler_metadata_adapter as adapter
    import monitor_AF7001_yawsigma0025_learning as original
    import continue_saved_joint_pd_bundle as bundle
    from types import SimpleNamespace
    from audit_uniform_motion_milestone import audit
    plan = read(output / 'experiment_plan.json')
    api = adapter.private_module(original.SCRIPT, '_saved_bundle50_original11')
    api.SCRIPT, api.subprocess = SCRIPT, bundle.processes(subprocess)
    api.runtime = SimpleNamespace(profile=runtime.profile, DELTA=plan['next_configuration_changes'])
    api.CHECKS = tuple((
        'compare_uniform_control_recorded_joint_targets_v3.py' if name == 'compare_parallel8192_recorded_joint_targets.py'
        else 'audit_historical_source3001_count_bound.py' if name == 'audit_parallel8192_count_bound.py' else name,
        stem, options) for name, stem, options in original.CHECKS)
    assert len(api.CHECKS) == 11
    proof = original50_tick(api)
    result = api.tick(output, 'owned_saved_bundle50')
    assert result['completed_prefixes'] == [50] and result['state'] == 'all_owned_prefix_diagnostics_completed'
    audit(output, 50)
    write_json(output / 'original11_health50_complete.json', dict(result, adapter=proof,
        original11_numeric_readers_unchanged=True, training_modified=False))


def review(output, request, checkpoint):
    import review_frozen_joint_pd_command_matrix as matrix
    import review_joint_pd_robustness_suite as scores
    baseline = read(Path(request['matched_baseline']) / 'complete_frozen_command_matrix.json')
    assert baseline['checkpoint_sha256'] == runtime.recipe()['source_checkpoint_sha256']
    destination = output / 'independent_review189'
    destination.mkdir()
    jobs, pins = [], {}
    for job in read(Path(request['matched_baseline']) / 'qualification.json')['batches']:
        name = job['name']
        case_file = destination / (name + '_commands.json')
        write_json(case_file, job['cases'])
        target = destination / name
        argv = runtime.replay_command(checkpoint, target)
        argv[argv.index('--seconds') + 1] = '20'
        at = argv.index('--seeds') + 1
        argv[at:at + 3] = list(map(str, job['seeds']))
        argv += ['--command-probe-json', str(case_file)]
        jobs.append(dict(name=name, command=argv, replay=str(target), cases=job['cases'], seeds=job['seeds']))
        pins[str(case_file)] = sha(case_file)
    reader, previous = matrix.verify_reader()
    pins.update({str(path): sha(path) for path in (SCRIPT, runtime.SCRIPT, checkpoint,
        reader, previous, Path(request['recipe']), output / 'final_checkpoint_verification_readonly.json')})
    write_json(destination / 'qualification.json', dict(qualified=True,
        checkpoint_sha256=sha(checkpoint), actual_updates=runtime.saved_updates(checkpoint),
        trial_count=189, seconds_per_case=20, numerical_reader_extraction_AST_identical=True,
        batches=jobs, source_pins=pins, policy_accepted=False))
    after = matrix.run(destination)
    key = lambda value: (value['case']['name'], value['seed'])
    before_rows = {key(t): t for t in baseline['all_trials']}
    after_rows = {key(t): t for t in after['all_trials']}
    assert before_rows.keys() == after_rows.keys()
    result = dict(complete=True, source_summary=scores.summary(baseline['all_trials']),
        candidate_summary=scores.summary(after['all_trials']),
        newly_failed=sum(before_rows[k]['original_AND_literal10_passed'] and not after_rows[k]['original_AND_literal10_passed'] for k in before_rows),
        recovered=sum(not before_rows[k]['original_AND_literal10_passed'] and after_rows[k]['original_AND_literal10_passed'] for k in before_rows),
        newly_lost_survival=sum(before_rows[k]['survived'] and not after_rows[k]['survived'] for k in before_rows),
        all_pairs=[dict(before=before_rows[k], after=after_rows[k]) for k in before_rows],
        contains_all9_old_yaw_failed_conditions=True, checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), decision_required=True, policy_accepted=False,
        full735_and_sustained_qualification_still_required=True)
    write_json(output / 'complete50_matched189_agent_decision_required.json', result)


def run(request_path):
    import torch
    import train_joint_pd_standing_first as boundary
    import continue_joint_pd_observed100 as original_external
    import plain27_sampler_metadata_adapter as adapter
    import joint_pd_review_interval200 as interval
    from native_admitted_population_finalizer import install
    from train_uniform_control_direct8192_block200 import require_explicit_allocation, record_direct_start
    request = read(request_path)
    runtime.configure(request['recipe'])
    assert all(sha(path) == digest for path, digest in request['source_pins'].items())
    source, output = Path(request['source_checkpoint']), Path(request['output'])
    saved, cfg = runtime.require_source(source)
    require_explicit_allocation(Path(request['allocation']), block_id=output.name, source=source, num_envs=8192)
    source_actual, target = runtime.saved_updates(source), runtime.saved_updates(source) + 50
    expected = runtime.expected_configuration(cfg, 8192)
    assert not output.exists()
    output.mkdir()
    run_name = f'ordinary_sampler_n8192_{output.name}_u{source_actual}'
    delta = dict(rewards={}, commands={}, exploration={}, controller={}, physics={})
    plan = dict(created_at=datetime.now(timezone.utc).isoformat(), source_checkpoint=str(source),
        source_sha256=sha(source), source_actual_ppo_updates=source_actual,
        source_Adam_step=source_actual * 20, source_Adam_states=17,
        source_optimizer_LR=[group['lr'] for group in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(),
        target_actual_ppo_updates=target, bounded_training_updates=50,
        expected_terminal_Adam_step=target * 20, native_num_envs=8192,
        fixed_standing_environments=2048, motion_training_environments=6144,
        native_transition_batch=8192 * 48, run_name=run_name, training_config=expected,
        source_replay=request['source_mapping_replay'], profile=runtime.profile(),
        next_configuration_changes=delta, only_reward_change={}, only_command_change={},
        only_controller_change={}, only_exploration_change={},
        only_runtime_sampling_change=dict(retired_targeted_sampler_disabled=True,
                                          original_class_continuous_sampler=True),
        ordinary_full_policy_PPO=True, auxiliary_losses_enabled=False,
        optimizer_reset=False, learned_std_reset=False, actual_saved_LR_reset=False,
        source_files=request['source_pins'], checkpoint_save_interval=1,
        all_six_Gaussian_std_entries_remain_trainable=True,
        original_observation_and_action_maps_and_saved_controller_retained=True,
        capacity_tests_performed=False, automatic_4096_fallback=False,
        no_total_training_update_limit=True, relative_error_target=.1, acceptance_changed=False,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    write_json(output / 'experiment_plan.json', plan)
    record_direct_start(output, plan)
    argv, final, _ = runtime.command(source, cfg, run_name, 50, cfg['commands'], output / 'STOP')
    assert final == target - 1
    write_json(output / 'launch.json', dict(command=argv, source_sha256=sha(source)))
    with (output / 'training.log').open('x') as log:
        subprocess.run(argv, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    runs = list(RUNS.glob('*_' + run_name))
    assert len(runs) == 1
    checkpoint = runs[0] / f'model_{final}.pt'
    proof = boundary.verify_update(source, checkpoint, target, expected, source)
    write_json(output / 'resume_verification.json', proof)
    interval.UPDATES = 50
    external = adapter.private_module(original_external.__file__, '_saved_bundle50_external_finalizer')
    interval.install_finalizer(external)
    install(external, 8192)
    external.finalize_observations(output)
    after, _ = runtime.require_source(checkpoint)
    assert len(after['optimizer_state_dict']['state']) == 17
    assert all(float(v['step']) == target * 20 for v in after['optimizer_state_dict']['state'].values())
    write_json(output / 'final_checkpoint_verification_readonly.json', dict(verified=True,
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), actual_ppo_updates=target,
        source_checkpoint=str(source), source_checkpoint_sha256=sha(source),
        all17_Adam_step=target * 20, all50_native_updates_verified=True,
        saved_std=after['model_state_dict']['std'].tolist(),
        saved_actual_LR=[g['lr'] for g in after['optimizer_state_dict']['param_groups']],
        source_initial_NN_std_Adam_and_LR_preserved=True, policy_accepted=False))
    health50(output)
    review(output, request, checkpoint)
    assert all(sha(path) == digest for path, digest in request['source_pins'].items())
    write_json(output / 'status.json', dict(state='complete50_independently_reviewed_agent_decision_required',
        actual_ppo_updates=target, checkpoint=str(checkpoint), training_active=False,
        goal_remains_active=True, policy_accepted=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.request.resolve())
    except BaseException as error:
        output = Path(read(args.request)['output'])
        output.mkdir(exist_ok=True)
        write_json(output / 'inspection_required.json', dict(error=repr(error),
            checkpoints_preserved=True, no_automatic_blind_restart=True,
            goal_remains_active=True, policy_accepted=False))
        raise
