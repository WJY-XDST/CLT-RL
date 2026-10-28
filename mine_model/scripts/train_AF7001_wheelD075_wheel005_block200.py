"""One admitted200 ordinary-PPO block with a declared wheel velocity-drive D.

Prepare first, qualify all original post200 review readers, then launch. The
matched exploration trial lowers only the future wheel std floor .01 -> .005;
legs retain .01. The legacy reward-block carrier is neutral: its existing yaw coefficient remains
-200. No reward is revised, and no second controller feedback is introduced.
"""
import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import time
from types import SimpleNamespace

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import plain27_sampler_metadata_adapter as adapter
import joint_pd_AF7001_wheelD075_wheel005_runtime as runtime
import joint_pd_review_interval200 as interval
import joint_pd_acceptance_10pct_20261017 as acceptance
import train_joint_pd_reward_block as original_block
import continue_joint_pd_observed100 as original_external
import verify_AF7001_wheelD075_wheel005_completed200 as final_check


SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_AF7001_wheelD075_wheel005_qualification_20261024/post200_complete_qualification.json'


def read(path):
    return json.loads(Path(path).read_text())


def prepare(output, previous):
    output, previous = Path(output).resolve(), Path(previous).resolve()
    decision = read(runtime.ADMISSION)
    runtime.NUM_ENVS = count = decision['selected_num_envs']
    source = Path(decision['source_checkpoint'])
    runtime.review_ready(source)
    assert previous == runtime.SOURCE_REPORT.parent and output.parent == RESULTS
    assert str(output) == decision['planned_production_output']
    existing = output / 'experiment_plan.json'
    if existing.exists():
        plan = read(existing)
        assert plan['source_checkpoint'] == str(source) and plan['next_configuration_changes'] == runtime.DELTA
        assert plan['profile'] == runtime.profile() and plan['native_num_envs'] == count
        assert all(sha(path) == digest for path, digest in plan['source_files'].items())
        return plan, copy.deepcopy(plan['training_config'])
    assert not output.exists()
    cpu, frozen = [read(decision[key]) for key in ('CPU_qualification', 'frozen27_qualification')]
    assert cpu['qualified'] and frozen['qualified']
    assert cpu['source_sha256'] == frozen['source_checkpoint_sha256'] == sha(source)
    assert cpu['implementation_sha256'] == sha(runtime.SCRIPT)
    assert cpu['target_native_environment_count'] == frozen['target_training_environment_count'] == count
    assert frozen['all_initial27_inputs_and_initial6_deterministic_means_bitwise_equal']
    assert frozen['all_other_resolved_environment_fields_unchanged']
    assert not cpu['actual_PPO_updates_executed'] and not frozen['PPO_updates_performed']
    saved, cfg = runtime.require_source(source)
    expected = runtime.expected_configuration(cfg, count)
    plan = copy.deepcopy(read(previous / 'experiment_plan.json'))
    pins = {**plan['source_files'], **decision['source_pins'], **cpu['source_files'], **frozen['source_pins']}
    for path in (SCRIPT, runtime.SCRIPT, runtime.DECISION, runtime.ADMISSION, final_check.SCRIPT,
        Path(original_block.__file__), Path(original_external.__file__), Path(interval.__file__),
        SCRIPT.with_name('native_child_reported_environment_count.py'),
        SCRIPT.with_name('native_admitted_population_finalizer.py'),
        SCRIPT.with_name('review_AF7001_wheelD075_wheel005_block200.py'),
        SCRIPT.with_name('recover_AF7001_wheelD075_completed200_boundary.py'),
        SCRIPT.with_name('watch_joint_pd_scratch.py'),
        SCRIPT.with_name('train_joint_pd_standing_first.py'), Path(decision['CPU_qualification']),
        Path(decision['frozen27_qualification']), Path(decision['capacity_verified_summary']), previous / 'status.json'):
        pins[str(path)] = sha(path)
    assert all(sha(path) == digest for path, digest in pins.items())
    plan.update(created_at=datetime.now().astimezone().isoformat(), previous_output=str(previous),
        source_checkpoint=str(source), source_sha256=sha(source), source_actual_ppo_updates=7001,
        source_Adam_step=140020, source_Adam_states=17,
        source_optimizer_LR=[group['lr'] for group in saved['optimizer_state_dict']['param_groups']],
        source_std=saved['model_state_dict']['std'].tolist(), source_replay=str(previous / 'replay'),
        target_actual_ppo_updates=7201, expected_terminal_Adam_step=144020,
        bounded_training_updates=200, native_num_envs=count, fixed_standing_environments=count // 4,
        motion_training_environments=count * 3 // 4, native_transition_batch=count * 48,
        optimizer_minibatches_per_actual_update=20, checkpoint_save_interval=1,
        run_name=f'plain_wheelD075_wheel005_n{count}_' + output.name + '_u7001',
        reward_key='yaw_rate_error_sq', reward_value=-200., neutral_legacy_reward_carrier_only=True,
        only_reward_change={}, only_command_change={}, only_exploration_change=copy.deepcopy(runtime.DELTA['exploration']),
        only_controller_change=runtime.DELTA['controller'],
        next_configuration_changes=runtime.DELTA, authoritative_current_delta=runtime.DELTA,
        only_parallelism_change={key: [cfg[key]['num_envs'], count] for key in ('scene', 'terrain')},
        training_config=expected, profile=runtime.profile(), source_files=pins,
        optimizer_reset=False, learned_std_reset=False, actual_saved_LR_reset=False,
        source_model200_reloaded=False, source_model_and_optimizer_reloaded_from_original200=False,
        ordinary_full_policy_PPO=True, architecture_migrated=False, auxiliary_losses_enabled=False,
        action_map_or_controller_changed=True, action_map_changed=False, controller_change_declared=True,
        early_learning_inspection_actual_updates=[50, 100], full_native_independent_review_after_actual_updates=200,
        all_six_Gaussian_std_entries_remain_trainable=True, no_total_training_update_limit=True,
        relative_error_target=.1, acceptance_changed=False, capacity_admission=decision['capacity_verified_summary'],
        CPU_qualification=decision['CPU_qualification'], frozen27_qualification=decision['frozen27_qualification'],
        full735_static810_original120_signed_and_small_yaw_review_required=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    output.mkdir()
    write_json(existing, plan)
    return plan, expected


def wait_resources(output, plan):
    from check_plain27_environment_capacity import resources
    capacity = read(plan['capacity_admission'])
    measured = next(row for row in capacity['all_results'] if row['num_envs'] == plan['native_num_envs'])
    assert measured['passed']
    policy = read(RESULTS.parent / 'config/training_environment_capacity.json')
    reserve = int(policy['minimum_host_reserve_gib'] * 1024 ** 3)
    allowance = int(policy['maximum_probe_swap_gib'] * 1024 ** 3)
    gpu_increment = measured['gpu_used_mib_peak'] - measured['resources_before']['gpu_used_mib']
    while True:
        current = resources()
        needed_host = measured['MemoryPeak_observed_peak'] + reserve - min(current['swap_free_bytes'], allowance)
        needed_gpu = gpu_increment + policy['minimum_gpu_reserve_mib']
        ready = current['host_available_bytes'] >= needed_host and current['gpu_free_mib'] >= needed_gpu
        write_json(Path(output) / 'current_capacity_resources_before_training.json', dict(
            observed_at=datetime.now().astimezone().isoformat(), ready=ready,
            source_checkpoint_sha256=plan['source_sha256'], planned_num_envs=plan['native_num_envs'],
            current_resources=current, required_host_available_bytes=needed_host,
            required_gpu_free_mib=needed_gpu, exact_source_capacity_summary=plan['capacity_admission'],
            fresh_check_required_for_each_block=True, native_training_started=False, policy_accepted=False))
        if ready:
            return
        time.sleep(10.)


def neutral_carrier_command(*args, **kwargs):
    assert kwargs.pop('reward_overrides') == {'rewards.yaw_rate_error_sq': -200.}
    assert runtime.baseline()[2]['rewards']['yaw_rate_error_sq'] == -200.
    return runtime.command(*args, reward_overrides={}, **kwargs)


def run(output, previous, unit):
    plan, expected = prepare(output, previous)
    proof = read(QUALIFICATION)
    assert proof['qualified'] and proof['all_original_scopes_and_numeric_readers_retained']
    assert proof['native_num_envs'] == runtime.NUM_ENVS
    assert proof['prepared_plan_path'] == str(Path(output).resolve() / 'experiment_plan.json')
    assert all(sha(path) == digest for path, digest in proof['source_pins'].items())
    block = adapter.private_module(original_block.__file__, '_wheelD075_wheel005_original_reward_block')
    # Admit the exact declared D=.5 -> .75 contract before training, while
    # preserving every original saved-state and complete-configuration guard.
    block.verify_update = final_check.verify_boundary
    external = adapter.private_module(original_external.__file__, '_wheelD075_original_external_finalizer')
    from native_child_reported_environment_count import install as install_reporter
    from native_admitted_population_finalizer import install as install_population_finalizer
    child_api = adapter.private_module(block.runtime.__file__, '_wheelD075_child_actual_count')
    reporter = install_reporter(child_api)
    block.runtime = child_api
    interval.install_finalizer(external)
    finalizer = install_population_finalizer(external, runtime.NUM_ENVS)
    write_json(Path(output) / 'admitted_population_metadata_adapters.json', dict(
        actual_native_num_envs=runtime.NUM_ENVS, reporter=reporter, finalizer=finalizer,
        original_subprocess_numeric_and_optimizer_checks_retained=True,
        unchanged_legacy_reward_carrier={'yaw_rate_error_sq': -200.}))
    external.record_resource_wait(output, plan)
    wait_resources(output, plan)
    block.continuation_command, block.replay_command = neutral_carrier_command, runtime.replay_command
    block.assess, block.assess_directory = acceptance.assess_current, acceptance.assess_current_legs
    pointer = dict(unit=unit, directory=str(Path(output).resolve()), source_checkpoint=plan['source_checkpoint'],
        controller_kind='ordinary_joint_pd_declared_wheelD075_future_wheel005', actual_updates_per_review=200,
        native_num_envs=runtime.NUM_ENVS, fixed_standing_environments=runtime.NUM_ENVS // 4,
        optimizer_reset=False, weights_reinitialized=False, no_total_training_update_limit=True, policy_accepted=False)
    for name in ('joint_pd_active_joint_tracking.json', 'joint_pd_active_standing_seed.json', 'joint_pd_active_resume.json'):
        write_json(RESULTS / name, pointer)
    block.run(SimpleNamespace(previous=Path(previous), output=Path(output), reward_key='yaw_rate_error_sq',
        value=-200., updates=200, prepare_only=False), prepare_function=lambda *args: (plan, expected))
    external.finalize_observations(output)
    return final_check.verify(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--unit')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.output, args.previous)
    else:
        run(args.output, args.previous, args.unit)
