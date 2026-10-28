"""Finish the saved200 block after the declared drive-contract check failed.

This entry has no training command or optimizer invocation. It retains the
failed owner's evidence, verifies the exact saved7201 boundary, runs the same
canonical independent replay, and completes the original persisted-data audit.
Only the explicitly declared .5 -> .75 wheel damping contract is admitted.
"""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch
import yaml

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, sha
import plain27_sampler_metadata_adapter as adapter
import joint_pd_AF7001_wheelD075_runtime as runtime
import train_joint_pd_standing_first as standing_original
import watch_joint_pd_scratch as contract_original
import train_joint_pd_reward_block as block
import continue_joint_pd_observed100 as external_original
import joint_pd_review_interval200 as interval
import joint_pd_acceptance_10pct_20261017 as acceptance
import verify_AF7001_wheelD075_completed200 as final_check
from native_admitted_population_finalizer import install as install_population
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M


SCRIPT = Path(__file__).resolve()
OUTPUT = ROOT / 'mine_model/results/joint_pd_optimization_20261023_192237_032943_fbd005e4_AF7001_wheelD075_200'
ARCHIVE = OUTPUT / 'boundary_recovery_20261023'
UNIT = 'clt-joint-pd-af7001-wheelD075-boundary-recovery-20261023.service'
OLD_UNIT = 'clt-joint-pd-af7001-wheelD075-200-20261023.service'
EXPECTED_SHA = 'e3460bc0eadfa5a7a4f008688c2205e8dd06b4b8e7177f47d0091c4581dcf7e1'


def read(path):
    return json.loads(Path(path).read_text())


def intact(pins):
    assert all(sha(path) == digest for path, digest in pins.items()), 'Pinned source changed'


def check_contract_pair(before, after):
    assert set(before) == set(after), 'Unexpected contract key change'
    differences = {key: [before[key], after[key]] for key in before
                   if key != 'fresh_weights' and before[key] != after[key]}
    assert differences == {'wheel_damping': [.5, .75]}, differences
    return differences


def declared_contract(run, reward_profile=None, *, fresh_weights=True, source_checkpoint=None):
    assert source_checkpoint is not None and fresh_weights is False
    before = yaml.safe_load((Path(source_checkpoint).parent / 'params/control_contract.yaml').read_text())
    after = yaml.safe_load((Path(run) / 'params/control_contract.yaml').read_text())
    check_contract_pair(before, after)
    # Every original mode/effort/precision/reward-profile guard still executes.
    # Source equality is checked above with exactly one declared exception.
    return contract_original.validate_contract(run, reward_profile,
        fresh_weights=fresh_weights, source_checkpoint=None)


def verified_boundary():
    plan = read(OUTPUT / 'experiment_plan.json')
    state = read(OUTPUT / 'status.json')
    assert plan['next_configuration_changes'] == runtime.DELTA
    assert plan['source_actual_ppo_updates'] == 7001 and plan['target_actual_ppo_updates'] == 7201
    assert plan['bounded_training_updates'] == 200 and plan['native_num_envs'] == 8192
    assert plan['only_reward_change'] == plan['only_command_change'] == plan['only_exploration_change'] == {}
    intact(plan['source_files'])
    checkpoint, source = Path(state['checkpoint']), Path(plan['source_checkpoint'])
    assert sha(checkpoint) == state['checkpoint_sha256'] == EXPECTED_SHA
    assert sha(source) == plan['source_sha256'] == runtime.ANCHOR_SHA
    saved, current = runtime.require_source(checkpoint)
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert all(float(row['step']) == 144020 for row in saved['optimizer_state_dict']['state'].values())
    assert len(list((OUTPUT / 'native_learning_signal').glob('audit_update_*.json'))) == 200
    previous, cfg = runtime.require_source(source)
    expected = runtime.expected_configuration(cfg, 8192)
    assert expected == plan['training_config']
    expected['log_dir'] = current['log_dir']
    assert current == expected
    # Import each original function once; never re-inspect a compiled adapter.
    original = adapter.private_module(standing_original.__file__, '_wheelD075_recovered_boundary_verifier')
    original.validate_contract = declared_contract
    proof = original.verify_update(source, checkpoint, 7201, plan['training_config'], source)
    start = yaml.safe_load((checkpoint.parent / 'params/curriculum_start.yaml').read_text())
    assert start['completed_ppo_updates'] == 7001 and start['initial_common_step_counter'] == 7001 * 48
    assert start['motion_progress'] > 0. and tuple(start['height_bounds_m']) == SAFE_BASE_HEIGHT_RANGE_M
    assert all(abs(start[key] - value) <= 1.e-6 for key, value in zip(
        ('sampled_command_height_min_m', 'sampled_command_height_max_m'), SAFE_BASE_HEIGHT_RANGE_M))
    proof.update(first_reset=start, only_reward_change={},
        physical_control_contract_unchanged=False,
        physical_control_contract_unchanged_except_explicitly_declared_wheel_damping=True,
        declared_control_contract_difference={'wheel_damping': [.5, .75]},
        complete_saved_configuration_matches_declared_recipe=True,
        actual_new_PPO_updates_during_recovery=0, recovery_entry=str(SCRIPT),
        original_failed_unit_preserved=OLD_UNIT)
    return plan, checkpoint, proof


def qualify():
    plan, checkpoint, proof = verified_boundary()
    assert proof['optimizer_step'] == 144020 and proof['optimizer_states'] == 17
    # Negative cases ensure the exception cannot hide any other physical edit.
    before = yaml.safe_load((Path(plan['source_checkpoint']).parent / 'params/control_contract.yaml').read_text())
    after = yaml.safe_load((checkpoint.parent / 'params/control_contract.yaml').read_text())
    rejected = []
    for name, mutate in (
        ('different_wheel_damping', lambda value: value.update(wheel_damping=.8)),
        ('changed_leg_effort', lambda value: value.update(leg_effort_limit=35.)),
        ('extra_contract_key', lambda value: value.update(unexpected=True)),
    ):
        invalid = dict(after)
        mutate(invalid)
        try:
            check_contract_pair(before, invalid)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Physical guard failed negative case: ' + name)
    receipt = dict(verified=True, observed_at=datetime.now().astimezone().isoformat(),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        actual_ppo_updates=7201, all17_Adam_step=144020,
        saved_std=torch.load(checkpoint, map_location='cpu', weights_only=True)['model_state_dict']['std'].tolist(),
        saved_optimizer_LR=[group['lr'] for group in torch.load(checkpoint, map_location='cpu', weights_only=True)['optimizer_state_dict']['param_groups']],
        original_verify_update_numeric_and_weight_checks_retained=True,
        only_contract_equality_exception={'wheel_damping': [.5, .75]},
        complete_saved_recipe_verified=True, rejected_negative_cases=rejected,
        PPO_updates_executed=0, optimizer_executed=False, native_replay_started=False,
        policy_accepted=False, source_pins={str(path): sha(path) for path in (
            SCRIPT, Path(standing_original.__file__), Path(contract_original.__file__),
            runtime.SCRIPT, final_check.SCRIPT, OUTPUT / 'experiment_plan.json', checkpoint)})
    ARCHIVE.mkdir(exist_ok=True)
    destination = ARCHIVE / 'recovery_preflight.json'
    assert not destination.exists()
    write_json(destination, receipt)
    return receipt


def run():
    qualification = read(ARCHIVE / 'recovery_preflight.json')
    intact(qualification['source_pins'])
    assert qualification['verified'] and qualification['checkpoint_sha256'] == EXPECTED_SHA
    assert not (ARCHIVE / 'completed.json').exists()
    plan, checkpoint, proof = verified_boundary()
    assert not (OUTPUT / 'resume_verification.json').exists() and not (OUTPUT / 'replay').exists()
    old = read(OUTPUT / 'status.json')
    assert old['state'] == 'agent_diagnosis_required' and not old['training_active']
    assert old['reason'] == "ValueError('Resumed physical contract changed: wheel_damping')"
    for name in ('status.json', 'controller.log', 'learning_monitor_wheelD075_v2_status.json',
                 'reference_std_diagnostics_queue_status.json'):
        path = OUTPUT / name
        if path.exists():
            destination = ARCHIVE / ('original_failed_' + name)
            assert not destination.exists()
            shutil.copy2(path, destination)
    state = dict(old)

    def save(phase, **values):
        state.update(state=phase, updated_at=datetime.now().astimezone().isoformat(),
            training_active=False, actual_ppo_updates=7201, live_ppo_updates=7201,
            checkpoint=str(checkpoint), checkpoint_sha256=EXPECTED_SHA,
            recovery_unit=UNIT, original_failed_unit=OLD_UNIT,
            PPO_updates_repeated=0, goal_remains_active=True, policy_accepted=False)
        state.pop('reason', None)
        state.update(values)
        write_json(OUTPUT / 'status.json', state)
        print(json.dumps(dict(state=phase, **values)), flush=True)

    with (OUTPUT / '.controller.lock').open('a') as owned, (RUNS / '.cumulative_learning.lock').open('a') as global_lock:
        for handle in (owned, global_lock):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_json(OUTPUT / 'resume_verification.json', proof)
            save('independent_replay', child_pid=None)
            child_api = adapter.private_module(block.runtime.__file__, '_wheelD075_recovered_original_child_runner')
            child_api.OUTPUT = OUTPUT
            environment = dict(os.environ, PYTHONPATH=':'.join(str(ROOT / path) for path in (
                'wheel_legged_isaaclab', 'mine_model/scripts', 'mine_model/experiments')),
                PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            replay = OUTPUT / 'replay'
            child_api.run_child(runtime.replay_command(checkpoint, replay), OUTPUT / 'replay.log',
                environment, reporter=save)
            intact(plan['source_files'])
            report = acceptance.assess_current(replay)
            assert report['checkpoint_sha256'] == EXPECTED_SHA
            legs = acceptance.assess_current_legs(replay)
            write_json(replay / 'height_leg_command_assessment.json', legs)
            block.diagnose_joint_pd(replay, replay / 'joint_pd_command_diagnosis.json')
            reward = block.audit_rewards(replay)
            samples = np.genfromtxt(replay / 'trace.csv', delimiter=',', names=True)
            discrepancy = block.quadratic_height_discrepancy(samples,
                read(replay / 'resolved_env.json'), read(replay / 'manifest.json')['policy_dt'])
            assert discrepancy <= 1.e-5
            reward['quadratic_height_cost_max_abs_error'] = discrepancy
            write_json(replay / 'reward_balance_audit.json', reward)
            comparison = block.repair_comparison(report,
                read(Path(plan['source_replay']) / 'assessment.json'), read(plan['protected_reference']))
            comparison.update(checkpoint=str(checkpoint), checkpoint_sha256=EXPECTED_SHA,
                actual_leg_static_command_tracking_passed=legs['actual_leg_static_command_tracking_passed'])
            write_json(OUTPUT / 'comparison.json', comparison)
            save('agent_review_required', comparison=str(OUTPUT / 'comparison.json'), child_pid=None)
            external = adapter.private_module(external_original.__file__, '_wheelD075_recovered_original_external_finalizer')
            counters = interval.install_finalizer(external)
            population = install_population(external, 8192)
            external.finalize_observations(OUTPUT)
            final = final_check.verify(OUTPUT)
            assert final['verified'] and final['all17_Adam_step'] == 144020
            intact(qualification['source_pins'])
            intact(plan['source_files'])
            write_json(ARCHIVE / 'completed.json', dict(completed=True,
                observed_at=datetime.now().astimezone().isoformat(), checkpoint_sha256=EXPECTED_SHA,
                actual_ppo_updates=7201, all17_Adam_step=144020,
                original_failed_unit_preserved=OLD_UNIT, successful_recovery_unit=UNIT,
                canonical_independent_replay_complete=True, persisted_original200_data_verified=True,
                counter_adapter=counters, population_adapter=population,
                actual_new_PPO_updates=0, repeated_PPO_updates=0,
                checkpoint_modified=False, optimizer_executed=False, acceptance_changed=False,
                full_independent_reviews_still_required=True, policy_accepted=False))
        except BaseException as error:
            save('agent_diagnosis_required', reason=repr(error), child_pid=None)
            raise


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualify-only', action='store_true')
    args = cli.parse_args()
    value = qualify() if args.qualify_only else run()
    if value:
        print(json.dumps({key: value[key] for key in ('verified', 'actual_ppo_updates', 'all17_Adam_step', 'saved_optimizer_LR')}))
