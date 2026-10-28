"""Bind the complete independent60s scope to a wholly passed saved policy.

The existing original735 launcher/scorer and overlapping-window reader remain
unchanged. Only the globally declared duration and fresh seed triples differ.
This has no training or hardware entry point and never promotes a policy.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    path = Path(path)
    assert not path.exists(), 'Preserve existing evidence and inspect its exact service'
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def contracts(batches):
    return [(v['name'], v['cases'], v['seeds']) for v in batches]


def make_plan(requirement, original_scope, output):
    old, scope = read(requirement), read(original_scope)
    assert old['scope_only_not_completion_evidence'] and not old['policy_accepted']
    assert scope['qualified'] and scope['trial_count'] == 735 and scope['unique_numeric_commands'] == 219
    assert old['literal_nonzero_relative_percent_less_than'] == 10.
    assert contracts(old['original735_matrix']['scopes']) == contracts(scope['batches'])
    nominal = old['original735_matrix']['scopes']
    fresh = old['independent_sustained_matrix']
    assert fresh['seconds_per_case'] == 60. and fresh['original_steady_start_seconds'] == 17.
    assert fresh['original_overlapping_window_seconds'] == 3. and fresh['original_stride_seconds'] == 1.
    known = {seed for batch in nominal for seed in batch['seeds']} | {43}
    offset = fresh['reset_seeds_offset']
    assert len(nominal) == len(fresh['scopes']) == 12
    for before, after in zip(nominal, fresh['scopes']):
        assert before['name'] == after['name'] and before['cases'] == after['cases']
        assert after['seeds'] == [seed + offset for seed in before['seeds']]
        assert known.isdisjoint(after['seeds'])
    assert sum(len(v['cases']) * len(v['seeds']) for v in fresh['scopes']) == 735
    value = copy.deepcopy(old)
    value.update(schema='full735_independent_sustained_candidate_binding_v2',
                 observed_at=datetime.now(timezone.utc).isoformat(),
                 original_scope_qualification=str(Path(original_scope).resolve()),
                 candidate_native_configuration_must_match_exactly=True,
                 nominal735_and_selection189_must_both_pass_before_execution=True,
                 independent_windows_per_trial=41)
    # The earlier scope's gain2 was a snapshot, not an instruction to retune
    # a later saved controller. Bind the candidate's whole physics contract.
    value['source_physics'] = dict(
        mass_kg=old['source_physics']['mass_kg'],
        feedforward_per_leg_N=old['source_physics']['feedforward_per_leg_N'],
        direct_joint_position_PD=True,
        common_wheel_gain_binding='exact verified candidate checkpoint and nominal replay',
        all_controller_reward_and_noise_settings_binding='exact candidate paired configuration')
    value['source_pins'].update({str(p.resolve()): sha(p) for p in
                               (Path(requirement), Path(original_scope), SCRIPT)})
    write(output, value)
    return value


def fully_passed(path, count):
    # This guard is intentionally before importing any PPO/runtime module.
    from summarize_frozen_joint_pd_matrix import load
    report, rows = load(Path(path))
    assert len(rows) == count
    if not all(v['survived'] and v['original_AND_literal10_passed'] for v in rows):
        raise ValueError(f'All {count} original trials must pass before sustained qualification')
    return report, rows


def qualify(*, plan_path, runtime_module, recipe, checkpoint, verification,
            nominal735, independent189, output):
    plan = read(plan_path)
    assert plan['schema'] == 'full735_independent_sustained_candidate_binding_v2'
    nominal, nominal_rows = fully_passed(nominal735, 735)
    independent, _ = fully_passed(independent189, 189)
    digest = sha(checkpoint)
    assert nominal['checkpoint_sha256'] == independent['checkpoint_sha256'] == digest
    expected = {(case['name'], seed): case
                for batch in plan['original735_matrix']['scopes']
                for case in batch['cases'] for seed in batch['seeds']}
    assert {(v['case']['name'], v['seed']): v['case'] for v in nominal_rows} == expected
    assert runtime_module.startswith('joint_pd_') and runtime_module.endswith('_runtime')
    assert runtime_module.replace('_', '').isalnum()
    located = importlib.util.find_spec(runtime_module)
    assert located and Path(located.origin).resolve().parent == ROOT / 'mine_model/experiments'
    runtime = importlib.import_module(runtime_module)
    runtime.configure(recipe)
    _, cfg = runtime.require_source(checkpoint)
    assert cfg['leg_control_mode'] == 'implicit_joint_position'
    assert cfg['feedforward_force'] == plan['source_physics']['feedforward_per_leg_N']
    nominal_jobs = read(Path(nominal735).parent / 'qualification.json')['batches']
    physical_pins = {}
    for job in nominal_jobs:
        directory = Path(job['replay'])
        physical = read(directory / 'source200_environment.json')
        drive = read(directory / 'uniform_global_control_native_contract.json')
        resolved = read(directory / 'resolved_env.json')
        assert physical['native_environment_verified'] and drive['verified']
        assert all(abs(v - plan['source_physics']['mass_kg']) < 1.e-5
                   for v in physical['actual_mass_range_kg'])
        assert drive['actual_PhysX_wheel_damping_min'] == drive['actual_PhysX_wheel_damping_max'] == cfg['wheel_damping']
        assert drive['actual_USD_damping'] == [cfg['wheel_damping']] * 2
        assert resolved['feedforward_force'] == cfg['feedforward_force']
        assert resolved['robot']['spawn']['usd_path'] == cfg['robot']['spawn']['usd_path']
        for name in ('source200_environment.json', 'uniform_global_control_native_contract.json', 'resolved_env.json'):
            path = directory / name
            physical_pins[str(path)] = sha(path)
    import review_saved_joint_pd_same_scope as same
    import review_frozen_joint_pd_command_matrix as matrix
    old_runtime = same.runtime
    same.runtime = runtime
    try:
        value = same.qualify(recipe=recipe, checkpoint=checkpoint, verification=verification,
                             original_scope=Path(plan['original_scope_qualification']), output=output)
    finally:
        same.runtime = old_runtime
    # Preserve the unexecuted original template as provenance. The new
    # qualification is finalized before any replay process can be launched.
    template = output / 'original20s_case_seed_template_qualification.json'
    write(template, value)
    independent_scope = plan['independent_sustained_matrix']
    for job, fresh in zip(value['batches'], independent_scope['scopes']):
        assert job['name'] == fresh['name'] and job['cases'] == fresh['cases']
        job['seeds'] = fresh['seeds']
        argv = job['command']
        argv[argv.index('--seconds') + 1] = str(independent_scope['seconds_per_case'])
        at = argv.index('--seeds') + 1
        argv[at:at + 3] = list(map(str, fresh['seeds']))
    value.update(seconds_per_case=60., selected_runtime_module=runtime_module,
                 independent_sustained=True, original735_numeric_case_contracts_identical=True,
                 original735_case_seed_contracts_identical=False,
                 reset_seeds_offset=independent_scope['reset_seeds_offset'],
                 fresh_seed_contracts_identical_to_requirement=True,
                 steady_start_s=17., window_s=3., stride_s=1., windows_per_trial=41,
                 candidate_physics_from_its_own_paired_configuration=True,
                 candidate_wheel_damping=cfg['wheel_damping'],
                 nominal735_report=str(nominal735), independent189_report=str(independent189),
                 sustained_scope_plan=str(plan_path), execution_started=False)
    value['source_pins'].update({str(p.resolve()): sha(p) for p in
                               (SCRIPT, plan_path, nominal735, independent189, template,
                                ROOT / 'mine_model/scripts/audit_historical_yaw_sustained_windows.py',
                                ROOT / 'mine_model/scripts/summarize_mine_trial.py')})
    value['source_pins'].update(physical_pins)
    matrix.verify_pins(value['source_pins'])
    (output / 'qualification.json').write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    return value


def run(output):
    import review_frozen_joint_pd_command_matrix as matrix
    import audit_historical_yaw_sustained_windows as historical
    qualification = read(output / 'qualification.json')
    assert qualification['qualified'] and qualification['independent_sustained']
    assert qualification['seconds_per_case'] == 60. and qualification['trial_count'] == 735
    nominal, _ = fully_passed(qualification['nominal735_report'], 735)
    independent, _ = fully_passed(qualification['independent189_report'], 189)
    assert nominal['checkpoint_sha256'] == independent['checkpoint_sha256'] == qualification['checkpoint_sha256']
    matrix.verify_pins(qualification['source_pins'])
    complete = matrix.run(output)
    rows, pins = [], dict(complete['source_pins'])
    for job in qualification['batches']:
        path, value = historical.audit(Path(job['replay']))
        assert value['duration_s'] == 60. and value['windows_per_trial'] == 41
        assert value['steady_interval_s'] == [17., 60.]
        assert value['complete_trace_survival_checked']
        rows.extend(value['trials'])
        pins[str(path)] = sha(path)
        pins.update(value['source_pins'])
    expected = {(case['name'], seed): case for job in qualification['batches']
                for case in job['cases'] for seed in job['seeds']}
    assert len(rows) == len(expected) == 735
    assert {(v['case']['name'], v['seed']): v['case'] for v in rows} == expected
    matrix.verify_pins(pins)
    count = sum(v['original_AND_literal10_all_windows_passed'] for v in rows)
    value = dict(complete=True, verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
                 checkpoint_sha256=qualification['checkpoint_sha256'],
                 actual_updates=qualification['actual_updates'], trial_count=735,
                 unique_numeric_commands=219, seconds_per_case=60., windows_per_trial=41,
                 steady_start_s=17., window_s=3., stride_s=1.,
                 survivors=sum(v['survived'] for v in rows),
                 original_AND_literal10_all_windows_passed=count,
                 independent_sustained_scope_passed=(count == 735),
                 all_trials=rows, all_failures_retained=True, source_pins=pins,
                 original7_and_original_static_checks_retained=True,
                 candidate_physics_preserved=True, acceptance_changed=False,
                 sampled_policy_actions=False, training_modified=False,
                 optimizer_executed=False, hardware_enabled=False, policy_accepted=False)
    write(output / 'complete_independent_sustained735.json', value)
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('plan', 'qualify', 'run'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requirement', type=Path)
    parser.add_argument('--original-scope', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--runtime-module')
    parser.add_argument('--recipe', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--verification', type=Path)
    parser.add_argument('--nominal735', type=Path)
    parser.add_argument('--independent189', type=Path)
    args = parser.parse_args()
    if args.mode == 'plan':
        assert args.requirement and args.original_scope
        result = make_plan(args.requirement.resolve(), args.original_scope.resolve(), args.output.resolve())
        print(json.dumps(dict(plan_prepared=True, candidate_qualified=False, execution_started=False,
                              trials=735, seconds=60, windows_per_trial=41)))
    elif args.mode == 'qualify':
        assert all((args.plan, args.runtime_module, args.recipe, args.checkpoint,
                    args.verification, args.nominal735, args.independent189))
        result = qualify(plan_path=args.plan.resolve(), runtime_module=args.runtime_module,
                         recipe=args.recipe.resolve(), checkpoint=args.checkpoint.resolve(),
                         verification=args.verification.resolve(), nominal735=args.nominal735.resolve(),
                         independent189=args.independent189.resolve(), output=args.output.resolve())
        print(json.dumps(dict(qualified=result['qualified'], trials=735, execution_started=False)))
    else:
        result = run(args.output.resolve())
        print(json.dumps(dict(complete=True, passed=result['original_AND_literal10_all_windows_passed'],
                              trials=735, policy_accepted=False)))
