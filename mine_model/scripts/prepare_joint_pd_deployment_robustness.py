"""Bind the original735 stress matrix only after nominal735 and189 pass.

The request file is a preparation artifact. Qualification freezes an accepted
candidate identity and makes independent one-factor/combined replay jobs. It
does not execute simulation, train, promote a policy or enable real hardware.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
STRESS_RUNTIME = ROOT / 'mine_model/experiments/joint_pd_deployment_robustness_runtime.py'
DR_LIBRARY = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/deployment_randomization.py'
DELAY_LIBRARY = ROOT / 'wheel_legged_isaaclab/wheel_legged_gym_isaaclab/robustness.py'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def write_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError('Preserve existing qualified results: ' + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require_passed(report_path, count):
    # This remains before importing torch, PPO, or a simulator runtime.
    import summarize_frozen_joint_pd_matrix as summaries
    report, rows = summaries.load(report_path)
    if len(rows) != count or not all(row['survived'] and row['original_AND_literal10_passed'] for row in rows):
        raise ValueError(f'All original {count} trials must survive and pass strict10 before robustness')
    return report, rows


def inspect_nominal(report_path, count, diagnostic_preacceptance):
    """Explicit diagnostic permission preserves failed gates instead of passing them."""
    if not diagnostic_preacceptance:
        return require_passed(report_path, count)
    import summarize_frozen_joint_pd_matrix as summaries
    report, rows = summaries.load(report_path)
    if (not report.get('complete') or len(rows) != count
            or len({(row['case']['name'], row['seed']) for row in rows}) != count
            or not report.get('all_failures_retained')):
        raise ValueError('Diagnostic testing still requires the complete unchanged nominal scope')
    return report, rows


def qualify(*, request_path, nominal735, independent189, runtime_module, recipe,
            checkpoint, verification, original_scope, output, diagnostic_preacceptance=False):
    request = read(request_path)
    if request['schema'] != 'deployment_robustness_request_v1':
        raise ValueError('Unknown preparation request')
    nominal, rows = inspect_nominal(nominal735, 735, diagnostic_preacceptance)
    independent, independent_rows = inspect_nominal(independent189, 189, diagnostic_preacceptance)
    if nominal['checkpoint_sha256'] != independent['checkpoint_sha256'] or sha(checkpoint) != nominal['checkpoint_sha256']:
        raise ValueError('Both nominal reviews must belong to this exact candidate')
    scope = read(original_scope)
    expected = {(case['name'], seed): case for batch in scope['batches']
                for case in batch['cases'] for seed in batch['seeds']}
    if (not scope['qualified'] or scope['trial_count'] != 735
            or scope['unique_numeric_commands'] != 219 or len(expected) != 735
            or {(row['case']['name'], row['seed']): row['case'] for row in rows} != expected):
        raise ValueError('A different or narrowed nominal matrix cannot qualify the original735 suite')
    if not runtime_module.startswith('joint_pd_') or not runtime_module.endswith('_runtime'):
        raise ValueError('A local saved candidate runtime is required')
    location = importlib.util.find_spec(runtime_module)
    if not location or Path(location.origin).resolve().parent != ROOT / 'mine_model/experiments':
        raise ValueError('The nominal runtime must belong to this project')
    runtime = importlib.import_module(runtime_module)
    runtime.configure(recipe)
    _, cfg = runtime.require_source(checkpoint)
    if cfg['leg_control_mode'] != 'implicit_joint_position':
        raise ValueError('Keep direct joint angle targets and implicit PD')
    if abs(float(cfg['feedforward_force']) - request['nominal_feedforward_per_leg_N']) > 1.e-6:
        raise ValueError('Inspect a changed nominal mass/feedforward contract')
    dt = float(cfg['sim']['dt'])
    if dt != request['nominal_simulation_dt_s'] or abs(dt * cfg['decimation'] - .02) > 1.e-10:
        raise ValueError('Inspect a changed physics timestep or 50Hz policy contract')
    import review_saved_joint_pd_same_scope as same
    import review_frozen_joint_pd_command_matrix as matrix
    from wheel_legged_gym_isaaclab.deployment_randomization import DeploymentRandomizationSpec
    specifications = [(item, DeploymentRandomizationSpec.from_dict(item['specification']))
                      for item in request['profiles']]
    if not specifications or specifications[0][0]['name'] != 'nominal' or specifications[0][1].enabled:
        raise ValueError('The paired reference must be an unperturbed nominal replay')
    if len({item['name'] for item, spec in specifications}) != len(specifications):
        raise ValueError('Stress profile names must be unique')
    if output.exists():
        raise FileExistsError('Do not overwrite a prepared or running suite')
    output.mkdir(parents=True)
    suite_path = output / 'suite_plan.json'
    source_files = [SCRIPT, STRESS_RUNTIME, request_path, nominal735, independent189,
                    recipe, checkpoint, verification, original_scope, Path(location.origin),
                    ROOT / 'mine_model/config/tracking_acceptance.json',
                    DR_LIBRARY, DELAY_LIBRARY]
    profiles = []
    old_runtime = same.runtime
    same.runtime = runtime
    try:
        for entry, spec in specifications:
            name = entry['name']
            if not name or any(value not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for value in name):
                raise ValueError('Profile names are labels, not paths')
            specification = output / (name + '_spec.json')
            write_new(specification, entry['specification'])
            destination = output / name
            qualification = same.qualify(recipe=recipe, checkpoint=checkpoint,
                verification=verification, original_scope=original_scope, output=destination)
            write_new(destination / 'original_unperturbed_qualification.json', qualification)
            for job in qualification['batches']:
                command = job['command']
                index = command.index(str(runtime.SCRIPT))
                if command[index + 1] != 'replay':
                    raise ValueError('Only the saved-policy replay may be wrapped')
                job['command'] = command[:index] + [str(STRESS_RUNTIME), 'replay',
                    '--suite-plan', str(suite_path), '--profile', name,
                    '--nominal-runtime-module', runtime_module] + command[index + 2:]
            qualification.update(stress_test_not_nominal_acceptance=True,
                                 nominal_checkpoint_sha256=sha(checkpoint))
            qualification['source_pins'].update({str(p): sha(p) for p in
                                                (STRESS_RUNTIME, specification, DR_LIBRARY, DELAY_LIBRARY)})
            matrix.write(destination / 'qualification.json', qualification)
            source_files.append(specification)
            source_files.append(destination / 'qualification.json')
            profiles.append(dict(name=name, output=str(destination), specification=str(specification)))
    finally:
        same.runtime = old_runtime
    result = dict(schema=('diagnostic_deployment_robustness_suite_v1' if diagnostic_preacceptance
                          else 'post_acceptance_deployment_robustness_suite_v1'), qualified=True,
        observed_at=datetime.now(timezone.utc).isoformat(), checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), actual_ppo_updates=runtime.saved_updates(checkpoint),
        recipe=str(recipe), nominal_runtime_module=runtime_module, profiles=profiles,
        nominal_mass_kg=request['nominal_mass_kg'],
        nominal_feedforward_per_leg_N=request['nominal_feedforward_per_leg_N'],
        original_scope_qualification=str(original_scope),
        trial_count_per_profile=735,
        nominal735_passed=all(row['survived'] and row['original_AND_literal10_passed'] for row in rows),
        independent189_passed=all(row['survived'] and row['original_AND_literal10_passed'] for row in independent_rows),
        diagnostic_preacceptance=diagnostic_preacceptance,
        nominal735_strict_passed=sum(row['survived'] and row['original_AND_literal10_passed'] for row in rows),
        independent189_strict_passed=sum(row['survived'] and row['original_AND_literal10_passed'] for row in independent_rows),
        diagnostic_authorization=('User explicitly requested robustness in parallel with NX host implementation; '
                                  'this is not final acceptance' if diagnostic_preacceptance else None),
        timing_resolution_ms=dt * 1000., engineering_ranges_not_measured_hardware_uncertainty=True,
        original_numeric_gates_unchanged=True, evaluation_cases_enter_training=False,
        parameters_chosen_from_commands=False, simulator_started=False,
        optimizer_executed=False, policy_accepted=False, hardware_enabled=False,
        fresh_independent_sustained735_still_required=True,
        source_pins={str(p.resolve()): sha(p) for p in source_files})
    write_new(suite_path, result)
    return result


def run(plan_path):
    import review_frozen_joint_pd_command_matrix as matrix
    import summarize_frozen_joint_pd_matrix as summaries
    from wheel_legged_gym_isaaclab.deployment_randomization import DeploymentRandomizationSpec
    plan = read(plan_path)
    diagnostic = (plan.get('schema') == 'diagnostic_deployment_robustness_suite_v1'
                  and plan.get('diagnostic_preacceptance') is True
                  and bool(plan.get('diagnostic_authorization')))
    if (not plan.get('qualified') or (not diagnostic and
            (plan.get('schema') != 'post_acceptance_deployment_robustness_suite_v1'
             or not plan['nominal735_passed'] or not plan['independent189_passed']))):
        raise ValueError('Qualify an actually passed saved candidate first')
    matrix.verify_pins(plan['source_pins'])
    original = read(plan['original_scope_qualification'])
    required = {(case['name'], seed): case for batch in original['batches']
                for case in batch['cases'] for seed in batch['seeds']}
    if len(required) != 735:
        raise ValueError('The original full acceptance scope was lost')
    baseline, reports, raw, physical_pins = None, {}, {}, {}
    for entry in plan['profiles']:
        destination = Path(entry['output'])
        complete = destination / 'complete_frozen_command_matrix.json'
        if not complete.exists():
            matrix.run(destination)
        report, rows = summaries.load(complete)
        if report['checkpoint_sha256'] != plan['checkpoint_sha256'] or len(rows) != 735:
            raise ValueError('A stress profile lost candidate identity or original735 coverage')
        expected = {(row['case']['name'], row['seed']): row for row in rows}
        if {key: row['case'] for key, row in expected.items()} != required:
            raise ValueError('The stress replay differs from the original735 case/seed scope')
        if baseline is None:
            if entry['name'] != 'nominal':
                raise ValueError('An unperturbed paired reference is required')
            if not diagnostic and not all(row['survived'] and row['original_AND_literal10_passed'] for row in rows):
                matrix.write(plan_path.parent / 'paired_nominal_inspection_required.json', dict(
                    error='Previously accepted candidate did not reproduce all735 nominal passes',
                    existing_nominal_evidence_preserved=True, stress_profiles_not_started=True,
                    policy_accepted=False, hardware_enabled=False))
                raise ValueError('Reproduce the paired nominal acceptance before perturbing physics')
            baseline = expected
        if expected.keys() != baseline.keys() or any(expected[key]['case'] != baseline[key]['case'] for key in expected):
            raise ValueError('Stress tests must preserve every nominal case and seed')
        specification = read(entry['specification'])
        spec = DeploymentRandomizationSpec.from_dict(specification)
        if tuple(spec.mass_scale) != (1., 1.):
            raise ValueError('This payload suite keeps the independently calibrated leg masses fixed')
        qualification = read(destination / 'qualification.json')
        from dataclasses import asdict
        full_spec = json.loads(json.dumps(asdict(spec)))
        for job in qualification['batches']:
            contract_path = Path(job['replay']) / 'robustness_contract.json'
            contract = read(contract_path)
            matrix.verify_pins(contract['source_pins'])
            if (not contract['nominal_saved_controller_verified_before_perturbation']
                    or contract['profile'] != entry['name'] or contract['specification'] != full_spec
                    or contract['nominal_checkpoint_sha256'] != plan['checkpoint_sha256']
                    or contract['optimizer_executed'] or contract['policy_weights_changed']):
                raise ValueError('Inspect actual stress-hook binding, not just its requested configuration')
            applied = contract['applied']
            if applied['enabled'] != spec.enabled:
                raise ValueError('Stress hooks were not applied as declared')
            if spec.enabled:
                if not all(applied[key] for key in ('physical_properties_readback_verified',
                        'mass_and_inertia_readback_verified', 'measured_wheel_masses_preserved',
                        'gravity_feedforward_retains_nominal_mass')):
                    raise ValueError('Actual physical properties or nominal feedforward are unverified')
                count = len(job['cases']) * len(job['seeds'])
                if len(applied['actual_mass_kg']) != count or len(applied['body_mass_offset_kg']) != count:
                    raise ValueError('Physical readback must cover every virtual robot')
                low, high = spec.body_mass_offset_kg
                for nominal, actual, delta in zip(applied['nominal_mass_kg'], applied['actual_mass_kg'],
                                                  applied['body_mass_offset_kg']):
                    if (abs(nominal - plan['nominal_mass_kg']) > 1.e-5
                            or not low - 1.e-6 <= delta <= high + 1.e-6
                            or abs(actual - nominal - delta) > 1.e-5):
                        raise ValueError('Actual total mass does not realize the requested payload')
                if applied['nominal_feedforward_per_leg_N'] != plan['nominal_feedforward_per_leg_N']:
                    raise ValueError('Feedforward must not use privileged randomized mass')
                for kind, interval in (('sensor', spec.sensor_delay_steps), ('action', spec.action_delay_steps)):
                    values = applied[kind + '_delay_ms']
                    resolution = plan['timing_resolution_ms']
                    if len(values) != count or any(not interval[0] * resolution - 1.e-5 <= value <= interval[1] * resolution + 1.e-5 for value in values):
                        raise ValueError('Actual latency differs from the declared physics-step interval')
            elif applied['hooks_installed'] or applied['RNG_draws'] or applied['physics_properties_written']:
                raise ValueError('The paired nominal profile must be non-invasive')
            physical_pins[str(contract_path)] = sha(contract_path)
        paired = dict(recovered=sum(not baseline[key]['original_AND_literal10_passed'] and
                                   expected[key]['original_AND_literal10_passed'] for key in expected),
            newly_failed=sum(baseline[key]['original_AND_literal10_passed'] and
                             not expected[key]['original_AND_literal10_passed'] for key in expected),
            newly_lost_survival=sum(baseline[key]['survived'] and not expected[key]['survived'] for key in expected))
        reports[entry['name']] = dict(summary=summaries.summary(rows), versus_nominal=paired,
                                     actual_native_properties_verified=True, output=str(destination))
        raw[entry['name']] = rows
        matrix.write(plan_path.parent / 'progress.json', dict(
            observed_at=datetime.now(timezone.utc).isoformat(), completed_profiles=list(reports),
            reports=reports, policy_accepted=False, training_active=False, hardware_enabled=False))
    result = dict(complete=True, observed_at=datetime.now(timezone.utc).isoformat(),
        diagnostic_preacceptance=diagnostic,
        checkpoint_sha256=plan['checkpoint_sha256'], profiles=reports, all_trials=raw,
        all_failures_retained=True, actual_native_properties_checked_for_every_batch=True,
        original_numeric_gates_unchanged=True, optimizer_executed=False, hardware_enabled=False,
        policy_accepted=False, measured_hardware_uncertainty_claim=False,
        fresh_independent_sustained735_and_deployment_checks_still_required=True,
        source_pins={**plan['source_pins'], **physical_pins, str(plan_path): sha(plan_path)})
    write_new(plan_path.parent / 'complete_deployment_robustness.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'run'))
    parser.add_argument('--plan', type=Path)
    for name in ('request', 'nominal735', 'independent189', 'recipe', 'checkpoint',
                 'verification', 'original-scope', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--runtime-module')
    parser.add_argument('--diagnostic-preacceptance', action='store_true',
                        help='Explicitly test a fully reviewed but unaccepted frozen policy; never promotes it')
    args = parser.parse_args()
    if args.mode == 'qualify':
        if not all(getattr(args, key) for key in ('request', 'nominal735', 'independent189', 'recipe',
                'checkpoint', 'verification', 'original_scope', 'output', 'runtime_module')):
            parser.error('All candidate, scope and request inputs are required for qualification')
        value = qualify(request_path=args.request.resolve(), nominal735=args.nominal735.resolve(),
            independent189=args.independent189.resolve(), runtime_module=args.runtime_module,
            recipe=args.recipe.resolve(), checkpoint=args.checkpoint.resolve(),
            verification=args.verification.resolve(), original_scope=args.original_scope.resolve(),
            output=args.output.resolve(), diagnostic_preacceptance=args.diagnostic_preacceptance)
        print(json.dumps(dict(qualified=value['qualified'], profiles=len(value['profiles']),
                             trials_per_profile=735, simulator_started=False)))
    else:
        if not args.plan:
            parser.error('--plan is required')
        value = run(args.plan.resolve())
        print(json.dumps(dict(complete=value['complete'], profiles=len(value['profiles']),
                             actual_native_properties_checked=True, policy_accepted=False)))
