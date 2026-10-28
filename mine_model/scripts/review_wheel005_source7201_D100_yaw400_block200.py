"""All original reviews, source7201 -> candidate7401, with fresh functions.

Only ownership/clock labels and the two declared expected drive fields are
adapted. Original scorers, cases, seeds, windows, failures and None remain.
Source data are reproduced before production; no compiled adapter is read
back as source and no historical runtime or evidence is modified.
"""
import argparse
import ast
import copy
from datetime import datetime
import inspect
import json
from pathlib import Path
import re
from types import SimpleNamespace

import plain27_sampler_metadata_adapter as adapter
import joint_pd_wheel005_source7201_D100_yaw400_runtime as native_runtime
import verify_wheel005_source7201_D100_yaw400_completed200 as final_check
import review_wheel125_training_block as core
import review_source6001_orientation060_block200 as known_template
import review_source6201_lin300_block200 as retained_template
import review_preserved6801_leg005_fullspeed_height_transitions_v2 as signed_template
import probe_source7001_yawcost200_unchanged_small_yaw120_v2 as small_template
import audit_completed_tracking_reports as error_template
import audit_independent_fullspeed_sustained_raw as long_template
from joint_pd_continuation import ROOT, sha
from qualify_source801_drift50_transfer import differences


SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
PREPARATION = native_runtime.QUALIFICATION
LAUNCH = PREPARATION / 'owned_block200_launch.json'
QUALIFICATION = PREPARATION / 'post200_complete_qualification.json'
SIGNED_QUALIFICATION = PREPARATION / 'post200_signed_qualification.json'
SMALL_QUALIFICATION = PREPARATION / 'post200_small_yaw120_qualification.json'
SOURCE_PARITY = PREPARATION / 'source_all_scopes_original_reader_parity.json'
OUTPUT = RESULTS / 'joint_pd_wheel005_source7201_D100_yaw400_review7401_20261024'
STATIC = RESULTS / 'joint_pd_static_retention_wheel005_D100_yaw400_source7401_20261024'
EXTRA = RESULTS / 'joint_pd_wheel005_D100_yaw400_signed_height7401_20261024'
SMALL = RESULTS / 'joint_pd_wheel005_D100_yaw400_small_yaw120_7401_20261024'
UNIT = 'clt-joint-pd-wheel005-source7201-D100-yaw400-full-reviews-20261024.service'
TRAINING_UNIT = 'clt-joint-pd-wheel005-source7201-D100-yaw400-200-20261024.service'
SOURCE_TRAINING = native_runtime.SOURCE_REPORT.parent
SOURCE_REVIEW = RESULTS / 'joint_pd_AF7001_D075_wheel005_review7201_20261024'
SOURCE_STATIC = RESULTS / 'joint_pd_static_retention_AF7001_D075_wheel005_source7201_20261024'
SOURCE_EXTRA = RESULTS / 'joint_pd_AF7001_D075_wheel005_signed_height_20261024/candidate7201'
SOURCE_SMALL = RESULTS / 'joint_pd_AF7001_D075_wheel005_small_yaw120_20261024'
SOURCE_QUALIFICATION = RESULTS / 'joint_pd_AF7001_wheelD075_wheel005_qualification_20261024'
ALIGNMENT_LABEL = {'all200_all4096_alignment_max_errors': 'all200_all8192_alignment_max_errors'}


def read(path):
    return json.loads(Path(path).read_text())


def ready():
    value = native_runtime.repair_ready()
    return dict(value,
        source_original_sustained120=str(SOURCE_TRAINING / 'independent120_raw_verified_paired_source7001_readonly_20261022.json'),
        source_all_errors=str(SOURCE_TRAINING / 'all_errors_all657_source7201_readonly_20261022.json'))


runtime = SimpleNamespace(**dict(native_runtime.__dict__, TRAINING=SOURCE_TRAINING,
    repair_ready=ready, anchor=native_runtime.baseline,
    REWARD_KEY='yaw_rate_error_sq', REWARD_VALUE=-400.))


def strings(function, names, static_name=None):
    values = {}
    pattern = re.compile('|'.join(re.escape(key) for key in sorted(names, key=len, reverse=True)))
    for node in ast.walk(ast.parse(inspect.getsource(function))):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        value = pattern.sub(lambda match: names[match.group()], node.value)
        value = value.replace('_20261020', '_20261024')
        if static_name and node.value == 'joint_pd_verified_static_source6001_wheel125_20261020.json':
            value = static_name
        if value != node.value:
            values[node.value] = value
    return values


errors = adapter.private_module(error_template.SCRIPT, '_D100_yaw400_original_fixed_raw_reader')
long_raw = adapter.private_module(long_template.SCRIPT, '_D100_yaw400_original_120_raw_reader')
static_raw = adapter.private_module(core.static_raw.SCRIPT, '_D100_yaw400_original_static_raw_reader')
ADAPTERS = []
for module in (errors, long_raw, static_raw):
    ADAPTERS += adapter.retarget(module, ('audit',), exact_strings=ALIGNMENT_LABEL)
core_api = SimpleNamespace(**dict(core.__dict__, errors=errors, static_raw=static_raw))

base = adapter.private_module(core.SCRIPT, '_D100_yaw400_fresh_original_all_scopes')
known = adapter.private_module(known_template.SCRIPT, '_D100_yaw400_fresh_original_known54')
retained = adapter.private_module(retained_template.SCRIPT, '_D100_yaw400_fresh_original_left24')
for module in (base, known, retained):
    module.SCRIPT, module.QUALIFICATION, module.OUTPUT, module.STATIC = SCRIPT, QUALIFICATION, OUTPUT, STATIC
    module.runtime = runtime
base.final_check, base.errors, base.static_raw = final_check, errors, static_raw
known._api, known.original = base, core_api
retained._base, retained._api, retained.core = base, known, core_api
retained.final_check, retained.long_raw = final_check, long_raw
ADAPTERS += adapter.retarget(base, ('run',),
    integers={'*': {6001: 7401, 120020: 148020}},
    exact_strings=strings(core.run, {'source6001': 'source7401'},
        'joint_pd_verified_static_source7401_wheel005_D100_yaw400_20261024.json'))


def expected_differences():
    before = native_runtime.baseline()[2]
    after = native_runtime.expected_configuration(before, native_runtime.NUM_ENVS)
    after['log_dir'] = 'THE_NEXT_OWNED_OUTPUT_DIRECTORY'
    return differences(before, after)


def install_retained():
    tree = ast.parse(inspect.getsource(retained_template.run))
    before, changes = ast.dump(tree), []
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Compare)
        and ast.unparse(node.left) == "final['saved_configuration_differences']"]
    assert len(matches) == 1
    assert ast.literal_eval(matches[0].comparators[0]) == ['/log_dir', '/rewards/tracking_lin_vel']
    changes.append((matches[0], 'comparators', matches[0].comparators))
    matches[0].comparators = [ast.Name(id='EXPECTED_CFG_DIFFERENCES', ctx=ast.Load())]
    labels = strings(retained_template.run,
        {'source6201': 'source7201', 'candidate6401': 'candidate7401', 'source6401': 'source7401'})
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (6201, 6401, 128020):
            changes.append((node, 'value', node.value))
            node.value = {6201: 7201, 6401: 7401, 128020: 148020}[node.value]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in labels:
            changes.append((node, 'value', node.value))
            node.value = labels[node.value]
    modified = copy.deepcopy(tree)
    for node, field, value in reversed(changes):
        setattr(node, field, value)
    assert ast.dump(tree) == before
    retained.EXPECTED_CFG_DIFFERENCES = expected_differences()
    exec(compile(ast.fix_missing_locations(modified), str(retained_template.SCRIPT), 'exec'), retained.__dict__)
    return dict(original_retained_run_AST_restored_identical=True,
        declared_saved_configuration_differences=retained.EXPECTED_CFG_DIFFERENCES)


ADAPTERS.append(install_retained())
signed = adapter.private_module(signed_template.SCRIPT, '_D100_yaw400_fresh_original_signed540')
signed.SCRIPT, signed.QUALIFICATION, signed.OUTPUT = SCRIPT, SIGNED_QUALIFICATION, EXTRA
signed.AFTER_UNIT, signed.runtime, signed.SOURCE = TRAINING_UNIT, runtime, SOURCE_EXTRA
ADAPTERS += adapter.retarget(signed, ('run',),
    integers={'*': {7001: 7401, 140020: 148020}},
    exact_strings=strings(signed_template.run, {'source6801': 'source7201', 'candidate7001': 'candidate7401'}),
    keyword_names={'source6801': 'source7201', 'candidate7001': 'candidate7401'})
small = adapter.private_module(small_template.SCRIPT, '_D100_yaw400_fresh_original_small120')
small.SCRIPT, small.QUALIFICATION, small.OUTPUT, small.runtime = SCRIPT, SMALL_QUALIFICATION, SMALL, runtime
small.CACHE = SOURCE_SMALL
small.SOURCE_DIRECTORY = SOURCE_SMALL / 'candidate7201/candidate'


def install_small():
    tree = ast.parse(inspect.getsource(small_template.run))
    before, changes = ast.dump(tree), []
    fn = tree.body[0]
    indices = [i for i, node in enumerate(fn.body) if isinstance(node, ast.Assign)
        and ast.unparse(node.value) == 'template.assess(directory, candidate, expected, proof)']
    assert len(indices) == 1
    index = indices[0]
    added = ast.parse("expected['wheel_damping'] = 1.\n"
                      "expected['robot']['actuators']['wheels']['damping'] = 1.\n"
                      "expected['rewards']['yaw_rate_error_sq'] = -400.").body
    fn.body[index:index] = added
    labels = strings(small_template.run, {'source7001': 'source7201',
        'candidate7001': 'candidate7201', 'candidate7201': 'candidate7401'})
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value in (7201, 144020):
            changes.append((node, 'value', node.value))
            node.value = {7201: 7401, 144020: 148020}[node.value]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in labels:
            changes.append((node, 'value', node.value))
            node.value = labels[node.value]
    modified = copy.deepcopy(tree)
    del fn.body[index:index + len(added)]
    for node, field, value in reversed(changes):
        setattr(node, field, value)
    assert ast.dump(tree) == before
    exec(compile(ast.fix_missing_locations(modified), str(small_template.SCRIPT), 'exec'), small.__dict__)
    return dict(original_small_run_AST_restored_identical=True,
        only_declared_candidate_expected_drive_and_reward_fields_added=True,
        original_numeric_assess_and_98_windows_unchanged=True)


ADAPTERS.append(install_small())


def assert_bindings():
    for module in (base, known, retained, signed, small):
        assert module.run.__globals__ is module.__dict__ and module.runtime is runtime
        assert module.SCRIPT == SCRIPT
    assert known._api is base and retained._api is known and retained._base is base
    assert base.errors is errors and known.original.errors is errors and retained.core.errors is errors
    assert retained.long_raw is long_raw and base.static_raw is static_raw
    assert signed.SOURCE == SOURCE_EXTRA
    assert small.SOURCE_DIRECTORY == SOURCE_SMALL / 'candidate7201/candidate'
    assert retained.EXPECTED_CFG_DIFFERENCES == expected_differences()
    return dict(all_five_original_run_modules_fresh_and_bound=True,
        all_source_fixtures_are_actual3964_source7201=True,
        original_physical_scorers_cases_seeds_and_numeric_code_unchanged=True,
        original_adapter_ASTs_restored_identical=True)


assert_bindings()


def source_parity():
    """Exercise the exact bound original readers on the completed source."""
    native_runtime.review_ready(native_runtime.baseline()[0])
    if SOURCE_PARITY.exists():
        result = read(SOURCE_PARITY)
        core.intact(result['source_pins'])
        return result
    PREPARATION.mkdir(exist_ok=True)
    source_final = native_runtime.SOURCE_REPORT
    records, outputs = [], []
    specs = [(SOURCE_TRAINING / 'original_all657_tracking_input_specification_20261022.json',
              SOURCE_TRAINING / 'all_errors_all657_source7201_readonly_20261022.json', 'fixed657'),
             (SOURCE_REVIEW / 'known_interior/input_specification.json',
              SOURCE_REVIEW / 'known_interior/all54_raw_verified.json', 'known54'),
             (SOURCE_REVIEW / 'retained_left_interior/input_specification.json',
              SOURCE_REVIEW / 'retained_left_interior/all24_raw_verified.json', 'left24')]
    for spec, original, label in specs:
        output = PREPARATION / ('source_' + label + '_raw_reproduction.json')
        rebuilt = read(output) if output.exists() else errors.audit(source_final, spec, output)
        prior = read(original)
        assert rebuilt['trials'] == prior['trials']
        assert rebuilt['actual_ppo_updates'] == 7201 and rebuilt['checkpoint_sha256'] == native_runtime.ANCHOR_SHA
        assert rebuilt['all7_metrics_and_limits_raw_reproduction_max_error'] == 0.
        outputs += [output, original, spec]
        records.append(dict(scope=label, trials=rebuilt['total_trials'], original_rows_reproduced=True))
        print(json.dumps(dict(source_reader_parity=label, trials=rebuilt['total_trials'])), flush=True)
    for directories, prior, label in [
        ([SOURCE_REVIEW / 'moving_height/deterministic', SOURCE_REVIEW / 'moving_height/saved_Gaussian',
          SOURCE_REVIEW / 'forward_steering/deterministic'], SOURCE_TRAINING / 'motion_and_steering_all810_phases_raw_verified_readonly_20261022.json', 'motion810'),
        ([SOURCE_EXTRA / mode for mode in ('deterministic', 'saved_Gaussian')], SOURCE_EXTRA / 'all540_phases_raw_verified.json', 'signed540')]:
        output = PREPARATION / ('source_' + label + '_raw_reproduction.json')
        rebuilt = read(output) if output.exists() else retained.phase_raw.audit(directories, output)
        assert rebuilt['trials'] == read(prior)['trials']
        outputs += [output, prior]
        records.append(dict(scope=label, phases=rebuilt['total_trials'], original_rows_reproduced=True))
        print(json.dumps(dict(source_reader_parity=label, phases=rebuilt['total_trials'])), flush=True)
    static_output = RESULTS / 'joint_pd_D100_yaw400_source7201_CPU_static_raw_reproduction_20261024.json'
    stat = read(static_output) if static_output.exists() else static_raw.audit(SOURCE_TRAINING, SOURCE_STATIC, static_output)
    assert stat['raw_integrity_verified'] and stat['checkpoint_sha256'] == native_runtime.ANCHOR_SHA
    assert stat['static_retention_verified']
    outputs.append(static_output)
    prior_long = SOURCE_TRAINING / 'independent120_raw_verified_paired_source7001_readonly_20261022.json'
    baseline_long = Path(read(prior_long)['paired_comparison']['baseline'])
    long_output = PREPARATION / 'source_original120_raw_reproduction.json'
    rebuilt = read(long_output) if long_output.exists() else long_raw.audit(SOURCE_TRAINING, SOURCE_REVIEW, long_output, baseline_long)
    assert rebuilt['trials'] == read(prior_long)['trials'] and rebuilt['survivors'] == 54
    outputs += [long_output, prior_long, baseline_long]
    proof = read(SOURCE_QUALIFICATION / 'post200_small_yaw120_qualification.json')
    proof = dict(proof, source_reference_resolved_env=str(small.SOURCE_DIRECTORY / 'resolved_env.json'))
    expected = read(small.SOURCE_DIRECTORY / 'resolved_env.json')
    report, _ = small.template.assess(small.SOURCE_DIRECTORY, native_runtime.baseline()[0], expected, proof)
    prior_small = SOURCE_SMALL / 'candidate7201/all54_raw_and_windows_verified.json'
    assert report == read(prior_small)
    outputs.append(prior_small)
    pins = {str(path): sha(path) for path in [SCRIPT, adapter.SCRIPT, native_runtime.SCRIPT,
        core.SCRIPT, known_template.SCRIPT, retained_template.SCRIPT, signed_template.SCRIPT,
        small_template.SCRIPT, error_template.SCRIPT, long_template.SCRIPT, core.static_raw.SCRIPT,
        source_final, native_runtime.baseline()[0], *outputs]}
    for output in outputs:
        value = read(output)
        if isinstance(value, dict):
            for key in ('source_pins', 'source_files'):
                if key in value:
                    core.intact(value[key])
    core.intact(pins)
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        source_checkpoint_sha256=native_runtime.ANCHOR_SHA, source_actual_updates=7201,
        fixed735_original_metric_limit_flags_reproduced=True, families=records,
        static_all_raw_timelines_verified=True, original120_all98_windows_and_flags_reproduced=True,
        small_yaw120_all54_and98_windows_exactly_reproduced=True,
        source_numeric_failures_and_None_preserved=True,
        bindings=assert_bindings(), adapters=ADAPTERS, source_pins=pins,
        PPO_updates=0, simulator_started=False, policy_accepted=False)
    core.save(SOURCE_PARITY, result)
    return result


def qualify():
    parity = source_parity()
    launch = read(LAUNCH)
    training = Path(launch['output'])
    plan_path = training / 'experiment_plan.json'
    plan = read(plan_path)
    assert plan['source_sha256'] == native_runtime.ANCHOR_SHA
    assert plan['source_actual_ppo_updates'] == 7201 and plan['target_actual_ppo_updates'] == 7401
    assert plan['expected_terminal_Adam_step'] == 148020 and plan['native_num_envs'] == 8192
    assert plan['profile'] == native_runtime.profile() and plan['next_configuration_changes'] == native_runtime.DELTA
    assert not any(path.exists() for path in (QUALIFICATION, SIGNED_QUALIFICATION, SMALL_QUALIFICATION, OUTPUT, STATIC, EXTRA, SMALL))
    old = read(SOURCE_QUALIFICATION / 'post200_complete_qualification.json')
    coverage = ('full_height_cases', 'full_seeds', 'combined_batches', 'combined_seeds', 'moving_plan',
        'moving_seeds', 'steering_plan', 'steering_seeds', 'original_static_height_plan',
        'sustained_cases', 'sustained_seeds', 'known_interior_cases', 'known_interior_seeds',
        'retained_left_cases', 'retained_left_seeds')
    proof = {key: copy.deepcopy(old[key]) for key in coverage}
    for scale in core.SCALES:
        m = read(SOURCE_REVIEW / ('full_height_' + str(scale)) / 'candidate/manifest.json')
        assert m['cases'] == proof['full_height_cases'][str(scale)] and m['seeds'] == proof['full_seeds']
    for level, cases in proof['combined_batches'].items():
        m = read(SOURCE_REVIEW / 'combined/candidate' / level / 'manifest.json')
        assert m['cases'] == cases and m['seeds'] == proof['combined_seeds']
    assert proof['original_static_height_plan'] == core.static_steps.step_plan()
    assert proof['sustained_cases'] == core.sustained.cases_for(plan['training_config'])
    pins = {**plan['source_files'], **parity['source_pins'],
        **read(native_runtime.ADMISSION)['source_pins'],
        str(LAUNCH): sha(LAUNCH), str(plan_path): sha(plan_path),
        str(SOURCE_PARITY): sha(SOURCE_PARITY), str(final_check.SCRIPT): sha(final_check.SCRIPT)}
    core.intact(pins)
    proof.update(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        training_launch=str(LAUNCH), training=str(training), after_unit=TRAINING_UNIT,
        output=str(OUTPUT), static_output=str(STATIC), source_actual_updates=7201,
        source_checkpoint_sha256=native_runtime.ANCHOR_SHA,
        target_actual_updates=7401, target_all17_Adam_step=148020,
        prepared_plan_path=str(plan_path), native_num_envs=8192, fixed_standing_environments=2048,
        known_interior_source_directory=str(SOURCE_REVIEW / 'known_interior/candidate'),
        known_interior_source_raw=str(SOURCE_REVIEW / 'known_interior/all54_raw_verified.json'),
        retained_left_source_directory=str(SOURCE_REVIEW / 'retained_left_interior/candidate'),
        retained_left_source_raw=str(SOURCE_REVIEW / 'retained_left_interior/all24_raw_verified.json'),
        complete_review_expected=str(OUTPUT / 'complete_with_retained_interior_reviews.json'),
        all_original_scopes_and_numeric_readers_retained=True,
        declared_expected_configuration_differences=expected_differences(),
        only_controller_change=native_runtime.DELTA['controller'], only_exploration_change={},
        only_reward_change=native_runtime.DELTA['rewards'],
        source_all_required_original_scopes_reproduced=True,
        fresh_module_binding_preflight=assert_bindings(), metadata_adapters=ADAPTERS,
        source_pins=pins, acceptance_changed=False, policy_accepted=False)
    core.save(QUALIFICATION, proof)
    signed_old = read(SOURCE_QUALIFICATION / 'post200_signed_qualification.json')
    signed_proof = {key: copy.deepcopy(signed_old[key]) for key in (
        'command_plan', 'seeds', 'episodes_per_run', 'phases_per_run', 'samples_per_run', 'modes',
        'signed_ramp_bounds', 'reset_velocity_amplitude_unchanged',
        'physical_min_root_height_unchanged', 'low_height_terminal_time_unchanged')}
    cached = read(SOURCE_EXTRA.parent / 'complete_paired_fullspeed_height_transitions.json')['results']['candidate7201']
    signed_proof.update(qualified=True, observed_at=proof['observed_at'], after_unit=TRAINING_UNIT,
        training=str(training), output=str(EXTRA), source_checkpoint=str(native_runtime.baseline()[0]),
        source_checkpoint_sha256=native_runtime.ANCHOR_SHA, source_actual_updates=7201,
        candidate_actual_updates=7401, candidate_all17_Adam_step=148020,
        cached_source_directory=str(SOURCE_EXTRA), cached_source_results=cached,
        source_completed_signed_transition_review=str(SOURCE_EXTRA.parent / 'complete_paired_fullspeed_height_transitions.json'),
        original_complete_required=str(OUTPUT / 'complete_with_retained_interior_reviews.json'),
        only_exploration_change={}, only_controller_change=native_runtime.DELTA['controller'],
        only_reward_change=native_runtime.DELTA['rewards'],
        original_signed_height_numeric_readers_and_all_failures_preserved=True,
        sampled_numeric_gates_descriptive_original_stress_gate_survival_only=True,
        source_pins=pins, native_num_envs=8192, policy_accepted=False)
    core.save(SIGNED_QUALIFICATION, signed_proof)
    small_old = read(SOURCE_QUALIFICATION / 'post200_small_yaw120_qualification.json')
    small_proof = {key: copy.deepcopy(small_old[key]) for key in (
        'cases', 'seeds', 'seconds_per_case', 'stable_seconds', 'original_windows_per_trial',
        'episodes_per_policy', 'total_paired_episodes', 'diagnostic_prefix_ends_s')}
    small_proof.update(qualified=True, observed_at=proof['observed_at'], after_unit=TRAINING_UNIT,
        training=str(training), output=str(SMALL), source_checkpoint=str(native_runtime.baseline()[0]),
        source_checkpoint_sha256=native_runtime.ANCHOR_SHA, source_actual_updates=7201,
        target_actual_updates=7401, target_all17_Adam_step=148020,
        source_reference_resolved_env=str(small.SOURCE_DIRECTORY / 'resolved_env.json'),
        cached_source_native_directory=str(small.SOURCE_DIRECTORY),
        complete_original_review_required=str(OUTPUT / 'complete_with_retained_interior_reviews.json'),
        signed_height_review_required=str(EXTRA / 'complete_paired_fullspeed_height_transitions.json'),
        candidate_expected_reward_difference=native_runtime.DELTA['rewards'], candidate_expected_controller_difference=native_runtime.DELTA['controller'],
        only_exploration_change={}, cached_source_runs_reused=1, new_candidate_native_runs=1,
        original_heldout_source_seeds_reused_for_paired_comparison=True,
        no_new_fresh_seed_claim_for_cached_source=True,
        original_numeric_score_function_and_all_failures_preserved=True,
        source_pins=pins, native_num_envs=8192, policy_accepted=False)
    core.save(SMALL_QUALIFICATION, small_proof)
    return proof


def run():
    proof = read(QUALIFICATION)
    core.intact(proof['source_pins'])
    result, extra, supplement = retained.run(), signed.run(), small.run()
    core.intact(proof['source_pins'])
    complete = dict(completed=True, observed_at=datetime.now().astimezone().isoformat(),
        checkpoint=result['checkpoint'], checkpoint_sha256=result['checkpoint_sha256'], actual_updates=7401,
        original735_static810_original120_complete=result['all_original_and_retained_reviews_complete'],
        signed_fullspeed_height_complete=extra['completed'], small_yaw120_complete=supplement['completed'],
        original_complete=str(OUTPUT / 'complete_with_retained_interior_reviews.json'),
        signed_complete=str(EXTRA / 'complete_paired_fullspeed_height_transitions.json'),
        small_yaw_complete=str(SMALL / 'complete_paired_small_yaw_sustained120.json'),
        numeric_gates_and_all_failures_preserved=True, source_pins=proof['source_pins'],
        can_select_or_extend_training=False, training_modified=False, policy_accepted=False)
    core.save(OUTPUT / 'complete_all_scopes_wheel005_D100_yaw400.json', complete)
    return complete


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source-only', action='store_true')
    cli.add_argument('--qualify-only', action='store_true')
    args = cli.parse_args()
    result = source_parity() if args.source_only else (qualify() if args.qualify_only else run())
    print(json.dumps({key: result[key] for key in ('qualified', 'completed', 'policy_accepted') if key in result}))
