"""Original full/static/travel/sustained review of exactly one wheel125 block.

Collect every failure. This worker cannot select, extend, roll back or accept
a policy. It uses the original numerical readers and saved normal controller.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
from audit_plain27_after_queued_reviews import wait_unit
from review_plain27_single_reward_block200 import rebound
import joint_pd_source5801_wheel125_runtime as runtime
import verify_completed_wheel125_joint_pd_trial as final_check
import review_source5201_motion075_combined_turns as combined
import diagnose_consolidation_motion_height_steps as numeric
import verify_static_height_steps10 as static_steps
import validate_quality50_static_retention10 as static_template
import audit_completed_joint_pd_retention_raw as static_raw
import audit_completed_tracking_reports as errors
import verify_independent_fullspeed_sustained as sustained

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_wheel125_post200_review_qualification_20261020.json'
OUTPUT = RESULTS / 'joint_pd_wheel125_review6001_20261020'
STATIC = RESULTS / 'joint_pd_static_retention_wheel125_20261020_source6001'
SCALES = (.25, .375, .5, .625, .75, 1.)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def intact(pins):
    changed = [p for p, h in pins.items() if sha(p) != h]
    if changed:
        raise ValueError('Preserve qualified current and archived source evidence: ' + repr(changed[:6]))


def qualify(launch_path):
    launch_path = Path(launch_path).resolve()
    launch = read(launch_path)
    training = Path(launch['output'])
    plan = read(training / 'experiment_plan.json')
    fixture = read(launch['native_fixture_verification'])
    if (QUALIFICATION.exists() or OUTPUT.exists() or STATIC.exists()
            or launch['source_checkpoint_sha256'] != runtime.ANCHOR_SHA
            or launch['target_actual_updates'] != 6001 or launch['target_all17_Adam_step'] != 120020
            or launch['planned_actual_updates'] != 200 or launch['next_configuration_changes'] != runtime.DELTA
            or not fixture['verified'] or plan['profile'] != runtime.profile()
            or plan['only_controller_change'] != runtime.DELTA['controller']
            or plan['early_learning_inspection_actual_updates'] != [50, 100]):
        raise ValueError('Qualify only the owned exact200 wheel-clamp trial')
    baseline = read(runtime.TRAINING / 'retained_six_motion_envelopes_review_20261019.json')
    full_cases = {}
    for scale in SCALES:
        directory = Path(baseline['suites'][str(scale)]['output']) / 'candidate'
        manifest = read(directory / 'manifest.json')
        if len(manifest['cases']) != 17 or manifest['seeds'] != [483, 484, 485]:
            raise ValueError('Preserve every original full-height case and reset seed')
        full_cases[str(scale)] = manifest['cases']
    old_combined = RESULTS / 'joint_pd_combined_turns_consolidation_20261020_source5601_candidate5801/candidate'
    batches = {level: read(old_combined / level / 'manifest.json')['cases'] for level in ('low', 'nominal', 'high')}
    moving_path = RESULTS / 'joint_pd_motion_height_steps_consolidation_20261020_source5801/deterministic_commands.json'
    steering_path = RESULTS / 'joint_pd_forward_steering_steps_consolidation_fullheight_20261020_source5801/deterministic_commands.json'
    moving, steering = read(moving_path), read(steering_path)
    # Exercise the unchanged numeric reader on the normal saved-controller
    # fixture, not only on a unit-bound historical policy.
    fixture_dir = RESULTS / 'joint_pd_source5801_wheel125_native_fixture_v2_20261020/candidate'
    m = read(fixture_dir / 'manifest.json')
    reader, writes, adapter_proof = combined.reader(m['cases'], m['seeds'])
    rebuilt = reader(fixture_dir, Path(fixture['untrained_fixture']))
    if len(writes) != 1 or rebuilt['total_trials'] != 108 or rebuilt['survivors'] != 108 or rebuilt['resets']:
        raise ValueError('Original108-trial numerical reader rejected the normal saved wheel125 controller')
    pins = dict(plan['source_files'])
    for p in (SCRIPT, launch_path, runtime.SCRIPT, final_check.SCRIPT, combined.SCRIPT,
            numeric.SCRIPT, static_raw.SCRIPT, errors.SCRIPT, sustained.SCRIPT,
            Path(static_steps.__file__), Path(static_template.__file__),
            Path(static_template.sustained.__file__), Path(numeric.acceptance.assessor.__file__),
            numeric.acceptance.CURRENT, numeric.acceptance.POLICY,
            runtime.TRAINING / 'retained_six_motion_envelopes_review_20261019.json',
            fixture_dir / 'manifest.json', fixture_dir / 'trace.csv', moving_path, steering_path):
        pins[str(p)] = sha(p)
    intact(pins)
    proof = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        training_launch=str(launch_path), training=str(training), after_unit=launch['unit'],
        output=str(OUTPUT), static_output=str(STATIC), target_actual_updates=6001,
        target_all17_Adam_step=120020, full_height_cases=full_cases, full_seeds=[483, 484, 485],
        combined_batches=batches, combined_seeds=[683, 684, 685],
        moving_plan=moving, moving_seeds=[583, 584, 585],
        steering_plan=steering, steering_seeds=[783, 784, 785],
        original_static_height_plan=static_steps.step_plan(),
        sustained_cases=sustained.cases_for(plan['training_config']), sustained_seeds=[1283, 1284, 1285],
        canonical27_plus_six51_plus_combined324_total=657,
        original_saved_controller_fixture_numeric_reader_verified=True,
        unchanged_numeric_reader_metadata_adapter=adapter_proof,
        original_metrics_thresholds_absolute_floors_windows_and_failed_None_preserved=True,
        source_pins=pins, acceptance_changed=False, policy_accepted=False,
        can_extend_training=False, can_select_or_rollback=False)
    save(QUALIFICATION, proof)
    return proof


def native(source, directory, seconds, seeds, commands=None, sampled=False):
    directory = Path(directory)
    if directory.exists():
        raise ValueError('Inspect existing native data before any retry: ' + str(directory))
    directory.parent.mkdir(parents=True, exist_ok=True)
    argv = runtime.replay_command(source, directory)
    argv[argv.index('--seconds') + 1] = str(seconds)
    i = argv.index('--seeds') + 1
    argv[i:i + 3] = list(map(str, seeds))
    if commands is not None:
        command_file = directory.parent / (directory.name + '_commands.json')
        save(command_file, commands)
        argv.extend(['--command-probe-json', str(command_file)])
    if sampled:
        argv.append('--sample-policy-actions')
    if '--record_policy_inputs' not in argv:
        argv.append('--record_policy_inputs')
    if any(x in argv for x in ('--overrides_json', '--wheel-reference-limit-probe', '--heading-hold',
            '--command-scale', '--export_dir', runtime.FLAG)):
        raise ValueError('Review the normal saved controller without feedback or diagnostic overrides')
    save(directory.parent / (directory.name + '_launch.json'), dict(command=argv,
        checkpoint_sha256=sha(source), qualification_sha256=sha(QUALIFICATION)))
    with (directory.parent / (directory.name + '.log')).open('x') as stream:
        subprocess.run(argv, cwd=ROOT, env=numeric.native.environment(),
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    manifest, complete, cfg = [read(directory / n) for n in ('manifest.json', 'completed.json', 'resolved_env.json')]
    if (not complete['completed'] or manifest['checkpoint_sha256'] != sha(source)
            or manifest['saved_wheel_action_limit'] != 1.25 or cfg['wheel_action_limit'] != 1.25
            or cfg['action_scale_vel'] != 15. or manifest['wheel_reference_max_rad_s'] != 18.75
            or manifest['wheel_reference_limit_probe'] is not None or manifest['weights_updated']):
        raise ValueError('Actual normal saved wheel/leg execution or completion differs')
    return directory


def run():
    proof = read(QUALIFICATION)
    intact(proof['source_pins'])
    if not proof['qualified'] or OUTPUT.exists():
        raise ValueError('Preserve previous native reviews and their failures')
    OUTPUT.mkdir()
    status = OUTPUT / 'status.json'
    wait_unit(proof['after_unit'], status)
    training = Path(proof['training'])
    final_path = training / 'final_checkpoint_verification_readonly_20261019.json'
    final = final_check.verify(training) if not final_path.exists() else read(final_path)
    if not final['verified'] or final['actual_ppo_updates'] != 6001 or final['all17_Adam_step'] != 120020:
        raise ValueError('Require exact completed200 and all17 Adam clocks before frozen review')
    source = Path(final['checkpoint'])
    runtime.require_source(source)
    inputs = [dict(scope='canonical27', kind='canonical', directory=str(training / 'replay'),
        report=str(training / 'replay/assessment.json'))]
    full_reports = {}
    for scale in SCALES:
        write_json(status, dict(state='full_height_' + str(scale), training_active=False, policy_accepted=False))
        directory = native(source, OUTPUT / ('full_height_' + str(scale)) / 'candidate', 20.,
            proof['full_seeds'], proof['full_height_cases'][str(scale)])
        reader, _, _ = combined.reader(proof['full_height_cases'][str(scale)], proof['full_seeds'])
        report = reader(directory, source)
        path = directory / 'motion025_full_height_diagnosis.json'
        save(path, report)
        full_reports[str(scale)] = dict(output=str(directory.parent), total_trials=51,
            survivors=report['survivors'], resets=report['resets'],
            passed=sum(t['stage_constraints_passed'] for t in report['trials']))
        inputs.append(dict(scope='motion_scale' + str(scale), kind='fixed_command_probe',
            directory=str(directory), report=str(path)))
        intact(proof['source_pins'])
    save(training / 'retained_six_motion_envelopes_review_20261020.json', dict(
        checkpoint_sha256=sha(source), actual_updates=6001, suites=full_reports, policy_accepted=False))
    STATIC.mkdir()
    fixed = STATIC / 'fixed_height_review'
    fixed.mkdir()
    stages, steps = {}, {}
    with numeric.acceptance.use_current():
        stages['baseline'], baseline_manifest, baseline_cfg = static_template.sustained.assess_stage(
            training / 'replay', fixed / 'baseline', sha(source), 'baseline')
    for label, seconds, seeds, stable in (('heldout', 30., [183, 184, 185], 20.),
            ('sustained', 60., [283, 284, 285], 40.)):
        write_json(status, dict(state='static_' + label, training_active=False, policy_accepted=False))
        directory = native(source, STATIC / label, seconds, seeds)
        with numeric.acceptance.use_current():
            stages[label], _, _ = static_template.sustained.assess_stage(directory, fixed / label,
                sha(source), label, baseline_manifest, baseline_cfg)
    command_plan = STATIC / 'command_plan.json'
    save(command_plan, proof['original_static_height_plan'])
    for label, sampled in (('height_steps_deterministic', False), ('height_steps_sampled', True)):
        directory = native(source, STATIC / label, 50., [383, 384, 385],
            proof['original_static_height_plan'], sampled)
        steps[label] = static_steps.verify(directory, command_plan, sampled=sampled)
    deterministic, gaussian_alive = static_template.milestone.step_readiness(steps)
    static_report = dict(checkpoint=str(source), checkpoint_sha256=sha(source),
        static_retention_verified=bool(all(s['static_passed'] for s in stages.values()) and deterministic and gaussian_alive),
        stages=stages, deterministic_full_height_steps_passed=deterministic,
        sampled_exploration_survived=gaussian_alive, height_step_reports={label:
            str(STATIC / label / 'static_height_step_validation.json') for label in steps},
        same_original_thresholds_and_windows=True, training_modified=False, policy_accepted=False)
    save(STATIC / 'static_retention.json', static_report)
    static_raw.audit(training, STATIC, RESULTS / 'joint_pd_verified_static_source6001_wheel125_20261020.json')
    moving_reports = {}
    for label, sampled in (('deterministic', False), ('saved_Gaussian', True)):
        write_json(status, dict(state='moving_' + label, training_active=False, policy_accepted=False))
        directory = native(source, OUTPUT / 'moving_height' / label, 50., proof['moving_seeds'], proof['moving_plan'], sampled)
        reader = rebound(numeric.assess, plan=lambda: proof['moving_plan'], SEEDS=proof['moving_seeds'])
        moving_reports[label] = reader(directory, source, sampled)
    combined_reports = {}
    for level, cases in proof['combined_batches'].items():
        write_json(status, dict(state='combined_' + level, training_active=False, policy_accepted=False))
        directory = native(source, OUTPUT / 'combined' / 'candidate' / level, 20., proof['combined_seeds'], cases)
        reader, _, _ = combined.reader(cases, proof['combined_seeds'])
        report = reader(directory, source)
        path = directory / 'combined_turn_tracking_diagnosis.json'
        save(path, report)
        combined_reports[level] = report
        inputs.append(dict(scope='combined_' + level, kind='fixed_command_probe', directory=str(directory), report=str(path)))
    write_json(status, dict(state='forward_steering_steps', training_active=False, policy_accepted=False))
    directory = native(source, OUTPUT / 'forward_steering' / 'deterministic', 50., proof['steering_seeds'], proof['steering_plan'])
    reader = rebound(numeric.assess, plan=lambda: proof['steering_plan'], SEEDS=proof['steering_seeds'])
    steering_report = reader(directory, source, False)
    specification = training / 'original_all657_tracking_input_specification_20261020.json'
    save(specification, inputs)
    all_errors = errors.audit(final_path, specification, training / 'all_errors_all657_source6001_readonly_20261020.json')
    write_json(status, dict(state='independent_sustained120', training_active=False, policy_accepted=False))
    directory = native(source, OUTPUT / 'sustained120' / 'candidate', 120.,
        proof['sustained_seeds'], proof['sustained_cases'])
    data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True).reshape(6000, 54)
    trials = sustained.score(data, proof['sustained_cases'], proof['sustained_seeds'], 120., 100.)
    sustained_report = dict(checkpoint_sha256=sha(source), actual_updates=6001,
        total_trials=54, survivors=sum(t['survived'] for t in trials), passed=sum(t['passed'] for t in trials),
        raw_OR_resets=sum(t['raw_OR_resets'] for t in trials), trials=trials,
        original98_overlapping_150frame_windows_and_strict_limits=True, policy_accepted=False)
    save(OUTPUT / 'sustained120/independent_fullspeed_sustained.json', sustained_report)
    intact(proof['source_pins'])
    result = dict(observed_at=datetime.now().astimezone().isoformat(), actual_updates=6001,
        checkpoint=str(source), checkpoint_sha256=sha(source), all_required_review_data_complete=True,
        full_height=full_reports, static_retention_verified=static_report['static_retention_verified'],
        moving_height={k: {f: v[f] for f in ('total_trials', 'survivors', 'resets', 'all_numeric_phase_gates_passed')}
            for k, v in moving_reports.items()}, combined_turns={k: dict(total_trials=v['total_trials'],
                survivors=v['survivors'], resets=v['resets'], passed=sum(t['stage_constraints_passed'] for t in v['trials']))
            for k, v in combined_reports.items()},
        forward_steering=dict(total_trials=27, survivors=steering_report['survivors'],
            resets=steering_report['resets'], passed=sum(t['passed'] for t in steering_report['trials'])),
        sustained120={k: sustained_report[k] for k in ('total_trials', 'survivors', 'passed', 'raw_OR_resets')},
        original_all657_error_report=str(training / 'all_errors_all657_source6001_readonly_20261020.json'),
        source_pins=proof['source_pins'], training_modified=False, optimizer_executed=False,
        acceptance_changed=False, policy_accepted=False, can_extend_training=False)
    save(OUTPUT / 'complete_original_review.json', result)
    write_json(status, dict(state='agent_review_required', native_review_active=False,
        training_active=False, policy_accepted=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-launch', type=Path)
    args = parser.parse_args()
    result = qualify(args.qualify_launch) if args.qualify_launch else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'actual_updates', 'all_required_review_data_complete',
        'static_retention_verified', 'policy_accepted') if k in result}))
