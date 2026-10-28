"""Recheck static retention of a completed motion025 quality50 PPO branch.

Reuse the original full27 and full-height step readers and their10% limits.
This validates the current checkpoint, without calling a historical static
milestone current, releasing a faster curriculum, or starting another PPO run.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

import source1201_orientation40
import joint_pd_source2001_quality50_runtime as runtime
import joint_pd_acceptance_10pct_20261017 as acceptance
import run_static_reset_pair_diagnosis as boundary
import run_static_milestone_validation10 as milestone
import verify_stationary_motion_release10 as sustained
from verify_static_height_steps10 import step_plan, verify as verify_steps
from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_quality50_static_retention_qualification_20261018.json'
SPECS = (
    ('heldout', 30., [183, 184, 185], False, False),
    ('sustained', 60., [283, 284, 285], False, False),
    ('height_steps_deterministic', 50., [383, 384, 385], True, False),
    ('height_steps_sampled', 50., [383, 384, 385], True, True),
)


def dependencies():
    paths = (SCRIPT, runtime.SCRIPT, runtime.criteria_fix.SCRIPT,
        Path(boundary.__file__), Path(milestone.__file__), Path(sustained.__file__),
        Path(verify_steps.__code__.co_filename), acceptance.CURRENT, acceptance.POLICY)
    return {str(path): sha(path) for path in paths}


def reviewed_source(training_output):
    training_output = Path(training_output).resolve()
    state = json.loads((training_output / 'status.json').read_text())
    if state['training_active'] or state['state'] != 'agent_review_required':
        raise ValueError('Finish the exact200-update block and full27 review first')
    original = boundary.smooth
    boundary.smooth = runtime
    try:
        source, entry = boundary.reviewed_source(training_output)
    finally:
        boundary.smooth = original
    _, cfg = runtime.require_source(source)
    if (entry != runtime.SCRIPT or cfg['commands']['training_motion_scale'] != .25
            or cfg['scene']['num_envs'] != 4096 or cfg['terrain']['num_envs'] != 4096):
        raise ValueError('Require the intact motion025/quality50/full4096 descendant')
    return source


def command(source, output, label, command_plan):
    matches = [spec for spec in SPECS if spec[0] == label]
    if len(matches) != 1:
        raise ValueError('Use one of the original independent validation specifications')
    _, seconds, seeds, steps, sampled = matches[0]
    argv = runtime.replay_command(source, output)
    argv[argv.index('--seconds') + 1] = str(seconds)
    index = argv.index('--seeds') + 1
    argv[index:index + 3] = [str(seed) for seed in seeds]
    if steps:
        argv.extend(['--command-probe-json', str(command_plan)])
    if sampled:
        argv.append('--sample-policy-actions')
    banned = ('--command-scale', '--overrides_json', '--heading-hold', '--export_dir', runtime.FLAG)
    if (any(flag in argv for flag in banned) or '--headless' not in argv
            or argv.count(str(runtime.SCRIPT)) != 1
            or ('--sample-policy-actions' in argv) is not sampled):
        raise ValueError('No input, command, control, reward or checkpoint overrides in retention validation')
    return argv


def qualify(checkpoint, active_training, output):
    """CPU-only CLI and reader compatibility checks, never native validation."""
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    runtime.require_source(checkpoint)
    if output != QUALIFICATION or output.exists():
        raise ValueError('Preserve the owned qualification proof')
    try:
        reviewed_source(active_training)
    except ValueError as error:
        if str(error) != 'Finish the exact200-update block and full27 review first':
            raise
    else:
        raise ValueError('Active training was admitted as a completed review source')
    commands = {label: command(checkpoint, RESULTS / ('not_native_static_retention_' + label),
        label, RESULTS / 'not_native_static_retention_command_plan.json') for label, *_ in SPECS}
    # These validators must stay identical. Testing command construction does
    # not pretend that the intermediate checkpoint completed independent replays.
    if sustained.SPECS != dict(baseline=(20., [83, 84, 85], None),
            heldout=(30., [183, 184, 185], 20.), sustained=(60., [283, 284, 285], 40.)):
        raise ValueError('Original independent durations, seeds or windows changed')
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        dependencies=dependencies(), qualification_checkpoint=str(checkpoint),
        active_training_rejected=True, commands=commands, step_plan=step_plan(),
        same_full27_and_step_metric_readers=True, original_thresholds_and_windows_unchanged=True,
        source_guard='Original completed200/full27/full17/pinned-source selector; declared profile only',
        source_training_motion_scale=.25, training_modified=False, PPO_updates_performed=0,
        native_simulations_started=0, independent_validation_passed=False,
        policy_accepted=False, hardware_enabled=False)
    write_json(output, result)
    return result


def require_full51(directory, source):
    directory = Path(directory).resolve()
    status = json.loads((directory / 'status.json').read_text())
    candidate = directory / 'candidate'
    report = json.loads((candidate / 'motion025_full_height_diagnosis.json').read_text())
    complete = json.loads((candidate / 'completed.json').read_text())
    if (status['state'] != 'agent_review_required' or status['diagnostic_active']
            or report['checkpoint_sha256'] != sha(source)
            or report['total_trials'] != 51 or len(report['trials']) != 51
            or not complete['completed'] or complete['samples'] != 51000
            or any(sha(p) != digest for p, digest in report['source_files'].items())):
        raise ValueError('Finish and preserve the current checkpoint full51 review before retention validation')
    return directory


def run(training_output, full51, output):
    training_output, output = Path(training_output).resolve(), Path(output).resolve()
    qualified = json.loads(QUALIFICATION.read_text())
    if not qualified['qualified'] or qualified['dependencies'] != dependencies():
        raise ValueError('Requalify changed validation code or original criteria')
    source = reviewed_source(training_output)
    full51 = require_full51(full51, source)
    if (output.parent != RESULTS or output.exists()
            or not output.name.startswith('joint_pd_static_retention_')):
        raise ValueError('Use a fresh owned output; preserve all previous evidence')
    output.mkdir()
    digest = sha(source)
    write_json(output / 'plan.json', dict(observed_at=datetime.now().astimezone().isoformat(),
        training_output=str(training_output), full51_review=str(full51), checkpoint=str(source),
        checkpoint_sha256=digest, dependencies=dependencies(), acceptance_revision=acceptance.profile(),
        training_motion_scale=.25, source_training_is_not_all_static=True,
        PPO_updates_performed=0, training_modified=False, policy_accepted=False, hardware_enabled=False))
    try:
        milestone.baseline_ready(training_output)
    except ValueError as error:
        if str(error) != 'The completed canonical static stage still fails; diagnose before expensive validation':
            raise
        result = dict(state='not_run_canonical_static_failed', validation_active=False,
            checkpoint_sha256=digest, static_retention_verified=False, native_simulations_started=0,
            reason=str(error), policy_accepted=False, goal_remains_active=True)
        write_json(output / 'status.json', result)
        return result
    command_plan = output / 'command_plan.json'
    write_json(command_plan, step_plan())
    review = output / 'fixed_height_review'
    review.mkdir()
    baseline, manifest, cfg = sustained.assess_stage(training_output / 'replay',
        review / 'baseline', digest, 'baseline')
    stages, steps = dict(baseline=baseline), {}
    for label, _, seeds, changing, sampled in SPECS:
        destination = output / label
        argv = command(source, destination, label, command_plan)
        write_json(output / (label + '_launch.json'), dict(command=argv, checkpoint_sha256=digest))
        write_json(output / 'status.json', dict(state=label, validation_active=True,
            training_active=False, checkpoint_sha256=digest, policy_accepted=False))
        with (output / (label + '.log')).open('w') as log:
            subprocess.run(argv, cwd=ROOT, env=milestone.smooth.previous_controller.helper.base.environment(),
                stdout=log, stderr=subprocess.STDOUT, check=True)
        if changing:
            steps[label] = verify_steps(destination, command_plan, seeds=seeds, sampled=sampled)
        else:
            stages[label], _, _ = sustained.assess_stage(destination, review / label,
                digest, label, manifest, cfg)
        if sha(source) != digest or qualified['dependencies'] != dependencies():
            raise ValueError('Frozen checkpoint or validation implementation changed')
    deterministic, sampled_survived = milestone.step_readiness(steps)
    retained = bool(all(stage['static_passed'] for stage in stages.values())
        and deterministic and sampled_survived)
    result = dict(checkpoint=str(source), checkpoint_sha256=digest,
        static_retention_verified=retained, stages=stages,
        deterministic_full_height_steps_passed=deterministic,
        sampled_exploration_survived=sampled_survived,
        height_step_reports={label: str(output / label / 'static_height_step_validation.json') for label in steps},
        same_original_thresholds_and_windows=True, acceptance_revision=acceptance.profile(),
        full27_moving_failures_retained=True, full51_review=str(full51),
        source_training_motion_scale=.25, faster_curriculum_authorized=False,
        full_goal_acceptance_not_implied=True, independent_current_checkpoint=True,
        weights_updated=False, optimizer_updated=False, training_modified=False,
        policy_accepted=False, goal_remains_active=True, hardware_enabled=False, shutdown=False)
    write_json(output / 'static_retention.json', result)
    write_json(output / 'status.json', dict(state='agent_review_required', validation_active=False,
        training_active=False, checkpoint_sha256=digest, static_retention_verified=retained,
        policy_accepted=False, goal_remains_active=True))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualify-checkpoint', type=Path)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--full51', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.qualify_checkpoint:
        result = qualify(args.qualify_checkpoint, args.training_output, args.output)
    else:
        if args.full51 is None:
            parser.error('--full51 is required for current-model validation')
        result = run(args.training_output, args.full51, args.output)
    print(json.dumps(dict(qualified=result.get('qualified'),
        static_retention_verified=result.get('static_retention_verified'),
        native_simulations_started=result.get('native_simulations_started'),
        policy_accepted=False), ensure_ascii=False))
