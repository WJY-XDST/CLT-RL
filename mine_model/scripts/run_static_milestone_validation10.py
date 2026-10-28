"""Independently validate a completed static model before any velocity stage.

Requires canonical static pass, fresh30/60s full27 replays and from-reset
height steps at both deterministic and saved-Gaussian actions. No PPO here.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import joint_pd_acceptance_10pct_20261017 as acceptance
from joint_pd_stationary_stage import summary
from run_static_reset_pair_diagnosis import RESULTS, reviewed_source, replay_command
from verify_static_height_steps10 import step_plan, verify as verify_steps
from verify_stationary_motion_release10 import verify as verify_sustained
import source1001_reference_smooth10 as smooth


def baseline_ready(training_output):
    report = json.loads((Path(training_output)/'replay/assessment.json').read_text())
    supplemental = acceptance.assess_current_legs(Path(training_output)/'replay')
    if (report['acceptance_revision'] != acceptance.profile()
            or not summary(report)['stationary_all_original_metrics_passed']
            or not supplemental['static_stage_candidate_passed']):
        raise ValueError('The completed canonical static stage still fails; diagnose before expensive validation')


def reuse_step_pair(directory, checkpoint_sha, command_plan):
    """Reuse exact already-completed step evidence without rerunning or rewriting it."""
    directory = Path(directory).resolve()
    pair = json.loads((directory/'paired_reset_diagnosis.json').read_text())
    if (pair['checkpoint_sha256'] != checkpoint_sha
            or not pair['same_checkpoint_seeds_full_height_and_control']):
        raise ValueError('The paired reset05 step evidence does not qualify for this checkpoint')
    reports,paths = {},{}
    for label,subdir,sampled in [('height_steps_deterministic','deterministic_reset05',False),
                                 ('height_steps_sampled','sampled_reset05',True)]:
        path = directory/subdir/'static_height_step_validation.json'
        row = json.loads(path.read_text())
        if (row['checkpoint_sha256'] != checkpoint_sha
                or (not sampled and not row['height_steps_passed'])
                or row['seeds'] != [383,384,385] or row['sampled_policy_actions'] is not sampled
                or row['reset_velocity_amplitude'] != .05 or row['survivors'] != 9 or row['resets']
                or row['validator_sha256'] != sha(verify_steps.__code__.co_filename)
                or row['command_plan_sha256'] != sha(command_plan)
                or any(sha(name) != digest for name,digest in row['source_files'].items())):
            raise ValueError('Completed height-step evidence or its immutable inputs changed')
        reports[label],paths[label] = row,str(path)
    return reports,paths


def step_readiness(reports):
    """Acceptance uses deterministic inference; saved exploration is a stress diagnosis."""
    deterministic = reports.get('height_steps_deterministic',{})
    sampled = reports.get('height_steps_sampled',{})
    deterministic_passed = bool(deterministic.get('height_steps_passed'))
    sampled_survived = bool(sampled.get('survivors') == 9 and sampled.get('resets') == 0)
    return deterministic_passed,sampled_survived


def run(training_output, output, height_step_pair=None):
    training_output, output = Path(training_output).resolve(),Path(output).resolve()
    baseline_ready(training_output)
    source,entry = reviewed_source(training_output)
    if (output.parent != RESULTS or output.exists()
            or not output.name.startswith('joint_pd_static_milestone_')):
        raise ValueError('Preserve all prior evidence; use a fresh owned static validation output')
    output.mkdir()
    digest = sha(source)
    command_plan = output/'command_plan.json'
    write_json(command_plan,step_plan())
    write_json(output/'plan.json',dict(created_at=datetime.now().astimezone().isoformat(),
        training_output=str(training_output),checkpoint=str(source),checkpoint_sha256=digest,
        implementation_sha256=sha(__file__),steps_reader_sha256=sha(verify_steps.__code__.co_filename),
        sustained_reader_sha256=sha(verify_sustained.__code__.co_filename),
        acceptance_revision=acceptance.profile(),weights_updated=False,optimizer_updated=False,
        velocity_training_started=False,hardware_enabled=False,policy_accepted=False))
    specs = [('heldout',30.,[183,184,185],False,False),
             ('sustained',60.,[283,284,285],False,False),
             ('height_steps_deterministic',50.,[383,384,385],True,False),
             ('height_steps_sampled',50.,[383,384,385],True,True)]
    step_reports,step_paths = {},{}
    if height_step_pair is not None:
        step_reports,step_paths = reuse_step_pair(height_step_pair,digest,command_plan)
        specs = [spec for spec in specs if not spec[3]]
    for label,duration,seeds,steps,sampled in specs:
        destination = output/label
        command = replay_command(source,entry,destination)
        command[command.index('--seconds')+1] = str(duration)
        index = command.index('--seeds')+1
        command[index:index+3] = [str(seed) for seed in seeds]
        if steps:
            command.extend(['--command-probe-json',str(command_plan)])
        if sampled:
            command.append('--sample-policy-actions')
        write_json(output/(label+'_launch.json'),dict(command=command,checkpoint_sha256=digest))
        write_json(output/'status.json',dict(state=label,validation_active=True,training_active=False,
            checkpoint_sha256=digest,policy_accepted=False))
        with (output/(label+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=smooth.previous_controller.helper.base.environment(),
                stdout=log,stderr=subprocess.STDOUT,check=True)
        if steps:
            step_reports[label] = verify_steps(destination,command_plan,seeds=seeds,sampled=sampled)
            step_paths[label] = str(destination/'static_height_step_validation.json')
        if sha(source) != digest:
            raise ValueError('The frozen static checkpoint changed')
    validation_output = RESULTS/('joint_pd_stationary_validation_'+output.name)
    sustained = verify_sustained(training_output,validation_output,output/'heldout',output/'sustained')
    deterministic_steps,sampled_survived = step_readiness(step_reports)
    passed = bool(sustained['motion_release_allowed'] and deterministic_steps and sampled_survived)
    report = dict(checkpoint=str(source),checkpoint_sha256=digest,
        static_milestone_verified=passed,canonical_static_passed=True,
        independent_heldout_and_sustained_passed=sustained['motion_release_allowed'],
        repeated_from_reset_height_steps_passed=deterministic_steps,
        sampled_exploration_survives_without_reset=sampled_survived,
        sampled_exploration_strict_tracking_metrics_are_diagnostic=True,
        sustained_validation=str(validation_output/'motion_release.json'),
        height_step_reports=step_paths,
        existing_step_evidence_reused=height_step_pair is not None,
        acceptance_revision=acceptance.profile(),full_moving_goal_accepted=False,
        velocity_training_started=False,weights_updated=False,optimizer_updated=False,
        hardware_enabled=False,shutdown=False,policy_accepted=False)
    if passed:
        # This is a reviewable static milestone artifact, not a native resume
        # selector or real-robot deployment. Keep the original complete state.
        model = output/'verified_static_model'
        model.mkdir()
        shutil.copy2(source,model/source.name)
        shutil.copytree(source.parent/'params',model/'params')
        if sha(model/source.name) != digest:
            raise ValueError('The packaged full-state checkpoint differs from the validated model')
        write_json(model/'manifest.json',dict(**report,source_training_output=str(training_output),
            resume_source_remains_original_native_checkpoint=True,
            all17_Adam_and_saved_adaptive_LR_included=True,
            files_sha256={str(path.relative_to(model)):sha(path) for path in model.rglob('*') if path.is_file()}))
        report['verified_static_model_directory'] = str(model)
    write_json(output/'static_milestone.json',report)
    write_json(output/'status.json',dict(state='validation_complete',validation_active=False,
        training_active=False,static_milestone_verified=passed,checkpoint_sha256=digest,
        policy_accepted=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--height-step-pair',type=Path,
                        help='Reuse exact completed reset05 pair traces instead of repeating their simulation')
    options = parser.parse_args()
    print(json.dumps(run(options.training_output,options.output,options.height_step_pair),indent=2))
