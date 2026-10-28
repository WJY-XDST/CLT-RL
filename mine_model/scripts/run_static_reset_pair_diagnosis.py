"""Compare frozen static height starts at reset noise .05 and .5 after review.

Same checkpoint, seeds, full height commands and saved Gaussian actions.
No learning, gain changes, threshold changes or automatic source selection.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, sha, tensor_finite
from train_joint_pd_standing_seed import saved_updates
import source1001_reference_smooth10 as smooth
import joint_pd_stationary_reset_runtime as reset_stage
from verify_static_height_steps10 import step_plan, verify

RESULTS = ROOT/'mine_model/results'


def reviewed_source(training_output):
    training_output = Path(training_output).resolve()
    if training_output.parent != RESULTS:
        raise ValueError('Use the owned completed training output')
    read = lambda name: json.loads((training_output/name).read_text())
    state, native = read('status.json'), read('native_learning_signal/completed_external.json')
    report, complete = read('replay/assessment.json'), read('replay/completed.json')
    plan = read('experiment_plan.json')
    source = Path(state['checkpoint']).resolve()
    saved = torch.load(source, map_location='cpu', weights_only=True)
    digest = sha(source)
    if (state['training_active'] or state['state'] != 'agent_review_required'
            or not source.is_relative_to(RUNS.resolve())
            or digest != state['checkpoint_sha256'] or digest != native['checkpoint_sha256']
            or digest != report['checkpoint_sha256'] or not tensor_finite(saved)
            or native['actual_observed_updates'] != 200 or native['optimizer_minibatches'] != 4000
            or native['PPO_updates_repeated'] or native['actual_ppo_updates'] != saved_updates(source)
            or report['total_trials'] != 27 or not complete['completed'] or complete['samples'] != 27000
            or len(saved['optimizer_state_dict']['state']) != 17
            or any(float(item['step']) != 20*saved_updates(source)
                   for item in saved['optimizer_state_dict']['state'].values())
            or any(sha(name) != value for name,value in plan['source_files'].items())):
        raise ValueError('Require the intact completed200/full27/full17 checkpoint before diagnosis')
    declared = saved.get('infos',{}).get(smooth.KEY)
    if declared == smooth.profile():
        smooth.require_source(source)
        entry = smooth.SCRIPT
    elif declared == reset_stage.profile():
        reset_stage.require_source(source)
        entry = reset_stage.SCRIPT.with_name('stationary_reset_observed.py')
    else:
        raise ValueError('This driver diagnoses only the declared smooth10 or reset05 descendant')
    return source, entry


def replay_command(source, entry, output):
    command = smooth.replay_command(source, output)
    command[command.index(str(smooth.SCRIPT))] = str(entry)
    return command


def run(training_output, output, seeds):
    source, entry = reviewed_source(training_output)
    output = Path(output).resolve()
    if (output.parent != RESULTS or output.exists()
            or not output.name.startswith('joint_pd_static_reset_pair_')):
        raise ValueError('Use a fresh diagnosis output; never overwrite earlier evidence')
    output.mkdir()
    plan = output/'command_plan.json'
    write_json(plan, step_plan())
    digest = sha(source)
    write_json(output/'plan.json', dict(created_at=datetime.now().astimezone().isoformat(),
        source_training_output=str(Path(training_output).resolve()), checkpoint=str(source),
        checkpoint_sha256=digest, command_plan_sha256=sha(plan), seeds=seeds,
        implementation_sha256=sha(__file__), reader_sha256=sha(verify.__code__.co_filename),
        weights_updated=False, optimizer_updated=False, acceptance_unchanged=True,
        physical_control_unchanged=True, policy_accepted=False, hardware_enabled=False))
    results = {}
    for sampled in (False,True):
        for amplitude in (.05,.5):
            label = ('sampled' if sampled else 'deterministic')+('_reset05' if amplitude == .05 else '_reset50')
            destination = output/label
            command = replay_command(source,entry,destination)
            command[command.index('--seconds')+1] = '50'
            index = command.index('--seeds')+1
            command[index:index+3] = [str(seed) for seed in seeds]
            command.extend(['--command-probe-json',str(plan),'--reset-velocity-amplitude',str(amplitude)])
            if sampled:
                command.append('--sample-policy-actions')
            write_json(output/(label+'_launch.json'), dict(command=command, checkpoint_sha256=digest))
            write_json(output/'status.json', dict(state=label, diagnostic_active=True,
                training_active=False, checkpoint_sha256=digest, policy_accepted=False))
            with (output/(label+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,env=smooth.previous_controller.helper.base.environment(),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            result = verify(destination,plan,seeds=seeds,sampled=sampled,reset_amplitude=amplitude)
            results[label] = {key:result[key] for key in
                ('survivors','total_trials','resets','height_steps_passed','canonical_reset_distribution')}
            results[label]['worst_first_second_pitch_deg'] = max(
                phase['first_second']['pitch_peak_deg'] for trial in result['trials'] for phase in trial['phases'])
            results[label]['worst_first_second_roll_deg'] = max(
                phase['first_second']['roll_peak_deg'] for trial in result['trials'] for phase in trial['phases'])
            if sha(source) != digest:
                raise ValueError('Frozen checkpoint changed during diagnosis')
    report = dict(checkpoint=str(source), checkpoint_sha256=digest, results=results,
        static_stage_reset05_steps_passed=all(results[label]['height_steps_passed']
            for label in ('deterministic_reset05','sampled_reset05')),
        same_checkpoint_seeds_full_height_and_control=True, training_modified=False,
        acceptance_unchanged=True, policy_accepted=False,
        limitation='Three paired startup seeds per command. Compare disturbance recovery and '
            'command transitions; do not infer rare-failure rates or causal training convergence. '
            'Reset50 is diagnostic, not a relaxation or substitution of canonical reset05 validation.')
    write_json(output/'paired_reset_diagnosis.json',report)
    write_json(output/'status.json',dict(state='diagnosis_complete',diagnostic_active=False,
        training_active=False,checkpoint_sha256=digest,policy_accepted=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs=3,default=[383,384,385])
    options = parser.parse_args()
    print(json.dumps(run(options.training_output,options.output,options.seeds),indent=2))
