"""Reproduce completed motion-phase metrics, then apply the existing literal audit.

Keep original floor-aware gates and their inclusive151-frame windows. Report
nonzero command-relative errors separately, including failures and undefined
first-episode metrics. This starts no simulator and updates no policy.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

import audit_original_and_literal_relative_errors as literal
import diagnose_consolidation_motion_height_steps as numeric
from joint_pd_continuation import ROOT, sha

SCRIPT = Path(__file__).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def audit(directories, destination):
    destination = Path(destination).resolve()
    if destination.exists() or not destination.is_relative_to(ROOT / 'mine_model/results'):
        raise ValueError('Preserve previous immutable phase diagnostics')
    if sha(literal.GATE) != literal.GATE_SHA:
        raise ValueError('Keep the original ten-percent acceptance file')
    criteria = numeric.acceptance.profile()['criteria']
    pins = {str(p): sha(p) for p in (SCRIPT, literal.SCRIPT, literal.GATE,
        numeric.SCRIPT, Path(numeric.acceptance.assessor.__file__))}
    records, checkpoints, first_episodes, max_error = [], set(), 0, 0.
    scopes = set()
    for directory in map(lambda p: Path(p).resolve(), directories):
        manifest, complete, reported = [read(directory/name) for name in
            ('manifest.json','completed.json','motion_height_steps_diagnosis.json')]
        checkpoint = Path(manifest['checkpoint'])
        if (not complete['completed'] or complete['samples'] != 67500
                or manifest['weights_updated'] or manifest['policy_input_remapping']
                or manifest['action_bias_probes'] or manifest['wheel_probe']
                or manifest['wheel_reference_limit_probe'] is not None
                or manifest['command_scale'] != 1. or manifest['curriculum_progress'] is not None
                or manifest['seconds_per_case'] != 50. or manifest['policy_dt'] != .02
                or manifest['command_phase_seconds'] != 5.
                or manifest['leg_control_mode'] != 'implicit_joint_position'
                or manifest['checkpoint_sha256'] != sha(checkpoint)
                or reported['checkpoint_sha256'] != manifest['checkpoint_sha256']
                or reported['criteria'] != criteria or reported['total_trials'] != 27
                or manifest['cases'] != reported['command_plan']['phases'][0]):
            raise ValueError('Require the original completed unchanged-control50s motion phases')
        if any(sha(p) != digest for p,digest in reported['source_pins'].items()):
            raise ValueError('Original phase source evidence changed')
        checkpoints.add((str(checkpoint), manifest['checkpoint_sha256']))
        pins.update(reported['source_pins'])
        pins.update({str(p):sha(p) for p in (checkpoint,directory/'motion_height_steps_diagnosis.json')})
        data = np.genfromtxt(directory/'trace.csv', delimiter=',', names=True)
        if len(data) != 67500 or any(not np.isfinite(data[k]).all() for k in data.dtype.names):
            raise ValueError('Retain the complete finite raw phase timelines')
        scope = str(directory.relative_to(ROOT / 'mine_model/results'))
        if scope in scopes:
            raise ValueError('Duplicate phase scope')
        scopes.add(scope)
        for cid, case in enumerate(manifest['cases']):
            for sid,seed in enumerate(manifest['seeds']):
                trial = reported['trials'][cid*len(manifest['seeds'])+sid]
                rows = data[(data['case_id']==cid)&(data['seed']==seed)]
                if (trial['case'] != case or trial['seed'] != seed or len(rows) != 2500
                        or not np.allclose(rows['time_s'],np.arange(1,2501)*.02,rtol=0.,atol=1.e-8)):
                    raise ValueError('Original phase trial order or timeline changed')
                survived = not any(np.any(rows[k]) for k in ('episode','terminated','timeout'))
                if bool(survived) != trial['survived']:
                    raise ValueError('Original first-episode survival changed')
                phase_ids = np.floor((rows['time_s']-.02+1.e-7)/5.).astype(int)
                for phase in trial['phases']:
                    index, command = phase['index'], phase['command']
                    if command != reported['command_plan']['phases'][index][cid]:
                        raise ValueError('Original phase command changed')
                    part = rows[phase_ids==index]
                    tail = part[part['time_s']>=5.*(index+1)-3.]
                    if len(tail) != 151:
                        raise ValueError('Original inclusive3s phase window changed')
                    metrics = numeric.acceptance.assessor.metrics_for_window(tail,command,criteria) if survived else None
                    limits = numeric.limits(tail,command,criteria)
                    flags = numeric.acceptance.assessor.tracking_components(bool(survived),metrics,
                        {k:v-1.e-12 for k,v in limits.items()})
                    passed = flags['task_goal_passed'] and flags['reference_tracking_passed']
                    if (any(flags[k] != phase[k] for k in flags)
                            or bool(passed) != phase['phase_tracking_passed']
                            or (metrics is None) != (phase['metrics'] is None)):
                        raise ValueError('Original phase flags or failed None values changed')
                    for k,v in limits.items():
                        max_error=max(max_error,abs(v-phase['limits'][k]))
                        if metrics is not None:
                            max_error=max(max_error,abs(metrics[k]-phase['metrics'][k]))
                    if metrics is not None and any(metrics[k] != phase['metrics'][k]
                            for k in ('upright_support','command_reached')):
                        raise ValueError('Original physical prerequisite changed')
                    relative = {metric:100.*metrics[metric]/abs(command[key]) if command[key] else None
                        for metric,key in literal.CHANNELS} if metrics is not None else None
                    records.append(dict(scope=scope+'/phase'+str(index),case=command,seed=seed,
                        survived=bool(survived),metrics=metrics,limits=limits,flags=flags,
                        passed=bool(passed),nonzero_command_relative_error_percent=relative))
                first_episodes += 1
    if len(checkpoints)!=1 or len(records)!=270*len(directories) or max_error!=0.:
        raise ValueError('Require one unchanged checkpoint and exact reproduction of every original phase')
    checkpoint,digest=next(iter(checkpoints))
    if any(sha(p)!=v for p,v in pins.items()):
        raise ValueError('Read-only phase inputs changed')
    result=dict(observed_at=datetime.now().astimezone().isoformat(),verified=True,
        checkpoint=checkpoint,checkpoint_sha256=digest,actual_ppo_updates=None,
        actual_update_clock_not_inferred_from_filename=True,
        total_trials=len(records),trial_unit='original steady command phase',
        total_first_episodes=first_episodes,total_passed=sum(r['passed'] for r in records),
        original_metrics_undefined_trials=sum(r['metrics'] is None for r in records),
        all7_metrics_and_limits_raw_reproduction_max_error=max_error,
        original151_frame_windows_flags_and_failed_None_preserved=True,
        trials=records,source_pins=pins,simulator_started=False,optimizer_executed=False,
        training_modified=False,acceptance_changed=False,policy_accepted=False,
        limitation='Supplemental phase units are not the original657 fixed-command trials. '
            'Saved-Gaussian numerical flags remain descriptive; its existing stress gate remains survival.')
    with destination.open('x') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--replay',type=Path,action='append',required=True)
    cli.add_argument('--raw-output',type=Path,required=True)
    cli.add_argument('--literal-output',type=Path,required=True)
    args=cli.parse_args()
    result=audit(args.replay,args.raw_output)
    relative=literal.audit(args.raw_output,args.literal_output)
    print(json.dumps({k:relative[k] for k in ('verified','total_trials',
        'original_seven_metric_floor_aware_passed','original_gates_and_nonzero_relative10_both_passed',
        'nonzero_relative_failed_metric_counts','original_failed_undefined_trials')}))
