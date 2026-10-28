"""CPU-only static readiness for gradual motion; never full-policy acceptance.

Require full27 canonical replays at20/30/60seconds with three disjoint seed
sets. Keep all original10% metrics and the full height endpoints. Moving
cases remain visible even while only the static stage authorizes a curriculum.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, RUNS, sha, tensor_finite, optimizer_step
from train_joint_pd_standing_seed import saved_updates
import joint_pd_acceptance_10pct_20261017 as acceptance
import joint_pd_stationary_stage as stage
import assess_height_leg_command_tracking as legs
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M
from wheel_legged_gym_isaaclab.robot_model import validate_checkpoint_robot

RESULTS = ROOT / 'mine_model/results'
SPECS = dict(baseline=(20., [83,84,85], None),
             heldout=(30., [183,184,185], 20.),
             sustained=(60., [283,284,285], 40.))


def canonical_cases():
    return [dict(name=name, speed=speed, yaw=yaw, height=height) for name,speed,yaw,height in (
        ('stand',0.,0.,.3), ('forward_0p3',.3,0.,.3), ('forward_0p6',.6,0.,.3),
        ('reverse_0p3',-.3,0.,.3), ('reverse_0p6',-.6,0.,.3),
        ('spin_left',0.,.3,.3), ('spin_right',0.,-.3,.3),
        ('height_low',0.,0.,SAFE_BASE_HEIGHT_RANGE_M[0]),
        ('height_high',0.,0.,SAFE_BASE_HEIGHT_RANGE_M[1]))]


def require_manifest(manifest, checkpoint_sha, label):
    duration, seeds, _ = SPECS[label]
    if (manifest['checkpoint_sha256'] != checkpoint_sha
            or manifest['seconds_per_case'] != duration or manifest['seeds'] != seeds
            or manifest['cases'] != canonical_cases()
            or manifest.get('policy_acceptance_test') is not True
            or manifest.get('weights_updated') is not False
            or manifest.get('policy_input_remapping') is not False
            or manifest.get('leg_control_mode') != 'implicit_joint_position'
            or manifest.get('command_scale') != 1. or manifest.get('curriculum_progress') is not None
            or manifest.get('sampled_policy_actions') or manifest.get('command_probe')
            or manifest.get('wheel_probe') or manifest.get('action_bias_probes')
            or manifest.get('reset_velocity_uniform_amplitude') != .05
            or manifest.get('physics_dt') != .005 or manifest.get('policy_dt') != .02
            or manifest.get('solver_position_iterations') != 128
            or manifest.get('solver_velocity_iterations') != 8
            or manifest.get('enable_external_forces_every_iteration') is not True):
        raise ValueError('Require unchanged full27 targets, fullheight, frozen directPD and independent ' + label)


def data_view(source, destination):
    source = Path(source).resolve()
    destination.mkdir()
    for name in ('manifest.json','completed.json','resolved_env.json','trace.csv'):
        if not (source/name).is_file():
            raise ValueError('Incomplete replay: ' + str(source/name))
        (destination/name).symlink_to(source/name)
    return destination


def supplemental_windows(manifest, samples, stable_seconds, criteria):
    """The same actual-leg command limits in every sustained3s window."""
    duration, dt = manifest['seconds_per_case'], manifest['policy_dt']
    count = round(duration/dt)
    lookup = {}
    for cid,case in enumerate(manifest['cases']):
        for seed in manifest['seeds']:
            data = samples[(samples['case_id']==cid) & (samples['seed']==seed)]
            if len(data) != count or not np.allclose(data['time_s'],np.arange(1,count+1)*dt,atol=1.e-8,rtol=0.):
                raise ValueError('Incomplete full27 validation timeline')
            if case['name'] in legs.STATIONARY:
                lookup[(case['name'],seed)] = (case,data)
    target_low, target_high = (h-legs.UPRIGHT_HEIGHT_OFFSET_M for h in SAFE_BASE_HEIGHT_RANGE_M)
    minimum_span = target_high*(1.-criteria['relative_error'])-target_low*(1.+criteria['relative_error'])
    rows = []
    for start in np.arange(duration-stable_seconds,duration-3.+dt/2.,1.):
        metrics = {}
        for key,(case,data) in lookup.items():
            survived = not np.any(data['episode'] != 0) and not np.any(data['terminated']) and not np.any(data['timeout'])
            window = data[(data['time_s'] >= start-dt*.1) & (data['time_s'] < start+3.-dt*.1)]
            if len(window) != round(3./dt):
                raise ValueError('Missing complete supplemental3s window')
            result = legs.trial_metrics(window,case,start+3.,criteria)
            result['survived'] = bool(result['survived'] and survived)
            metrics[key] = result
        spans = []
        for seed in manifest['seeds']:
            low, high = (metrics[(name,seed)] for name in ('height_low','height_high'))
            for side in legs.SIDES:
                span = (high['metrics']['legs'][side]['measured_mean_m']-
                        low['metrics']['legs'][side]['measured_mean_m']) if low['survived'] and high['survived'] else None
                spans.append(bool(span is not None and span > minimum_span + legs.ROUND_OFF))
        passed = len(metrics) == 9 and all(r['survived'] and r['height_posture_passed']
            and r['leg_command_passed'] and r['stationary_drift_passed'] for r in metrics.values()) and all(spans)
        rows.append(dict(start_s=float(start), passed=bool(passed)))
    return dict(passed=bool(rows and all(r['passed'] for r in rows)),
        windows_per_static_trial=len(rows), window_seconds=3., stride_seconds=1.,
        stable_seconds=stable_seconds, necessary_endpoint_span_m=minimum_span, windows=rows,
        same_original_actual_leg_command_limits=True, measurement_reference_only=True)


def assess_stage(source, destination, checkpoint_sha, label, baseline_manifest=None, baseline_cfg=None):
    directory = data_view(source,destination)
    manifest = json.loads((directory/'manifest.json').read_text())
    complete = json.loads((directory/'completed.json').read_text())
    cfg = json.loads((directory/'resolved_env.json').read_text())
    require_manifest(manifest,checkpoint_sha,label)
    if (not complete['completed'] or complete['samples'] != round(SPECS[label][0]/.02)*27
            or sha(manifest['checkpoint']) != checkpoint_sha
            or not manifest.get('asset_chain_sha256')
            or any(sha(p) != v for p,v in manifest['asset_chain_sha256'].items())):
        raise ValueError('Frozen checkpoint, asset chain or completion mismatch')
    if baseline_manifest is not None:
        if manifest['asset_chain_sha256'] != baseline_manifest['asset_chain_sha256']:
            raise ValueError('Do not alter physics during independent validation')
        expected = copy.deepcopy(baseline_cfg)
        expected['episode_length_s'] = SPECS[label][0]+2.
        if cfg != expected:
            raise ValueError('Validation may change duration and independent seeds only')
    stable = SPECS[label][2]
    if stable is None:
        main = acceptance.assess_current(directory)
        supplementary = acceptance.assess_current_legs(directory)
        passed_leg = supplementary['static_stage_candidate_passed']
    else:
        main = acceptance.assess_current_stability(directory,stable)
        samples = np.genfromtxt(directory/'trace.csv',delimiter=',',names=True)
        supplementary = supplemental_windows(manifest,samples,stable,acceptance.profile()['criteria'])
        passed_leg = supplementary['passed']
    summary = stage.summary(main)
    result = dict(label=label, static_summary=summary,
        static_passed=bool(summary['stationary_all_original_metrics_passed'] and passed_leg),
        actual_leg_command_passed=bool(passed_leg), full27_passed=bool(main['passed']),
        full27_failures_retained=True, supplemental=supplementary,
        original_data_sha256={str(Path(source).resolve()/name):sha(directory/name)
            for name in ('manifest.json','completed.json','resolved_env.json','trace.csv')})
    write_json(directory/'static_validation.json',result)
    return result,manifest,cfg


def verify(training_output, output, heldout=None, sustained=None):
    training_output, output = Path(training_output).resolve(),Path(output).resolve()
    if (training_output.parent != RESULTS or output.parent != RESULTS
            or not output.name.startswith('joint_pd_stationary_validation_') or output.exists()):
        raise ValueError('Use completed owned training and a fresh validation report directory')
    state = json.loads((training_output/'status.json').read_text())
    native = json.loads((training_output/'native_learning_signal/completed_external.json').read_text())
    checkpoint = Path(state['checkpoint']).resolve()
    saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg = validate_checkpoint_robot(checkpoint)
    if (state['training_active'] or state['state'] != 'agent_review_required'
            or not checkpoint.is_relative_to(RUNS.resolve())
            or native['actual_observed_updates'] != 200 or native['optimizer_minibatches'] != 4000
            or native['PPO_updates_repeated'] or sha(checkpoint) != state['checkpoint_sha256']
            or native['checkpoint_sha256'] != sha(checkpoint)
            or native['actual_ppo_updates'] != saved_updates(checkpoint)
            or state['actual_ppo_updates'] != saved_updates(checkpoint)
            or not native['all_probe_RNG_unchanged']
            or max(native['all_environment_alignment_max_errors'].values()) != 0.
            or cfg['leg_control_mode'] != 'implicit_joint_position'
            or cfg['commands']['training_motion_scale'] != 0.
            or saved['model_state_dict']['actor.0.weight'].shape[1] != 27
            or saved['model_state_dict']['critic.0.weight'].shape[1] != 27
            or not tensor_finite(saved) or len(saved['optimizer_state_dict']['state']) != 17
            or optimizer_step(saved) != 20*saved_updates(checkpoint)):
        raise ValueError('Require a complete retained-state200-update stationary source')
    output.mkdir()
    baseline,manifest,baseline_cfg = assess_stage(training_output/'replay',output/'baseline',sha(checkpoint),'baseline')
    stages = dict(baseline=baseline)
    for label,directory in [('heldout',heldout),('sustained',sustained)]:
        if directory is None:
            stages[label] = dict(static_passed=False,missing=True)
        else:
            stages[label],_,_ = assess_stage(directory,output/label,sha(checkpoint),label,manifest,baseline_cfg)
    report = dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        source_actual_updates=saved_updates(checkpoint),source_all17_Adam_step=optimizer_step(saved),
        acceptance_revision=acceptance.profile(),stages=stages,
        motion_release_allowed=all(r['static_passed'] for r in stages.values()),
        scope='Static readiness to begin gradual motion; full27 moving failures remain reported',
        independent_heldout_passed=stages['heldout']['static_passed'],
        independent_sustained_passed=stages['sustained']['static_passed'],
        expected_validation_specs={k:list(v) for k,v in SPECS.items()},
        validator_sha256=sha(__file__),native_simulations_started=0,training_modified=False,
        hardware_enabled=False,policy_accepted=False,full_goal_acceptance_not_implied=True)
    write_json(output/'motion_release.json',report)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--heldout',type=Path)
    parser.add_argument('--sustained',type=Path)
    args=parser.parse_args()
    result=verify(args.training_output,args.output,args.heldout,args.sustained)
    print(json.dumps({k:v for k,v in result.items() if k!='stages'},indent=2))
