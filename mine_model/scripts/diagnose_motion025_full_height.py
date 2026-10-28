"""Paired low-speed diagnostics after a completed200-update motion block.

Canonical full27 replay must already exist. These extra frozen-policy cases
keep both full height endpoints and never replace final policy acceptance.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import joint_pd_acceptance_10pct_20261017 as acceptance
import source1201_orientation40 as stationary
import joint_pd_source1401_motion025_runtime as motion
import run_static_reset_pair_diagnosis as boundary
from train_source200_stationary_height200 import environment
import assess_height_leg_command_tracking as leg_assessor
from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_cases
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
SEEDS = [483, 484, 485]
SECONDS = 20.


def cases():
    low, high = SAFE_BASE_HEIGHT_RANGE_M
    targets = [('stand', 0., 0., .3),
               ('forward01', .1, 0., .3), ('forward02', .2, 0., .3),
               ('reverse01', -.1, 0., .3), ('reverse02', -.2, 0., .3),
               ('spin_left', 0., .075, .3), ('spin_right', 0., -.075, .3),
               ('height_low', 0., 0., low), ('height_high', 0., 0., high)]
    for height_name, height in (('low', low), ('high', high)):
        targets.extend((f'{height_name}_{name}', speed, yaw, height) for name, speed, yaw in
            [('forward', .2, 0.), ('reverse', -.2, 0.),
             ('spin_left', 0., .075), ('spin_right', 0., -.075)])
    return [dict(name='stage025_' + name, speed=speed, yaw=yaw, height=height)
            for name, speed, yaw, height in targets]


def source_for_diagnosis(training_output):
    previous = boundary.smooth
    boundary.smooth = motion
    try:
        source, entry = boundary.reviewed_source(training_output)
    finally:
        boundary.smooth = previous
    if entry != motion.SCRIPT:
        raise ValueError('Require the completed motion025 native checkpoint and full27 report')
    motion.require_source(source)
    return source


def command(source, output, command_file, *, baseline=False):
    entry = stationary if baseline else motion
    argv = entry.replay_command(source, output)
    argv[argv.index('--seconds') + 1] = str(SECONDS)
    i = argv.index('--seeds') + 1
    argv[i:i + 3] = [str(seed) for seed in SEEDS]
    argv.extend(['--command-probe-json', str(command_file)])
    if any(flag in argv for flag in ('--command-scale', '--overrides_json',
            '--sample-policy-actions', '--heading-hold', '--export_dir')):
        raise ValueError('Keep exact saved control and unscaled full height commands')
    return argv


def verify(directory, source):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    manifest, complete, cfg = read('manifest.json'), read('completed.json'), read('resolved_env.json')
    if (manifest['cases'] != cases() or manifest['seeds'] != SEEDS
            or manifest['seconds_per_case'] != SECONDS
            or not manifest['command_probe'] or manifest['policy_acceptance_test'] is not False
            or manifest['weights_updated'] or manifest['policy_input_remapping']
            or manifest['sampled_policy_actions'] or manifest['command_scale'] != 1.
            or manifest['curriculum_progress'] is not None or manifest['command_sequence'] is not None
            or manifest['reset_velocity_uniform_amplitude'] != .05
            or manifest['leg_control_mode'] != 'implicit_joint_position'
            or manifest['physics_dt'] != .005 or manifest['policy_dt'] != .02
            or manifest['solver_position_iterations'] != 128 or manifest['solver_velocity_iterations'] != 8
            or manifest['enable_external_forces_every_iteration'] is not True
            or manifest['action_bias_probes'] or manifest['wheel_probe']
            or manifest['wheel_reference_limit_probe'] is not None
            or manifest['checkpoint_sha256'] != sha(source)
            or not complete['completed'] or complete['samples'] != 51000
            or cfg['commands']['ranges_height'] != list(SAFE_BASE_HEIGHT_RANGE_M)):
        raise ValueError('Incomplete or changed low-speed/full-height diagnostic')
    for path, digest in manifest['asset_chain_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Calibrated replay asset changed')
    data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    if len(data) != 51000 or not all(np.isfinite(data[key]).all() for key in data.dtype.names):
        raise ValueError('Missing or nonfinite diagnostic trace')
    criteria = acceptance.profile()['criteria']
    rows = []
    for cid, case in enumerate(cases()):
        for seed in SEEDS:
            part = data[(data['case_id'] == cid) & (data['seed'] == seed)]
            if len(part) != 1000 or not np.allclose(part['time_s'], np.arange(1, 1001) * .02, rtol=0., atol=1.e-8):
                raise ValueError('Missing complete first-episode timeline')
            if not np.allclose(part['height_cmd_m'], case['height'], rtol=0., atol=1.e-7):
                raise ValueError('Full-height diagnostic target changed')
            survived = bool(not np.any(part['episode']) and not np.any(part['terminated']) and not np.any(part['timeout']))
            tail = part[part['time_s'] >= SECONDS - 3.]
            physical = acceptance.assessor.metrics_for_window(tail, case, criteria) if survived else None
            reference = tail if survived else part[part['episode'] == 0]
            leg_reference = max(.001, min(np.mean(abs(reference[f'{side}_length_ref_m'])) for side in ('left', 'right')))
            wheel_reference = min(np.mean(abs(reference[f'{side}_wheel_ref_rad_s'])) for side in ('left', 'right'))
            fraction = criteria['relative_error']
            limits = dict(speed_mae_m_s=max(criteria['speed_absolute_m_s'], fraction * abs(case['speed'])),
                yaw_mae_rad_s=max(criteria['yaw_absolute_rad_s'], fraction * abs(case['yaw'])),
                height_mae_m=fraction * case['height'], leg_length_mae_m=fraction * leg_reference,
                leg_angle_mae_rad=fraction * criteria['leg_angle_design_range_rad'],
                wheel_speed_mae_rad_s=max(criteria['wheel_absolute_rad_s'], fraction * wheel_reference),
                body_p95_deg=criteria['body_p95_deg'])
            components = acceptance.assessor.tracking_components(survived, physical,
                {key: value - 1.e-12 for key, value in limits.items()})
            actual_static_legs = None
            if case['speed'] == 0. and case['yaw'] == 0.:
                actual_static_legs = leg_assessor.trial_metrics(part, case, SECONDS, criteria)
            static_ok = actual_static_legs is None or all(actual_static_legs[key] for key in
                ('height_posture_passed', 'leg_command_passed', 'stationary_drift_passed'))
            rows.append(dict(case=case, seed=seed, survived=survived,
                metrics=physical, limits=limits, actual_static_leg_commands=actual_static_legs,
                **components, stage_constraints_passed=bool(components['task_goal_passed']
                    and components['reference_tracking_passed'] and static_ok)))
    result = dict(checkpoint=str(source), checkpoint_sha256=sha(source),
        total_trials=51, survivors=sum(row['survived'] for row in rows), resets=complete['resets'],
        stage_all_constraints_passed=bool(not complete['resets'] and all(row['stage_constraints_passed'] for row in rows)),
        criteria=criteria, limits_use_original_relative_and_absolute_floor_rules=True,
        full_height_range=list(SAFE_BASE_HEIGHT_RANGE_M), trials=rows,
        source_files={str(directory / name): sha(directory / name) for name in
            ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv')},
        canonical_acceptance_replaced=False, policy_accepted=False,
        limitation='Extra deterministic low-speed evidence, including motion at both height endpoints. '
                   'The canonical full27 and independent sustained full-scope gates remain required. '
                   'Passing these original absolute-floor limits is not a claim that every small nonzero command has <10% relative error.')
    write_json(directory / 'motion025_full_height_diagnosis.json', result)
    return result


def run(training_output, output):
    output = Path(output).resolve()
    if output.parent != RESULTS or output.exists() or not output.name.startswith('joint_pd_motion025_pair_'):
        raise ValueError('Use a fresh owned output and preserve earlier reports')
    candidate = source_for_diagnosis(training_output)
    baseline = stationary.RESULTS / 'joint_pd_static_milestone_20261017_source1401/verified_static_model/model_1400.pt'
    if sha(baseline) != motion.ANCHOR_SHA:
        raise ValueError('Require the intact independently verified static baseline')
    stationary.require_source(baseline)
    output.mkdir()
    command_file = output / 'command_plan.json'
    write_json(command_file, cases())
    source_pins = {str(path): sha(path) for path in
        (SCRIPT, candidate, baseline, motion.SCRIPT, stationary.SCRIPT, acceptance.CURRENT,
         ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py', command_file)}
    write_json(output / 'plan.json', dict(created_at=datetime.now().astimezone().isoformat(),
        training_output=str(Path(training_output).resolve()), checkpoint=str(candidate),
        baseline_checkpoint=str(baseline), cases=cases(), seeds=SEEDS, seconds=SECONDS,
        source_pins=source_pins, canonical_full27_already_required=True,
        acceptance_replaced=False, model_updated=False, optimizer_updated=False, hardware_enabled=False))
    results = {}
    for label, source in (('baseline1401', baseline), ('candidate', candidate)):
        destination = output / label
        argv = command(source, destination, command_file, baseline=label == 'baseline1401')
        write_json(output / (label + '_launch.json'), dict(command=argv, checkpoint_sha256=sha(source)))
        write_json(output / 'status.json', dict(state=label, diagnostic_active=True,
            training_active=False, policy_accepted=False))
        with (output / (label + '.log')).open('w') as log:
            subprocess.run(argv, cwd=ROOT, env=environment(),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        results[label] = verify(destination, source)
        if any(sha(path) != digest for path, digest in source_pins.items()):
            raise ValueError('Frozen diagnosis source changed')
    before_cfg = json.loads((output / 'baseline1401/resolved_env.json').read_text())
    after_cfg = json.loads((output / 'candidate/resolved_env.json').read_text())
    comparable = json.loads(json.dumps(before_cfg))
    comparable['log_dir'] = after_cfg['log_dir']
    comparable['commands']['training_motion_scale'] = after_cfg['commands']['training_motion_scale']
    if (before_cfg['commands']['training_motion_scale'] != 0.
            or after_cfg['commands']['training_motion_scale'] != .25 or comparable != after_cfg):
        raise ValueError('Paired native physical/control configurations differ beyond motion metadata')
    paired = []
    for old, new in zip(results['baseline1401']['trials'], results['candidate']['trials']):
        if (old['case'], old['seed']) != (new['case'], new['seed']):
            raise ValueError('Paired diagnostic commands or seeds differ')
        delta = ({key: new['metrics'][key] - old['metrics'][key] for key in new['limits']}
                 if old['survived'] and new['survived'] else None)
        paired.append(dict(case=new['case']['name'], seed=new['seed'], metric_changes=delta,
            baseline_survived=old['survived'], candidate_survived=new['survived']))
    result = dict(baseline_checkpoint_sha256=sha(baseline), checkpoint_sha256=sha(candidate),
        results={key: {field: value[field] for field in
            ('total_trials', 'survivors', 'resets', 'stage_all_constraints_passed')} for key, value in results.items()},
        paired_metric_changes=paired, source_pins=source_pins, training_modified=False,
        paired_native_configurations_equal_except_motion_metadata_and_log_directory=True,
        canonical_acceptance_replaced=False, automatic_source_selection=False, policy_accepted=False)
    write_json(output / 'paired_motion025_full_height_diagnosis.json', result)
    write_json(output / 'status.json', dict(state='diagnosis_complete', diagnostic_active=False,
        training_active=False, policy_accepted=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-plan', action='store_true')
    args = parser.parse_args()
    if args.check_plan:
        anchor = stationary.RESULTS / 'joint_pd_static_milestone_20261017_source1401/verified_static_model/model_1400.pt'
        _, cfg = stationary.require_source(anchor)
        validated = command_probe_cases(cases(), SimpleNamespace(**cfg['commands']))
        argv = command(anchor, RESULTS / 'motion025_cpu_plan_not_a_native_replay',
                       RESULTS / 'motion025_cpu_plan_not_a_native_replay/commands.json', baseline=True)
        print(json.dumps(dict(cases=len(validated), trials=len(validated) * len(SEEDS),
            full_height_range=list(SAFE_BASE_HEIGHT_RANGE_M), PPO_updates_performed=0,
            frozen_baseline_command=argv, simulator_started=False, policy_accepted=False)))
    elif args.training_output is None or args.output is None:
        parser.error('--training-output and --output are required')
    else:
        print(json.dumps(run(args.training_output, args.output), ensure_ascii=False))
