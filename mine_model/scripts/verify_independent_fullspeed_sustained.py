"""Supplemental signed full-speed sustained evidence; never replace original gates.

Only the commanded cases, fresh reset seeds and replay duration differ. The
ordinary saved controller and the original seven metrics/limits remain intact.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from auto_train_mine import write_json
from audit_plain27_after_queued_reviews import wait_unit
from joint_pd_continuation import ROOT, sha
import joint_pd_source5401_motion100_runtime as runtime
import diagnose_consolidation_motion_height_steps as numeric
from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_plan

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
SEEDS = [1283, 1284, 1285]
SECONDS, STABLE_SECONDS = 120., 100.


def read(path):
    return json.loads(Path(path).read_text())


def intact(pins):
    changed = [p for p, digest in pins.items() if sha(p) != digest]
    if changed:
        raise ValueError('Preserve qualified source evidence: ' + repr(changed[:8]))


def cases_for(cfg):
    heights = [cfg['commands']['ranges_height'][0], .3, cfg['commands']['ranges_height'][1]]
    return [dict(name=f'{height_label}_{direction}_{turn}', speed=speed, yaw=yaw, height=height)
        for height_label, height in zip(('low', 'nominal', 'high'), heights)
        for direction, speed in (('reverse08', -.8), ('forward08', .8))
        for turn, yaw in (('right03', -.3), ('straight', 0.), ('left03', .3))]


def score(data, cases, seeds, seconds, stable_seconds):
    """Reuse original metrics/limits and original one-second overlapping windows."""
    criteria = numeric.acceptance.profile()['criteria']
    dt, steps = .02, round(seconds / .02)
    if data.shape != (steps, len(cases) * len(seeds)):
        raise ValueError('Require every complete case/seed timeline')
    timeline = np.arange(1, steps + 1) * dt
    starts = np.arange(seconds - stable_seconds, seconds - 3. + dt / 2., 1.)
    result = []
    for cid, case in enumerate(cases):
        for sid, seed in enumerate(seeds):
            part = data[:, cid * len(seeds) + sid]
            if (not np.allclose(part['time_s'], timeline, atol=1.e-8, rtol=0.)
                    or not np.all(part['case_id'] == cid) or not np.all(part['seed'] == seed)):
                raise ValueError('Missing or reordered case/seed samples')
            if (not np.allclose(part['height_cmd_m'], case['height'], atol=1.e-7, rtol=0.)
                    or not np.allclose(part['yaw_cmd_rad_s'], case['yaw'], atol=1.e-7, rtol=0.)
                    or np.any(abs(part['speed_cmd_m_s']) > abs(case['speed']) + 1.e-7)
                    or np.any(part['speed_cmd_m_s'] * case['speed'] < -1.e-7)):
                raise ValueError('Keep the signed case commands and the original bounded speed ramp')
            failed = (part['terminated'] != 0) | (part['timeout'] != 0)
            survived = not failed.any() and not np.any(part['episode'])
            tail = part[part['time_s'] >= seconds - 3.]
            if len(tail) != 151:
                raise ValueError('Keep the original inclusive final151-frame limit reference')
            bound = numeric.limits(tail, case, criteria)
            metrics = None
            if survived:
                windows = []
                for start in starts:
                    window = part[(part['time_s'] >= start - dt * .1)
                        & (part['time_s'] < start + 3. - dt * .1)]
                    if len(window) != 150:
                        raise ValueError('Keep every original150-frame overlapping window')
                    windows.append(numeric.acceptance.assessor.metrics_for_window(window, case, criteria))
                metrics = {k: max(v[k] for v in windows) for k in bound}
                metrics.update({k: all(v[k] for v in windows)
                    for k in ('command_reached', 'upright_support')})
            strict = {k: value - 1.e-12 for k, value in bound.items()}
            flags = numeric.acceptance.assessor.tracking_components(bool(survived), metrics, strict)
            passed = bool(survived and metrics['command_reached'] and metrics['upright_support']
                and all(metrics[k] < limit for k, limit in strict.items()))
            result.append(dict(case=case, seed=seed, survived=bool(survived),
                first_failure_s=None if not failed.any() else float(part['time_s'][failed][0]),
                raw_OR_resets=int(failed.sum()), metrics=metrics, limits=bound, **flags, passed=passed,
                original_steady_windows_per_trial=len(starts)))
    return result


def qualify(training_launch, after_launch, output, qualification):
    training_launch, after_launch, output, qualification = [Path(p).resolve()
        for p in (training_launch, after_launch, output, qualification)]
    launch, after = read(training_launch), read(after_launch)
    training = Path(launch['output'])
    plan = read(training / 'experiment_plan.json')
    source = Path(launch['source_checkpoint'])
    state, cfg = runtime.require_source(source)
    if (qualification.exists() or output.exists() or not output.is_relative_to(RESULTS)
            or not qualification.is_relative_to(RESULTS)
            or launch['planned_actual_updates'] != 200 or launch['target_actual_updates'] != 5801
            or launch['next_configuration_changes'] != dict(rewards={}, commands={}, exploration={}, controller={}, physics={})
            or after['training_output'] != str(training) or after['target_actual_updates'] != 5801
            or after['target_all17_Adam_step'] != 116020 or not after['queued']
            or cfg['commands']['training_motion_scale'] != 1.
            or cfg['action_scale_vel'] != 15. or cfg['leg_control_mode'] != 'implicit_joint_position'
            or len(state['optimizer_state_dict']['state']) != 17):
        raise ValueError('Qualify the unchanged5601→5801 ordinary200-update descendant only')
    cases = cases_for(cfg)
    parsed, sequence, period = command_probe_plan(cases, SimpleNamespace(**cfg['commands']))
    if len(parsed) != 18 or sequence is not None or period is not None:
        raise ValueError('Require all3 heights × both0.8 directions × all3 yaw commands')
    old = RESULTS / 'joint_pd_static_retention_motion100_20261019_source5601/fixed_height_review/sustained'
    old_manifest, old_report = read(old / 'manifest.json'), read(old / 'assessment_stability.json')
    raw_manifest = RESULTS / 'joint_pd_verified_static_source5601_motion100_20261019.json'
    raw = read(raw_manifest)
    if not raw['raw_integrity_verified'] or raw['checkpoint_sha256'] != sha(source):
        raise ValueError('Require the exact source own sustained raw evidence')
    data = np.genfromtxt(old / 'trace.csv', delimiter=',', names=True).reshape(3000, 27)
    reproduced = score(data, old_manifest['cases'], old_manifest['seeds'], 60., 40.)
    fields = ('survived', 'metrics', 'limits', 'passed', 'task_goal_passed',
        'reference_tracking_passed', 'failed_metrics', 'failed_prerequisites')
    for trial in reproduced:
        original = next(t for c in old_report['conditions'] if all(c[k] == trial['case'][k]
            for k in ('name', 'speed', 'yaw', 'height')) for t in c['trials'] if t['seed'] == trial['seed'])
        if any(trial[k] != original[k] for k in fields) or trial['original_steady_windows_per_trial'] != 38:
            raise ValueError('Original sustained metrics, limits, strict flags or38 windows differ')
    known_seeds = set()
    for path in plan['source_files']:
        if Path(path).name == 'manifest.json':
            known_seeds.update(read(path).get('seeds', []))
    if known_seeds.intersection(SEEDS):
        raise ValueError('Choose unused independent reset seeds before freezing the qualification')
    pins = dict(plan['source_files'])
    pins.update(raw['source_pins'])
    for p in (SCRIPT, training_launch, after_launch, training / 'experiment_plan.json', source,
            raw_manifest, old / 'manifest.json', old / 'trace.csv', old / 'assessment_stability.json',
            old / 'resolved_env.json', runtime.SCRIPT, numeric.SCRIPT,
            Path(numeric.acceptance.assessor.__file__), numeric.acceptance.CURRENT,
            numeric.acceptance.POLICY, ROOT / 'wheel_legged_isaaclab/scripts/evaluate_mine_policy.py',
            Path(command_probe_plan.__code__.co_filename)):
        pins[str(p)] = sha(p)
    intact(pins)
    result = dict(qualified=True, observed_at=datetime.now().astimezone().isoformat(),
        output=str(output), training_output=str(training), after_unit=after['unit'],
        target_actual_updates=5801, target_all17_Adam_step=116020, baseline_checkpoint=str(source),
        baseline_resolved_config=str(old / 'resolved_env.json'), cases=cases, seeds=SEEDS,
        seconds_per_case=SECONDS, stable_seconds=STABLE_SECONDS, windows_per_trial=98,
        total_trials=54, expected_rows=324000, original27_sustained_report_exactly_reproduced=True,
        reset_seeds_absent_from_all_source_manifest_seeds=True, source_pins=pins,
        action_controller_physics_reward_overrides=False, original657_and_static_gates_not_replaced=True,
        training_modified=False, optimizer_executed=False, simulator_started=False,
        acceptance_changed=False, policy_accepted=False)
    with qualification.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def run(qualification):
    qualification = Path(qualification).resolve()
    proof = read(qualification)
    intact(proof['source_pins'])
    output, training = Path(proof['output']), Path(proof['training_output'])
    if not proof['qualified'] or output.exists():
        raise ValueError('Preserve qualified requests and previous native results')
    output.mkdir()
    wait_unit(proof['after_unit'], output / 'status.json')
    final = read(training / 'final_checkpoint_verification_readonly_20261019.json')
    all_errors = read(training / 'all_errors_all657_source5801_readonly_20261019.json')
    source = Path(final['checkpoint'])
    state, cfg = runtime.require_source(source)
    if (not final['verified'] or final['actual_ppo_updates'] != proof['target_actual_updates']
            or final['all17_Adam_step'] != proof['target_all17_Adam_step']
            or not all_errors['verified'] or all_errors['total_trials'] != 657
            or all_errors['checkpoint_sha256'] != sha(source)
            or cases_for(cfg) != proof['cases'] or len(state['optimizer_state_dict']['state']) != 17
            or any(float(s['step']) != proof['target_all17_Adam_step']
                for s in state['optimizer_state_dict']['state'].values())):
        raise ValueError('Require this exact completed checkpoint and all original657 raw trials')
    intact(proof['source_pins'])
    command_file, directory = output / 'commands.json', output / 'candidate'
    write_json(command_file, proof['cases'])
    argv = runtime.replay_command(source, directory)
    argv[argv.index('--seconds') + 1] = str(proof['seconds_per_case'])
    i = argv.index('--seeds') + 1
    argv[i:i + 3] = list(map(str, proof['seeds']))
    argv.extend(['--command-probe-json', str(command_file)])
    if '--record_policy_inputs' not in argv:
        argv.append('--record_policy_inputs')
    if any(x in argv for x in ('--overrides_json', '--command-scale', '--heading-hold',
            '--export_dir', '--sample-policy-actions', '--wheel-reference-limit-probe', runtime.FLAG)):
        raise ValueError('Use the exact ordinary saved controller without interventions')
    write_json(output / 'launch.json', dict(command=argv, qualification_sha256=sha(qualification)))
    write_json(output / 'status.json', dict(state='native_independent_fullspeed_sustained',
        training_active=False, diagnostic_active=True, policy_accepted=False))
    with (output / 'candidate.log').open('x') as log:
        subprocess.run(argv, cwd=ROOT, env=numeric.native.environment(), stdout=log,
            stderr=subprocess.STDOUT, check=True)
    manifest, complete, resolved = [read(directory / p) for p in
        ('manifest.json', 'completed.json', 'resolved_env.json')]
    reference = read(proof['baseline_resolved_config'])
    reference['log_dir'], reference['episode_length_s'] = resolved['log_dir'], SECONDS + 2.
    reference['scene']['num_envs'] = reference['terrain']['num_envs'] = 54
    if (resolved != reference or manifest['checkpoint_sha256'] != sha(source)
            or manifest['cases'] != proof['cases'] or manifest['seeds'] != proof['seeds']
            or manifest['seconds_per_case'] != SECONDS or manifest['command_sequence'] is not None
            or not manifest['command_probe'] or manifest['policy_acceptance_test']
            or manifest['sampled_policy_actions'] or manifest['weights_updated']
            or manifest['policy_input_remapping'] or manifest['action_bias_probes'] or manifest['wheel_probe']
            or manifest['wheel_reference_limit_probe'] is not None
            or any(manifest.get(k) is not None for k in ('tracking_goal_reward_probe',
                'tracking_goal_component_reward_probe', 'joint_saturation_reward_probe',
                'wheel_action_saturation_reward_probe'))
            or manifest['reset_velocity_uniform_amplitude'] != .05 or not complete['completed']
            or complete['samples'] != proof['expected_rows']):
        raise ValueError('Sustained replay changed beyond declared case/seed/duration metadata')
    for path, digest in manifest['asset_chain_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Physical asset changed')
    data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True).reshape(6000, 54)
    if any(not np.isfinite(data[k]).all() for k in data.dtype.names):
        raise ValueError('Keep every finite native row, including every failed episode')
    with np.load(directory / 'policy_inputs.npz') as z:
        means, observations = z['means'], z['observations']
        if means.shape != (6000, 54, 6) or observations.shape != (6000, 54, 27):
            raise ValueError('Keep the exact full policy input timeline')
        if not np.isfinite(means).all() or not np.isfinite(observations).all():
            raise ValueError('Nonfinite policy inputs or means cannot provide complete evidence')
        refs = np.stack([data[k] for k in ('left_wheel_ref_rad_s', 'right_wheel_ref_rad_s')], -1)
        mapping_error = float(abs(refs - cfg['action_scale_vel'] * np.clip(means[..., [2, 5]], -1., 1.)).max())
    resets = int(np.count_nonzero((data['terminated'] != 0) | (data['timeout'] != 0)))
    if resets != complete['resets'] or mapping_error > 1.e-7:
        raise ValueError('Raw OR resets or actual wheel execution mapping differ')
    trials = score(data, proof['cases'], proof['seeds'], SECONDS, STABLE_SECONDS)
    intact(proof['source_pins'])
    result = dict(observed_at=datetime.now().astimezone().isoformat(), raw_integrity_verified=True,
        checkpoint=str(source), checkpoint_sha256=sha(source), actual_updates=5801,
        all17_Adam_step=116020, total_trials=54, survivors=sum(t['survived'] for t in trials),
        passed=sum(t['passed'] for t in trials), raw_OR_resets=resets, trials=trials,
        seconds_per_case=SECONDS, stable_seconds=STABLE_SECONDS, overlapping_windows_per_trial=98,
        original7_metrics_limits_strict_flags_and150_frame_windows_preserved=True,
        original657_and_static_gates_not_replaced=True, wheel_mapping_max_error_rad_s=mapping_error,
        independent_sustained_scope_passed=all(t['passed'] for t in trials),
        original657_all_passed=all_errors['total_passed'] == 657,
        training_modified=False, optimizer_executed=False, acceptance_changed=False, policy_accepted=False,
        source_pins={str(directory / p): sha(directory / p) for p in
            ('manifest.json', 'completed.json', 'trace.csv', 'resolved_env.json', 'policy_inputs.npz')},
        limitation='Finite independent signed fullspeed constant-turn sustained evidence. Existing static, height transitions and moving steering transitions remain separate required gates.')
    write_json(output / 'independent_fullspeed_sustained.json', result)
    write_json(output / 'status.json', dict(state='agent_review_required', diagnostic_active=False,
        independent_sustained_scope_passed=result['independent_sustained_scope_passed'],
        training_active=False, policy_accepted=False))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('qualify', 'run'))
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--training-launch', type=Path)
    parser.add_argument('--after-launch', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'qualify':
        if any(v is None for v in (args.training_launch, args.after_launch, args.output)):
            parser.error('Qualification requires exact training/after launch files and a new output')
        result = qualify(args.training_launch, args.after_launch, args.output, args.qualification)
    else:
        result = run(args.qualification)
    print(json.dumps({k: result[k] for k in ('qualified', 'total_trials', 'survivors',
        'passed', 'independent_sustained_scope_passed', 'policy_accepted') if k in result}))
