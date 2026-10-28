"""Supplemental frozen moving-height transitions using original10% metric functions.

This follows a nonstartup native low-height event. It changes no training,
physical termination, original static validation or acceptance configuration.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import ROOT, sha
import train_source4401_consolidation200 as controller
import review_source4401_consolidation_full_height_and_retention_v2 as review
import joint_pd_acceptance_10pct_20261017 as acceptance
import verify_static_height_steps10 as static
import diagnose_motion025_full_height as native
from wheel_legged_gym_isaaclab.action_bias_probe import command_probe_plan

SCRIPT = Path(__file__).resolve()
RESULTS = ROOT / 'mine_model/results'
QUALIFICATION = RESULTS / 'joint_pd_source4401_consolidation_motion_height_qualification_20261019.json'
OUTPUT = RESULTS / 'joint_pd_motion_height_steps_consolidation_20261019_source4601'
SEEDS = [583, 584, 585]


def plan():
    base = static.step_plan()
    motions = [('standing', 0., 0.), ('forward', .5, 0.), ('spin_right', 0., -.08112095296382904)]
    phases = [[dict(name=label + '_' + case['name'], speed=speed, yaw=yaw, height=case['height'])
        for label, speed, yaw in motions for case in phase] for phase in base['phases']]
    return dict(phase_seconds=base['phase_seconds'], phases=phases)


def limits(tail, case, criteria):
    fraction = criteria['relative_error']
    length = max(.001, min(np.mean(abs(tail[f'{side}_length_ref_m'])) for side in ('left', 'right')))
    wheel = min(np.mean(abs(tail[f'{side}_wheel_ref_rad_s'])) for side in ('left', 'right'))
    return dict(speed_mae_m_s=max(criteria['speed_absolute_m_s'], fraction * abs(case['speed'])),
        yaw_mae_rad_s=max(criteria['yaw_absolute_rad_s'], fraction * abs(case['yaw'])),
        height_mae_m=fraction * case['height'], leg_length_mae_m=fraction * length,
        leg_angle_mae_rad=fraction * criteria['leg_angle_design_range_rad'],
        wheel_speed_mae_rad_s=max(criteria['wheel_absolute_rad_s'], fraction * wheel),
        body_p95_deg=criteria['body_p95_deg'])


def argv(source, output, sampled):
    command = controller.runtime.replay_command(source, output)
    command[command.index('--seconds') + 1] = '50'
    i = command.index('--seeds') + 1
    command[i:i + 3] = [str(seed) for seed in SEEDS]
    command.extend(['--command-probe-json', str(OUTPUT / 'command_plan.json')])
    if sampled:
        command.append('--sample-policy-actions')
    if any(x in command for x in ('--overrides_json', '--command-scale', '--heading-hold',
            '--export_dir', controller.runtime.FLAG)):
        raise ValueError('No physical, input, action or heading-feedback overrides')
    return command


def qualify():
    launch_path = controller.QUALIFICATION / 'training_launch.json'
    queued_path = controller.QUALIFICATION / 'four51_static_review_launch_v2.json'
    launch, queued = [json.loads(p.read_text()) for p in (launch_path, queued_path)]
    source = Path(launch['source_checkpoint'])
    _, cfg = controller.runtime.require_source(source)
    checked, sequence, period = command_probe_plan(plan(), SimpleNamespace(**cfg['commands']))
    if (len(checked) != 9 or len(sequence) != 10 or period != 5.
            or cfg['commands']['training_motion_scale'] != .625
            or cfg['min_root_height'] != .22 or cfg['low_height_terminal_time_s'] != .15
            or max(abs(c['speed']) for c in plan()['phases'][0]) > .625 * .8):
        raise ValueError('Preserve the current envelope and fixed physical support termination')
    prior = Path(controller.ready()['training_output']) / 'replay'
    assessment = json.loads((prior / 'assessment.json').read_text())
    manifest = json.loads((prior / 'manifest.json').read_text())
    trace = np.genfromtxt(prior / 'trace.csv', delimiter=',', names=True)
    criteria = acceptance.profile()['criteria']
    max_error, checked_trials = 0., 0
    for cid, case in enumerate(assessment['conditions']):
        for trial in case['trials']:
            tail = trace[(trace['case_id'] == cid) & (trace['seed'] == trial['seed']) & (trace['time_s'] >= 17.)]
            physical = acceptance.assessor.metrics_for_window(tail, case, criteria)
            bound = limits(tail, case, criteria)
            for key, value in bound.items():
                max_error = max(max_error, abs(value - trial['limits'][key]), abs(physical[key] - trial['metrics'][key]))
            strict = {key: value - 1.e-12 for key, value in bound.items()}
            flags = acceptance.assessor.tracking_components(trial['survived'], physical, strict)
            if any(flags[k] != trial[k] for k in flags):
                raise ValueError('Original canonical component flags differ')
            checked_trials += 1
    if checked_trials != 27 or max_error > 1.e-12 or manifest['checkpoint_sha256'] != sha(source):
        raise ValueError('Require all original27 metric/limit/flag parity checks')
    event = Path(launch['output']) / 'terminal_event_ages_prefix_0040_readonly.json'
    rows = json.loads(event.read_text())['events']
    observed = next(e for e in rows if e['environment_id'] == 2992 and e['update'] == 37)
    files = [SCRIPT, launch_path, queued_path, prior / 'assessment.json', prior / 'manifest.json',
        prior / 'trace.csv', source, event, Path(static.__file__), Path(acceptance.assessor.__file__),
        acceptance.CURRENT, acceptance.POLICY, controller.runtime.SCRIPT,
        Path(review.__file__), Path(command_probe_plan.__code__.co_filename)]
    result = dict(qualified=True, created_at=datetime.now().astimezone().isoformat(),
        output=str(OUTPUT), training_output=launch['output'], training_unit=launch['unit'],
        after_review_unit=queued['unit'], training_launch=str(launch_path),
        target_actual_updates=4601, target_all17_Adam_step=92020,
        cases=9, seeds=SEEDS, phases=10, phase_seconds=5., samples_per_native_run=67500,
        command_plan=plan(), original27_numeric_limit_flag_parity_max_error=max_error,
        original27_trials_checked=checked_trials, motivating_native_event=observed,
        causal_mechanism_unproven=True, fixed_min_root_height_unchanged=.22,
        low_height_terminal_time_unchanged=.15, source_pins={str(p): sha(p) for p in files},
        command_construction={label: argv(source, OUTPUT / label, sampled)
            for label, sampled in [('deterministic', False), ('saved_Gaussian', True)]},
        PPO_updates_performed=0, simulator_started=False, acceptance_changed=False,
        original_canonical_and_static_retention_not_replaced=True, policy_accepted=False)
    with QUALIFICATION.open('x') as stream:
        json.dump(result, stream, indent=2)
    return result


def assess(directory, source, sampled):
    directory = Path(directory)
    manifest, complete, cfg = [json.loads((directory / n).read_text())
        for n in ('manifest.json', 'completed.json', 'resolved_env.json')]
    spec = plan()
    sequence = [[[c[k] for k in ('speed', 'yaw', 'height')] for c in phase] for phase in spec['phases']]
    if (manifest['checkpoint_sha256'] != sha(source) or not complete['completed'] or complete['samples'] != 67500
            or manifest['cases'] != spec['phases'][0] or manifest['command_sequence'] != sequence
            or manifest['command_phase_seconds'] != 5. or manifest['seconds_per_case'] != 50.
            or manifest['seeds'] != SEEDS or manifest['sampled_policy_actions'] is not sampled
            or manifest['reset_velocity_uniform_amplitude'] != .05 or manifest['physics_dt'] != .005
            or manifest['policy_dt'] != .02 or manifest['solver_position_iterations'] != 128
            or manifest['solver_velocity_iterations'] != 8 or manifest['enable_external_forces_every_iteration'] is not True
            or manifest['command_scale'] != 1. or manifest['curriculum_progress'] is not None
            or manifest['policy_input_remapping'] or manifest['action_bias_probes'] or manifest['wheel_probe']
            or manifest['wheel_reference_limit_probe'] is not None or manifest['weights_updated']
            or manifest['leg_control_mode'] != 'implicit_joint_position'
            or cfg['min_root_height'] != .22 or cfg['low_height_terminal_time_s'] != .15):
        raise ValueError('Require the complete unchanged-control moving-height probe')
    for p, digest in manifest['asset_chain_sha256'].items():
        if sha(p) != digest:
            raise ValueError('Physical asset changed')
    data = np.genfromtxt(directory / 'trace.csv', delimiter=',', names=True)
    if len(data) != 67500 or any(not np.isfinite(data[k]).all() for k in data.dtype.names):
        raise ValueError('Retain all finite episodes and failures')
    criteria, trials = acceptance.profile()['criteria'], []
    timeline = np.arange(1, 2501) * .02
    phase_ids = np.floor((timeline - .02 + 1.e-7) / 5.).astype(int)
    for cid, initial in enumerate(spec['phases'][0]):
        for seed in SEEDS:
            rows = data[(data['case_id'] == cid) & (data['seed'] == seed)]
            if len(rows) != 2500 or not np.allclose(rows['time_s'], timeline, atol=1.e-8, rtol=0.):
                raise ValueError('Missing complete50s timeline')
            target = np.asarray([sequence[i][cid] for i in phase_ids])
            if (not np.allclose(rows['height_cmd_m'], target[:, 2], atol=1.e-7, rtol=0.)
                    or not np.allclose(rows['yaw_cmd_rad_s'], target[:, 1], atol=1.e-7, rtol=0.)
                    or np.any(rows['speed_cmd_m_s'] < -1.e-7)
                    or np.any(rows['speed_cmd_m_s'] > initial['speed'] + 1.e-7)):
                raise ValueError('Commands differ from planned heights/yaw or allowed original speed ramp')
            survived = not any(np.any(rows[k]) for k in ('episode', 'terminated', 'timeout'))
            phases = []
            for index in range(10):
                part = rows[phase_ids == index]
                case = spec['phases'][index][cid]
                tail = part[part['time_s'] >= 5. * (index + 1) - 3.]
                if len(tail) != 151:
                    raise ValueError('Original inclusive3s steady phase window differs')
                bound = limits(tail, case, criteria)
                physical = acceptance.assessor.metrics_for_window(tail, case, criteria) if survived else None
                flags = acceptance.assessor.tracking_components(survived, physical,
                    {k: v - 1.e-12 for k, v in bound.items()})
                first = part[part['time_s'] <= 5. * index + 1. + .002]
                phases.append(dict(index=index, command=case, metrics=physical, limits=bound, **flags,
                    phase_tracking_passed=flags['task_goal_passed'] and flags['reference_tracking_passed'],
                    first_second=dict(root_height_min_m=float(first['height_m'].min()),
                        pitch_peak_deg=float(np.rad2deg(abs(first['pitch_rad'])).max()),
                        roll_peak_deg=float(np.rad2deg(abs(first['roll_rad'])).max()))))
            failed = rows[(rows['terminated'] > 0) | (rows['timeout'] > 0)]
            trials.append(dict(case=initial, seed=seed, survived=bool(survived),
                first_failure_s=None if not len(failed) else float(failed['time_s'][0]),
                resets=int(rows['terminated'].sum()), phases=phases,
                passed=bool(survived and all(p['phase_tracking_passed'] for p in phases))))
    result = dict(checkpoint=str(source), checkpoint_sha256=sha(source), sampled=sampled,
        total_trials=27, survivors=sum(t['survived'] for t in trials), resets=complete['resets'],
        all_numeric_phase_gates_passed=all(t['passed'] for t in trials), trials=trials,
        criteria=criteria, original_metric_function_and_limits_reused=True,
        first_second_transients_recorded_separately=True, command_plan=spec,
        original_canonical_and_static_retention_not_replaced=True, policy_accepted=False,
        source_pins={str(directory / n): sha(directory / n) for n in
            ('manifest.json', 'completed.json', 'trace.csv', 'resolved_env.json')},
        limitation='Supplemental finite heldout transitions, not exact reproduction of the native event or proof against rare failures. Gaussian numerical errors are reported, while its stress criterion remains survival only.')
    write_json(directory / 'motion_height_steps_diagnosis.json', result)
    return result


def run():
    proof = json.loads(QUALIFICATION.read_text())
    if not proof['qualified'] or any(sha(p) != h for p, h in proof['source_pins'].items()):
        raise ValueError('Require intact numeric parity and planned within-envelope commands')
    if OUTPUT.exists():
        raise ValueError('Preserve earlier diagnostic outputs')
    while True:
        raw = subprocess.check_output(['systemctl', '--user', 'show', proof['after_review_unit'],
            '-p', 'MainPID', '-p', 'SubState', '-p', 'Result'], text=True)
        state = dict(line.split('=', 1) for line in raw.splitlines() if '=' in line)
        if state.get('MainPID') == '0' and state.get('SubState') == 'exited' and state.get('Result') == 'success':
            break
        if state.get('SubState') in ('failed', 'dead'):
            raise ValueError('Inspect the specific preceding review unit; never restart PPO from an observation timeout')
        write_json(OUTPUT.with_name(OUTPUT.name + '_waiter.json'), dict(
            observed_at=datetime.now().astimezone().isoformat(), state='waiting_for_own_original_reviews',
            unit=proof['after_review_unit'], unit_state=state, training_modified=False))
        time.sleep(15.)
    training = Path(proof['training_output'])
    source = review.retention().reviewed_source(training)
    final = json.loads((training / 'final_checkpoint_verification_readonly_20261019.json').read_text())
    if not final['verified'] or final['actual_ppo_updates'] != 4601 or final['all17_Adam_step'] != 92020:
        raise ValueError('Require the exact completed target; this probe runs no PPO')
    OUTPUT.mkdir()
    write_json(OUTPUT / 'command_plan.json', plan())
    write_json(OUTPUT / 'plan.json', dict(qualification=str(QUALIFICATION), qualification_sha256=sha(QUALIFICATION),
        checkpoint=str(source), checkpoint_sha256=sha(source), actual_updates=4601,
        fixed_physical_termination_not_changed=True, acceptance_changed=False, policy_accepted=False))
    reports = {}
    for label, sampled in [('deterministic', False), ('saved_Gaussian', True)]:
        command = argv(source, OUTPUT / label, sampled)
        write_json(OUTPUT / (label + '_launch.json'), dict(command=command, checkpoint_sha256=sha(source)))
        write_json(OUTPUT / 'status.json', dict(state=label, diagnostic_active=True, training_active=False,
            checkpoint_sha256=sha(source), policy_accepted=False))
        with (OUTPUT / (label + '.log')).open('x') as log:
            subprocess.run(command, cwd=ROOT, env=native.environment(), stdout=log, stderr=subprocess.STDOUT, check=True)
        reports[label] = assess(OUTPUT / label, source, sampled)
        if any(sha(p) != h for p, h in proof['source_pins'].items()):
            raise ValueError('Original code, criteria or qualified inputs changed')
    result = dict(checkpoint=str(source), checkpoint_sha256=sha(source), actual_updates=4601,
        results={label: {k: report[k] for k in ('total_trials', 'survivors', 'resets', 'all_numeric_phase_gates_passed')}
            for label, report in reports.items()},
        supplemental_steps_ready=reports['deterministic']['all_numeric_phase_gates_passed']
            and reports['saved_Gaussian']['survivors'] == 27 and reports['saved_Gaussian']['resets'] == 0,
        same_saved_weights_and_commands=True, training_modified=False, optimizer_executed=False,
        acceptance_changed=False, original_canonical_and_static_retention_not_replaced=True,
        policy_accepted=False, hardware_enabled=False, shutdown=False)
    write_json(OUTPUT / 'motion_height_steps_summary.json', result)
    write_json(OUTPUT / 'status.json', dict(state='agent_review_required', diagnostic_active=False,
        training_active=False, checkpoint_sha256=sha(source), policy_accepted=False))
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--qualify', action='store_true')
    args = cli.parse_args()
    result = qualify() if args.qualify else run()
    print(json.dumps({k: result[k] for k in ('qualified', 'actual_updates', 'supplemental_steps_ready', 'policy_accepted') if k in result}))
