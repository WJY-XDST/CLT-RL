"""Read frozen from-reset height steps using the unchanged strict10% metrics.

This adds changing-command evidence to canonical20/30/60-second validation.
It is never a replacement for the original full27 acceptance report.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import sha
import joint_pd_acceptance_10pct_20261017 as acceptance
import assess_height_leg_command_tracking as legs
from summarize_joint_pd_height_transitions import moments
from wheel_legged_gym_isaaclab.height_commands import SAFE_BASE_HEIGHT_RANGE_M


def step_plan():
    low, high = SAFE_BASE_HEIGHT_RANGE_M
    return dict(phase_seconds=5., phases=[
        [dict(name=name, speed=0., yaw=0., height=height) for name, height in (
            ('nominal', .3), ('from_low', low if i % 2 == 0 else high),
            ('from_high', high if i % 2 == 0 else low))] for i in range(10)])


def phase_metrics(rows, case, end_time, criteria, survived):
    """Same final3s metrics and limits, plus actual legs vs height command."""
    if not survived:
        return dict(passed=False, survived=False, metrics=None,
                    reason='A failed first episode cannot supply a passing steady window')
    tail = rows[rows['time_s'] >= end_time-3.]
    if len(tail) < 150:
        raise ValueError('Missing final3s of a5s command phase')
    physical = acceptance.assessor.metrics_for_window(tail, case, criteria)
    fraction = criteria['relative_error']
    leg_reference = max(.001, min(np.mean(abs(tail[f'{side}_length_ref_m'])) for side in legs.SIDES))
    wheel_reference = min(np.mean(abs(tail[f'{side}_wheel_ref_rad_s'])) for side in legs.SIDES)
    limits = dict(speed_mae_m_s=criteria['speed_absolute_m_s'],
        yaw_mae_rad_s=criteria['yaw_absolute_rad_s'], height_mae_m=fraction*case['height'],
        leg_length_mae_m=fraction*leg_reference,
        leg_angle_mae_rad=fraction*criteria['leg_angle_design_range_rad'],
        wheel_speed_mae_rad_s=max(criteria['wheel_absolute_rad_s'], fraction*wheel_reference),
        body_p95_deg=criteria['body_p95_deg'])
    strict = {key: value-1.e-12 for key, value in limits.items()}
    parts = acceptance.assessor.tracking_components(True, physical, strict)
    command = legs.trial_metrics(rows, case, end_time, criteria)
    passed = (parts['task_goal_passed'] and parts['reference_tracking_passed']
              and command['height_posture_passed'] and command['leg_command_passed']
              and command['stationary_drift_passed'])
    return dict(passed=bool(passed), survived=True, metrics=physical, limits=limits,
                components=parts, actual_leg_command=command,
                settling_seconds=2., steady_window_seconds=3.)


def verify(directory, plan_file, *, seeds=(383,384,385), sampled=False, reset_amplitude=.05):
    directory, plan_file = Path(directory).resolve(), Path(plan_file).resolve()
    manifest = json.loads((directory/'manifest.json').read_text())
    complete = json.loads((directory/'completed.json').read_text())
    plan = json.loads(plan_file.read_text())
    sequence = [[[case[key] for key in ('speed','yaw','height')] for case in phase]
                for phase in plan['phases']]
    if (reset_amplitude not in (.05,.5) or plan != step_plan()
            or manifest.get('policy_acceptance_test') is not False
            or not manifest.get('command_probe') or manifest.get('weights_updated') is not False
            or manifest.get('leg_control_mode') != 'implicit_joint_position'
            or manifest.get('policy_input_remapping') or manifest.get('command_scale') != 1.
            or manifest.get('curriculum_progress') is not None or manifest.get('action_bias_probes')
            or manifest.get('wheel_probe') or manifest.get('wheel_reference_limit_probe') is not None
            or manifest.get('reset_velocity_uniform_amplitude') != reset_amplitude
            or manifest.get('sampled_policy_actions') is not sampled
            or manifest.get('physics_dt') != .005 or manifest.get('policy_dt') != .02
            or manifest.get('solver_position_iterations') != 128
            or manifest.get('solver_velocity_iterations') != 8
            or manifest.get('enable_external_forces_every_iteration') is not True
            or manifest['cases'] != plan['phases'][0] or manifest['command_sequence'] != sequence
            or manifest['command_phase_seconds'] != 5. or manifest['seconds_per_case'] != 50.
            or manifest['seeds'] != list(seeds) or len(seeds) != 3 or len(set(seeds)) != 3
            or not complete['completed'] or complete['samples'] != 22500
            or sha(manifest['checkpoint']) != manifest['checkpoint_sha256']):
        raise ValueError('Require complete unchanged from-reset extrema and full-range5s steps')
    for path, digest in manifest['asset_chain_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed calibrated replay asset: '+path)
    samples = np.genfromtxt(directory/'trace.csv', delimiter=',', names=True)
    if len(samples) != 22500 or not all(np.isfinite(samples[key]).all() for key in samples.dtype.names):
        raise ValueError('Incomplete or nonfinite height-step evidence')
    criteria = acceptance.profile()['criteria']
    trials = []
    for cid, case in enumerate(plan['phases'][0]):
        for seed in seeds:
            rows = samples[(samples['case_id'] == cid) & (samples['seed'] == seed)]
            timeline = np.arange(1,2501)*.02
            if len(rows) != 2500 or not np.allclose(rows['time_s'], timeline, atol=1.e-8, rtol=0.):
                raise ValueError('Missing complete first-episode50s timeline')
            phase_ids = np.floor((timeline-.02+1.e-7)/5.).astype(int)
            targets = np.array([sequence[i][cid] for i in phase_ids])
            actual = np.column_stack([rows[key] for key in ('speed_cmd_m_s','yaw_cmd_rad_s','height_cmd_m')])
            if not np.allclose(targets, actual, atol=1.e-7, rtol=0.):
                raise ValueError('Recorded commands changed or narrowed the declared full-height steps')
            survived = bool(not np.any(rows['episode']) and not np.any(rows['terminated'])
                            and not np.any(rows['timeout']))
            phases = []
            for i, commands in enumerate(plan['phases']):
                part = rows[phase_ids == i]
                age = part['time_s']-5.*i
                phases.append(dict(index=i, target_height_m=commands[cid]['height'],
                    first_second=moments(part[age <= 1.+.002]),
                    **phase_metrics(part, commands[cid], 5.*(i+1), criteria, survived)))
            trials.append(dict(case=case['name'], seed=seed, survived=survived,
                resets=int(rows['terminated'].sum()), phases=phases,
                passed=bool(survived and all(phase['passed'] for phase in phases))))
    passed = bool(len(trials) == 9 and all(trial['passed'] for trial in trials) and complete['resets'] == 0)
    report = dict(checkpoint=manifest['checkpoint'], checkpoint_sha256=manifest['checkpoint_sha256'],
        sampled_policy_actions=sampled, seeds=list(seeds), reset_velocity_amplitude=reset_amplitude,
        canonical_reset_distribution=reset_amplitude == .05,
        first_command_includes_both_full_height_endpoints=True, nominal_settling_phase_removed=True,
        full_height_range=list(SAFE_BASE_HEIGHT_RANGE_M), repeated_changes_per_extreme_trial=9,
        criteria=criteria, total_trials=9, survivors=sum(trial['survived'] for trial in trials),
        resets=complete['resets'], height_steps_passed=passed, trials=trials,
        source_files={str(directory/name):sha(directory/name)
                      for name in ('manifest.json','completed.json','trace.csv','resolved_env.json')},
        command_plan_sha256=sha(plan_file), validator_sha256=sha(__file__),
        canonical_acceptance_replaced=False, policy_accepted=False, training_modified=False,
        limitation='Three paired reset seeds. Saved-Gaussian and deterministic runs are separate. '
                   'No-reset50s steps and steady3s metrics do not prove absence of rare failures, '
                   'replace canonical or independent sustained reports, or authorize real hardware.')
    write_json(directory/'static_height_step_validation.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs=3, default=[383,384,385])
    parser.add_argument('--sampled', action='store_true')
    parser.add_argument('--reset-amplitude', type=float, choices=[.05,.5], default=.05)
    options = parser.parse_args()
    result = verify(options.directory, options.plan, seeds=options.seeds, sampled=options.sampled,
                    reset_amplitude=options.reset_amplitude)
    print(json.dumps({key:result[key] for key in ('checkpoint_sha256','total_trials','survivors',
                                               'resets','height_steps_passed','policy_accepted')}))
