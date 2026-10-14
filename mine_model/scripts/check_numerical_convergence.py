"""Assess fixed-controller solver sweeps, without treating survival as accuracy.

Consumes native diagnostic JSON only. The screening bounds live in the sweep
manifest and are written before simulation, not chosen after looking at curves.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def finite_numeric_tree(value):
    if isinstance(value, dict):
        return all(finite_numeric_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_numeric_tree(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(np.isfinite(value))
    return True


def tree_is_connected(body_names, hinges):
    neighbours = {name: set() for name in body_names}
    edges = [h for h in hinges if not h['excluded']]
    for hinge in edges:
        left, right = hinge['bodies']
        if left not in neighbours or right not in neighbours:
            return False
        neighbours[left].add(right)
        neighbours[right].add(left)
    visited, todo = set(), [body_names[0]]
    while todo:
        name = todo.pop()
        if name not in visited:
            visited.add(name)
            todo.extend(neighbours[name] - visited)
    return len(edges) == len(body_names) - 1 and len(visited) == len(body_names)


def velocity_disagreement(times, positions, velocities, mask):
    """Central differences at the SAME state timestamps; exclude endpoints."""
    rates = np.gradient(positions, times, axis=0, edge_order=2)
    error = abs(rates - velocities)
    mask = np.asarray(mask, dtype=bool).copy()
    mask[[0, -1]] = False
    if not np.any(mask):
        return None
    return dict(rms_rad_s=float(np.sqrt(np.mean(error[mask] ** 2))),
                p95_rad_s=float(np.percentile(error[mask], 95)),
                max_rad_s=float(np.max(error[mask])))


def freefall_metrics(report):
    if report['initial']['freefall_position_m'] is None:
        return None
    records = report['physics_records']
    times = np.array([r['state_time_s'] for r in records])
    keep = times <= 1. + 1e-12
    t = times[keep]
    z = np.array([r['freefall_position_m'][0][2] for r in records])[keep]
    v = np.array([r['freefall_velocity_m_s'][0][2] for r in records])[keep]
    z0, v0 = report['initial']['freefall_position_m'][0][2], report['initial']['freefall_velocity_m_s'][0][2]
    gravity = report['gravity_m_s2'][2]
    return dict(duration_s=float(t[-1]), acceleration_fit_m_s2=float(np.polyfit(t, v, 1)[0]),
                velocity_error_max_m_s=float(np.max(abs(v - (v0 + gravity * t)))),
                position_error_max_m=float(np.max(abs(z - (z0 + v0 * t + .5 * gravity * t * t)))))


def response_metrics(times, values, step_time, delta):
    if delta == 0:
        return None
    before = values[(times >= step_time - .5) & (times < step_time)]
    after = values[times >= times[-1] - 1.]
    if not len(before) or not np.any(times >= step_time):
        # A failed run that never reaches its stimulus is not a zero response.
        return None
    baseline, final = np.mean(before, axis=0), np.mean(after, axis=0)
    change = final - baseline
    post = times >= step_time
    responses = []
    for leg in range(values.shape[1]):
        if abs(change[leg]) < 1e-8:
            responses.append(dict(actual_change=0., gain=0., rise_time_s=None, overshoot_fraction=None))
            continue
        progress = (values[post, leg] - baseline[leg]) / change[leg]
        response_time = times[post] - step_time
        t10 = response_time[np.flatnonzero(progress >= .1)[0]] if np.any(progress >= .1) else None
        t90 = response_time[np.flatnonzero(progress >= .9)[0]] if np.any(progress >= .9) else None
        responses.append(dict(actual_change=float(change[leg]), gain=float(change[leg] / delta),
                              rise_time_s=float(t90 - t10) if t10 is not None and t90 is not None else None,
                              overshoot_fraction=float(max(0., np.max(progress) - 1.)),
                              final_value=float(final[leg])))
    return responses


def assess_case(report, criteria):
    records, physics = report['records'], report['physics_records']
    times = np.array([r['elapsed_s'] for r in records])
    physics_times = np.array([r['state_time_s'] for r in physics])
    steady = times >= times[-1] - 1.
    steady_physics = physics_times >= max(2., physics_times[-1] - 1.)
    pos = np.array([r['joint_pos_rad'][0] for r in physics])
    vel = np.array([r['joint_vel_rad_s'][0] for r in physics])
    body_names, hinges = report['initial']['body_names'], report['initial']['hinges']
    gap = np.array([r['hinge_gap_m'][0] for r in physics])
    axes = np.rad2deg(np.array([r['hinge_axis_error_rad'][0] for r in physics]))
    torque = np.array([r['motor_target_nm'][0] for r in physics])
    update = np.array([r['control_update'] for r in physics])
    held = (~update)[1:]
    held_error = float(np.max(abs(np.diff(torque, axis=0)[held]))) if np.any(held) else 0.
    control_times = physics_times[update]
    ticks_ok = (len(control_times) >= 2 and
                np.allclose(np.diff(control_times), report['control_dt'], atol=1e-12) and
                report['controller_updates'] == len(control_times))
    mass = sum(report['initial']['masses_kg'])
    weight = mass * abs(report['gravity_m_s2'][2])
    contact_z = np.array([sum(f[2] for f in r['contact_forces_world_n'][0]) for r in records])
    contact_mean = float(np.mean(contact_z[steady]))
    contact_error = abs(contact_mean - weight) / weight if weight else 0.
    vdis = velocity_disagreement(physics_times, pos, vel, steady_physics)
    freefall = freefall_metrics(report)
    height = np.array([r['height_m'][0] for r in records])
    length = np.array([r['fk_length_m'][0] for r in records])
    angle = np.array([r['fk_angle_rad'][0] for r in records])
    fkintegrity = dict(
        length_error_max_m=float(np.max(abs(length - np.array([r['physical_length_m'][0] for r in records])))),
        angle_error_max_rad=float(np.max(abs(angle - np.array([r['physical_angle_rad'][0] for r in records])))),
    )
    checks = dict(
        finite=finite_numeric_tree(report),
        fixed_controller_clock=bool(ticks_ok and held_error < 1e-6 and
                                    abs(report['control_dt'] - 1 / criteria['controller_hz']) < 1e-12 and
                                    abs(report['policy_dt'] - 1 / criteria['policy_hz']) < 1e-12),
        tree_connected=tree_is_connected(body_names, hinges),
        four_external_hinges=sum(h['excluded'] for h in hinges) == 4,
        leg_builtin_pd_disabled=all(x == 0 for key in ('leg_drive_kp', 'leg_drive_kd') for x in report['initial'][key]),
        survived=report['survived_without_reset'],
        hinge_positions=bool(np.max(gap) <= criteria['hinge_gap_max_m']),
        hinge_axes=bool(np.max(axes) <= criteria['hinge_axis_max_deg']),
        steady_velocity_consistent=bool(vdis is not None and vdis['p95_rad_s'] <= criteria['steady_joint_velocity_disagreement_p95_rad_s']),
        contact_weight_balance=bool(report['survived_without_reset'] and contact_error <= criteria['steady_contact_weight_relative_error']),
        gravity_velocity=bool(freefall and freefall['velocity_error_max_m_s'] <= criteria['freefall_velocity_error_m_s']),
        gravity_position=bool(freefall and freefall['position_error_max_m'] <= criteria['freefall_position_error_m']),
    )
    return dict(checks=checks, passed=all(checks.values()),
                first_failure_s=report['first_failure_s'], mass_kg=mass, expected_weight_n=weight,
                steady_contact_z_n=contact_mean, contact_weight_relative_error=float(contact_error),
                hinge_gap_max_m=float(np.max(gap)), hinge_axis_max_deg=float(np.max(axes)),
                steady_height_m=float(np.mean(height[steady])),
                steady_length_m=np.mean(length[steady], axis=0).tolist(),
                steady_angle_rad=np.mean(angle[steady], axis=0).tolist(),
                steady_motor_nm=np.mean(np.array([r['motor_target_nm'][0] for r in records])[steady], axis=0).tolist(),
                steady_joint_velocity_disagreement=vdis,
                transient_joint_velocity_disagreement=velocity_disagreement(physics_times, pos, vel, physics_times >= .1),
                fk_vs_physical=fkintegrity, freefall=freefall,
                controller_updates=report['controller_updates'], held_torque_error_nm=held_error,
                length_response=response_metrics(times, length, report['reference_step']['step_time_s'], report['reference_step']['length_step_m']),
                angle_response=response_metrics(times, angle, report['reference_step']['step_time_s'], report['reference_step']['angle_step_rad']))


def compare_waveforms(left, right):
    # Output records occur every 20 ms in ALL experiments, including fine physics.
    keys = {'height_m': 'height_rmse_m', 'fk_length_m': 'length_rmse_m',
            'fk_angle_rad': 'angle_rmse_rad', 'vertical_velocity_m_s': 'vertical_speed_rmse_m_s',
            'motor_target_nm': 'torque_rmse_nm'}
    times = [[r['elapsed_s'] for r in item['records']] for item in (left, right)]
    if len(times[0]) != len(times[1]) or not np.allclose(*times, atol=1e-12):
        return dict(comparable=False)
    result = dict(comparable=True)
    for field, label in keys.items():
        a, b = (np.array([r[field][0] for r in item['records']]) for item in (left, right))
        result[label] = float(np.sqrt(np.mean((a - b) ** 2)))
    return result


def assess(directory, partial=False):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    criteria = manifest['criteria']
    reports, per_case = {}, {}
    for case in manifest['cases']:
        path = directory / (case['name'] + '.json')
        if path.exists():
            reports[case['name']] = json.loads(path.read_text())
            per_case[case['name']] = assess_case(reports[case['name']], criteria)
        elif not partial:
            raise ValueError('Missing diagnostic: ' + case['name'])
    pairs = []
    for kind in ('static', 'length_step', 'angle_step'):
        for fine, coarse in ((400, 200), (800, 400)):
            a, b = f'true_{kind}_{coarse}hz_p64', f'true_{kind}_{fine}hz_p64'
            if a in reports and b in reports:
                pairs.append(dict(group='timestep', left=a, right=b, **compare_waveforms(reports[a], reports[b])))
    for iterations in (32, 128):
        a, b = 'true_static_200hz_p64', f'true_static_200hz_p{iterations}'
        if a in reports and b in reports:
            pairs.append(dict(group='iterations', left=a, right=b, **compare_waveforms(reports[a], reports[b])))
    for pair in pairs:
        pair['passed'] = pair['comparable'] and all(pair[key] <= criteria[limit] for key, limit in (
            ('height_rmse_m', 'waveform_height_rmse_m'), ('length_rmse_m', 'waveform_length_rmse_m'),
            ('angle_rmse_rad', 'waveform_angle_rmse_rad'), ('torque_rmse_nm', 'waveform_torque_rmse_nm')))
    static = [value for name, value in per_case.items() if name.startswith('true_static_')]
    static_height_spread = float(np.ptp([v['steady_height_m'] for v in static])) if static else None
    static_torque_spread = float(np.max(np.ptp([v['steady_motor_nm'] for v in static], axis=0))) if static else None
    completed = len(per_case) == len(manifest['cases'])
    true_passed = all(value['passed'] for name, value in per_case.items() if name.startswith('true_'))
    reliable = (completed and true_passed and all(pair['passed'] for pair in pairs) and
                static_height_spread <= criteria['static_height_spread_m'] and
                static_torque_spread <= criteria['static_torque_spread_nm'])
    result = dict(completed=completed, numerical_screen_passed=bool(reliable), criteria=criteria,
                  static_height_spread_m=static_height_spread, static_torque_spread_nm=static_torque_spread,
                  per_case=per_case, comparisons=pairs,
                  boundary='Fixed explicit VMC under a physical Z guide. Not a free-balance or sim2real acceptance.',
                  policy_or_training_modified=False)
    (directory / ('partial_assessment.json' if partial else 'assessment.json')).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    result = assess(args.directory, args.partial)
    print(json.dumps({k: v for k, v in result.items() if k not in ('per_case', 'criteria')}, indent=2))


if __name__ == '__main__':
    main()
