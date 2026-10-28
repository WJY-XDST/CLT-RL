"""Project recorded independent wheel Gaussians through the saved clamp.

This measures instantaneous wheel-reference exploration, not body tracking,
closed-loop variance, policy quality, or an acceptance probability. No new
random samples, simulator, model updates, or acceptance changes are used.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def moments(mean, std, limit):
    a, b = (-limit - mean) / std, (limit - mean) / std
    pa, pb = ndtr(a), ndtr(b)
    fa, fb = np.exp(-a * a / 2.) / np.sqrt(2. * np.pi), np.exp(-b * b / 2.) / np.sqrt(2. * np.pi)
    interior = pb - pa
    first = -limit * pa + mean * interior + std * (fa - fb) + limit * (1. - pb)
    second = (limit * limit * (pa + 1. - pb) + (mean * mean + std * std) * interior
        + 2. * mean * std * (fa - fb) + std * std * (a * fa - b * fb))
    variance = second - first * first
    if np.min(variance) < -1.e-12:
        raise ValueError('Investigate clipped-Gaussian moment cancellation')
    return first, np.maximum(variance, 0.), pa + 1. - pb


def quadrature_error():
    error = 0.
    for mean in (-1.4, -1.25, -.3, 0., 1.243, 1.25, 1.4):
        std, limit = .01, 1.25
        first, variance, _ = moments(np.array(mean), np.array(std), limit)
        points = [v for v in ((-limit - mean) / std, (limit - mean) / std) if -12. < v < 12.]
        integrals = [quad(lambda z: float(np.clip(mean + std * z, -limit, limit)) ** power
            * np.exp(-z * z / 2.) / np.sqrt(2. * np.pi), -12., 12.,
            points=points, epsabs=1.e-12, epsrel=1.e-12)[0] for power in (1, 2)]
        error = max(error, abs(float(first) - integrals[0]),
            abs(float(variance) - (integrals[1] - integrals[0] ** 2)))
    if error > 1.e-11:
        raise ValueError('Analytical moments do not reproduce independent quadrature')
    return error


def audit(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'wheel_gaussian_reference_precision_prefix_{through:04d}_readonly.json'
    if directory.parent != ROOT / 'mine_model/results' or not 1 <= through <= 200 or destination.exists():
        raise ValueError('Require an owned committed prefix and a fresh diagnostic')
    plan_path = directory / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    cfg = plan['training_config']
    if cfg['leg_control_mode'] != 'implicit_joint_position' or cfg['observation_space'] != 27:
        raise ValueError('Keep the original direct-joint plain27 contract')
    limit, scale = cfg['wheel_action_limit'], cfg['action_scale_vel']
    radius, track = cfg['wheel_radius'], cfg['wheel_track_width']
    if limit != 1.25 or scale != 15. or radius <= 0. or track <= 0.:
        raise ValueError('Read the exact saved wheel geometry and software bound')
    gate_path = ROOT / 'mine_model/config/tracking_acceptance.json'
    gate = json.loads(gate_path.read_text())
    pins = {str(p): sha(p) for p in (SCRIPT, plan_path, gate_path)}
    arrays, environment_ids = [], None
    for i in range(1, through + 1):
        audit_path = directory / f'native_learning_signal/audit_update_{i:04d}.json'
        data_path = directory / f'native_learning_signal/rollout_{i:04d}.npz'
        record = json.loads(audit_path.read_text())
        if (record['update'] != i or record['total_environments'] != 4096
                or any(record['all_environment_alignment_max_errors'].values())
                or not record['probe_RNG_unchanged'] or record['added_loss_or_backward']):
            raise ValueError('Resolve native alignment before interpreting exploration')
        if environment_ids is None:
            environment_ids = record['sampled_environment_ids']
        if environment_ids != record['sampled_environment_ids']:
            raise ValueError('Recorded population changed')
        with np.load(data_path, allow_pickle=False) as data:
            mean, std = [data[k][..., [2, 5]].astype('float64') for k in ('mu', 'sigma')]
            commands = data['native_commands'].astype('float64')
            if mean.shape != (48, 32, 2) or std.shape != mean.shape or np.any(std <= 0.):
                raise ValueError('Require original48-by32 independent wheel Gaussians')
            _, variance, clipping = moments(mean, std, limit)
            total_variance = variance.sum(-1) * scale ** 2
            arrays.append(dict(commands=commands, std=std,
                yaw_std=radius / track * np.sqrt(total_variance),
                speed_std=radius / 2. * np.sqrt(total_variance), clipping=clipping))
        pins.update({str(p): sha(p) for p in (audit_path, data_path)})
    data = {k: np.concatenate([a[k] for a in arrays], axis=0) for k in arrays[0]}
    magnitude = abs(data['commands'][..., 1])
    groups = {}
    for name, mask in (('zero_yaw', magnitude == 0.),
            ('small_yaw_0p05_to_0p10', (magnitude >= .05) & (magnitude < .10)),
            ('yaw_0p10_to_0p20', (magnitude >= .10) & (magnitude < .20)),
            ('yaw_0p20_to_0p30', (magnitude >= .20) & (magnitude <= .300001))):
        count = int(mask.sum())
        if not count:
            groups[name] = dict(recorded_states=0)
            continue
        ratio = data['yaw_std'][mask] / (gate['relative_error'] * magnitude[mask]) if name != 'zero_yaw' else None
        groups[name] = dict(recorded_states=count,
            instantaneous_yaw_reference_std_P5_P50_P95_rad_s=np.percentile(data['yaw_std'][mask], [5, 50, 95]).tolist(),
            instantaneous_speed_reference_std_P5_P50_P95_m_s=np.percentile(data['speed_std'][mask], [5, 50, 95]).tolist(),
            wheel_clipping_probability_mean_by_head=data['clipping'][mask].mean(0).tolist(),
            reference_yaw_std_divided_by_literal_body_MAE_threshold_P50=None if ratio is None else float(np.median(ratio)))
    if any(sha(p) != h for p, h in pins.items()):
        raise ValueError('A committed input changed')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_prefix=through,
        source_actual_updates=plan['source_actual_ppo_updates'], sampled_environment_ids=environment_ids,
        saved_geometry=dict(wheel_radius_m=radius, wheel_track_m=track),
        normal_wheel_cap_rad_s=limit * scale, recorded_raw_std_min=float(data['std'].min()),
        recorded_raw_std_max=float(data['std'].max()),
        unclipped_wheel_floor_0p01_yaw_reference_std_rad_s=radius / track * scale * np.sqrt(2.) * .01,
        analytic_moments_independent_quadrature_max_error=quadrature_error(), groups=groups,
        reference_variance_formula='yaw=(radius/track)^2*scale^2*(var_clip_left+var_clip_right); speed=(radius/2)^2*scale^2*(var_clip_left+var_clip_right)',
        limitations=['Instantaneous reference noise is filtered by actuator and body dynamics; it is not body-yaw MAE or a predicted acceptance failure.',
            'Only32 recorded environments; not a population failure-rate estimate.',
            'Comparison to the body-MAE threshold is dimensional context, not a replacement metric or gate.',
            'No floor or reward change is selected by this diagnostic.'],
        source_pins=pins, RNG_draws_added=0, training_modified=False, optimizer_executed=False,
        simulator_started=False, acceptance_changed=False, policy_accepted=False)
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return destination, result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    path, result = audit(args.directory, args.through)
    print(json.dumps(dict(report=str(path), groups=result['groups'], optimizer_executed=False)))
