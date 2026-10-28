"""Read local command, measured-motion and action-history actor sensitivities.

Every trial in a completed frozen matrix contributes three recorded steady
states. Autograd differentiates cached observations only; checkpoint tensors,
optimizer, simulator and original acceptance decisions are never modified.
Kinematic perturbations below define input directions, not a robot plant model.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_height_response import actor


SCRIPT = Path(__file__).resolve()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def distribution(values):
    x = np.asarray(values, dtype=np.float64)
    assert x.size and np.isfinite(x).all()
    return dict(mean=float(x.mean()), minimum=float(x.min()),
                p05=float(np.quantile(x, .05)), p95=float(np.quantile(x, .95)),
                maximum=float(x.max()))


def audit(matrix, output):
    assert not output.exists(), 'Preserve completed diagnostics'
    plan = read(matrix / 'qualification.json')
    complete = read(matrix / 'complete_frozen_command_matrix.json')
    assert complete['complete'] and plan['trial_count'] == len(complete['all_trials'])
    scored = {(row['case']['name'], row['seed']): row for row in complete['all_trials']}
    assert len(scored) == plan['trial_count']
    pins = {str(p): sha(p) for p in (SCRIPT, Path(actor.__code__.co_filename),
            matrix / 'qualification.json', matrix / 'complete_frozen_command_matrix.json')}
    inputs, recorded, identities = [], [], []
    config, checkpoint = None, None
    for job in plan['batches']:
        directory = Path(job['replay'])
        manifest = read(directory / 'manifest.json')
        cfg = read(directory / 'resolved_env.json')
        done = read(directory / 'completed.json')
        assert done['completed'] and not done['resets']
        assert not manifest['weights_updated'] and not manifest['sampled_policy_actions']
        assert not manifest['policy_input_remapping']
        assert manifest['checkpoint_sha256'] == complete['checkpoint_sha256']
        here = Path(manifest['checkpoint']).resolve()
        checkpoint = checkpoint or here
        assert here == checkpoint
        contract = {key: cfg[key] for key in ('obs_scales', 'wheel_action_limit',
                    'action_scale_vel', 'wheel_radius', 'wheel_track_width')}
        config = config or contract
        assert config == contract
        width = len(manifest['cases']) * len(manifest['seeds'])
        steps = round(manifest['seconds_per_case'] / manifest['policy_dt'])
        assert manifest['policy_dt'] == .02 and manifest['seconds_per_case'] >= 20.
        frames = [steps - 150, steps - 75, steps - 1]
        with np.load(directory / 'policy_inputs.npz', allow_pickle=False) as archive:
            assert archive['observations'].shape == (steps, width, 27)
            assert archive['means'].shape == (steps, width, 6)
            inputs.append(archive['observations'][frames].reshape(-1, 27).copy())
            recorded.append(archive['means'][frames].reshape(-1, 6).copy())
        identities.extend((case['name'], seed) for _ in frames
                          for case in manifest['cases'] for seed in manifest['seeds'])
        for name in ('manifest.json', 'resolved_env.json', 'completed.json', 'policy_inputs.npz'):
            pins[str(directory / name)] = sha(directory / name)
    assert len(identities) == 3 * len(scored) and set(identities) == set(scored)
    assert all(identities.count(key) == 3 for key in scored)
    assert sha(checkpoint) == complete['checkpoint_sha256']
    pins[str(checkpoint)] = sha(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    weights = saved['model_state_dict']
    assert len(saved['optimizer_state_dict']['state']) == 17
    assert all(not value.requires_grad for value in weights.values())
    torch.set_num_threads(1)
    rng = torch.random.get_rng_state().clone()
    x = torch.from_numpy(np.concatenate(inputs)).requires_grad_(True)
    raw = actor(x, weights)
    parity = float((raw.detach() - torch.from_numpy(np.concatenate(recorded))).abs().max())
    assert parity < 1.e-4, ('Actor does not reproduce the recorded mean', parity)
    radius, track, scale, bound = [config[k] for k in
        ('wheel_radius', 'wheel_track_width', 'action_scale_vel', 'wheel_action_limit')]
    wheels = raw[:, [2, 5]].clamp(-bound, bound) * scale
    output_tensors = dict(speed=radius * wheels.mean(-1),
                         yaw=radius / track * (wheels[:, 1] - wheels[:, 0]))
    gradients = {key: torch.autograd.grad(value.sum(), x, retain_graph=True)[0]
                 for key, value in output_tensors.items()}
    assert x.grad is None and all(value.grad is None for value in weights.values())
    obs = config['obs_scales']
    directions = {}
    for name, entries in (
        ('command_speed', {6:obs['lin_vel']}),
        ('command_yaw', {7:obs['ang_vel']}),
        ('measured_speed_only', {17:obs['lin_vel']}),
        ('measured_yaw_only', {2:obs['ang_vel']}),
        ('measured_ideal_rolling_speed', {17:obs['lin_vel'],
             19:obs['dof_vel'] / radius, 20:obs['dof_vel'] / radius}),
        ('measured_ideal_rolling_yaw', {2:obs['ang_vel'],
             19:-obs['dof_vel'] * track / (2. * radius),
             20:obs['dof_vel'] * track / (2. * radius)}),
        ('previous_wheel_reference_speed', {23:1. / (radius * scale),
                                           26:1. / (radius * scale)}),
        ('previous_wheel_reference_yaw', {23:-track / (2. * radius * scale),
                                         26:track / (2. * radius * scale)}),
    ):
        direction = torch.zeros(27)
        for column, value in entries.items():
            direction[column] = value
        directions[name] = direction
    sensitivities = {key:{name:(gradient * direction).sum(-1).detach().numpy()
                         for name, direction in directions.items()}
                     for key, gradient in gradients.items()}
    epsilon = 1.e-3
    numerical_errors = {}
    with torch.no_grad():
        for name, direction in directions.items():
            references = []
            for sign in (-1., 1.):
                mu = actor(x.detach() + sign * epsilon * direction, weights)
                w = mu[:, [2, 5]].clamp(-bound, bound) * scale
                references.append(dict(speed=radius * w.mean(-1),
                    yaw=radius / track * (w[:, 1] - w[:, 0])))
            numerical_errors[name] = {}
            for key in output_tensors:
                difference = (references[1][key] - references[0][key]) / (2. * epsilon)
                error = float(np.max(abs(difference.numpy() - sensitivities[key][name])))
                assert error < .01, (key, name, error)
                numerical_errors[name][key] = error
    indices = defaultdict(list)
    for index, identity in enumerate(identities):
        indices[identity].append(index)
    trials = []
    for identity, locations in indices.items():
        score = scored[identity]
        trials.append(dict(case=score['case'], seed=score['seed'], survived=score['survived'],
            original_passed=score['original_passed'],
            strict10_passed=score['original_AND_literal10_passed'],
            nonzero_relative_percent=score['nonzero_relative_percent'],
            local_wheel_reference_response={key:{name:float(values[locations].mean())
                for name, values in entries.items()} for key, entries in sensitivities.items()}))
    groups = {}
    for axis in ('speed', 'yaw'):
        groups[axis] = {}
        for label, failed in (('passed_relative10', False), ('failed_relative10', True)):
            selected = [row for row in trials if row['nonzero_relative_percent'][axis] is not None
                and (row['nonzero_relative_percent'][axis] >= 10.) == failed]
            groups[axis][label] = dict(trials=len(selected), response={
                name:distribution([row['local_wheel_reference_response'][axis][name] for row in selected])
                for name in directions} if selected else None)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(sha(path) == digest for path, digest in pins.items())
    result = dict(verified=True, observed_at=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint), checkpoint_sha256=complete['checkpoint_sha256'],
        original_trials_retained=len(trials), cached_frames_per_trial=3,
        cached_state_count=len(identities), recorded_actor_mean_parity_max_error=parity,
        central_difference_step_in_physical_input_units=epsilon,
        derivative_central_difference_max_errors=numerical_errors,
        derivative_units='Output rolling-speed/yaw-reference proxy per physical input unit.',
        input_direction_columns={name:direction.tolist() for name,direction in directions.items()},
        parameter_gradients_computed=False, input_derivatives_computed=True,
        all_checkpoint_optimizer_and_source_files_unchanged=True, all_RNG_unchanged=True,
        simulator_started=False, optimizer_executed=False, training_modified=False,
        acceptance_changed=False, policy_accepted=False,
        groups=groups, all_trials=trials, source_pins=pins,
        limitations=[
            'Local actor derivatives hold unselected observations fixed; they are not closed-loop robot gains.',
            'Ideal rolling measurement directions define a kinematic sensor perturbation, not a physical simulation.',
            'History-response derivatives do not prove an integral controller or identify actuator/friction dynamics.',
            'Passed/failed labels are original replay evidence used for diagnosis only; they never enter training.',
            'These cached states are correlated. This report cannot establish a causal learning defect or policy acceptance.'])
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False) + '\n')
    print(json.dumps(dict(verified=True, original_trials=len(trials),
        recorded_mean_max_error=parity, groups=groups, optimizer_executed=False)))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.matrix.resolve(),args.output.resolve())
