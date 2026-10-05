"""Measure saved actor feedback; trace reconstruction is approximate, not physics linearization."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replay', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.replay / 'manifest.json').read_text())
    checkpoint = Path(manifest['checkpoint'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    state = saved['model_state_dict']
    layers = []
    for i in (0, 2, 4, 6):
        w = state[f'actor.{i}.weight']
        linear = torch.nn.Linear(w.shape[1], w.shape[0])
        linear.weight.data.copy_(w)
        linear.bias.data.copy_(state[f'actor.{i}.bias'])
        layers.append(linear)
        if i != 6:
            layers.append(torch.nn.ELU())
    actor = torch.nn.Sequential(*layers).eval()
    exact = np.load(args.replay / 'policy_inputs.npz') if (args.replay / 'policy_inputs.npz').exists() else None
    parity_max = None
    if exact is not None:
        with torch.no_grad():
            predicted = actor(torch.tensor(exact['observations'])).numpy()
        parity_max = float(np.max(np.abs(predicted - exact['means'])))
        if parity_max > 1e-4:
            raise ValueError('Recorded actor inference does not match saved checkpoint')
    groups = {}
    with (args.replay / 'trace.csv').open() as f:
        for row in csv.DictReader(f):
            if int(row['episode']) == 0:
                groups.setdefault((row['case_id'], row['seed']), []).append(row)
    results = []
    for (case, seed), rows in groups.items():
        a = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]
             if k not in ('case_id',)}
        t = a['time_s']
        pitch, roll = a['pitch_rad'], a['roll_rad']
        obs = np.zeros((len(t), 27), np.float32)
        # Euler derivatives approximate body rates; yaw is measured directly.
        obs[:, 0] = np.gradient(roll, t) * .25
        obs[:, 1] = np.gradient(pitch, t) * .25
        obs[:, 2] = a['yaw_rate_rad_s'] * .25
        obs[:, 3] = np.sin(pitch)
        obs[:, 4] = -np.sin(roll) * np.cos(pitch)
        obs[:, 5] = -np.cos(roll) * np.cos(pitch)
        for col, key, scale in ((6, 'speed_cmd_m_s', 2), (7, 'yaw_cmd_rad_s', .25),
                (8, 'height_cmd_m', 5), (17, 'speed_m_s', 2),
                (18, 'lateral_speed_m_s', 2), (19, 'left_wheel_rad_s', .05),
                (20, 'right_wheel_rad_s', .05)):
            obs[:, col] = a[key] * scale
        for side, offset in (('left', 0), ('right', 1)):
            obs[:, 9 + offset] = a[side + '_angle_rad']
            obs[:, 11 + offset] = np.gradient(a[side + '_angle_rad'], t) * .05
            obs[:, 13 + offset] = a[side + '_length_m'] * 5
            obs[:, 15 + offset] = np.gradient(a[side + '_length_m'], t) * .25
            obs[:, 21 + offset * 3] = a[side + '_angle_ref_rad'] / .2
            obs[:, 22 + offset * 3] = (a[side + '_length_ref_m'] - .27) / .09
            obs[:, 23 + offset * 3] = a[side + '_wheel_ref_rad_s'] / 15
        indices = np.flatnonzero(t >= t[-1] - 3)[::5]
        if exact is not None:
            eid = int(case) * 3 + list(exact['seeds']).index(int(seed))
            obs = exact['observations'][:len(t), eid]
        jacobians = []
        for i in indices:
            x = torch.tensor(obs[i], requires_grad=True)
            j = torch.autograd.functional.jacobian(actor, x).detach().numpy()
            jacobians.append(j[[2, 5]])
        j = np.array(jacobians)
        previous = j[:, :, [23, 26]]
        wheel = j[:, :, [19, 20]] * .75
        eig = np.linalg.eigvals(previous)
        delta_obs = obs[indices] - obs[np.maximum(indices - 1, 0)]
        contributions = j * delta_obs[:, None, :] * 15
        # Compare reconstructed inference with the *next* applied wheel action.
        predicted = actor(torch.tensor(obs[:-1])).detach().numpy()[:, [2, 5]] * 15
        actual = obs[1:, [23, 26]] * 15
        mask = t[:-1] >= t[-1] - 3
        results.append({'case': case, 'seed': int(seed),
            'reconstructed_next_reference_mae_rad_s': np.abs(predicted[mask] - actual[mask]).mean(0).tolist(),
            'previous_wheel_action_jacobian_mean': previous.mean(0).tolist(),
            'previous_action_spectral_radius_p95': float(np.quantile(np.abs(eig).max(1), .95)),
            'wheel_speed_to_reference_jacobian_mean': wheel.mean(0).tolist(),
            'yaw_rate_to_wheel_reference_gain_mean': (j[:, :, 2] * 3.75).mean(0).tolist(),
            'input_delta_contribution_rms_rad_s': np.sqrt(np.square(contributions).mean(0)).tolist(),
            'wheel_reference_delta_rms_rad_s': np.sqrt(np.square(np.diff(actual[mask], axis=0)).mean(0)).tolist()})
    report = {'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'method': ('autograd actor Jacobian at exact recorded inference observations' if exact is not None else
                   'autograd actor Jacobian at approximately reconstructed steady trace observations'),
        'actor_inference_parity_max': parity_max,
        'limitations': ('This is not the combined actor/PhysX stability Jacobian.' if exact is not None else
                       'Euler-derived body xy rates and finite-difference virtual leg velocities are approximate. This is not the combined actor/PhysX stability Jacobian.'),
        'trials': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
