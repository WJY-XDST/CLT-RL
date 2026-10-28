"""Read-only FK and clipping diagnostics on completed signed-turn replays.

The hypothetical unclipped angles are never applied. This does not install IK,
change an action range, run simulation or select a checkpoint.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch

from joint_pd_continuation import ROOT, sha
from wheel_legged_gym_isaaclab import five_bar_vmc as geometry
from wheel_legged_gym_isaaclab import leg_control as mapping

SCRIPT = Path(__file__).resolve()


def audit(replay, limit_report, destination):
    replay, limit_report, destination = map(lambda p: Path(p).resolve(),
                                           (replay, limit_report, destination))
    if destination.exists() or not destination.is_relative_to(ROOT / 'mine_model/results'):
        raise ValueError('Preserve immutable owned diagnostic output')
    read = lambda p: json.loads(Path(p).read_text())
    manifest, complete, cfg = [read(replay / name) for name in
                              ('manifest.json', 'completed.json', 'resolved_env.json')]
    prior = read(limit_report)
    if (not complete['completed'] or complete['samples'] != 108000 or complete['resets']
            or len(manifest['cases']) != 36 or manifest['seeds'] != [683, 684, 685]
            or manifest['sampled_policy_actions'] or manifest['weights_updated']
            or cfg['leg_control_mode'] != mapping.DIRECT_JOINT_POSITION_MODE
            or cfg['leg_joint_names'] != prior['motor_order']
            or prior['original_native_motor_target_parity_max_error_rad'] > 1.e-7
            or prior['original_native_FK_reference_parity_max_error_m'] > 2.e-7):
        raise ValueError('Require the completed original108-case direct-joint recording and native limits')
    checkpoint = Path(manifest['checkpoint'])
    if sha(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Source checkpoint identity changed')
    with np.load(replay / 'policy_inputs.npz', allow_pickle=False) as data:
        means = data['means'].copy()
    if means.shape != (1000, 108, 6) or not np.isfinite(means).all():
        raise ValueError('Require all recorded policy means')
    trace = np.genfromtxt(replay / 'trace.csv', delimiter=',', names=True)
    frame = np.rint(trace['time_s'] / .02).astype(int) - 1
    ids = trace['case_id'].astype(int) * 3 + np.searchsorted(manifest['seeds'], trace['seed'].astype(int))
    seen = np.zeros((1000, 108), np.uint8)
    np.add.at(seen, (frame, ids), 1)
    if not np.all(seen == 1):
        raise ValueError('Raw recording identities are not one-to-one')
    refs = np.zeros((1000, 108, 2), np.float32)
    refs[frame, ids, 0], refs[frame, ids, 1] = trace['left_length_ref_m'], trace['right_length_ref_m']
    # time17.00s through20.00s, inclusive151 frames; no failure exclusion.
    raw = torch.from_numpy(means[849:, :, mapping.LEG_ACTION_COLUMNS].reshape(-1, 4))
    limits = raw.new_tensor(prior['USD_composed_joint_limits_rad']).expand(len(raw), -1, -1)
    rng = torch.random.get_rng_state().clone()
    target, _ = mapping.joint_position_targets(raw, cfg['leg_joint_position_offset'],
        cfg['leg_joint_position_scale'], limits, cfg['leg_joint_target_limit_margin'])
    hypothetical = raw.new_tensor(cfg['leg_joint_position_offset']) + raw * raw.new_tensor(cfg['leg_joint_position_scale'])
    length, angle, _, valid = geometry.five_bar_state(target[:, ::2], target[:, 1::2], **cfg['five_bar_geometry'])
    other_length, other_angle, _, other_valid = geometry.five_bar_state(
        hypothetical[:, ::2], hypothetical[:, 1::2], **cfg['five_bar_geometry'])
    parity = float(abs(length.reshape(151, 108, 2).numpy() - refs[849:]).max())
    if parity > 2.e-7 or not valid.all():
        raise ValueError('Exact original clipped FK does not reproduce all native references')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    std = saved['model_state_dict']['std'].numpy()[list(mapping.LEG_ACTION_COLUMNS)]
    if std.shape != (4,) or not np.isfinite(std).all() or np.any(std <= 0.):
        raise ValueError('Require four finite positive saved joint standard deviations')
    mu = raw.numpy().reshape(151, 108, 4).astype(np.float64)
    atoms = np.where(mu >= 1., ndtr((mu - 1.) / std),
                     np.where(mu <= -1., ndtr((-1. - mu) / std), 0.))
    length, angle, other_length, other_angle, other_valid = [
        a.numpy().reshape(151, 108, 2) for a in (length, angle, other_length, other_angle, other_valid)]
    q = target.numpy().reshape(151, 108, 4)
    hq = hypothetical.numpy().reshape(151, 108, 4)
    margin = cfg['leg_joint_target_limit_margin']
    unsafe_q = ((hq < limits[0, :, 0].numpy() + margin)
                | (hq > limits[0, :, 1].numpy() - margin))
    rows = []
    for cid, case in enumerate(manifest['cases']):
        for sid, seed in enumerate(manifest['seeds']):
            env = cid * 3 + sid
            good = bool(other_valid[:, env].all())
            rows.append(dict(case=case, seed=seed, steady_frames=151,
                raw_leg_mean=mu[:, env].mean(0).tolist(),
                raw_leg_abs_max=abs(mu[:, env]).max(0).tolist(),
                deterministic_head_clipping_fraction=(abs(mu[:, env]) > 1.).mean(0).tolist(),
                same_boundary_atom_probability_by_motor=atoms[:, env].mean(0).tolist(),
                both_motor_same_boundary_probability_by_leg=(atoms[:, env, ::2] * atoms[:, env, 1::2]).mean(0).tolist(),
                executed_joint_target_mean_rad=q[:, env].mean(0).tolist(),
                executed_reference_length_mean_m=length[:, env].mean(0).tolist(),
                executed_reference_angle_mean_rad=angle[:, env].mean(0).tolist(),
                hypothetical_joint_target_mean_rad=hq[:, env].mean(0).tolist(),
                hypothetical_joint_outside_USD_margin_fraction=unsafe_q[:, env].mean(0).tolist(),
                hypothetical_invalid_FK_fraction=(~other_valid[:, env]).mean(0).tolist(),
                hypothetical_length_mean_m=other_length[:, env].mean(0).tolist() if good else None,
                hypothetical_length_delta_mean_mm=(1000 * (other_length[:, env] - length[:, env]).mean(0)).tolist() if good else None,
                hypothetical_angle_delta_mean_rad=(other_angle[:, env] - angle[:, env]).mean(0).tolist() if good else None))
    if not torch.equal(rng, torch.random.get_rng_state()):
        raise ValueError('Read-only diagnostics must not draw random samples')
    files = [SCRIPT, Path(mapping.__file__), Path(geometry.__file__), limit_report, checkpoint,
             *(replay / name for name in ('manifest.json', 'completed.json', 'resolved_env.json',
                                         'trace.csv', 'policy_inputs.npz'))]
    result = dict(observed_at=datetime.now().astimezone().isoformat(), checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), original108_trials=108, motor_order=cfg['leg_joint_names'],
        full151_frame_window_preserved=True, original_FK_reference_parity_max_error_m=parity,
        exact_original_motor_map_and_FK_reused=True, rows=rows,
        source_pins={str(p): sha(p) for p in files}, RNG_draws_added=0,
        simulator_started=False, optimizer_executed=False, training_modified=False,
        controller_modified=False, action_envelope_modified=False, policy_accepted=False,
        limitations=['Hypothetical unexecuted FK is a geometric diagnostic only; no contact, collision, motor torque or stability was evaluated.',
            'Per-motor boundary probability is not loss of all leg exploration; the other motor can still change virtual length and angle.',
            'USD joint limits do not prove clearance from the physical limit blocks or that a wider action envelope is safe.',
            'The original raw-action saturation penalty remains active; this report does not establish a reward bug or justify changing control.'])
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ('replay', 'limit-report', 'output'):
        cli.add_argument('--' + name, type=Path, required=True)
    args = cli.parse_args()
    result = audit(args.replay, args.limit_report, args.output)
    print(json.dumps(dict(trials=result['original108_trials'],
        FK_parity=result['original_FK_reference_parity_max_error_m'], training_modified=False)))
