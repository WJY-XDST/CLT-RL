"""Read actual saved PPO actions; no rollout, reward change or optimizer step."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import torch

from joint_pd_continuation import ROOT, sha
from wheel_legged_gym_isaaclab.reward_signals import wheel_action_overshoot_penalty

SCRIPT = Path(__file__).resolve()
COEFFICIENTS = (0., -.025, -.05, -.1)


def audit(training, destination):
    training, destination = Path(training).resolve(), Path(destination).resolve()
    final_path = training / 'final_checkpoint_verification_readonly_20261019.json'
    final = json.loads(final_path.read_text())
    raw = training / 'native_learning_signal'
    complete = json.loads((raw / 'completed_external.json').read_text())
    if (destination.exists() or not destination.is_relative_to(ROOT / 'mine_model/results')
            or not final['verified'] or final['exact_actual_update_delta'] != 200
            or complete['actual_observed_updates'] != 200 or not complete['verified_outside_Kit']
            or final['checkpoint_sha256'] != sha(final['checkpoint'])):
        raise ValueError('Require a completed original200 block and fresh diagnostic output')
    import yaml
    cfg_path = Path(final['checkpoint']).parent / 'params/env.yaml'
    # Isaac's YAML contains tuple tags; only read scalar fields, without
    # constructing any Python objects from those tags.
    cfg = yaml.load(cfg_path.read_text(), Loader=yaml.BaseLoader)
    if float(cfg['rewards']['wheel_action_saturation']) != 0.:
        raise ValueError('This counterfactual compares the currently disabled existing wheel cost')
    pins = {str(p): sha(p) for p in (SCRIPT, final_path, cfg_path, raw / 'completed_external.json',
        Path(wheel_action_overshoot_penalty.__code__.co_filename))}
    payloads, ids, flat_count, unchanged, positive_signal = [], None, 0, True, 0
    rng = torch.random.get_rng_state().clone()
    dt = float(cfg['sim']['dt']) * int(cfg['decimation'])
    cap = float(cfg['rewards']['wheel_action_saturation_penalty_clip'])
    for index in range(1, 201):
        path, ap = raw / f'rollout_{index:04d}.npz', raw / f'audit_update_{index:04d}.json'
        record = json.loads(ap.read_text())
        current_ids = np.asarray(record['sampled_environment_ids'])
        if (record['update'] != index or record['total_environments'] != 4096
                or any(record['all_environment_alignment_max_errors'].values())
                or not record['probe_RNG_unchanged'] or record['added_loss_or_backward']
                or sha(ap) != complete['records_sha256'][str(ap)]):
            raise ValueError('Keep actual native alignment, original RNG and PPO accounting')
        if ids is None:
            ids = current_ids
        if not np.array_equal(ids, current_ids):
            raise ValueError('Report any change in sampled native environment identities')
        with np.load(path, allow_pickle=False) as z:
            a, mu, sigma = [z[k].copy().reshape(-1, 6) for k in ('actions', 'mu', 'sigma')]
            native = z['native_raw_actions'].copy().reshape(-1, 6)
            command = z['native_commands'].copy().reshape(-1, 3)
            native_reward = z['native_reward'].copy().reshape(-1)
        if not np.array_equal(a, native) or any(not np.isfinite(x).all() for x in (a, mu, sigma)) or np.any(sigma <= 0):
            raise ValueError('Use actual finite original sampled Gaussian actions')
        acts, std = torch.from_numpy(a), torch.from_numpy(sigma)
        toward, away = acts.clone(), acts.clone()
        wheels = acts[:, [2, 5]]
        spread = std[:, [2, 5]]
        deep = (np.abs(mu[:, [2, 5]]) - 1.) > 6. * sigma[:, [2, 5]]
        flat = torch.from_numpy(deep) & (wheels.abs() > 1. + spread) & (wheels.abs() < 10. - spread)
        direction = torch.sign(wheels) * spread * flat
        toward[:, [2, 5]], away[:, [2, 5]] = wheels - direction, wheels + direction
        unchanged &= torch.equal(toward.clamp(-1., 1.), away.clamp(-1., 1.))
        flat_count += int(flat.sum())
        cost, near_cost, far_cost = [wheel_action_overshoot_penalty(x).numpy() for x in (acts, toward, away)]
        positive_signal += int(np.count_nonzero((far_cost > near_cost) & flat.any(-1).numpy()))
        payloads.append(dict(cost=cost, near_cost=near_cost, far_cost=far_cost,
            command=command, native_reward=native_reward, clipped=(np.abs(a[:, [2, 5]]) >= 1.),
            deep=deep, flat=flat.numpy(), wheel_std=sigma[:, [2, 5]], raw_abs=np.abs(a[:, [2, 5]])))
        pins.update({str(path): sha(path), str(ap): sha(ap)})
    if not unchanged or not torch.equal(rng, torch.random.get_rng_state()):
        raise ValueError('A deterministic reward comparison must not alter execution or draw RNG')
    all_data = {key: np.concatenate([v[key] for v in payloads]) for key in payloads[0]}
    count = len(all_data['cost'])
    cmd = all_data['command']
    masks = dict(all=np.ones(count, dtype=bool),
        zero_command=(abs(cmd[:, 0]) < .01) & (abs(cmd[:, 1]) < .01),
        moving=abs(cmd[:, 0]) > .05, either_wheel_clipped=all_data['clipped'].any(-1),
        either_wheel_mean_over6std_beyond_clamp=all_data['deep'].any(-1))
    candidates = {}
    for coefficient in COEFFICIENTS:
        delta = np.maximum(coefficient * all_data['cost'], -cap) * dt
        sensitivity = np.maximum(coefficient * all_data['near_cost'], -cap) * dt - np.maximum(coefficient * all_data['far_cost'], -cap) * dt
        candidates[str(coefficient)] = dict(
            groups={k: dict(frames=int(mask.sum()), mean_added_reward_per_step=float(delta[mask].mean()) if mask.any() else None,
                maximum_cost_per_step=float(-delta[mask].min()) if mask.any() else None,
                mean_toward_vs_away_reward_difference=float(sensitivity[mask].mean()) if mask.any() else None)
                for k, mask in masks.items()}, per_term_clip_fraction=float(np.mean(coefficient * all_data['cost'] <= -cap)))
    if any(sha(p) != digest for p, digest in pins.items()):
        raise ValueError('Preserve completed original PPO data and reward implementation')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), actual_updates=final['actual_ppo_updates'],
        checkpoint_sha256=final['checkpoint_sha256'], original_actual_update_count=200,
        sampled_native_environment_ids=ids.tolist(), sampled_transition_count=count,
        sampled_wheel_clipped_coordinate_fraction=float(all_data['clipped'].mean()),
        sampled_wheel_mean_over6std_beyond_clamp_fraction=float(all_data['deep'].mean()),
        max_recorded_raw_wheel_action_abs=float(all_data['raw_abs'].max()),
        recorded_wheel_std_range=[float(all_data['wheel_std'].min()), float(all_data['wheel_std'].max())],
        fully_clipped_coordinates_with_bitwise_identical_execution_under_1std_perturbations=flat_count,
        frames_with_distinguishable_existing_raw_cost_but_identical_execution=positive_signal,
        original_current_wheel_cost_coefficient=0., recorded_dt=dt, counterfactual_coefficients=candidates,
        source_pins=pins, native_execution_bitwise_unchanged_for_paired_perturbations=unchanged,
        RNG_unchanged=True, optimizer_executed=False, simulator_started=False, training_modified=False,
        candidate_coefficient_selected=False, policy_accepted=False, acceptance_changed=False,
        limitation='32 recorded environments per update, not all4096. Immediate reward sensitivity only; '
            'not changed advantages, a physical intervention, convergence proof, or proof of a global local optimum.')
    with destination.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.training_output, args.output)
    print(json.dumps({k: result[k] for k in ('sampled_transition_count',
        'sampled_wheel_clipped_coordinate_fraction', 'sampled_wheel_mean_over6std_beyond_clamp_fraction',
        'fully_clipped_coordinates_with_bitwise_identical_execution_under_1std_perturbations',
        'frames_with_distinguishable_existing_raw_cost_but_identical_execution', 'policy_accepted')}))
