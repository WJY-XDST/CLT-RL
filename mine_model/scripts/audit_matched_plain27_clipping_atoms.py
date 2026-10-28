"""Compare Gaussian clipping atoms on identical recorded source observations.

The probability of both motors remaining at their deterministic clipping
boundaries is evaluated analytically. No new action samples or dynamics run.
This separates changes in policy clipping from changes in visited states.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtr
import torch

from diagnose_height_response import actor
from joint_pd_continuation import ROOT, sha
from wheel_legged_gym_isaaclab import leg_control as mapping

SCRIPT = Path(__file__).resolve()


def audit(directory, through):
    directory = Path(directory).resolve()
    destination = directory / f'matched_clipping_atoms_prefix_{through:04d}_readonly.json'
    if directory.parent != ROOT / 'mine_model/results' or destination.exists():
        raise ValueError('Preserve one owned immutable diagnostic')
    comparison_path = directory / f'recorded_source_joint_targets_prefix_{through:04d}_readonly.json'
    geometry_path = directory / f'joint_target_geometric_exploration_prefix_{through:04d}_readonly.json'
    plan_path = directory / 'experiment_plan.json'
    comparison, geometry, plan = [json.loads(path.read_text())
        for path in (comparison_path, geometry_path, plan_path)]
    if (comparison['committed_prefix'] != through or geometry['committed_updates'] != through
            or comparison['source_observation_input_columns_changed']
            or not comparison['all_original9_cases_and27_trials_retained']
            or comparison['source_actor_recorded_mean_parity_max_error'] > 1.e-4
            or comparison['source_recorded_FK_length_parity_max_error_m'] > 2.e-7
            or plan['policy_class'] != 'ActorCritic' or not plan['ordinary_full_policy_PPO']
            or plan['auxiliary_losses_enabled'] or plan['external_observation_size'] != 27):
        raise ValueError('Require the qualified original plain27 fixed-input comparison')
    cfg = plan['training_config']
    limits = np.asarray(geometry['USD_composed_joint_limits_rad'])
    offset = np.asarray(cfg['leg_joint_position_offset'])
    scale = np.abs(np.asarray(cfg['leg_joint_position_scale']))
    margin = cfg['leg_joint_target_limit_margin']
    if (np.any(offset - scale < limits[:, 0] + margin)
            or np.any(offset + scale > limits[:, 1] - margin)):
        raise ValueError('Additional USD clamps require a different atom calculation')
    replay = Path(comparison['fixed_input_source'])
    manifest = json.loads((replay / 'manifest.json').read_text())
    with np.load(replay / 'policy_inputs.npz', allow_pickle=False) as data:
        observations, recorded = [data[key][-150:].copy() for key in ('observations', 'means')]
    if observations.shape != (150, 27, 27) or recorded.shape != (150, 27, 6):
        raise ValueError('Require the original9 cases/3 seeds/last3s observations')
    pins = {**comparison['source_pins'], **{str(path): sha(path)
        for path in (SCRIPT, comparison_path, geometry_path, plan_path, replay / 'manifest.json')}}
    rng = torch.random.get_rng_state().clone()
    results, parity = [], None
    for label, path in (('source', comparison['source_checkpoint']),
                        ('candidate', comparison['candidate_checkpoint'])):
        saved = torch.load(path, map_location='cpu', weights_only=True)['model_state_dict']
        with torch.inference_mode():
            means = actor(torch.from_numpy(observations), saved).numpy().astype(np.float64)
        std = saved['std'].numpy().astype(np.float64)
        if std.shape != (6,) or np.any(std <= 0.) or not np.isfinite(means).all():
            raise ValueError('Require six positive independent Gaussian scales')
        if label == 'source':
            parity = float(abs(means - recorded).max())
            if parity > 1.e-4:
                raise ValueError('Source actor no longer reproduces the original recording')
        # A clipped Gaussian has probability mass at +/-1. Inside the open
        # interval it has a density, so equality to its deterministic mean has
        # probability zero. The six original action heads are independent.
        atom = np.where(means >= 1., ndtr((means - 1.) / std),
            np.where(means <= -1., ndtr((-1. - means) / std), 0.))
        motor_atoms = atom[..., mapping.LEG_ACTION_COLUMNS]
        both = motor_atoms[..., ::2] * motor_atoms[..., 1::2]
        cases = []
        for index, case in enumerate(manifest['cases']):
            selection = slice(index * 3, (index + 1) * 3)
            cases.append(dict(case=case, recorded_states=450,
                deterministic_mean_outside_motor_bounds_fraction=(
                    abs(means[:, selection, mapping.LEG_ACTION_COLUMNS]) > 1.).mean((0, 1)).tolist(),
                same_motor_boundary_atom_probability=motor_atoms[:, selection].mean((0, 1)).tolist(),
                both_motors_same_boundary_probability_by_leg=both[:, selection].mean((0, 1)).tolist()))
        results.append(dict(label=label, checkpoint=path, saved_std=std.tolist(), cases=cases))
    if not torch.equal(rng, torch.random.get_rng_state()) or any(sha(p) != h for p, h in pins.items()):
        raise ValueError('Read-only inputs or RNG changed')
    result = dict(observed_at=datetime.now().astimezone().isoformat(), committed_prefix=through,
        source_actual_updates=comparison['source_actual_updates'],
        candidate_actual_updates=comparison['candidate_actual_updates'],
        source_actor_recorded_mean_parity_max_error=parity, policies=results,
        formula='p(mu>=1)=Phi((mu-1)/sigma); p(mu<=-1)=Phi((-1-mu)/sigma); otherwise0; multiply the two independent motor atom probabilities per leg',
        original_USD_clamps_inactive_across_full_clipped_action_envelope=True,
        identical_recorded_observations_and_histories=True, source_pins=pins,
        RNG_draws_added=0, RNG_state_unchanged=True, simulator_started=False,
        optimizer_executed=False, training_modified=False, acceptance_changed=False, policy_accepted=False,
        limitations=[
            'Conditional probabilities on the older deterministic source trajectory, not the new candidate trajectory or full4096 training population.',
            'Both motors at the same clipping atoms implies unchanged motor targets and FK length. This is not the total numerical zero-FK probability or actual body-height exploration.',
            'These calculations do not establish local optimality, stability or independent acceptance.'])
    with destination.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    return destination, result


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('directory', type=Path)
    cli.add_argument('--through', type=int, required=True)
    args = cli.parse_args()
    torch.set_num_threads(1)
    path, result = audit(args.directory, args.through)
    print(json.dumps(dict(report=str(path), committed_prefix=result['committed_prefix'],
        RNG_draws_added=0, training_modified=False)))
