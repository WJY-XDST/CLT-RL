"""Verify the frozen full27 transfer changed only the existing standing cost."""
import argparse
import json
from pathlib import Path

import numpy as np

from auto_train_mine import write_json
from joint_pd_continuation import sha


def differences(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return [prefix + '/keys']
        return [v for key in left for v in differences(left[key], right[key], prefix + '/' + key)]
    return [] if left == right else [prefix]


def qualify(previous, directory):
    previous, directory = Path(previous).resolve(), Path(directory).resolve()
    before, after = previous / 'replay', directory / 'native_reward_transfer'
    old_manifest = json.loads((before / 'manifest.json').read_text())
    manifest = json.loads((after / 'manifest.json').read_text())
    completed = json.loads((after / 'completed.json').read_text())
    if (manifest['checkpoint_sha256'] != old_manifest['checkpoint_sha256']
            or sha(manifest['checkpoint']) != manifest['checkpoint_sha256']
            or manifest['cases'] != old_manifest['cases'] or manifest['seeds'] != old_manifest['seeds']
            or not completed['completed'] or completed['samples'] != 27000 or completed['resets']):
        raise ValueError('Require the identical frozen checkpoint and full original27 replay')
    old_cfg = json.loads((before / 'resolved_env.json').read_text())
    cfg = json.loads((after / 'resolved_env.json').read_text())
    changes = differences(old_cfg, cfg)
    if (changes != ['/rewards/standing_velocity'] or old_cfg['rewards']['standing_velocity'] != -25.
            or cfg['rewards']['standing_velocity'] != -50.):
        raise ValueError('Native configuration changed outside the one declared reward coefficient: ' + str(changes))
    old = np.genfromtxt(before / 'trace.csv', delimiter=',', names=True)
    new = np.genfromtxt(after / 'trace.csv', delimiter=',', names=True)
    if old.dtype.names != new.dtype.names or len(old) != 27000 or len(new) != 27000:
        raise ValueError('Incomplete frozen native trace')
    excluded = {'reward_total', 'reward_standing_velocity'}
    identical = [key for key in old.dtype.names if key not in excluded]
    if any(not np.array_equal(old[key], new[key]) for key in identical):
        raise ValueError('Frozen policy physical output or another reward changed')
    with np.load(before / 'policy_inputs.npz', allow_pickle=False) as a, np.load(after / 'policy_inputs.npz', allow_pickle=False) as b:
        if set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files):
            raise ValueError('Frozen original27 observations or deterministic actions changed')
    target_speed = np.array([case['speed'] for case in manifest['cases']])[new['case_id'].astype(int)]
    standing = (np.abs(new['speed_cmd_m_s']) < .01) & (np.abs(target_speed) < .01)
    signal = (new['speed_m_s']**2 + new['lateral_speed_m_s']**2) * standing
    dt = manifest['policy_dt']
    expected = np.maximum(-50. * signal, -cfg['rewards']['clip_single_reward']) * dt
    formula_error = float(np.max(np.abs(new['reward_standing_velocity'] - expected)))
    delta_error = float(np.max(np.abs(new['reward_total'] - old['reward_total']
        - new['reward_standing_velocity'] + old['reward_standing_velocity'])))
    if formula_error > 1.e-7 or delta_error > 1.e-7:
        raise ValueError('The native reward delta does not equal the independently reconstructed standing cost')
    pins = {str(replay / name):sha(replay / name) for replay in (before, after)
            for name in ('manifest.json', 'completed.json', 'resolved_env.json', 'trace.csv', 'policy_inputs.npz')}
    result = dict(qualified=True, source_sha256=manifest['checkpoint_sha256'],
        source_checkpoint=manifest['checkpoint'], only_reward_change={'standing_velocity':[-25., -50.]},
        exact_native_config_changes=changes, full27_samples=27000, resets=0,
        physical_outputs_original27_observations_and_deterministic_actions_bitwise_equal=True,
        all_other_reward_terms_bitwise_equal=True,
        independent_standing_reward_formula_max_error=formula_error,
        total_reward_delta_equals_only_standing_term_max_error=delta_error,
        data_sha256=pins, qualifier_sha256=sha(__file__), PPO_updates_performed=0,
        policy_accepted=False, limitation='Frozen reward/physics equivalence qualification only. '
        'The source still fails height/drift acceptance; convergence requires the next actual200 PPO updates and independent replay.')
    write_json(directory / 'native_reward_transfer_qualification.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({k:v for k, v in qualify(args.previous, args.qualification).items() if k != 'data_sha256'}, indent=2))
